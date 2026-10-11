"""22-state error model of the strapdown INS, shared by the ES-EKF and the ES-UKF.

Error state  dx = [ dr (3), dv (3), dtheta (3), dba (3), dbg (3), dsg (3), dgp (3) ]
  dr      position error in NED [m]                      r_true = r_est + dr
  dv      velocity error in NED [m/s]                    v_true = v_est + dv
  dtheta  attitude error, small rotation in NED [rad]    C_nb_true = (I + [dtheta]x) C_nb_est
                                                         q_true    = q(dtheta) ⊗ q_est
  dba     accelerometer bias error [m/s^2]
  dbg     gyro bias error [rad/s]
  dsg     gyro scale-factor error [-]
  dgp     time-correlated GNSS position error, NED [m]   (colored measurement noise, see below)
  dhb     dual-antenna heading error [rad]: baseline misalignment + slowly varying multipath error

Sensor models
  f_meas = f_true + ba + n_a
  w_meas = (1 + sg) * w_true + bg + n_g          (element-wise scale factor)
  =>  w_true ~ (w_meas - bg) * (1 - sg)
  GNSS position = antenna position + gp + white noise, gp first-order Gauss-Markov (RTX errors are
  correlated over 5-10 s; treating them as white makes the filter overconfident).

Linearised dynamics  dx_dot = F dx + G n:
  dr_dot     = dv
  dv_dot     = -[f_n]x dtheta - C dba - (2 w_ie + w_en) x dv + (2 g / R) dD e_D - C n_a
  dtheta_dot = -w_in x dtheta - C dbg - C diag(w_b) dsg - C n_g
  dba_dot    = w_ba ;   dbg_dot = w_bg ;   dsg_dot = w_sg      (random walks: the nominal values are
                                                               constant between updates, so the error
                                                               model must not decay either)
  dgp_dot    = -dgp / tau_gp + w_gp                            (the nominal gp decays the same way:
                                                               see propagate_nominal_aux)
  dhb_dot    = w_hb                                            (random walk)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attitude.rotations import quat_conj, quat_mult, quat_normalize, quat_to_dcm, quat_to_rotvec, rotvec_to_quat, skew
from ..geodesy.wgs84 import OMEGA_IE, normal_gravity, radii_of_curvature

N_STATES = 22
IR, IV, ITH = slice(0, 3), slice(3, 6), slice(6, 9)
IBA, IBG, ISG, IGP, IHB = slice(9, 12), slice(12, 15), slice(15, 18), slice(18, 21), slice(21, 22)
MG = 9.80665e-3
PPM = 1e-6


@dataclass
class NoiseParams:
    """Continuous-time noise of the error model (isotropic per sensor, horizontal/vertical for GNSS)."""
    vrw: float          # accelerometer velocity random walk [m/s/sqrt(s)]
    arw: float          # gyro angle random walk [rad/sqrt(s)]
    ba_sigma: float     # accelerometer bias instability [m/s^2] over ba_tau
    ba_tau: float       # [s]   -> random-walk PSD 2 sigma^2 / tau
    bg_sigma: float     # gyro bias instability [rad/s] over bg_tau
    bg_tau: float       # [s]
    sg_rw: float        # gyro scale-factor random walk [1/sqrt(s)]
    gp_sigma: np.ndarray  # GNSS correlated position error sigma N/E/D [m]
    gp_tau: np.ndarray    # GNSS correlated position error time constants N/E/D [s]
    hb_rw: float = 0.0    # dual-antenna heading error random walk [rad/sqrt(s)]

    @classmethod
    def from_config(cls, c: dict) -> "NoiseParams":
        """Config units: vrw [m/s/sqrt(h)], arw [deg/sqrt(h)], ba [mg], bg [deg/h], sg_rw [ppm/sqrt(h)],
        gnss_pos sigma [m] and tau [s] given as horizontal / vertical."""
        g = c["gnss_pos_error"]
        return cls(vrw=c["vrw"] / 60.0, arw=np.radians(c["arw"]) / 60.0,
                   ba_sigma=c["ba_sigma"] * MG, ba_tau=c["ba_tau"],
                   bg_sigma=np.radians(c["bg_sigma"]) / 3600.0, bg_tau=c["bg_tau"],
                   sg_rw=c["sg_rw"] * PPM / 60.0,
                   gp_sigma=np.array([g["sigma_h"], g["sigma_h"], g["sigma_v"]], float),
                   gp_tau=np.array([g["tau_h"], g["tau_h"], g["tau_v"]], float),
                   hb_rw=np.radians(c["heading_error_rw"]))

    def discrete_q(self, dt: float) -> np.ndarray:
        """Discrete process noise Qd ~ G Qc G^T dt. With isotropic IMU noise, C_nb drops out: Qd is diagonal."""
        q = np.zeros(N_STATES)
        q[IV] = self.vrw**2 * dt
        q[ITH] = self.arw**2 * dt
        q[IBA] = 2.0 * self.ba_sigma**2 / self.ba_tau * dt
        q[IBG] = 2.0 * self.bg_sigma**2 / self.bg_tau * dt
        q[ISG] = self.sg_rw**2 * dt
        q[IGP] = 2.0 * self.gp_sigma**2 / self.gp_tau * dt
        q[IHB] = self.hb_rw**2 * dt
        return np.diag(q)


@dataclass
class Nominal:
    """Nominal (total) state estimate."""
    lat: float
    lon: float
    h: float
    v: np.ndarray                                   # NED [m/s]
    q: np.ndarray                                   # q_nb
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hb: np.ndarray = field(default_factory=lambda: np.zeros(1))

    def copy(self) -> "Nominal":
        return Nominal(self.lat, self.lon, self.h, self.v.copy(), self.q.copy(), self.ba.copy(),
                       self.bg.copy(), self.sg.copy(), self.gp.copy(), self.hb.copy())

    @property
    def C_nb(self) -> np.ndarray:
        return quat_to_dcm(self.q)

    def gyro(self, w_meas):
        """Bias and scale-factor corrected angular rate."""
        return (w_meas - self.bg) * (1.0 - self.sg)

    def accel(self, f_meas):
        return f_meas - self.ba


def inject(x: Nominal, dx: np.ndarray) -> Nominal:
    """Return x ⊕ dx (error injection into the nominal state)."""
    r_n, r_e = radii_of_curvature(x.lat)
    lat = x.lat + dx[0] / (r_n + x.h)
    lon = x.lon + dx[1] / ((r_e + x.h) * np.cos(x.lat))
    h = x.h - dx[2]
    q = quat_normalize(quat_mult(rotvec_to_quat(dx[ITH]), x.q))
    return Nominal(float(lat), float(lon), float(h), x.v + dx[IV], q, x.ba + dx[IBA], x.bg + dx[IBG],
                   x.sg + dx[ISG], x.gp + dx[IGP], x.hb + dx[IHB])


def difference(a: Nominal, b: Nominal) -> np.ndarray:
    """Error state dx such that b ⊕ dx = a (first order). Used by the UKF to average sigma points."""
    r_n, r_e = radii_of_curvature(b.lat)
    dx = np.empty(N_STATES)
    dx[0] = (a.lat - b.lat) * (r_n + b.h)
    dx[1] = (a.lon - b.lon) * (r_e + b.h) * np.cos(b.lat)
    dx[2] = -(a.h - b.h)
    dx[IV] = a.v - b.v
    dx[ITH] = quat_to_rotvec(quat_mult(a.q, quat_conj(b.q)))
    dx[IBA] = a.ba - b.ba
    dx[IBG] = a.bg - b.bg
    dx[ISG] = a.sg - b.sg
    dx[IGP] = a.gp - b.gp
    dx[IHB] = a.hb - b.hb
    return dx


def transition_matrix(x: Nominal, f_b: np.ndarray, w_b: np.ndarray, noise: NoiseParams, dt: float) -> np.ndarray:
    """First-order discrete transition Phi = I + F dt over one IMU interval.

    ``f_b`` / ``w_b`` are the corrected specific force and angular rate of the interval.
    """
    C = quat_to_dcm(x.q)
    r_n, r_e = radii_of_curvature(x.lat)
    slat, clat = np.sin(x.lat), np.cos(x.lat)
    w_ie = OMEGA_IE * np.array([clat, 0.0, -slat])
    w_en = np.array([x.v[1] / (r_e + x.h), -x.v[0] / (r_n + x.h), -x.v[1] * slat / clat / (r_e + x.h)])
    f_n = C @ f_b

    F = np.zeros((N_STATES, N_STATES))
    F[0, 3] = F[1, 4] = F[2, 5] = 1.0
    F[5, 2] = 2.0 * normal_gravity(x.lat, x.h) / np.sqrt(r_n * r_e)
    F[IV, IV] = -skew(2.0 * w_ie + w_en)
    F[IV, ITH] = -skew(f_n)
    F[IV, IBA] = -C
    F[ITH, ITH] = -skew(w_ie + w_en)
    F[ITH, IBG] = -C * (1.0 - x.sg)              # w_b = (w_meas - bg)(1 - sg)
    F[ITH, ISG] = -C * (w_b / (1.0 - x.sg))      # -C diag(w_meas - bg)
    F[IGP, IGP] = -np.diag(1.0 / noise.gp_tau)
    return np.eye(N_STATES) + F * dt


def propagate_nominal_aux(x: Nominal, noise: NoiseParams, dt: float) -> None:
    """Deterministic part of the augmented states in the nominal state (in place).

    The correlated GNSS error is a zero-mean Gauss-Markov process: its estimate decays with tau.
    Biases and scale factors are random walks: their estimates stay constant between updates.
    """
    x.gp = x.gp * np.exp(-dt / noise.gp_tau)


def reset_jacobian(dtheta: np.ndarray) -> np.ndarray:
    """Covariance reset after injection: the attitude error is re-expressed about the new estimate."""
    G = np.eye(N_STATES)
    G[ITH, ITH] = np.eye(3) - 0.5 * skew(dtheta)
    return G
