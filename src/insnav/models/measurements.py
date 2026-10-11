"""GNSS measurement models (loosely coupled), shared by the ES-EKF and the ES-UKF.

Each model provides:
  * ``predict(x, w_b)``   nonlinear prediction of the measurement from a nominal state (used by both filters;
                          the UKF evaluates it on every sigma point),
  * ``residual(z, zp)``   measurement minus prediction, in a vector space (meters, m/s, wrapped radians),
  * ``jacobian(x, w_b)``  H = d residual / d dx, for the ES-EKF,
  * ``covariance(std)``   measurement noise R from the receiver-reported 1-sigma, with a floor.

``w_b`` is the bias-corrected body angular rate (needed by the velocity lever-arm term).
"""
from __future__ import annotations

import numpy as np

from ..attitude.rotations import skew, wrap_angle
from ..geodesy.wgs84 import radii_of_curvature
from .error_state import IBG, IGP, IHB, ISG, ITH, IR, IV, N_STATES, Nominal


class GnssPosition:
    """GNSS position = antenna position + correlated error gp (state) + white noise.

    lla_meas = lla_imu (+) (C_nb l_b + gp). Residual in NED meters.
    """
    name = "pos"
    dim = 3

    def __init__(self, lever_arm, std_floor=0.02, std_scale=1.0):
        self.l = np.asarray(lever_arm, float)
        self.floor = std_floor
        self.scale = np.asarray(std_scale, float)   # scalar or per-axis

    def predict(self, x: Nominal, w_b=None) -> np.ndarray:
        d = x.C_nb @ self.l + x.gp
        r_n, r_e = radii_of_curvature(x.lat)
        return np.array([x.lat + d[0] / (r_n + x.h),
                         x.lon + d[1] / ((r_e + x.h) * np.cos(x.lat)),
                         x.h - d[2]])

    @staticmethod
    def residual(z, zp):
        r_n, r_e = radii_of_curvature(zp[0])
        return np.array([(z[0] - zp[0]) * (r_n + zp[2]),
                         (z[1] - zp[1]) * (r_e + zp[2]) * np.cos(zp[0]),
                         -(z[2] - zp[2])])

    def jacobian(self, x: Nominal, w_b=None) -> np.ndarray:
        H = np.zeros((3, N_STATES))
        H[:, IR] = np.eye(3)
        H[:, ITH] = -skew(x.C_nb @ self.l)          # (I + [dθ]x) C l  =>  -[C l]x dθ
        H[:, IGP] = np.eye(3)
        return H

    def covariance(self, std):
        s = np.maximum(np.asarray(std) * self.scale, self.floor)
        return np.diag(s**2)


class GnssVelocity:
    """NED velocity of the GNSS antenna: v_ant = v + C_nb (w_b x l_b) (Earth-rate term < 1e-4 m/s neglected)."""
    name = "vel"
    dim = 3

    def __init__(self, lever_arm, std_floor=0.03, std_scale=1.0):
        self.l = np.asarray(lever_arm, float)
        self.floor = std_floor
        self.scale = np.asarray(std_scale, float)   # scalar or per-axis

    def predict(self, x: Nominal, w_b) -> np.ndarray:
        return x.v + x.C_nb @ np.cross(w_b, self.l)

    @staticmethod
    def residual(z, zp):
        return np.asarray(z) - zp

    def jacobian(self, x: Nominal, w_b) -> np.ndarray:
        C = x.C_nb
        H = np.zeros((3, N_STATES))
        H[:, IV] = np.eye(3)
        H[:, ITH] = -skew(C @ np.cross(w_b, self.l))
        # corrected rate w_b = (w_meas - bg)(1 - sg):  d w_b / d bg = -(1 - sg),  d w_b / d sg = -(w_meas - bg)
        H[:, IBG] = C @ skew(self.l) * (1.0 - x.sg)
        H[:, ISG] = C @ skew(self.l) * (w_b / (1.0 - x.sg))
        return H

    def covariance(self, std):
        s = np.maximum(np.asarray(std) * self.scale, self.floor)
        return np.diag(s**2)


class DualAntennaHeading:
    """Azimuth of the dual-antenna baseline b_n = C_nb b_b (b_b = secondary - primary lever arm).

    Using the full baseline geometry handles the +90 deg mounting offset and the roll/pitch coupling
    without any special case. The measured azimuth also contains the (estimated) misalignment hb of the
    hand-placed baseline:  psi_meas = azimuth(C_nb b_b) + hb.
    """
    name = "hdg"
    dim = 1

    def __init__(self, lever_primary, lever_secondary, std_floor_deg=0.2, std_scale=1.0):
        self.b = np.asarray(lever_secondary, float) - np.asarray(lever_primary, float)
        self.floor = np.radians(std_floor_deg)
        self.scale = np.asarray(std_scale, float)   # scalar or per-axis

    def predict(self, x: Nominal, w_b=None) -> np.ndarray:
        b_n = x.C_nb @ self.b
        return np.array([np.arctan2(b_n[1], b_n[0]) + x.hb[0]])

    @staticmethod
    def residual(z, zp):
        return wrap_angle(np.asarray(z) - zp)

    def jacobian(self, x: Nominal, w_b=None) -> np.ndarray:
        b_n = x.C_nb @ self.b
        n2 = b_n[0] ** 2 + b_n[1] ** 2
        dpsi_db = np.array([-b_n[1], b_n[0], 0.0]) / n2
        H = np.zeros((1, N_STATES))
        H[0, ITH] = dpsi_db @ (-skew(b_n))           # b_true = b_n + dθ x b_n
        H[0, IHB] = 1.0
        return H

    def covariance(self, std):
        s = np.maximum(np.asarray(std) * self.scale, self.floor)
        return np.diag(np.atleast_1d(s) ** 2)
