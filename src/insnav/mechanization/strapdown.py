"""Strapdown inertial navigation equations in the local-level NED frame (WGS84, curvilinear position).

State of the nominal navigation solution:
    lat, lon [rad], h [m] (ellipsoidal) | v_n = [vN, vE, vD] [m/s] | q_nb (body FRD -> NED, scalar-first)

Continuous-time model (Groves, *Principles of GNSS, Inertial and Multisensor Integrated Navigation*, ch. 5):
    q_nb_dot = 1/2 q_nb ⊗ [0, w_ib^b]  -  1/2 [0, w_in^n] ⊗ q_nb          with w_in = w_ie + w_en
    v_dot    = C_nb f^b + g^n - (2 w_ie^n + w_en^n) x v_n
    lat_dot  = vN / (R_N + h),   lon_dot = vE / ((R_E + h) cos lat),   h_dot = -vD

Discretisation (one IMU interval dt):
    * IMU rates are averaged over the interval (trapezoidal increments),
    * attitude: quaternion exponentials of the body increment (right) and of the frame rotation (left),
    * velocity: specific force rotated with the mid-interval attitude (both half rotations),
      Coriolis / transport terms from the previous epoch,
    * position: trapezoidal integration of the velocity.
At 200 Hz on a land vehicle the neglected coning / sculling terms are far below the IMU noise.

Implementation note
-------------------
The core (`_step`) is written component-wise ("structure of arrays"): every quantity is either a
Python float or a numpy array of shape (B,). The *same* code therefore propagates a single solution
with fast scalar arithmetic (ES-EKF, ~10x faster than small numpy arrays) or all the sigma points
of the unscented filter in one vectorised call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geodesy.wgs84 import A, E2, F, GAMMA_E, K_SOMIGLIANA, M_RATIO, OMEGA_IE

def _qexp_small(ax, ay, az):
    """Quaternion of a small rotation vector (|a| < 0.1 rad): 4th-order series, error < 1e-10.

    IMU increments over 5 ms are < 0.05 rad even at 10 rad/s; series avoid branching so the
    function works unchanged on scalars and arrays.
    """
    a2 = ax * ax + ay * ay + az * az
    w = 1.0 - a2 / 8.0 + a2 * a2 / 384.0
    k = 0.5 - a2 / 48.0 + a2 * a2 / 3840.0
    return w, k * ax, k * ay, k * az


def _qmul(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return (pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw)


def _qnormalize(q, sqrt):
    w, x, y, z = q
    n = sqrt(w * w + x * x + y * y + z * z)
    return w / n, x / n, y / n, z / n


def _qrot(q, vx, vy, vz):
    """C(q) @ v for a unit quaternion."""
    w, x, y, z = q
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def _radii(slat, sqrt):
    den = 1.0 - E2 * slat * slat
    sd = sqrt(den)
    return A * (1.0 - E2) / (den * sd), A / sd


def _gravity(slat, h, sqrt):
    s2 = slat * slat
    g0 = GAMMA_E * (1.0 + K_SOMIGLIANA * s2) / sqrt(1.0 - E2 * s2)
    return g0 * (1.0 - 2.0 / A * (1.0 + F + M_RATIO - 2.0 * F * s2) * h + 3.0 / (A * A) * h * h)


def _step(lat, lon, h, vn, ve, vd, q, wx, wy, wz, fx, fy, fz, dt, earth_rate=True, lib=np):
    """One strapdown interval. ``lib`` is ``math`` for Python floats or ``numpy`` for (B,) arrays."""
    sin, cos, sqrt = lib.sin, lib.cos, lib.sqrt
    slat, clat = sin(lat), cos(lat)
    r_n, r_e = _radii(slat, sqrt)

    # Earth rotation and transport rate in NED
    if earth_rate:
        wie_n, wie_d = OMEGA_IE * clat, -OMEGA_IE * slat
        wen_n = ve / (r_e + h)
        wen_e = -vn / (r_n + h)
        wen_d = -ve * slat / clat / (r_e + h)
    else:
        wie_n = wie_d = wen_n = wen_e = wen_d = 0.0
    win_n, win_e, win_d = wie_n + wen_n, wen_e, wie_d + wen_d

    # attitude: q_new = q(-w_in dt) ⊗ q ⊗ q(w_ib dt); mid-interval attitude with half increments
    q_body = _qexp_small(wx * dt, wy * dt, wz * dt)
    q_nav = _qexp_small(-win_n * dt, -win_e * dt, -win_d * dt)
    q_new = _qnormalize(_qmul(_qmul(q_nav, q), q_body), sqrt)
    q_mid = _qmul(_qmul(_qexp_small(-0.5 * win_n * dt, -0.5 * win_e * dt, -0.5 * win_d * dt), q),
                  _qexp_small(0.5 * wx * dt, 0.5 * wy * dt, 0.5 * wz * dt))

    # velocity
    fn, fe, fd = _qrot(q_mid, fx, fy, fz)
    cn, ce, cd = 2.0 * wie_n + wen_n, wen_e, 2.0 * wie_d + wen_d        # (2 w_ie + w_en)
    cor_n = ce * vd - cd * ve
    cor_e = cd * vn - cn * vd
    cor_d = cn * ve - ce * vn
    g = _gravity(slat, h, sqrt)
    vn2 = vn + (fn - cor_n) * dt
    ve2 = ve + (fe - cor_e) * dt
    vd2 = vd + (fd + g - cor_d) * dt

    # position (trapezoidal)
    van, vae, vad = 0.5 * (vn + vn2), 0.5 * (ve + ve2), 0.5 * (vd + vd2)
    h2 = h - vad * dt
    hm = 0.5 * (h + h2)
    lat2 = lat + van / (r_n + hm) * dt
    latm = 0.5 * (lat + lat2)
    _, r_em = _radii(sin(latm), sqrt)
    lon2 = lon + vae / ((r_em + hm) * cos(latm)) * dt
    return lat2, lon2, h2, vn2, ve2, vd2, q_new


# --- public API ---------------------------------------------------------------------------------------
@dataclass
class NavState:
    """Nominal navigation state. Fields are floats (single solution) or arrays of shape (B,)."""
    lat: float | np.ndarray
    lon: float | np.ndarray
    h: float | np.ndarray
    v: np.ndarray        # (3,) or (B, 3) NED velocity
    q: np.ndarray        # (4,) or (B, 4) q_nb

    @classmethod
    def from_arrays(cls, lla, v, q):
        lla = np.asarray(lla, float)
        return cls(lla[..., 0], lla[..., 1], lla[..., 2], np.asarray(v, float), np.asarray(q, float))

    def copy(self):
        return NavState(np.copy(self.lat), np.copy(self.lon), np.copy(self.h), self.v.copy(), self.q.copy())

    @property
    def lla(self):
        return np.stack([self.lat, self.lon, self.h], axis=-1)


def strapdown_step(x: NavState, w_ib, f_ib, dt, earth_rate: bool = True) -> NavState:
    """Propagate the navigation state over one IMU interval.

    ``w_ib`` / ``f_ib``: bias-corrected body angular rate [rad/s] and specific force [m/s^2] over the
    interval, shape (3,) for a single state or (B, 3) for a batch.
    ``earth_rate=False`` drops the Earth rotation and transport-rate terms (only to show their effect).
    """
    v, q = x.v, x.q
    w = np.asarray(w_ib, float)
    f = np.asarray(f_ib, float)
    if q.ndim == 1:      # scalar fast path: plain Python floats
        args = [float(x.lat), float(x.lon), float(x.h), *map(float, v), tuple(map(float, q)),
                *map(float, w), *map(float, f)]
    else:
        args = [x.lat, x.lon, x.h, v[:, 0], v[:, 1], v[:, 2], (q[:, 0], q[:, 1], q[:, 2], q[:, 3]),
                w[..., 0], w[..., 1], w[..., 2], f[..., 0], f[..., 1], f[..., 2]]
    lat, lon, h, vn, ve, vd, qn = _step(*args, dt, earth_rate, math if q.ndim == 1 else np)
    if q.ndim == 1:
        return NavState(lat, lon, h, np.array([vn, ve, vd]), np.array(qn))
    return NavState(lat, lon, h, np.stack([vn, ve, vd], axis=-1), np.stack(qn, axis=-1))


def run_free_inertial(x0: NavState, t, gyr, acc, gyro_bias=None, accel_bias=None, earth_rate=True):
    """Pure inertial navigation (no aiding) of a single solution over the IMU samples.

    Returns (lla (N, 3), v (N, 3), q (N, 4)).
    """
    n = len(t)
    bg = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, float)
    ba = np.zeros(3) if accel_bias is None else np.asarray(accel_bias, float)
    w_avg = 0.5 * (gyr[:-1] + gyr[1:]) - bg          # trapezoidal rates over each interval
    f_avg = 0.5 * (acc[:-1] + acc[1:]) - ba
    dts = np.diff(t)
    out_lla, out_v, out_q = np.empty((n, 3)), np.empty((n, 3)), np.empty((n, 4))
    lat, lon, h = float(x0.lat), float(x0.lon), float(x0.h)
    vn, ve, vd = map(float, x0.v)
    q = tuple(map(float, x0.q))
    out_lla[0], out_v[0], out_q[0] = (lat, lon, h), (vn, ve, vd), q
    W, Fm, D = w_avg.tolist(), f_avg.tolist(), dts.tolist()
    for k in range(1, n):
        wx, wy, wz = W[k - 1]
        fx, fy, fz = Fm[k - 1]
        lat, lon, h, vn, ve, vd, q = _step(lat, lon, h, vn, ve, vd, q, wx, wy, wz, fx, fy, fz,
                                           D[k - 1], earth_rate, math)
        out_lla[k] = lat, lon, h
        out_v[k] = vn, ve, vd
        out_q[k] = q
    return out_lla, out_v, out_q
