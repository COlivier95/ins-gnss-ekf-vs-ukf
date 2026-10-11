"""Autonomous coarse alignment: no information from the reference trajectory is used.

* roll / pitch: levelling from the mean specific force while the vehicle is at rest,
* heading:      first dual-antenna GNSS heading, inverted through the baseline geometry,
* position:     first GNSS fix moved from the antenna to the IMU with the lever arm,
* velocity:     first GNSS velocity.
"""
from __future__ import annotations

import numpy as np

from ..attitude.rotations import euler_to_dcm, euler_to_quat, wrap_angle
from ..geodesy.wgs84 import radii_of_curvature
from ..models.error_state import Nominal


def level(acc_mean):
    """Roll and pitch [rad] from the mean specific force at rest (FRD body frame: f = -C_bn g)."""
    fx, fy, fz = acc_mean
    roll = np.arctan2(-fy, -fz)
    pitch = np.arctan2(fx, np.hypot(fy, fz))
    return roll, pitch


def heading_from_baseline(psi_baseline, roll, pitch, baseline_b):
    """Vehicle yaw such that the azimuth of C_nb(roll, pitch, yaw) @ b_b equals the measured one."""
    yaw = psi_baseline - np.arctan2(baseline_b[1], baseline_b[0])
    for _ in range(5):                      # Newton iterations on a 1-D smooth function
        def az(y):
            b = euler_to_dcm(roll, pitch, y) @ baseline_b
            return np.arctan2(b[1], b[0])
        err = wrap_angle(psi_baseline - az(yaw))
        d = (az(yaw + 1e-6) - az(yaw - 1e-6)) / 2e-6
        yaw = yaw + err / d
    return float(wrap_angle(yaw))


def coarse_alignment(ds, lever_primary, lever_secondary, static_duration=5.0):
    """Return (nominal state, t_init, info) at the first GNSS epoch after the static window."""
    t0 = ds.imu.t[0]
    m = ds.imu.t < t0 + static_duration
    roll, pitch = level(ds.imu.acc[m].mean(axis=0))

    # first GNSS epoch with position, velocity and heading after the static window
    t_min = t0 + static_duration
    ip = int(np.searchsorted(ds.pos.t, t_min))
    ih = int(np.searchsorted(ds.hdt.t, ds.pos.t[ip]))
    while not np.isclose(ds.hdt.t[ih], ds.pos.t[ip], atol=1e-3):
        ip += 1
        ih = int(np.searchsorted(ds.hdt.t, ds.pos.t[ip]))
    t_init = float(ds.pos.t[ip])
    baseline = np.asarray(lever_secondary, float) - np.asarray(lever_primary, float)
    yaw = heading_from_baseline(ds.hdt.heading[ih], roll, pitch, baseline)

    q = euler_to_quat(roll, pitch, yaw)
    C = euler_to_dcm(roll, pitch, yaw)
    lat, lon, h = ds.pos.lla[ip]
    d = C @ np.asarray(lever_primary, float)                 # IMU -> antenna in NED
    r_n, r_e = radii_of_curvature(lat)
    lla_imu = (lat - d[0] / (r_n + h), lon - d[1] / ((r_e + h) * np.cos(lat)), h + d[2])
    iv = int(np.searchsorted(ds.vel.t, t_init))
    x = Nominal(float(lla_imu[0]), float(lla_imu[1]), float(lla_imu[2]), ds.vel.vel[iv].copy(), q)
    info = {"roll": roll, "pitch": pitch, "yaw": yaw, "pos_std": ds.pos.std[ip].copy()}
    return x, t_init, info
