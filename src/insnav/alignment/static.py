"""Static-period detection and gyro bias estimation from standstills (used for validation only)."""
from __future__ import annotations

import numpy as np

from ..attitude.rotations import euler_to_dcm
from ..geodesy.wgs84 import OMEGA_IE


def static_periods(t_imu, t_vel, vel_ned, speed_max=0.1, min_duration=20.0, trim=2.0):
    """Index ranges [(i0, i1), ...] of the IMU samples where the vehicle is at rest.

    The horizontal GNSS speed (smoothed over 2 s) is compared to ``speed_max``; ``trim`` seconds are
    removed at both ends of each period to discard the stop/start transients.
    """
    spd = np.linalg.norm(vel_ned[:, :2], axis=1)
    spd = np.convolve(spd, np.ones(10) / 10, mode="same")
    still = np.interp(t_imu, t_vel, spd) < speed_max
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
    out = []
    for a, b in zip(edges[::2], edges[1::2] - 1):
        a2, b2 = np.searchsorted(t_imu, t_imu[a] + trim), np.searchsorted(t_imu, t_imu[b] - trim)
        if t_imu[b2 - 1] - t_imu[a2] >= min_duration - 2 * trim:
            out.append((a2, b2))
    return out


def gyro_bias_from_static(imu, ref, periods):
    """Gyro bias per static period: mean rate minus the Earth rotation seen in the body frame.

    Uses the reference attitude, so it is an *independent check* of the filters' bias estimates,
    never an input to them. Returns an array (n_periods, 3) [rad/s].
    """
    out = []
    for a, b in periods:
        t = imu.t[a:b]
        eul = [np.interp(t, ref.t, np.unwrap(ref.euler[:, i])).mean() for i in range(3)]
        lat = np.interp(t.mean(), ref.t, ref.lla[:, 0])
        w_ie_b = euler_to_dcm(*eul).T @ (OMEGA_IE * np.array([np.cos(lat), 0.0, -np.sin(lat)]))
        out.append(imu.gyr[a:b].mean(axis=0) - w_ie_b)
    return np.array(out)
