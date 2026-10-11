"""WGS84 geodesy: ellipsoid constants, curvilinear <-> ECEF <-> local NED conversions, normal gravity.

Conventions
-----------
* Geodetic coordinates are (latitude [rad], longitude [rad], ellipsoidal height [m]).
* The local navigation frame is North-East-Down (NED).
* All functions are vectorised over a leading axis when given (N, 3) arrays.
"""
from __future__ import annotations

import numpy as np

# --- WGS84 ellipsoid -------------------------------------------------------------------------
A = 6378137.0                    # semi-major axis [m]
F = 1.0 / 298.257223563          # flattening
B = A * (1.0 - F)                # semi-minor axis [m]
E2 = F * (2.0 - F)               # first eccentricity squared
OMEGA_IE = 7.292115e-5           # Earth rotation rate [rad/s]
GM = 3.986004418e14              # Earth gravitational constant [m^3/s^2]

# Somigliana normal gravity constants
GAMMA_E = 9.7803253359           # normal gravity at the equator [m/s^2]
K_SOMIGLIANA = 0.00193185265241  # (b*gamma_p - a*gamma_e) / (a*gamma_e)
M_RATIO = OMEGA_IE**2 * A**2 * B / GM


def radii_of_curvature(lat):
    """Return (R_N, R_E): meridian and prime-vertical radii of curvature [m]."""
    s2 = np.sin(lat) ** 2
    den = 1.0 - E2 * s2
    r_e = A / np.sqrt(den)                 # prime vertical (transverse)
    r_n = A * (1.0 - E2) / den**1.5        # meridian
    return r_n, r_e


def lla_to_ecef(lla):
    """Geodetic (lat, lon, h) -> ECEF (x, y, z)."""
    lla = np.asarray(lla, dtype=float)
    lat, lon, h = lla[..., 0], lla[..., 1], lla[..., 2]
    _, r_e = radii_of_curvature(lat)
    clat, slat = np.cos(lat), np.sin(lat)
    x = (r_e + h) * clat * np.cos(lon)
    y = (r_e + h) * clat * np.sin(lon)
    z = (r_e * (1.0 - E2) + h) * slat
    return np.stack([x, y, z], axis=-1)


def ecef_to_lla(xyz):
    """ECEF (x, y, z) -> geodetic (lat, lon, h). Bowring initial guess + Newton refinement."""
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    ep2 = (A**2 - B**2) / B**2
    theta = np.arctan2(z * A, p * B)
    lat = np.arctan2(z + ep2 * B * np.sin(theta) ** 3, p - E2 * A * np.cos(theta) ** 3)
    for _ in range(3):
        _, r_e = radii_of_curvature(lat)
        h = p / np.cos(lat) - r_e
        lat = np.arctan2(z, p * (1.0 - E2 * r_e / (r_e + h)))
    _, r_e = radii_of_curvature(lat)
    h = p / np.cos(lat) - r_e
    return np.stack([lat, lon, h], axis=-1)


def C_en(lat, lon):
    """Rotation matrix from ECEF to NED at (lat, lon): v_ned = C_en @ v_ecef."""
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array([[-sl * co, -sl * so, cl],
                     [-so, co, 0.0],
                     [-cl * co, -cl * so, -sl]])


def lla_to_ned(lla, lla0):
    """Geodetic points -> local NED coordinates [m] relative to the origin lla0 (exact, via ECEF)."""
    d = lla_to_ecef(lla) - lla_to_ecef(lla0)
    return d @ C_en(lla0[0], lla0[1]).T


def ned_to_lla(ned, lla0):
    """Local NED coordinates [m] relative to lla0 -> geodetic points (exact, via ECEF)."""
    ecef = np.asarray(ned) @ C_en(lla0[0], lla0[1]) + lla_to_ecef(lla0)
    return ecef_to_lla(ecef)


def normal_gravity(lat, h):
    """Magnitude of normal gravity [m/s^2] (Somigliana + second-order free-air height correction).

    Positive scalar; in NED the gravity vector is [0, 0, +g].
    """
    s2 = np.sin(lat) ** 2
    g0 = GAMMA_E * (1.0 + K_SOMIGLIANA * s2) / np.sqrt(1.0 - E2 * s2)
    return g0 * (1.0 - 2.0 / A * (1.0 + F + M_RATIO - 2.0 * F * s2) * h + 3.0 / A**2 * h**2)
