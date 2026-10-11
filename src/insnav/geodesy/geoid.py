"""Local geoid undulation model, used to convert MSL altitudes to WGS84 ellipsoidal heights.

The GNSS receiver and the reference do not use the same geoid:
  * the Trimble receiver reports MSL heights with **EGM96** (its undulation field matches EGM96
    to a few centimeters along the whole drive),
  * the Qinertia reference was exported in MSL with **EGM2008**.
EGM2008 - EGM96 reaches 0.16 to 0.27 m over the test area, which showed up as a slowly varying
vertical offset between GNSS and reference. Everything inside insnav works with ellipsoidal heights,
so each source is converted with its own geoid.

Over a few kilometers the geoid is smooth: a linear model fitted on control points (geoid heights
from GeographicLib GeoidEval) is accurate to the centimeter level.
"""
from __future__ import annotations

import numpy as np


class LinearGeoid:
    """N(lat, lon) interpolated linearly from control points ``[[lat_deg, lon_deg, N_m], ...]``.

    * 1 point: constant undulation.
    * 2 points: linear along the segment joining them (constant across it).
    * >= 3 points: least-squares plane in (lat, lon).
    """

    def __init__(self, control_points):
        pts = np.asarray(control_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            raise ValueError("control_points must be a non-empty list of [lat_deg, lon_deg, N]")
        self.pts = pts

    def __call__(self, lat_rad, lon_rad):
        lat, lon = np.degrees(lat_rad), np.degrees(lon_rad)
        p = self.pts
        if len(p) == 1:
            return np.full(np.shape(lat), p[0, 2])
        if len(p) == 2:
            u = p[1, :2] - p[0, :2]
            s = ((np.stack([lat, lon], axis=-1) - p[0, :2]) @ u) / (u @ u)
            return p[0, 2] + s * (p[1, 2] - p[0, 2])
        A = np.column_stack([np.ones(len(p)), p[:, 0], p[:, 1]])
        c, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
        return c[0] + c[1] * lat + c[2] * lon
