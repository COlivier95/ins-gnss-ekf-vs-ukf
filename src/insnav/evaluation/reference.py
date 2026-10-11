"""Interpolation of the reference trajectory at arbitrary epochs."""
from __future__ import annotations

import numpy as np

from ..attitude.rotations import euler_to_dcm, wrap_angle


def interp_reference(ref, t):
    """Interpolate the reference at times ``t`` (GPS TOW [s]).

    Returns (lla, euler, C_nb). Angles are unwrapped before linear interpolation so that the
    yaw does not jump across +/-180 deg. Epochs outside the reference span are NaN.
    """
    t = np.asarray(t, dtype=float)
    lla = np.column_stack([np.interp(t, ref.t, ref.lla[:, i], left=np.nan, right=np.nan)
                           for i in range(3)])
    eul = np.column_stack([np.interp(t, ref.t, np.unwrap(ref.euler[:, i]), left=np.nan, right=np.nan)
                           for i in range(3)])
    eul = wrap_angle(eul)
    C_nb = euler_to_dcm(eul[:, 0], eul[:, 1], eul[:, 2])
    return lla, eul, C_nb
