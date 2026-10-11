"""Comparison of a filter output with the reference trajectory."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..attitude.rotations import quat_to_dcm, wrap_angle
from ..geodesy.wgs84 import lla_to_ned
from .reference import interp_reference


@dataclass
class Errors:
    t: np.ndarray            # GPS TOW [s]
    pos_ned: np.ndarray      # (N, 3) filter - reference [m]
    att: np.ndarray          # (N, 3) roll / pitch / yaw error [rad]
    sigma_pos: np.ndarray    # (N, 3) filter 1-sigma position [m]
    sigma_att: np.ndarray    # (N, 3) filter 1-sigma attitude error (NED small angles) [rad]
    ref_sigma_pos: np.ndarray | None = None   # (N, 3) reference 1-sigma N/E/D [m]
    ref_sigma_att: np.ndarray | None = None   # (N, 3) reference 1-sigma roll/pitch/yaw [rad]

    def combined_sigma(self):
        """1-sigma of (filter - reference): filter and reference uncertainties combined (independent)."""
        sp, sa = self.sigma_pos, self.sigma_att
        if self.ref_sigma_pos is not None:
            sp = np.sqrt(sp**2 + self.ref_sigma_pos**2)
            sa = np.sqrt(sa**2 + self.ref_sigma_att**2)
        return sp, sa

    @property
    def horizontal(self):
        return np.linalg.norm(self.pos_ned[:, :2], axis=1)


def compute_errors(result, ref, decimate=1) -> Errors:
    """Errors of the filter solution w.r.t. the reference, at the filter epochs inside the reference span."""
    sl = slice(None, None, decimate)
    t = result.t[sl]
    lla_r, eul_r, C_r = interp_reference(ref, t)
    ok = np.isfinite(lla_r[:, 0])
    t, lla_r, eul_r, C_r = t[ok], lla_r[ok], eul_r[ok], C_r[ok]
    lla = result.lla[sl][ok]
    origin = lla_r[0]
    pos = lla_to_ned(lla, origin) - lla_to_ned(lla_r, origin)
    # attitude error as Euler differences (readable) computed from the DCMs
    C = np.array([quat_to_dcm(q) for q in result.q[sl][ok]])
    roll = np.arctan2(C[:, 2, 1], C[:, 2, 2])
    pitch = -np.arcsin(np.clip(C[:, 2, 0], -1, 1))
    yaw = np.arctan2(C[:, 1, 0], C[:, 0, 0])
    att = wrap_angle(np.column_stack([roll, pitch, yaw]) - eul_r)
    ref_sp = ref_sa = None
    if getattr(ref, "pos_std", None) is not None:
        ref_sp = np.column_stack([np.interp(t, ref.t, ref.pos_std[:, i]) for i in range(3)])
        ref_sa = np.column_stack([np.interp(t, ref.t, ref.att_std[:, i]) for i in range(3)])
    return Errors(t, pos, att, result.sigma[sl][ok, 0:3], result.sigma[sl][ok, 6:9], ref_sp, ref_sa)


def summary(err: Errors, t_from: float | None = None, mask=None) -> dict:
    """RMS / max / 95th percentile statistics and 3-sigma consistency."""
    m = np.ones(len(err.t), bool) if mask is None else mask.copy()
    if t_from is not None:
        m &= err.t >= t_from
    sp, sa = err.combined_sigma()
    m &= np.isfinite(sp).all(axis=1) & np.isfinite(sa).all(axis=1)
    h = err.horizontal[m]
    v = np.abs(err.pos_ned[m, 2])
    a = np.degrees(np.abs(err.att[m]))
    within = np.abs(err.pos_ned[m]) <= 3 * err.sigma_pos[m]
    within_att = np.abs(err.att[m]) <= 3 * err.sigma_att[m]
    return {
        "horiz_rms_m": float(np.sqrt(np.mean(h**2))),
        "horiz_p95_m": float(np.percentile(h, 95)),
        "horiz_max_m": float(h.max()),
        "vert_rms_m": float(np.sqrt(np.mean(v**2))),
        "roll_rms_deg": float(np.sqrt(np.mean(a[:, 0] ** 2))),
        "pitch_rms_deg": float(np.sqrt(np.mean(a[:, 1] ** 2))),
        "yaw_rms_deg": float(np.sqrt(np.mean(a[:, 2] ** 2))),
        "pos_within_3sigma_ned": within.mean(axis=0).round(4).tolist(),
        "att_within_3sigma_rpy": within_att.mean(axis=0).round(4).tolist(),
        # consistency with filter + reference uncertainty (the meaningful test when the reference is not perfect)
        "pos_within_3sigma_ned_combined": (np.abs(err.pos_ned[m]) <= 3 * sp[m]).mean(axis=0).round(4).tolist(),
        "att_within_3sigma_rpy_combined": (np.abs(err.att[m]) <= 3 * sa[m]).mean(axis=0).round(4).tolist(),
        "pos_nees_ratio_ned": np.sqrt(np.mean((err.pos_ned[m] / sp[m]) ** 2, axis=0)).round(3).tolist(),
        "att_nees_ratio_rpy": np.sqrt(np.mean((err.att[m] / sa[m]) ** 2, axis=0)).round(3).tolist(),
    }
