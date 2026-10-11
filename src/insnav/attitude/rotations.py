"""Attitude representations: unit quaternions, direction cosine matrices and Euler angles.

Conventions (used consistently in the whole package)
-----------------------------------------------------
* Quaternions are Hamilton, scalar-first ``q = [w, x, y, z]``.
* ``q_nb`` / ``C_nb`` rotate a vector from the body frame b to the navigation frame n:
  ``v_n = C_nb @ v_b``.
* Euler angles are aerospace ZYX (yaw psi, pitch theta, roll phi):
  ``C_nb = Rz(psi) @ Ry(theta) @ Rx(phi)``.
"""
from __future__ import annotations

import numpy as np


def skew(v):
    """Skew-symmetric matrix [v]x such that skew(a) @ b == cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# --- Euler <-> DCM -----------------------------------------------------------------------------
def euler_to_dcm(roll, pitch, yaw):
    """ZYX Euler angles [rad] -> C_nb. Vectorised: returns (..., 3, 3)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    C = np.empty(np.shape(roll) + (3, 3))
    C[..., 0, 0] = cp * cy
    C[..., 0, 1] = sr * sp * cy - cr * sy
    C[..., 0, 2] = cr * sp * cy + sr * sy
    C[..., 1, 0] = cp * sy
    C[..., 1, 1] = sr * sp * sy + cr * cy
    C[..., 1, 2] = cr * sp * sy - sr * cy
    C[..., 2, 0] = -sp
    C[..., 2, 1] = sr * cp
    C[..., 2, 2] = cr * cp
    return C


def dcm_to_euler(C):
    """C_nb -> (roll, pitch, yaw) [rad], yaw in (-pi, pi]."""
    C = np.asarray(C)
    roll = np.arctan2(C[..., 2, 1], C[..., 2, 2])
    pitch = -np.arcsin(np.clip(C[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(C[..., 1, 0], C[..., 0, 0])
    return roll, pitch, yaw


# --- Quaternions -------------------------------------------------------------------------------
def quat_mult(p, q):
    """Hamilton product p ⊗ q (scalar-first)."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q      # keep the scalar part positive (unique representative)


def rotvec_to_quat(phi):
    """Rotation vector [rad] -> unit quaternion (exact exponential map, safe near zero)."""
    phi = np.asarray(phi, dtype=float)
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        # second-order Taylor expansion
        return quat_normalize(np.concatenate([[1.0 - angle**2 / 8.0], 0.5 * phi]))
    axis = phi / angle
    return np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])


def quat_to_rotvec(q):
    """Unit quaternion -> rotation vector [rad] (logarithmic map)."""
    q = quat_normalize(q)
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-8:
        return 2.0 * v
    return 2.0 * np.arctan2(s, q[0]) * v / s


def quat_to_dcm(q):
    """Unit quaternion q_nb -> C_nb."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def dcm_to_quat(C):
    """C_nb -> unit quaternion q_nb (Shepperd's method, numerically robust)."""
    C = np.asarray(C, dtype=float)
    tr = np.trace(C)
    cands = np.array([tr, C[0, 0], C[1, 1], C[2, 2]])
    i = int(np.argmax(cands))
    if i == 0:
        w = 0.5 * np.sqrt(1.0 + tr)
        q = [w, (C[2, 1] - C[1, 2]) / (4 * w), (C[0, 2] - C[2, 0]) / (4 * w), (C[1, 0] - C[0, 1]) / (4 * w)]
    elif i == 1:
        x = 0.5 * np.sqrt(1.0 + 2 * C[0, 0] - tr)
        q = [(C[2, 1] - C[1, 2]) / (4 * x), x, (C[0, 1] + C[1, 0]) / (4 * x), (C[0, 2] + C[2, 0]) / (4 * x)]
    elif i == 2:
        y = 0.5 * np.sqrt(1.0 + 2 * C[1, 1] - tr)
        q = [(C[0, 2] - C[2, 0]) / (4 * y), (C[0, 1] + C[1, 0]) / (4 * y), y, (C[1, 2] + C[2, 1]) / (4 * y)]
    else:
        z = 0.5 * np.sqrt(1.0 + 2 * C[2, 2] - tr)
        q = [(C[1, 0] - C[0, 1]) / (4 * z), (C[0, 2] + C[2, 0]) / (4 * z), (C[1, 2] + C[2, 1]) / (4 * z), z]
    return quat_normalize(np.array(q))


def euler_to_quat(roll, pitch, yaw):
    return dcm_to_quat(euler_to_dcm(roll, pitch, yaw))


def quat_to_euler(q):
    return dcm_to_euler(quat_to_dcm(q))


def wrap_angle(a):
    """Wrap angles [rad] to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(a)))
