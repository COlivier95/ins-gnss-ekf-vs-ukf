"""Error-State Unscented Kalman Filter (ES-UKF), loosely-coupled INS/GNSS, 22 states.

Same nominal state, error state, noise model, measurement models and tuning as the ES-EKF: only the
way uncertainty goes through the nonlinear models differs.

* Sigma points live in the **error space** (2n + 1 = 45 points for n = 22) and are mapped onto the
  manifold with the injection x_i = x ⊕ chi_i (attitude errors become quaternion multiplications, so
  no quaternion is ever averaged).
* Time update: every sigma point is propagated through the full nonlinear strapdown mechanization with
  its own biases and scale factors (one vectorised call for all points). To keep the cost reasonable the
  sigma points are regenerated every ``ukf.propagation_steps`` IMU samples (and before each GNSS update);
  the IMU samples in between are buffered. The nominal state is still propagated at 200 Hz for the outputs.
* Measurement update: the nonlinear measurement models are evaluated on the sigma points (no Jacobian).

Unscented transform weights: scaled UT (alpha, beta, kappa) from the configuration. alpha = 1, beta = 2,
kappa = 0 gives non-negative weights, so the covariance stays positive semi-definite by construction.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky

from ..geodesy.wgs84 import radii_of_curvature
from ..mechanization.strapdown import NavState, strapdown_step
from ..models.error_state import (IBA, IBG, IGP, IHB, IR, ISG, ITH, IV, N_STATES, Nominal, difference,
                                  inject, propagate_nominal_aux, reset_jacobian)
from .base import ErrorStateFilter


# --- batched quaternion helpers (B, 4) ------------------------------------------------------------
def _qmul(p, q):
    pw, px, py, pz = p.T
    qw, qx, qy, qz = q.T
    return np.stack([pw * qw - px * qx - py * qy - pz * qz,
                     pw * qx + px * qw + py * qz - pz * qy,
                     pw * qy - px * qz + py * qw + pz * qx,
                     pw * qz + px * qy - py * qx + pz * qw], axis=1)


def _qexp(phi):
    a = np.linalg.norm(phi, axis=1)
    k = 0.5 * np.sinc(a / (2.0 * np.pi))
    return np.column_stack([np.cos(0.5 * a), k[:, None] * phi])


def _qlog(q):
    q = q * np.where(q[:, :1] < 0, -1.0, 1.0)
    v = q[:, 1:]
    s = np.linalg.norm(v, axis=1)
    ang = 2.0 * np.arctan2(s, q[:, 0])
    k = np.where(s > 1e-12, ang / np.where(s > 1e-12, s, 1.0), 2.0)
    return k[:, None] * v


class SigmaSet:
    """Batch of nominal states, one per sigma point (structure of arrays)."""

    def __init__(self, x: Nominal, chi: np.ndarray):
        """x ⊕ chi_i for every row of chi (B, N_STATES)."""
        r_n, r_e = radii_of_curvature(x.lat)
        self.lat = x.lat + chi[:, 0] / (r_n + x.h)
        self.lon = x.lon + chi[:, 1] / ((r_e + x.h) * np.cos(x.lat))
        self.h = x.h - chi[:, 2]
        self.v = x.v + chi[:, IV]
        self.q = _qmul(_qexp(chi[:, ITH]), np.broadcast_to(x.q, (len(chi), 4)))
        self.ba = x.ba + chi[:, IBA]
        self.bg = x.bg + chi[:, IBG]
        self.sg = x.sg + chi[:, ISG]
        self.gp = x.gp + chi[:, IGP]
        self.hb = x.hb + chi[:, IHB]

    def errors_wrt(self, i0: int = 0) -> np.ndarray:
        """Error-state coordinates of all points relative to point i0 (B, N_STATES)."""
        lat0, h0 = self.lat[i0], self.h[i0]
        r_n, r_e = radii_of_curvature(lat0)
        d = np.empty((len(self.lat), N_STATES))
        d[:, 0] = (self.lat - lat0) * (r_n + h0)
        d[:, 1] = (self.lon - self.lon[i0]) * (r_e + h0) * np.cos(lat0)
        d[:, 2] = -(self.h - h0)
        d[:, IV] = self.v - self.v[i0]
        q0c = self.q[i0] * np.array([1.0, -1.0, -1.0, -1.0])
        d[:, ITH] = _qlog(_qmul(self.q, np.broadcast_to(q0c, self.q.shape)))
        d[:, IBA] = self.ba - self.ba[i0]
        d[:, IBG] = self.bg - self.bg[i0]
        d[:, ISG] = self.sg - self.sg[i0]
        d[:, IGP] = self.gp - self.gp[i0]
        d[:, IHB] = self.hb - self.hb[i0]
        return d

    def nominal(self, i: int) -> Nominal:
        return Nominal(float(self.lat[i]), float(self.lon[i]), float(self.h[i]), self.v[i].copy(),
                       self.q[i].copy(), self.ba[i].copy(), self.bg[i].copy(), self.sg[i].copy(),
                       self.gp[i].copy(), self.hb[i].copy())


class ESUKF(ErrorStateFilter):
    name = "ES-UKF"

    def __init__(self, system_cfg, filter_cfg):
        super().__init__(system_cfg, filter_cfg)
        u = filter_cfg["ukf"]
        n = N_STATES
        alpha, beta, kappa = u["alpha"], u["beta"], u["kappa"]
        lam = alpha**2 * (n + kappa) - n
        self.gamma = np.sqrt(n + lam)
        self.wm = np.full(2 * n + 1, 0.5 / (n + lam))
        self.wc = self.wm.copy()
        self.wm[0] = lam / (n + lam)
        self.wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
        self.max_steps = int(u.get("propagation_steps", 20))

    # ---- helpers -----------------------------------------------------------------------------------
    def _sigma_offsets(self, P):
        """chi (2n+1, n): 0, +gamma*S_j, -gamma*S_j with P = S S^T."""
        try:
            S = cholesky(P, lower=True)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(0.5 * (P + P.T))
            S = V * np.sqrt(np.maximum(w, 0.0))
        cols = self.gamma * S.T
        return np.vstack([np.zeros(N_STATES), cols, -cols])

    # ---- filter interface --------------------------------------------------------------------------
    def initialize(self, x0: Nominal, P0: np.ndarray):
        self.x = x0.copy()
        self.P = P0.copy()
        self._start = self.x.copy()
        self._buf = []            # raw IMU (w_meas, f_meas, dt) since the last sigma-point regeneration

    def propagate(self, w_b, f_b, dt):
        x = self.x
        # raw measurements (the sigma points apply their own bias / scale-factor corrections)
        self._buf.append((w_b / (1.0 - x.sg) + x.bg, f_b + x.ba, dt))
        s = strapdown_step(NavState(x.lat, x.lon, x.h, x.v, x.q), w_b, f_b, dt)
        x.lat, x.lon, x.h, x.v, x.q = s.lat, s.lon, s.h, s.v, s.q
        propagate_nominal_aux(x, self.noise, dt)
        if len(self._buf) >= self.max_steps:
            self._flush()

    def _flush(self):
        """Unscented time update over the buffered IMU samples."""
        if not self._buf:
            return
        chi = self._sigma_offsets(self.P)
        sp = SigmaSet(self._start, chi)
        Q = np.zeros((N_STATES, N_STATES))
        for w_raw, f_raw, dt in self._buf:
            w = (w_raw - sp.bg) * (1.0 - sp.sg)
            f = f_raw - sp.ba
            s = strapdown_step(NavState(sp.lat, sp.lon, sp.h, sp.v, sp.q), w, f, dt)
            sp.lat, sp.lon, sp.h, sp.v, sp.q = s.lat, s.lon, s.h, s.v, s.q
            sp.gp = sp.gp * np.exp(-dt / self.noise.gp_tau)
            Q += self.noise.discrete_q(dt)
        d = sp.errors_wrt(0)
        dmean = self.wm @ d
        dd = d - dmean
        P = (self.wc[:, None] * dd).T @ dd + Q
        self.x = inject(sp.nominal(0), dmean)
        self.P = 0.5 * (P + P.T)
        self._start = self.x.copy()
        self._buf = []

    def correct(self, model, z, R, w_b, gate):
        self._flush()
        x = self.x
        w_raw = w_b / (1.0 - x.sg) + x.bg
        chi = self._sigma_offsets(self.P)
        sp = SigmaSet(x, chi)
        zp = []
        for i in range(len(chi)):
            xi = sp.nominal(i)
            zp.append(model.predict(xi, xi.gyro(w_raw)))
        z0 = zp[0]
        dz = np.array([model.residual(zi, z0) for zi in zp])          # in the residual vector space
        dz_mean = self.wm @ dz
        e = dz - dz_mean
        Pzz = (self.wc[:, None] * e).T @ e + R
        Pxz = (self.wc[:, None] * chi).T @ e
        y = model.residual(z, z0) - dz_mean
        S_inv = np.linalg.inv(Pzz)
        nis = float(y @ S_inv @ y)
        if gate is not None and nis > gate:
            return False, nis
        K = Pxz @ S_inv
        dx = K @ y
        P = self.P - K @ Pzz @ K.T
        G = reset_jacobian(dx[ITH])
        self.P = G @ (0.5 * (P + P.T)) @ G.T
        self.x = inject(x, dx)
        self._start = self.x.copy()
        return True, nis
