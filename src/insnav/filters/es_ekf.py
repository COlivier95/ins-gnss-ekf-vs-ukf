"""Error-State Extended Kalman Filter (ES-EKF), loosely-coupled INS/GNSS, 22 states (see models/error_state.py).

Prediction: nominal state through the full nonlinear strapdown mechanization; error covariance
through the linearised error dynamics  P <- Phi P Phi^T + Qd.
Update: linearised measurement model H, Joseph-form covariance update, injection of the estimated
error into the nominal state and covariance reset.
"""
from __future__ import annotations

import numpy as np

from ..mechanization.strapdown import NavState, strapdown_step
from ..models.error_state import ITH, N_STATES, Nominal, inject, propagate_nominal_aux, reset_jacobian, transition_matrix
from .base import ErrorStateFilter


class ESEKF(ErrorStateFilter):
    name = "ES-EKF"

    def initialize(self, x0: Nominal, P0: np.ndarray):
        self.x = x0.copy()
        self.P = P0.copy()
        self._I = np.eye(N_STATES)

    def propagate(self, w_b, f_b, dt):
        x = self.x
        Phi = transition_matrix(x, f_b, w_b, self.noise, dt)
        s = strapdown_step(NavState(x.lat, x.lon, x.h, x.v, x.q), w_b, f_b, dt)
        x.lat, x.lon, x.h, x.v, x.q = s.lat, s.lon, s.h, s.v, s.q
        propagate_nominal_aux(x, self.noise, dt)
        P = Phi @ self.P @ Phi.T + self.noise.discrete_q(dt)
        self.P = 0.5 * (P + P.T)

    def correct(self, model, z, R, w_b, gate):
        x = self.x
        y = model.residual(z, model.predict(x, w_b))
        H = model.jacobian(x, w_b)
        PHt = self.P @ H.T
        S = H @ PHt + R
        S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)
        if gate is not None and nis > gate:
            return False, nis
        K = PHt @ S_inv
        dx = K @ y
        IKH = self._I - K @ H
        P = IKH @ self.P @ IKH.T + K @ R @ K.T          # Joseph form: stays symmetric positive definite
        G = reset_jacobian(dx[ITH])
        self.P = G @ P @ G.T
        self.x = inject(x, dx)
        return True, nis
