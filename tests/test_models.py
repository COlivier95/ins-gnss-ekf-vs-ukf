"""Jacobians of the measurement models and error dynamics checked against finite differences."""
import numpy as np

from insnav.attitude.rotations import euler_to_quat
from insnav.mechanization.strapdown import NavState, strapdown_step
from insnav.models.error_state import N_STATES, NoiseParams, Nominal, difference, inject, transition_matrix
from insnav.models.measurements import DualAntennaHeading, GnssPosition, GnssVelocity

LP, LS = [-0.90, -0.77, -1.20], [-0.90, 0.00, -1.20]
X0 = Nominal(np.radians(33.7), np.radians(-117.9), -23.0, np.array([12.0, -5.0, 0.3]),
             euler_to_quat(np.radians(3.0), np.radians(-2.0), np.radians(140.0)),
             np.array([0.01, -0.02, 0.005]), np.radians([120.0, 117.0, 175.0]) / 3600,
             np.array([-2e-3, 1e-3, -5.7e-3]), np.array([0.03, -0.02, 0.05]), np.array([np.radians(-0.3)]))
W_MEAS = np.array([0.02, -0.01, 0.3])          # raw gyro measurement
# finite-difference steps per state (position steps must stay large w.r.t. float64 resolution of lat/lon)
EPS = np.array([1e-2] * 3 + [1e-4] * 3 + [1e-4] * 3 + [1e-4] * 3 + [1e-6] * 3 + [1e-4] * 3 + [1e-3] * 3 + [1e-4])


def numerical_jacobian(model, x, w_meas):
    zp = model.predict(x, x.gyro(w_meas))
    H = np.zeros((model.dim, N_STATES))
    for i in range(N_STATES):
        dx = np.zeros(N_STATES)
        dx[i] = EPS[i]
        # central differences; the corrected gyro rate depends on the perturbed bias / scale factor
        xp, xm = inject(x, dx), inject(x, -dx)
        zp_plus = model.predict(xp, xp.gyro(w_meas))
        zp_minus = model.predict(xm, xm.gyro(w_meas))
        H[:, i] = (model.residual(zp_plus, zp) - model.residual(zp_minus, zp)) / (2 * EPS[i])
    return H


def test_measurement_jacobians():
    for model in (GnssPosition(LP), GnssVelocity(LP), DualAntennaHeading(LP, LS)):
        Hn = numerical_jacobian(model, X0, W_MEAS)
        Ha = model.jacobian(X0, X0.gyro(W_MEAS))
        assert np.allclose(Ha, Hn, atol=1e-5), (model.name, np.abs(Ha - Hn).max())


def test_heading_model_offset():
    # level vehicle heading North: baseline points East -> measured azimuth 90 deg
    x = X0.copy()
    x.q = euler_to_quat(0.0, 0.0, 0.0)
    x.hb = np.zeros(1)
    assert np.isclose(DualAntennaHeading(LP, LS).predict(x)[0], np.pi / 2)


def test_inject_difference_roundtrip():
    rng = np.random.default_rng(5)
    dx = rng.normal(0, [1, 1, 1, .1, .1, .1, .01, .01, .01, 1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5,
                        1e-3, 1e-3, 1e-3, .05, .05, .05, .01])
    assert np.allclose(difference(inject(X0, dx), X0), dx, rtol=1e-6, atol=1e-9)


def test_transition_matrix_matches_mechanization():
    """Propagate a perturbed and an unperturbed state through the mechanization: the difference must
    follow Phi @ dx to first order."""
    noise = NoiseParams(vrw=1e-3, arw=1e-4, ba_sigma=1e-3, ba_tau=3600, bg_sigma=1e-6, bg_tau=3600, sg_rw=1e-6,
                        gp_sigma=np.array([.05, .05, .08]), gp_tau=np.array([7., 7., 12.]))
    f_meas, w_meas, dt = np.array([0.5, -0.3, -9.7]), W_MEAS, 0.005
    dx = np.array([0.3, -0.2, 0.1, 0.02, -0.01, 0.03, 2e-4, -1e-4, 3e-4, 1e-3, -2e-3, 1e-3, 1e-5, -2e-5, 1e-5,
                   2e-3, -1e-3, 3e-3, 0.01, -0.02, 0.03, 0.005])

    def step(x):
        s = NavState.from_arrays([x.lat, x.lon, x.h], x.v, x.q)
        s2 = strapdown_step(s, x.gyro(w_meas), x.accel(f_meas), dt)
        return Nominal(float(s2.lat), float(s2.lon), float(s2.h), s2.v, s2.q, x.ba, x.bg, x.sg, x.gp, x.hb)

    x1, x1p = step(X0), step(inject(X0, dx))
    Phi = transition_matrix(X0, X0.accel(f_meas), X0.gyro(w_meas), noise, dt)
    predicted = Phi @ dx
    actual = difference(x1p, x1)
    # navigation states (position, velocity, attitude) must follow Phi to first order
    assert np.allclose(actual[:9], predicted[:9], atol=2e-6), np.abs(actual[:9] - predicted[:9]).max()
