import numpy as np

from insnav.attitude import rotations as r
from insnav.attitude.rotations import euler_to_quat, quat_to_dcm
from insnav.geodesy.wgs84 import OMEGA_IE, lla_to_ned, normal_gravity, radii_of_curvature
from insnav.mechanization.strapdown import NavState, run_free_inertial, strapdown_step

LLA0 = np.array([np.radians(33.7), np.radians(-117.9), -23.0])


def perfect_static_imu(lla, q_nb):
    """Exact IMU output of a platform at rest on the rotating Earth."""
    lat = lla[0]
    C_bn = quat_to_dcm(q_nb).T
    w_ie_n = OMEGA_IE * np.array([np.cos(lat), 0.0, -np.sin(lat)])
    g_n = np.array([0.0, 0.0, normal_gravity(lat, lla[2])])
    return C_bn @ w_ie_n, C_bn @ (-g_n)


def test_scalar_and_batch_paths_agree():
    """The scalar fast path (ES-EKF) and the vectorised path (UKF sigma points) are the same equations."""
    rng = np.random.default_rng(4)
    B = 7
    lla = LLA0 + np.column_stack([rng.normal(0, 1e-4, B), rng.normal(0, 1e-4, B), rng.normal(0, 5, B)])
    v = rng.normal(0, 10, (B, 3))
    q = np.array([r.rotvec_to_quat(p) for p in rng.normal(0, 1, (B, 3))])
    w = rng.normal(0, 0.5, (B, 3))
    f = rng.normal(0, 3, (B, 3)) + [0, 0, -9.8]
    batch = strapdown_step(NavState.from_arrays(lla, v, q), w, f, 0.005)
    for i in range(B):
        one = strapdown_step(NavState.from_arrays(lla[i], v[i], q[i]), w[i], f[i], 0.005)
        assert np.allclose(one.lla, batch.lla[i], rtol=0, atol=1e-12)
        assert np.allclose(one.v, batch.v[i], atol=1e-12)
        assert np.allclose(one.q, batch.q[i], atol=1e-12)


def test_static_platform_stays_put():
    """10 min of perfect static IMU data: the solution must not move (< 1 cm, < 1 mm/s, < 0.2 arcsec)."""
    q0 = euler_to_quat(np.radians(2.0), np.radians(-3.0), np.radians(140.0))
    w, f = perfect_static_imu(LLA0, q0)
    n = 200 * 600
    t = np.arange(n) * 0.005
    lla, v, q = run_free_inertial(NavState.from_arrays(LLA0, np.zeros(3), q0), t,
                                  np.tile(w, (n, 1)), np.tile(f, (n, 1)))
    assert np.linalg.norm(lla_to_ned(lla[-1], LLA0)) < 0.01
    assert np.abs(v[-1]).max() < 1e-3
    dq = r.quat_to_rotvec(r.quat_mult(r.quat_conj(q0), q[-1]))
    assert np.linalg.norm(dq) < 1e-6


def test_earth_rate_matters_for_static_platform():
    """Ignoring the Earth rotation on a static platform makes the solution drift."""
    q0 = euler_to_quat(0.0, 0.0, 0.0)
    w, f = perfect_static_imu(LLA0, q0)
    n = 200 * 60
    t = np.arange(n) * 0.005
    lla, _, _ = run_free_inertial(NavState.from_arrays(LLA0, np.zeros(3), q0), t, np.tile(w, (n, 1)),
                                  np.tile(f, (n, 1)), earth_rate=False)
    assert np.linalg.norm(lla_to_ned(lla[-1], LLA0)) > 0.1


def test_constant_velocity_east():
    """Level vehicle driving East at 20 m/s along a parallel for 60 s, with the exact IMU signals."""
    lat, h, ve, dt, n = LLA0[0], LLA0[2], 20.0, 0.005, 200 * 60
    _, r_e = radii_of_curvature(lat)
    w_ie = OMEGA_IE * np.array([np.cos(lat), 0.0, -np.sin(lat)])
    w_en = np.array([ve / (r_e + h), 0.0, -ve * np.tan(lat) / (r_e + h)])
    v_n = np.array([0.0, ve, 0.0])
    # attitude rotates with the local-level frame along the parallel: constant yaw of 90 deg
    q = euler_to_quat(0.0, 0.0, np.pi / 2)
    C_bn = quat_to_dcm(q).T
    f_n = -np.array([0.0, 0.0, normal_gravity(lat, h)]) + np.cross(2 * w_ie + w_en, v_n)
    w_b, f_b = C_bn @ (w_ie + w_en), C_bn @ f_n
    t = np.arange(n) * dt
    lla, v, _ = run_free_inertial(NavState.from_arrays(LLA0, v_n, q), t, np.tile(w_b, (n, 1)), np.tile(f_b, (n, 1)))
    assert abs(lla[-1, 0] - lat) < 1e-9                                    # stays on the parallel
    assert abs(lla[-1, 2] - h) < 0.01
    expected_dlon = ve * t[-1] / ((r_e + h) * np.cos(lat))
    assert abs((lla[-1, 1] - LLA0[1]) - expected_dlon) * (r_e + h) * np.cos(lat) < 0.01   # < 1 cm over 1.2 km
    assert np.allclose(v[-1], v_n, atol=1e-4)
