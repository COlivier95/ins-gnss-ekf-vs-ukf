import numpy as np

from insnav.attitude import rotations as r

RNG = np.random.default_rng(1)


def random_euler(n=100):
    return (RNG.uniform(-np.pi, np.pi, n), RNG.uniform(-1.5, 1.5, n), RNG.uniform(-np.pi, np.pi, n))


def test_dcm_is_rotation():
    C = r.euler_to_dcm(*random_euler())
    eye = np.einsum("nij,nkj->nik", C, C)
    assert np.allclose(eye, np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(C), 1.0)


def test_euler_dcm_roundtrip():
    roll, pitch, yaw = random_euler()
    rr, pp, yy = r.dcm_to_euler(r.euler_to_dcm(roll, pitch, yaw))
    assert np.allclose(r.wrap_angle(rr - roll), 0, atol=1e-10)
    assert np.allclose(pp, pitch, atol=1e-10)
    assert np.allclose(r.wrap_angle(yy - yaw), 0, atol=1e-10)


def test_yaw_convention():
    # yaw = +90 deg: body x (forward) points East in NED
    C = r.euler_to_dcm(0.0, 0.0, np.pi / 2)
    assert np.allclose(C @ [1, 0, 0], [0, 1, 0])
    # pitch = +10 deg (nose up): body x has a negative Down component
    C = r.euler_to_dcm(0.0, np.radians(10), 0.0)
    assert (C @ [1, 0, 0])[2] < 0


def test_quat_dcm_consistency():
    for roll, pitch, yaw in zip(*random_euler(50)):
        C = r.euler_to_dcm(roll, pitch, yaw)
        q = r.dcm_to_quat(C)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(r.quat_to_dcm(q), C, atol=1e-12)


def test_quat_mult_matches_dcm_product():
    for _ in range(50):
        q1 = r.rotvec_to_quat(RNG.normal(size=3))
        q2 = r.rotvec_to_quat(RNG.normal(size=3))
        assert np.allclose(r.quat_to_dcm(r.quat_mult(q1, q2)), r.quat_to_dcm(q1) @ r.quat_to_dcm(q2))


def test_rotvec_exp_log():
    for phi in [RNG.normal(size=3), np.array([1e-10, -2e-10, 3e-10]), np.array([0.0, 0.0, 3.0])]:
        assert np.allclose(r.quat_to_rotvec(r.rotvec_to_quat(phi)), phi, atol=1e-12)


def test_small_rotation_dcm():
    # exp([phi]x) ~ I + [phi]x for small phi
    phi = np.array([1e-4, -2e-4, 3e-4])
    assert np.allclose(r.quat_to_dcm(r.rotvec_to_quat(phi)), np.eye(3) + r.skew(phi), atol=1e-7)
