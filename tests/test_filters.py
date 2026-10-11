"""Filter tests on synthetic data with a known truth (no real data needed)."""
from types import SimpleNamespace

import numpy as np
import yaml
from pathlib import Path

from insnav.attitude.rotations import euler_to_dcm
from insnav.geodesy.wgs84 import OMEGA_IE, normal_gravity, radii_of_curvature

ROOT = Path(__file__).resolve().parents[1]
SYS = yaml.safe_load(open(ROOT / "config" / "system.yaml"))
FILT = yaml.safe_load(open(ROOT / "config" / "filters.yaml"))
LLA = np.array([np.radians(33.7), np.radians(-117.9), -24.0])
ATT = np.radians([2.0, 3.0, 160.0])
BG = np.radians([100.0, -50.0, 150.0]) / 3600.0      # true gyro bias [rad/s]
BA = np.array([0.003, -0.002, 0.001])                # true accel bias [m/s^2]


def synthetic_static_dataset(duration=120.0, seed=0):
    """Vehicle at rest: exact IMU signals + biases + white noise; GNSS position/velocity/heading."""
    rng = np.random.default_rng(seed)
    C_nb = euler_to_dcm(*ATT)
    lat, h = LLA[0], LLA[2]
    w_ie = OMEGA_IE * np.array([np.cos(lat), 0.0, -np.sin(lat)])
    f_b = C_nb.T @ np.array([0.0, 0.0, -normal_gravity(lat, h)])
    w_b = C_nb.T @ w_ie
    t = 490000.0 + np.arange(0.0, duration, 0.005)
    n = len(t)
    imu = SimpleNamespace(t=t, acc=f_b + BA + rng.normal(0, 0.005, (n, 3)),
                          gyr=w_b + BG + rng.normal(0, 2e-4, (n, 3)), temperature=np.zeros(n))
    tg = np.arange(t[0] + 0.2, t[-1], 0.2)
    m = len(tg)
    lp = np.array(SYS["gnss"]["lever_arm_primary"])
    ls = np.array(SYS["gnss"]["lever_arm_secondary"])
    d_ant = C_nb @ lp
    r_n, r_e = radii_of_curvature(lat)
    lla_ant = LLA + np.array([d_ant[0] / (r_n + h), d_ant[1] / ((r_e + h) * np.cos(lat)), -d_ant[2]])
    b_n = C_nb @ (ls - lp)
    pos_std = np.tile([0.02, 0.02, 0.05], (m, 1))
    noise = rng.normal(0, 1, (m, 3)) * pos_std
    pos = SimpleNamespace(t=tg, lla=lla_ant + noise * [1 / (r_n + h), 1 / ((r_e + h) * np.cos(lat)), -1],
                          std=pos_std, sol_type=np.full(m, 7), num_sv=np.full(m, 20), undulation=np.zeros(m))
    vel = SimpleNamespace(t=tg, vel=rng.normal(0, 0.02, (m, 3)), std=np.full((m, 3), 0.02))
    hdt = SimpleNamespace(t=tg, heading=np.arctan2(b_n[1], b_n[0]) + rng.normal(0, np.radians(0.3), m),
                          heading_std=np.full(m, np.radians(0.3)), pitch=np.zeros(m), pitch_std=np.ones(m))
    return SimpleNamespace(imu=imu, pos=pos, vel=vel, hdt=hdt, ref=None, t0=t[0])


def _check_static_convergence(filter_cls):
    ds = synthetic_static_dataset()
    res = filter_cls(SYS, FILT).run(ds, verbose=False)
    # attitude converges close to the truth
    from insnav.attitude.rotations import quat_to_euler
    eul = np.array(quat_to_euler(res.q[-1]))
    assert np.all(np.abs(np.degrees(eul - ATT)) < [0.05, 0.05, 0.3])
    # horizontal gyro biases are observable at rest (tilt -> velocity), and estimated within 3 sigma
    bg_err = res.bg[-1] - BG
    assert np.all(np.abs(bg_err[:2]) < 3 * res.sigma[-1, 12:14]), np.degrees(bg_err) * 3600
    assert np.all(np.abs(np.degrees(bg_err[:2]) * 3600) < 10.0)
    # position: close to the truth and consistent with the filter covariance (|error| < 3 sigma)
    r_n, r_e = radii_of_curvature(LLA[0])
    err = np.array([(res.lla[-1, 0] - LLA[0]) * r_n, (res.lla[-1, 1] - LLA[1]) * r_e * np.cos(LLA[0]),
                    -(res.lla[-1, 2] - LLA[2])])
    assert np.all(np.abs(err) < 3 * res.sigma[-1, 0:3]), (err, res.sigma[-1, 0:3])
    assert np.hypot(err[0], err[1]) < 0.10


def test_es_ekf_static_convergence():
    from insnav.filters.es_ekf import ESEKF
    _check_static_convergence(ESEKF)


def test_es_ukf_static_convergence():
    from insnav.filters.es_ukf import ESUKF
    _check_static_convergence(ESUKF)
