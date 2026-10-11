"""Integrity checks on the real acquisition (skipped when the data folder is absent)."""
from pathlib import Path

import numpy as np
import pytest

from insnav.io import load_dataset

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "raw" / "imuShort.txt.gz").exists(), reason="data not available")


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA)


def test_rates_and_monotonic_time(ds):
    for t, rate in [(ds.imu.t, 200), (ds.pos.t, 5), (ds.vel.t, 5), (ds.ref.t, 200)]:
        dt = np.diff(t)
        assert np.all(dt > 0)
        assert abs(np.median(dt) - 1.0 / rate) < 1e-4


def test_common_time_base(ds):
    # IMU, GNSS and reference all cover the same GPS-time window (within one IMU sample of each other)
    assert abs(ds.imu.t[0] - ds.ref.t[0]) < 0.01
    assert ds.pos.t[0] >= ds.imu.t[0]


def test_imu_static_start(ds):
    # vehicle at rest at the beginning: specific force norm ~ gravity, z axis down
    acc = ds.imu.acc[:400]
    assert abs(np.linalg.norm(acc, axis=1).mean() - 9.79) < 0.05
    assert acc[:, 2].mean() < -9.5
