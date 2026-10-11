"""Validation of the strapdown mechanization on the real IMU data (pure inertial, no aiding).

The navigation is initialised from the reference (position, velocity, attitude) at several epochs
of the drive and propagated with the raw IMU for 60 s. Three variants are compared:
  * raw IMU,
  * gyro bias removed (bias measured independently on the standstills, see insnav.alignment.static),
  * gyro bias removed but Earth rotation / transport rate ignored.

Usage:  python scripts/validate_mechanization.py [--data data] [--out docs/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from insnav.alignment.static import gyro_bias_from_static, static_periods
from insnav.attitude.rotations import euler_to_quat
from insnav.evaluation.reference import interp_reference
from insnav.geodesy.wgs84 import lla_to_ned
from insnav.io import load_dataset
from insnav.mechanization.strapdown import NavState, run_free_inertial
from insnav.plotting import style

STARTS = [100, 300, 600, 950, 1100]     # seconds since start of log, all while driving
DURATION = 60.0


def reference_velocity(ref, half_window=20):
    """NED velocity of the reference by central differences over +/- half_window samples (0.1 s)."""
    p = lla_to_ned(ref.lla, ref.lla[0])
    k = half_window
    v = np.full_like(p, np.nan)
    v[k:-k] = (p[2 * k:] - p[:-2 * k]) / (ref.t[2 * k:] - ref.t[:-2 * k])[:, None]
    return v


def free_inertial_error(d, v_ref, t_start, gyro_bias, earth_rate):
    i0 = np.searchsorted(d.imu.t, d.t0 + t_start)
    i1 = np.searchsorted(d.imu.t, d.t0 + t_start + DURATION)
    t = d.imu.t[i0:i1]
    lla0, eul0, _ = interp_reference(d.ref, t[:1])
    v0 = np.array([np.interp(t[0], d.ref.t, v_ref[:, i]) for i in range(3)])
    x0 = NavState.from_arrays(lla0[0], v0, euler_to_quat(*eul0[0]))
    lla, _, _ = run_free_inertial(x0, t, d.imu.gyr[i0:i1], d.imu.acc[i0:i1],
                                  gyro_bias=gyro_bias, earth_rate=earth_rate)
    lla_r, _, _ = interp_reference(d.ref, t)
    err = lla_to_ned(lla, lla0[0]) - lla_to_ned(lla_r, lla0[0])
    return t - t[0], np.linalg.norm(err[:, :2], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style.apply()
    C = style.COLORS

    d = load_dataset(args.data)
    periods = static_periods(d.imu.t, d.vel.t, d.vel.vel)
    biases = gyro_bias_from_static(d.imu, d.ref, periods)
    bg = np.median(biases, axis=0)
    print("Standstills [s]:", [(round(d.imu.t[a] - d.t0), round(d.imu.t[b - 1] - d.t0)) for a, b in periods])
    print("Gyro bias from standstills [deg/h]:", np.round(np.degrees(bg) * 3600, 1),
          "| spread between standstills:", np.round(np.degrees(biases.std(0)) * 3600, 1))

    v_ref = reference_velocity(d.ref)
    variants = [("Raw IMU", None, True, C["ukf"]),
                ("Gyro bias removed", bg, True, C["es_ekf"]),
                ("Gyro bias removed, no Earth rate", bg, False, C["gnss"])]
    results = {}
    for name, b, earth, _ in variants:
        curves = [free_inertial_error(d, v_ref, s, b, earth) for s in STARTS]
        tt = curves[0][0]
        n = min(len(c[1]) for c in curves)
        results[name] = (tt[:n], np.array([c[1][:n] for c in curves]))

    print(f"\nHorizontal error after 10 / 30 / 60 s (median over {len(STARTS)} starts):")
    for name, (tt, e) in results.items():
        med = np.median(e, axis=0)
        at = [med[np.searchsorted(tt, s) - 1] for s in (10, 30, 60)]
        print(f"  {name:34s} {at[0]:7.2f} {at[1]:7.2f} {at[2]:7.2f} m")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, _, _, color in variants:
        tt, e = results[name]
        ax.semilogy(tt, np.median(e, axis=0), color=color, label=name, lw=2)
        ax.fill_between(tt, e.min(0), e.max(0), color=color, alpha=0.12, lw=0)
    ax.set_xlabel("Time since start of pure inertial navigation [s]")
    ax.set_ylabel("Horizontal error [m]")
    ax.set_ylim(1e-2, 1e3)
    ax.set_xlim(0, DURATION)
    ax.set_title("Free-inertial drift on real data (median and range over 5 starts)")
    ax.legend(loc="lower right")
    fig.savefig(out / "04_free_inertial.png")
    plt.close(fig)
    print(f"Figure written to {out}/04_free_inertial.png")


if __name__ == "__main__":
    main()
