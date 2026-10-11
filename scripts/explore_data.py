"""Data exploration and sanity checks against the reference trajectory.

Produces the figures in docs/figures/ and prints the checks:
  * lever-arm validation (GNSS antenna position vs reference IMU position + C_nb * lever arm)
  * GNSS velocity vs reference-derived antenna velocity
  * dual-antenna heading vs reference yaw (+90 deg baseline offset)
  * GNSS quality around the bridge underpass (t ~ 449 s)

Usage:  python scripts/explore_data.py [--data data] [--out docs/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from insnav.attitude.rotations import wrap_angle
from insnav.evaluation.reference import interp_reference
from insnav.geodesy.wgs84 import lla_to_ned
from insnav.io import load_dataset
from insnav.plotting import style

BRIDGE = (445.0, 470.0)   # seconds since start of log, bridge underpass and receiver recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--config", default="config/system.yaml")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style.apply()
    C = style.COLORS

    cfg = yaml.safe_load(open(args.config))
    lever = np.array(cfg["gnss"]["lever_arm_primary"])
    hdg_off = np.radians(cfg["gnss"]["heading_offset_deg"])

    d = load_dataset(args.data)
    t0, origin = d.t0, d.ref.lla[0]

    # ---- reference & GNSS in local NED --------------------------------------------------------
    p_ref = lla_to_ned(d.ref.lla, origin)
    lla_r, _, C_nb = interp_reference(d.ref, d.pos.t)
    p_imu_at_gnss = lla_to_ned(lla_r, origin)
    p_ant_ref = p_imu_at_gnss + np.einsum("nij,j->ni", C_nb, lever)
    p_gnss = lla_to_ned(d.pos.lla, origin)
    tg = d.pos.t - t0
    res_no = p_gnss - p_imu_at_gnss
    res_la = p_gnss - p_ant_ref
    nominal = d.pos.std[:, 0] < 0.1

    # ---- velocity: differentiate the reference antenna position -------------------------------
    _, _, C_full = interp_reference(d.ref, d.ref.t)
    p_ant_full = p_ref + np.einsum("nij,j->ni", C_full, lever)
    v_ant = np.gradient(p_ant_full, d.ref.t, axis=0)
    v_ref = np.column_stack([np.interp(d.vel.t, d.ref.t, v_ant[:, i]) for i in range(3)])
    res_v = d.vel.vel - v_ref

    # ---- heading ------------------------------------------------------------------------------
    yaw_at_hdt = np.interp(d.hdt.t, d.ref.t, np.unwrap(d.ref.euler[:, 2]))
    res_h = np.degrees(wrap_angle(d.hdt.heading - hdg_off - yaw_at_hdt))
    th = d.hdt.t - t0

    print("=== Checks (nominal GNSS epochs, horizontal std < 0.1 m) ===")
    for name, r in [("no lever arm", res_no), ("with lever arm", res_la)]:
        rh = np.sqrt((r[nominal, :2] ** 2).sum(1).mean())
        print(f"position residual {name:15s}: mean NED {np.round(r[nominal].mean(0), 3)} m, "
              f"horizontal RMS {rh:.3f} m")
    print(f"velocity residual: mean NED {np.round(res_v[nominal].mean(0), 3)}, "
          f"std {np.round(res_v[nominal].std(0), 3)} m/s")
    mad = 1.4826 * np.median(np.abs(res_h - np.median(res_h)))
    print(f"heading residual (HDT - 90 deg - yaw): median {np.median(res_h):.3f} deg, robust std {mad:.3f} deg, "
          f"{(np.abs(res_h) > 3).sum()} epochs > 3 deg (max {np.abs(res_h).max():.1f} deg)")

    # =========================================================================================
    # Figure 1 - trajectory map
    # =========================================================================================
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    ax.plot(p_ref[:, 1], p_ref[:, 0], color=C["reference"], lw=1.2, label="Reference trajectory")
    tr = d.ref.t - t0
    m = (tr > BRIDGE[0]) & (tr < BRIDGE[1])
    ax.plot(p_ref[m, 1], p_ref[m, 0], color=C["outage"], lw=4, solid_capstyle="round",
            label="Bridge underpass (GNSS degraded)")
    ax.plot(p_ref[0, 1], p_ref[0, 0], "o", color=C["es_ekf"], ms=8, label="Start / end")
    ax.set_aspect("equal")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Test drive - 9.9 km, 23 min, up to 70 km/h")
    ax.legend(loc="lower left")
    fig.savefig(out / "01_trajectory.png")
    plt.close(fig)

    # =========================================================================================
    # Figure 2 - GNSS quality over the log and around the bridge
    # =========================================================================================
    fig, axs = plt.subplots(3, 2, figsize=(11, 6.5), sharex="col",
                            gridspec_kw={"width_ratios": [2.2, 1]})
    for col, (xlim, title) in enumerate([((tg[0], tg[-1]), "Whole log"),
                                          ((435, 480), "Zoom: bridge underpass")]):
        a0, a1, a2 = axs[:, col]
        hstd = np.hypot(d.pos.std[:, 0], d.pos.std[:, 1])
        herr = np.linalg.norm(res_la[:, :2], axis=1)
        a0.semilogy(tg, hstd, color=C["gnss"], label="Receiver 1σ (horizontal)")
        a0.semilogy(tg, np.maximum(herr, 1e-3), color=C["es_ekf"], lw=1.0, label="Actual error vs reference")
        a0.set_ylabel("Horizontal [m]")
        a0.set_title(title)
        a1.plot(tg, d.pos.num_sv, color=C["gnss"])
        a1.set_ylabel("Satellites used")
        a2.plot(th, res_h, ".", color=C["es_ekf"], ms=2.5)
        a2.set_ylabel("Heading error [deg]")
        a2.set_xlabel("Time since start [s]")
        if col == 0:
            a2.set_ylim(-5, 5)
        for a in (a0, a1, a2):
            a.axvspan(*BRIDGE, color=C["outage"], alpha=0.18, lw=0)
            a.set_xlim(*xlim)
    axs[0, 0].legend(loc="upper left")
    fig.suptitle("GNSS quality (Trimble, 5 Hz) - shaded: bridge underpass and receiver recovery",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "02_gnss_quality.png")
    plt.close(fig)

    # =========================================================================================
    # Figure 3 - lever-arm validation
    # =========================================================================================
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    hn = np.linalg.norm(res_no[nominal, :2], axis=1)
    hl = np.linalg.norm(res_la[nominal, :2], axis=1)
    bins = np.linspace(0, 2.5, 60)
    axs[0].hist(hn, bins=bins, color=C["ukf"], alpha=0.85, label=f"Without lever arm (RMS {np.sqrt((hn**2).mean()):.2f} m)")
    axs[0].hist(hl, bins=bins, color=C["es_ekf"], alpha=0.85, label=f"With lever arm (RMS {np.sqrt((hl**2).mean()):.3f} m)")
    axs[0].set_xlabel("Horizontal GNSS − reference [m]")
    axs[0].set_ylabel("Epochs")
    axs[0].set_title("Lever-arm compensation of the GNSS antenna")
    axs[0].legend()
    axs[1].plot(d.vel.t - t0, np.linalg.norm(res_v[:, :2], axis=1), color=C["es_ekf"], lw=0.8)
    axs[1].axvspan(*BRIDGE, color=C["outage"], alpha=0.18, lw=0)
    axs[1].set_ylim(0, 0.5)
    axs[1].set_xlabel("Time since start [s]")
    axs[1].set_ylabel("Horizontal [m/s]")
    axs[1].set_title("GNSS velocity − reference antenna velocity")
    fig.tight_layout()
    fig.savefig(out / "03_lever_arm_check.png")
    plt.close(fig)
    print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
