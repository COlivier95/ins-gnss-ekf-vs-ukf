"""Run one filter over the drive, print the accuracy summary and draw the analysis figures.

Usage:
    python scripts/run_filter.py --filter ekf
    python scripts/run_filter.py --filter ekf --outage 600 60      # simulated 60 s GNSS outage at t = 600 s
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from insnav.alignment.static import gyro_bias_from_static, static_periods
from insnav.evaluation.metrics import compute_errors, summary
from insnav.io import load_dataset
from insnav.plotting import style

BRIDGE = (445.0, 470.0)


def make_filter(name, system_cfg, filter_cfg):
    if name == "ekf":
        from insnav.filters.es_ekf import ESEKF
        return ESEKF(system_cfg, filter_cfg)
    if name == "ukf":
        from insnav.filters.es_ukf import ESUKF
        return ESUKF(system_cfg, filter_cfg)
    raise ValueError(name)


def plot_run(d, res, err, out: Path, tag: str):
    style.apply()
    C = style.COLORS
    color = C["es_ekf"] if res.name == "ES-EKF" else C["ukf"]
    sp_comb, sa_comb = err.combined_sigma()
    t = err.t - d.t0

    # --- position error with 3-sigma bounds -------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    labels = ["North", "East", "Down"]
    for i, ax in enumerate(axs):
        ax.plot(t, err.pos_ned[:, i], color=color, lw=0.9, label=f"{res.name} − reference")
        ax.fill_between(t, -3 * sp_comb[:, i], 3 * sp_comb[:, i], color=color, alpha=0.10, lw=0,
                        label="±3σ (filter + reference)")
        ax.plot(t, 3 * err.sigma_pos[:, i], color=color, lw=0.6, ls="--", label="±3σ (filter only)")
        ax.plot(t, -3 * err.sigma_pos[:, i], color=color, lw=0.6, ls="--")
        ax.set_ylabel(f"{labels[i]} [m]")
        ax.set_ylim(-0.6, 0.6)
        ax.axvspan(*BRIDGE, color=C["outage"], alpha=0.18, lw=0)
        for a, b in res.outages:
            ax.axvspan(a - d.t0, b - d.t0, color=C["outage"], alpha=0.3, lw=0)
    axs[0].legend(loc="upper right", ncol=3)
    axs[0].set_title(f"{res.name} position error vs PPK reference (shaded: bridge underpass / outages)")
    axs[-1].set_xlabel("Time since start [s]")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_position_error.png")
    plt.close(fig)

    # --- attitude error ------------------------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True)
    for i, (ax, lab) in enumerate(zip(axs, ["Roll", "Pitch", "Yaw"])):
        ax.plot(t, np.degrees(err.att[:, i]), color=color, lw=0.9)
        ax.fill_between(t, -3 * np.degrees(sa_comb[:, i]), 3 * np.degrees(sa_comb[:, i]),
                        color=color, alpha=0.10, lw=0, label="±3σ (filter + reference)")
        ax.plot(t, 3 * np.degrees(err.sigma_att[:, i]), color=color, lw=0.6, ls="--", label="±3σ (filter only)")
        ax.plot(t, -3 * np.degrees(err.sigma_att[:, i]), color=color, lw=0.6, ls="--")
        ax.set_ylabel(f"{lab} [deg]")
        ax.set_ylim(*([-0.2, 0.2] if i < 2 else [-0.6, 0.6]))
        ax.axvspan(*BRIDGE, color=C["outage"], alpha=0.18, lw=0)
    axs[0].legend(loc="upper right", ncol=2)
    axs[0].set_title(f"{res.name} attitude error vs reference")
    axs[-1].set_xlabel("Time since start [s]")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_attitude_error.png")
    plt.close(fig)

    # --- gyro bias estimate vs independent standstill estimates --------------------------------------
    periods = static_periods(d.imu.t, d.vel.t, d.vel.vel)
    bs = gyro_bias_from_static(d.imu, d.ref, periods)
    tr = res.t - d.t0
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for i, (ax, lab) in enumerate(zip(axs, "xyz")):
        b = np.degrees(res.bg[:, i]) * 3600
        s = np.degrees(res.sigma[:, 12 + i]) * 3600
        ax.plot(tr, b, color=color, lw=1.4, label="Filter estimate")
        ax.fill_between(tr, b - 3 * s, b + 3 * s, color=color, alpha=0.15, lw=0, label="±3σ")
        for k, (a, bb) in enumerate(periods):
            ta, tb = d.imu.t[a] - d.t0, d.imu.t[bb - 1] - d.t0
            ax.plot([ta, tb], [np.degrees(bs[k, i]) * 3600] * 2, color=C["reference"], lw=3,
                    label="Standstill estimate (independent)" if k == 0 else None, solid_capstyle="butt")
        mid = np.degrees(np.median(bs[:, i])) * 3600
        ax.set_ylim(mid - 40, mid + 40)
        ax.set_title(f"Gyro bias {lab}")
        ax.set_xlabel("Time since start [s]")
    axs[0].set_ylabel("[deg/h]")
    axs[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_gyro_bias.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="ekf", choices=["ekf", "ukf"])
    ap.add_argument("--outage", nargs=2, type=float, action="append", default=[],
                    metavar=("START", "DURATION"), help="simulated GNSS outage [s since start]")
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--figures", default="docs/figures")
    ap.add_argument("--runs", default="results/runs")
    args = ap.parse_args()

    d = load_dataset(args.data)
    sysc = yaml.safe_load(open("config/system.yaml"))
    fc = yaml.safe_load(open("config/filters.yaml"))
    filt = make_filter(args.filter, sysc, fc)
    res = filt.run(d, outages=args.outage, t_end=args.t_end)
    err = compute_errors(res, d.ref, decimate=5)

    s = summary(err, t_from=d.t0 + 60)
    s["runtime_s"] = res.runtime_s
    s["rejected"] = {k: int((~v["accepted"]).sum()) for k, v in res.updates.items() if k != "reacquisitions"}
    s["reacquisitions"] = len(res.updates.get("reacquisitions", []))
    s["gyro_bias_final_deg_h"] = (np.degrees(res.bg[-1]) * 3600).round(2).tolist()
    s["accel_bias_final_mg"] = (res.ba[-1] / 9.80665e-3).round(3).tolist()
    print(json.dumps(s, indent=2))

    tag = f"{args.filter}" + "".join(f"_out{int(a)}-{int(b)}" for a, b in args.outage)
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    with open(runs / f"{tag}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    json.dump(s, open(runs / f"{tag}_summary.json", "w"), indent=2)
    if not args.outage:
        out = Path(args.figures)
        out.mkdir(parents=True, exist_ok=True)
        plot_run(d, res, err, out, f"{'05' if args.filter == 'ekf' else '06'}_{args.filter}")
        print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
