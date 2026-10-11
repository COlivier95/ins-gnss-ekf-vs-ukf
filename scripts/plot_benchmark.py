"""Figures and summary tables of the benchmark (reads results/benchmark/*.json).

Usage:  python scripts/plot_benchmark.py [--out docs/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from insnav.plotting import style

RES = Path("results/benchmark")
NAMES = {"ekf": "ES-EKF", "ukf": "ES-UKF"}
LABELS = {100.0: "urban, slow", 320.0: "mixed", 520.0: "turns", 680.0: "turns + stop", 1040.0: "straight 65 km/h"}


def load_outages():
    data = {}
    for f in sorted(RES.glob("outages_*.json")):
        r = json.load(open(f))
        data[(r["filter"], r["duration"])] = r
    return data


def outage_summary(data):
    rows = []
    durations = sorted({d for _, d in data})
    for dur in durations:
        for flt in ("ekf", "ukf"):
            r = data.get((flt, dur))
            if r is None:
                continue
            end = np.array([o["horiz"][-1] for o in r["outages"]])
            peak = np.array([max(o["horiz"]) for o in r["outages"]])
            # consistency: fraction of samples with |error| <= 3 sigma (filter + reference), N and E
            inside = np.concatenate([
                (np.abs(o["err_n"]) <= 3 * np.array(o["sig_n"])) & (np.abs(o["err_e"]) <= 3 * np.array(o["sig_e"]))
                for o in r["outages"]])
            ratio = np.array([o["horiz"][-1] / max(o["sigma_h_comb"][-1], 1e-6) for o in r["outages"]])
            yaw_end = np.array([o["yaw_deg"][-1] for o in r["outages"]])
            rows.append({"duration": dur, "filter": flt, "end_median": float(np.median(end)),
                         "end_max": float(end.max()), "peak_max": float(peak.max()),
                         "inside_3sigma": float(inside.mean()), "err_over_sigma_end_median": float(np.median(ratio)),
                         "yaw_end_median": float(np.median(yaw_end)), "per_outage_end": end.round(3).tolist()})
    return rows


def plot_outages(data, rows, out):
    C = style.COLORS
    durations = sorted({r["duration"] for r in rows})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    for flt, off in (("ekf", -0.04), ("ukf", 0.04)):
        rr = [r for r in rows if r["filter"] == flt]
        x = np.array([r["duration"] for r in rr])
        med = np.array([r["end_median"] for r in rr])
        lo = np.array([min(r["per_outage_end"]) for r in rr])
        hi = np.array([max(r["per_outage_end"]) for r in rr])
        color = C["es_ekf"] if flt == "ekf" else C["ukf"]
        ax.errorbar(x * (1 + off), med, yerr=[med - lo, hi - med], color=color, marker="o", ms=6, lw=2,
                    capsize=4, label=f"{NAMES[flt]} (median, min-max over 5 outages)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(durations)
    ax.set_xticklabels([f"{int(d)} s" for d in durations])
    ax.set_xlabel("GNSS outage duration")
    ax.set_ylabel("Horizontal error at the end of the outage [m]")
    ax.set_title("Drift during GNSS outages")
    ax.legend(loc="upper left")

    # example: longest outage on the straight segment and in the turns, both filters, with 3-sigma
    ax = axs[1]
    dur = max(durations)
    for flt in ("ekf", "ukf"):
        r = data.get((flt, dur))
        if r is None:
            continue
        color = C["es_ekf"] if flt == "ekf" else C["ukf"]
        for o, ls in zip([o for o in r["outages"] if o["start"] in (520.0, 1040.0)], ("-", "--")):
            t = np.array(o["t"])
            ax.plot(t, o["horiz"], color=color, lw=1.8, ls=ls,
                    label=f"{NAMES[flt]} – {LABELS[o['start']]}")
            ax.plot(t, 3 * np.array(o["sigma_h_comb"]), color=color, lw=0.8, ls=ls, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Time since the start of the outage [s]")
    ax.set_ylabel("Horizontal error [m]  (thin: filter 3σ)")
    ax.set_title(f"{int(dur)} s outages: error and predicted 3σ")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "07_outage_drift.png")
    plt.close(fig)


def plot_heading_mc(out):
    files = sorted(RES.glob("heading_mc_*.json"))
    if not files:
        return None
    mc = json.load(open(files[-1]))
    C = style.COLORS
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    stats = {}
    for ax, flt in zip(axs, ("ekf", "ukf")):
        runs = [r for r in mc["runs"] if r["filter"] == flt]
        color = C["es_ekf"] if flt == "ekf" else C["ukf"]
        final, conv_t = [], []
        for r in runs:
            t = np.array(r["t"])
            y = np.abs(np.array(r["yaw_deg"]))
            ax.semilogy(t, np.maximum(y, 1e-3), color=color, lw=0.9, alpha=0.7)
            final.append(y[-1])
            bad = np.flatnonzero(y > 1.0)
            conv_t.append(t[bad[-1]] if len(bad) else t[0])
        final = np.array(final)
        ok = final < 1.0
        stats[flt] = {"runs": len(runs), "converged": int(ok.sum()),
                      "median_convergence_s": float(np.median(np.array(conv_t)[ok])) if ok.any() else None,
                      "final_yaw_median_deg": float(np.median(final))}
        ax.axhline(1.0, color=style.INK_2, lw=0.8, ls=":")
        ax.set_title(f"{NAMES[flt]}: {ok.sum()}/{len(runs)} runs below 1° at t = {t[-1]:.0f} s")
        ax.set_xlabel("Time since start [s]")
    axs[0].set_ylabel("|Yaw error| [deg]")
    fig.suptitle(f"Monte Carlo: initial heading error ~ N(0, {mc['sigma0']:.0f}°), no dual-antenna heading "
                 f"(same {len(runs)} draws for both filters)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "08_heading_monte_carlo.png")
    plt.close(fig)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style.apply()
    data = load_outages()
    summary = {}
    if data:
        rows = outage_summary(data)
        plot_outages(data, rows, out)
        summary["outages"] = rows
        print(f"{'outage':>7} {'filter':>7} {'end median':>11} {'end max':>9} {'inside 3σ':>10} {'err/σ end':>10} {'yaw end':>8}")
        for r in rows:
            print(f"{r['duration']:6.0f}s {NAMES[r['filter']]:>7} {r['end_median']:10.2f}m {r['end_max']:8.2f}m "
                  f"{100 * r['inside_3sigma']:9.1f}% {r['err_over_sigma_end_median']:10.2f} {r['yaw_end_median']:7.2f}°")
    mc = plot_heading_mc(out)
    if mc:
        summary["heading_mc"] = mc
        print("Heading Monte Carlo:", json.dumps(mc))
    json.dump(summary, open(RES / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
