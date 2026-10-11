"""ES-EKF vs ES-UKF benchmark.

1. ``outages``    simulated GNSS outages (position, velocity and heading all removed) of 10 / 30 / 60 / 120 s
                  at the same five epochs of the drive (slow urban, mixed, turns, straight at 65 km/h).
                  Each duration is one run per filter with the five outages; the filter re-converges
                  between them. Output: error and filter sigma during every outage.
2. ``heading-mc`` Monte Carlo on the initial heading without the dual-antenna heading (single-antenna case):
                  the coarse heading is perturbed by a random error drawn from N(0, sigma0) and the filter is
                  initialised with that sigma0. Same random draws for both filters (paired comparison).

Results are written to results/benchmark/*.json (small, versionable); figures with plot_benchmark.py.

Usage:
    python scripts/benchmark.py outages    [--workers 2]
    python scripts/benchmark.py heading-mc [--runs 20 --sigma0 30 --workers 2]
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

OUTAGE_STARTS = [100.0, 320.0, 520.0, 680.0, 1040.0]     # [s since start of log]
OUTAGE_DURATIONS = [10.0, 30.0, 60.0, 120.0]
OUT = Path("results/benchmark")

_DS = None


def _dataset():
    global _DS
    if _DS is None:
        from insnav.io import load_dataset
        _DS = load_dataset("data")
    return _DS


def _filter(name, sysc, fc):
    if name == "ekf":
        from insnav.filters.es_ekf import ESEKF
        return ESEKF(sysc, fc)
    from insnav.filters.es_ukf import ESUKF
    return ESUKF(sysc, fc)


def _configs():
    return yaml.safe_load(open("config/system.yaml")), yaml.safe_load(open("config/filters.yaml"))


# --- 1. outages ----------------------------------------------------------------------------------------
def outage_job(args):
    name, duration = args
    from insnav.evaluation.metrics import compute_errors
    d = _dataset()
    sysc, fc = _configs()
    tic = time.time()
    res = _filter(name, sysc, fc).run(d, outages=[(s, duration) for s in OUTAGE_STARTS], verbose=False)
    err = compute_errors(res, d.ref, decimate=20)          # 10 Hz
    sp, sa = err.combined_sigma()
    t = err.t - d.t0
    out = []
    for s in OUTAGE_STARTS:
        m = (t >= s) & (t <= s + duration)
        out.append({
            "start": s,
            "t": (t[m] - s).round(2).tolist(),
            "horiz": err.horizontal[m].round(4).tolist(),
            "vert": np.abs(err.pos_ned[m, 2]).round(4).tolist(),
            "yaw_deg": np.degrees(np.abs(err.att[m, 2])).round(4).tolist(),
            # filter-predicted horizontal 1-sigma (radial) and combined with the reference
            "sigma_h": np.hypot(err.sigma_pos[m, 0], err.sigma_pos[m, 1]).round(4).tolist(),
            "sigma_h_comb": np.hypot(sp[m, 0], sp[m, 1]).round(4).tolist(),
            "err_n": err.pos_ned[m, 0].round(4).tolist(), "err_e": err.pos_ned[m, 1].round(4).tolist(),
            "sig_n": sp[m, 0].round(4).tolist(), "sig_e": sp[m, 1].round(4).tolist(),
        })
    return {"filter": name, "duration": duration, "runtime_s": time.time() - tic, "outages": out}


# --- 2. heading Monte Carlo ------------------------------------------------------------------------------
def heading_job(args):
    name, k, err_deg, sigma0, t_end = args
    from insnav.evaluation.metrics import compute_errors
    d = _dataset()
    sysc, fc = _configs()
    fc = copy.deepcopy(fc)
    fc["measurements"]["heading"]["use"] = False
    fc["alignment"]["heading_perturbation_deg"] = float(err_deg)
    fc["initial_uncertainty"]["heading"] = float(sigma0)
    res = _filter(name, sysc, fc).run(d, t_end=t_end, verbose=False)
    err = compute_errors(res, d.ref, decimate=200)         # 1 Hz
    t = err.t - d.t0
    return {"filter": name, "run": k, "init_err_deg": float(err_deg), "t": t.round(2).tolist(),
            "yaw_deg": np.degrees(err.att[:, 2]).round(4).tolist(),
            "yaw_sigma_deg": np.degrees(err.sigma_att[:, 2]).round(4).tolist(),
            "horiz": err.horizontal.round(3).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["outages", "heading-mc"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--sigma0", type=float, default=30.0)
    ap.add_argument("--t-end", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--filters", default="ekf,ukf")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    filters = args.filters.split(",")

    if args.what == "outages":
        # UKF jobs first (longest) to balance the workers
        jobs = [(f, dur) for f in sorted(filters, reverse=True) for dur in OUTAGE_DURATIONS]
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(outage_job, jobs):
                fn = OUT / f"outages_{r['filter']}_{int(r['duration'])}s.json"
                json.dump(r, open(fn, "w"))
                worst = max(max(o["horiz"]) for o in r["outages"])
                print(f"{r['filter']} {r['duration']:5.0f} s outages: worst horizontal drift {worst:7.2f} m "
                      f"({r['runtime_s']:.0f} s)", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        errs = rng.normal(0.0, args.sigma0, args.runs)
        jobs = [(f, k, e, args.sigma0, args.t_end) for f in sorted(filters, reverse=True) for k, e in enumerate(errs)]
        results = []
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(heading_job, jobs):
                results.append(r)
                print(f"{r['filter']} run {r['run']:2d} init {r['init_err_deg']:+7.1f} deg -> final yaw error "
                      f"{r['yaw_deg'][-1]:+7.2f} deg, horiz {r['horiz'][-1]:8.2f} m", flush=True)
        tag = f"heading_mc_sigma{int(args.sigma0)}"
        json.dump({"sigma0": args.sigma0, "seed": args.seed, "runs": results}, open(OUT / f"{tag}.json", "w"))


if __name__ == "__main__":
    main()
