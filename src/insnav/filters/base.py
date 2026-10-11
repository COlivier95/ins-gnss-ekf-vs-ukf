"""Common machinery of the error-state filters: event loop, alignment, gating, outages, logging.

The ES-EKF and the ES-UKF only differ in two methods:
  * ``propagate(w_b, f_b, dt)``  - nominal + covariance prediction over an IMU interval,
  * ``correct(model, z, R, w_b)`` - measurement update, returning the innovation and its covariance
                                    *before* the update so the base class can gate it.
Everything else (models, tuning, timing, gating, outages, outputs) is shared, so the comparison is fair.

Timing: GNSS epochs are processed at their exact measurement time. When an epoch falls inside an IMU
interval, the interval is split (same IMU rates on both parts), the update is applied, and the
propagation resumes - at 20 m/s a 5 ms mismatch would otherwise be a 10 cm error.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from ..alignment.coarse import coarse_alignment
from ..models.error_state import IBA, IBG, IGP, IHB, IR, ISG, ITH, IV, MG, N_STATES, PPM, NoiseParams, Nominal
from ..models.measurements import DualAntennaHeading, GnssPosition, GnssVelocity



@dataclass
class FilterResult:
    name: str
    t: np.ndarray                # GPS TOW of the IMU epochs [s]
    lla: np.ndarray              # (N, 3) IMU position [rad, rad, m]
    vel: np.ndarray              # (N, 3) NED velocity [m/s]
    q: np.ndarray                # (N, 4) attitude q_nb
    ba: np.ndarray               # (N, 3) accelerometer bias [m/s^2]
    bg: np.ndarray               # (N, 3) gyro bias [rad/s]
    sg: np.ndarray               # (N, 3) gyro scale-factor error [-]
    gp: np.ndarray               # (N, 3) correlated GNSS position error NED [m]
    hb: np.ndarray               # (N, 1) dual-antenna baseline azimuth misalignment [rad]
    sigma: np.ndarray            # (N, N_STATES) 1-sigma of the error state
    updates: dict = field(default_factory=dict)   # per measurement: t, nis, accepted
    outages: list = field(default_factory=list)   # [(t_start, t_end)] GPS TOW
    runtime_s: float = 0.0
    info: dict = field(default_factory=dict)


class ErrorStateFilter(ABC):
    name = "base"

    def __init__(self, system_cfg: dict, filter_cfg: dict):
        g = system_cfg["gnss"]
        self.lever_p = np.asarray(g["lever_arm_primary"], float)
        self.lever_s = np.asarray(g["lever_arm_secondary"], float)
        self.cfg = filter_cfg
        self.noise = NoiseParams.from_config(filter_cfg["imu_noise"])
        mc = filter_cfg["measurements"]
        self.models = {
            "pos": GnssPosition(self.lever_p, mc["position"]["std_floor"], mc["position"]["std_scale"]),
            "vel": GnssVelocity(self.lever_p, mc["velocity"]["std_floor"], mc["velocity"]["std_scale"]),
            "hdg": DualAntennaHeading(self.lever_p, self.lever_s, mc["heading"]["std_floor"],
                                      mc["heading"]["std_scale"]),
        }
        self.use = {"pos": mc["position"]["use"], "vel": mc["velocity"]["use"], "hdg": mc["heading"]["use"]}
        gc = filter_cfg["gating"]
        self.gating = gc["enabled"]
        self.gate = {d: chi2.ppf(gc["probability"], d) for d in (1, 3)}
        # Re-acquisition: after this many consecutive rejections of the GNSS *position* (the absolute
        # anchor), the filter admits it has drifted (e.g. after a long GNSS outage), inflates its position
        # covariance to the size of the innovation and accepts the fix. Without it, an overconfident filter
        # can lock out the GNSS forever. Velocity and heading are never forced (bad velocities under the
        # bridge must stay rejected); they are accepted again naturally once the position is re-anchored.
        self.reacquire_after = int(gc.get("reacquire_after", 10))
        self._rejections = {k: 0 for k in ("pos", "vel", "hdg")}
        self._block = {"pos": IR, "vel": IV, "hdg": slice(8, 9)}
        self.x: Nominal | None = None
        self.P: np.ndarray | None = None

    # ---- filter-specific -------------------------------------------------------------------------
    @abstractmethod
    def initialize(self, x0: Nominal, P0: np.ndarray):
        ...

    @abstractmethod
    def propagate(self, w_b: np.ndarray, f_b: np.ndarray, dt: float):
        """Predict over dt with bias-corrected rates (nominal state and covariance)."""

    @abstractmethod
    def correct(self, model, z, R, w_b, gate: float | None):
        """Measurement update. Returns (accepted, nis)."""

    # ---- shared ----------------------------------------------------------------------------------------
    def initial_covariance(self, pos_std) -> np.ndarray:
        c = self.cfg["initial_uncertainty"]
        s = np.zeros(N_STATES)
        s[IR] = np.maximum(pos_std, 0.02) if c["position"] == "from_gnss" else c["position"]
        s[IV] = c["velocity"]
        s[6:8] = np.radians(c["roll_pitch"])
        s[8] = np.radians(c["heading"])
        s[IBA] = c["accel_bias"] * MG
        s[IBG] = np.radians(c["gyro_bias"]) / 3600.0
        s[ISG] = c["gyro_scale_factor"] * PPM
        s[IGP] = self.noise.gp_sigma
        s[IHB] = np.radians(c["baseline_azimuth"])
        return np.diag(s**2)

    def run(self, ds, outages=(), t_end=None, verbose=True) -> FilterResult:
        """Run the filter over the dataset.

        ``outages``: [(start, duration)] in seconds since the start of the log; all GNSS measurements
        (position, velocity, heading) are discarded inside these windows.
        """
        x0, t_init, info = coarse_alignment(ds, self.lever_p, self.lever_s,
                                            self.cfg["alignment"]["static_duration"])
        dpsi = np.radians(self.cfg["alignment"].get("heading_perturbation_deg", 0.0))
        if dpsi:      # stress test: start with a wrong heading (rotation about the NED down axis)
            from ..attitude.rotations import quat_mult, rotvec_to_quat
            x0.q = quat_mult(rotvec_to_quat([0.0, 0.0, dpsi]), x0.q)
        self.initialize(x0, self.initial_covariance(info["pos_std"]))
        t_log0 = ds.imu.t[0]
        out_windows = [(t_log0 + a, t_log0 + a + d) for a, d in outages]

        # GNSS epochs (position and velocity share epochs; heading mostly does too)
        epochs = np.union1d(np.round(ds.pos.t, 3), np.round(ds.hdt.t, 3))
        ipos = {round(t, 3): i for i, t in enumerate(ds.pos.t)}
        ihdg = {round(t, 3): i for i, t in enumerate(ds.hdt.t)}

        t_imu = ds.imu.t
        k0 = int(np.searchsorted(t_imu, t_init))
        k1 = len(t_imu) if t_end is None else int(np.searchsorted(t_imu, t_log0 + t_end))
        n = k1 - k0
        res = {"lla": np.empty((n, 3)), "vel": np.empty((n, 3)), "q": np.empty((n, 4)),
               "ba": np.empty((n, 3)), "bg": np.empty((n, 3)), "sg": np.empty((n, 3)), "gp": np.empty((n, 3)), "hb": np.empty((n, 1)),
               "sigma": np.empty((n, N_STATES))}
        upd = {k: {"t": [], "nis": [], "accepted": []} for k in self.models}
        self._rejections = {k: 0 for k in self._rejections}

        w_avg = 0.5 * (ds.imu.gyr[:-1] + ds.imu.gyr[1:])
        f_avg = 0.5 * (ds.imu.acc[:-1] + ds.imu.acc[1:])

        # propagate from t_init to the first IMU epoch after it
        t_cur = t_init
        j = int(np.searchsorted(epochs, t_init, side="right"))
        tic = time.time()
        for k in range(k0, k1):
            t_k = t_imu[k]
            wi, fi = w_avg[k - 1], f_avg[k - 1]
            while j < len(epochs) and epochs[j] <= t_k + 1e-9:
                t_m = epochs[j]
                j += 1
                if t_m <= t_cur:
                    continue
                self.propagate(self.x.gyro(wi), self.x.accel(fi), t_m - t_cur)
                t_cur = t_m
                if any(a <= t_m <= b for a, b in out_windows):
                    continue
                w_b = self.x.gyro(wi)
                key = round(t_m, 3)
                if key in ipos:
                    i = ipos[key]
                    if ds.pos.sol_type[i] != 0:
                        for name, z, std in (("pos", ds.pos.lla[i], ds.pos.std[i]),
                                             ("vel", ds.vel.vel[i], ds.vel.std[i])):
                            if self.use[name]:
                                self._update(name, z, std, w_b, t_m, upd)
                if key in ihdg and self.use["hdg"]:
                    i = ihdg[key]
                    self._update("hdg", np.array([ds.hdt.heading[i]]), ds.hdt.heading_std[i], w_b, t_m, upd)
            if t_k > t_cur:
                self.propagate(self.x.gyro(wi), self.x.accel(fi), t_k - t_cur)
                t_cur = t_k
            r = k - k0
            x = self.x
            res["lla"][r] = x.lat, x.lon, x.h
            res["vel"][r] = x.v
            res["q"][r] = x.q
            res["ba"][r] = x.ba
            res["bg"][r] = x.bg
            res["sg"][r] = x.sg
            res["gp"][r] = x.gp
            res["hb"][r] = x.hb
            res["sigma"][r] = np.sqrt(np.maximum(np.diag(self.P), 0.0))
            if verbose and r % 40000 == 0 and r:
                print(f"  [{self.name}] {t_k - t_log0:7.1f} s  ({time.time() - tic:5.1f} s elapsed)")
        runtime = time.time() - tic
        reacq = upd.pop("reacquisitions", [])
        updates = {k: {"t": np.asarray(v["t"], float), "nis": np.asarray(v["nis"], float),
                       "accepted": np.asarray(v["accepted"], bool)} for k, v in upd.items()}
        updates["reacquisitions"] = reacq
        return FilterResult(self.name, t_imu[k0:k1].copy(), updates=updates,
            outages=out_windows, runtime_s=runtime, info=info, **res)

    def _update(self, name, z, std, w_b, t, log):
        model = self.models[name]
        R = model.covariance(std)
        gate = self.gate[model.dim] if self.gating else None
        accepted, nis = self.correct(model, z, R, w_b, gate)
        if accepted:
            self._rejections[name] = 0
        else:
            self._rejections[name] += 1
            if name == "pos" and self._rejections[name] >= self.reacquire_after:
                y = model.residual(z, model.predict(self.x, w_b))
                b = self._block[name]
                self.P[b, b] += np.diag(np.atleast_1d(y) ** 2)
                accepted, nis = self.correct(model, z, R, w_b, None)
                self._rejections[name] = 0
                log.setdefault("reacquisitions", []).append((t, name))
        log[name]["t"].append(t)
        log[name]["nis"].append(nis)
        log[name]["accepted"].append(accepted)
