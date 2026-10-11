"""Readers for the SBG Systems text exports (sbgCenter) and the Qinertia reference trajectory.

Every stream is returned on a common time base: **GPS time of week [s]**.

* GNSS messages carry their measurement epoch in ``gpsTow`` [ms]: it is used directly, so the
  receiver output latency does not bias the time tagging.
* The IMU only carries the device timestamp [us]. It is converted to GPS time with the constant
  offset ``gpsTow - timestamp`` observed on the GNSS position messages (it is constant to the
  microsecond over the whole log because the device clock is disciplined by GNSS PPS).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..geodesy.geoid import LinearGeoid

DEG = np.pi / 180.0

# sbgECom GNSS position solution types (bits 6..11 of the status word)
GNSS_POS_TYPES = {
    0: "NO_SOLUTION", 1: "UNKNOWN", 2: "SINGLE", 3: "PSRDIFF", 4: "SBAS", 5: "OMNISTAR",
    6: "RTK_FLOAT", 7: "RTK_INT", 8: "PPP_FLOAT", 9: "PPP_INT", 10: "FIXED",
}


def _read_sbg_table(path: Path) -> pd.DataFrame:
    """Read a whitespace-separated sbgCenter export (header line + units line)."""
    return pd.read_csv(path, sep=r"\s+", skiprows=[1], dtype={"status": str, "statusExt": str})


def _hex(col: pd.Series) -> np.ndarray:
    return np.array([int(s, 16) for s in col], dtype=np.int64)


@dataclass
class ImuData:
    t: np.ndarray          # GPS TOW [s], (N,)
    acc: np.ndarray        # specific force in body FRD [m/s^2], (N, 3)
    gyr: np.ndarray        # angular rate in body FRD [rad/s], (N, 3)
    temperature: np.ndarray


@dataclass
class GnssPos:
    t: np.ndarray          # GPS TOW of the measurement epoch [s]
    lla: np.ndarray        # (lat [rad], lon [rad], ellipsoidal height [m]) of the primary antenna
    std: np.ndarray        # 1-sigma (north, east, down) [m]
    sol_type: np.ndarray   # sbgECom solution type code (see GNSS_POS_TYPES)
    num_sv: np.ndarray     # satellites used
    undulation: np.ndarray # geoid undulation N [m] (h_ellipsoid = h_msl + N)


@dataclass
class GnssVel:
    t: np.ndarray
    vel: np.ndarray        # NED velocity of the primary antenna [m/s]
    std: np.ndarray        # 1-sigma NED [m/s]


@dataclass
class GnssHdt:
    t: np.ndarray
    heading: np.ndarray    # dual-antenna baseline azimuth [rad] (true north, clockwise)
    heading_std: np.ndarray
    pitch: np.ndarray      # baseline elevation [rad]
    pitch_std: np.ndarray


@dataclass
class Reference:
    t: np.ndarray          # GPS TOW [s]
    euler: np.ndarray      # (roll, pitch, yaw) [rad], ZYX
    lla: np.ndarray        # (lat [rad], lon [rad], ellipsoidal height [m]) of the IMU
    altitude_type: str     # altitude delivered in the file: "ellipsoid" or "msl" (converted)
    pos_std: np.ndarray | None = None   # (N, 3) 1-sigma N/E/D [m] reported by Qinertia (if exported)
    att_std: np.ndarray | None = None   # (N, 3) 1-sigma roll/pitch/yaw [rad]


@dataclass
class Dataset:
    imu: ImuData
    pos: GnssPos
    vel: GnssVel
    hdt: GnssHdt
    ref: Reference
    tow_offset: float = field(default=0.0)   # GPS TOW - device time [s]

    @property
    def t0(self) -> float:
        return float(self.imu.t[0])


def load_dataset(data_dir: str | Path = "data", config: str | Path | dict | None = None) -> Dataset:
    """Load the full acquisition (IMU, GNSS position/velocity/heading, reference).

    ``config`` is the system configuration (path or dict, default ``<data_dir>/../config/system.yaml``).
    Its ``reference.geoid_control_points`` define the geoid used to convert the reference MSL altitude
    to ellipsoidal height. Without it, the receiver's own undulation is used (not recommended: the
    reference and the receiver may use different geoid models).
    """
    d = Path(data_dir)
    if config is None:
        config = d.resolve().parent / "config" / "system.yaml"
    if not isinstance(config, dict):
        config = yaml.safe_load(open(config)) if Path(config).exists() else {}
    raw, refdir = d / "raw", d / "reference"

    pos_df = _read_sbg_table(raw / "gnss2Pos.txt.gz")
    offsets = pos_df["gpsTow"].to_numpy() * 1e-3 - pos_df["timestamp"].to_numpy() * 1e-6
    tow_offset = float(np.median(offsets))
    if np.ptp(offsets) > 1e-3:
        raise ValueError(f"Device time / GPS time offset is not constant (spread {np.ptp(offsets):.4f} s)")

    # --- GNSS position --------------------------------------------------------------------------
    st = _hex(pos_df["status"])
    und = pos_df["undulation"].to_numpy()
    pos = GnssPos(
        t=pos_df["gpsTow"].to_numpy() * 1e-3,
        lla=np.column_stack([pos_df["latitude"] * DEG, pos_df["longitude"] * DEG,
                             pos_df["altitude"] + und]),
        std=pos_df[["latitudeStd", "longitudeStd", "altitudeStd"]].to_numpy(),
        sol_type=(st >> 6) & 0x3F,
        num_sv=pos_df["numSvUsed"].to_numpy(),
        undulation=und,
    )

    # --- GNSS velocity --------------------------------------------------------------------------
    vel_df = _read_sbg_table(raw / "gnss2Vel.txt.gz")
    vel = GnssVel(
        t=vel_df["gpsTow"].to_numpy() * 1e-3,
        vel=vel_df[["velN", "velE", "velD"]].to_numpy(),
        std=vel_df[["velStdN", "velStdE", "velStdD"]].to_numpy(),
    )

    # --- GNSS dual-antenna heading -------------------------------------------------------------
    hdt_df = _read_sbg_table(raw / "gnss2Hdt.txt.gz")
    hdt = GnssHdt(
        t=hdt_df["gpsTow"].to_numpy() * 1e-3,
        heading=hdt_df["heading"].to_numpy() * DEG,
        heading_std=hdt_df["headingStd"].to_numpy() * DEG,
        pitch=hdt_df["pitch"].to_numpy() * DEG,
        pitch_std=hdt_df["pitchStd"].to_numpy() * DEG,
    )

    # --- IMU ------------------------------------------------------------------------------------
    imu_df = _read_sbg_table(raw / "imuShort.txt.gz")
    imu = ImuData(
        t=imu_df["timestamp"].to_numpy() * 1e-6 + tow_offset,
        acc=imu_df[["accelX", "accelY", "accelZ"]].to_numpy(),
        gyr=imu_df[["gyroX", "gyroY", "gyroZ"]].to_numpy() * DEG,
        temperature=imu_df["temperature"].to_numpy(),
    )

    # --- Reference (Qinertia export) -----------------------------------------------------------
    # Preferred: ellipsoidal height export. Fallback: MSL export converted with the geoid given in the
    # configuration (Qinertia uses EGM2008 while the Trimble receiver uses EGM96).
    ref_files = sorted(refdir.glob("Reference_Trajectory*.txt*"))
    if not ref_files:
        raise FileNotFoundError(f"No reference trajectory in {refdir}")
    ref_df = pd.read_csv(ref_files[0], sep="\t", skiprows=[1], na_values=["N/A"])
    alt_col = ref_df.columns[6]
    has_std = ref_df.shape[1] >= 13
    std_df = ref_df.iloc[:, 7:13].to_numpy(float) if has_std else None
    ref_df = ref_df.iloc[:, :7]
    ref_df.columns = ["t", "roll", "pitch", "yaw", "lat", "lon", "alt"]
    t_ref = ref_df["t"].to_numpy()
    lat_r, lon_r = ref_df["lat"].to_numpy() * DEG, ref_df["lon"].to_numpy() * DEG
    alt = ref_df["alt"].to_numpy()
    if "ellipsoid" in alt_col.lower():
        h_ref = alt
    else:
        pts = config.get("reference", {}).get("geoid_control_points")
        und = LinearGeoid(pts)(lat_r, lon_r) if pts else np.interp(t_ref, pos.t, pos.undulation)
        h_ref = alt + und
    ref = Reference(
        t=t_ref,
        euler=ref_df[["roll", "pitch", "yaw"]].to_numpy() * DEG,
        lla=np.column_stack([lat_r, lon_r, h_ref]),
        altitude_type="ellipsoid" if "ellipsoid" in alt_col.lower() else "msl",
    )
    if has_std:
        # columns: roll, pitch, yaw std [deg] then latitude, longitude, altitude std [m]; the last epochs
        # are N/A or 0 (end of the smoothed solution) -> NaN
        pos_std = std_df[:, 3:6].copy()
        pos_std[pos_std <= 0] = np.nan
        ref.pos_std = pos_std
        ref.att_std = np.radians(std_df[:, 0:3])

    return Dataset(imu=imu, pos=pos, vel=vel, hdt=hdt, ref=ref, tow_offset=tow_offset)
