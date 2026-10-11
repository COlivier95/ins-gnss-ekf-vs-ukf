"""Shared matplotlib style for all figures (consistent colors and recessive axes).

Series colors follow a fixed categorical order validated for color-vision deficiency
(first three slots are safe even when all series overlap). Colors follow the entity:
the reference, the ES-EKF and the UKF always keep the same color in every figure.
"""
from __future__ import annotations

import matplotlib as mpl

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

COLORS = {
    "reference": "#52514e",   # neutral ink: the reference is context, not a competitor
    "es_ekf": "#2a78d6",      # slot 1 - blue
    "ukf": "#eb6834",         # slot 2 - orange
    "gnss": "#1baf7a",        # slot 3 - aqua
    "outage": "#eda100",      # shaded spans only (never a line color next to orange)
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "font.size": 9,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
