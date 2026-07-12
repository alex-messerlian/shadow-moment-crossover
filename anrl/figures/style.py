"""Shared publication style: colorblind-safe palette, print-legible fonts, savers.

All figures import :func:`apply_style` and the palette here so fonts and colors are
consistent across the set.  Targets a 3.5-inch single-column width in a two-column
paper (small multiples go full 7-inch width); tick/axis/legend text stays legible
at that size.  :func:`save_figure` writes a vector PDF (paper) and a 300-dpi PNG
(slides/web); :func:`write_csv` exports the exact plotted data (tidy, one row per
point) so the numbers are recoverable without rerunning anything.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "figures"

# Okabe-Ito colorblind-safe palette (https://jfly.uni-koeln.de/color/).
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
# Fixed semantic assignments, consistent across every figure.
C_SINGLE = OKABE_ITO["vermillion"]   # single-copy estimator
C_COLL = OKABE_ITO["green"]          # collective estimator
C_TRUE = OKABE_ITO["black"]          # the true quantity Tr(rho^k)
C_K = {2: OKABE_ITO["blue"], 3: OKABE_ITO["orange"], 4: OKABE_ITO["purple"]}  # moment order
C_ENS = {"haar_pure": OKABE_ITO["blue"], "low_rank": OKABE_ITO["orange"], "ghz_noisy": OKABE_ITO["green"]}
NOISE_LABEL = {"depolarizing": "Depolarizing", "amplitude_damping": "Amp. damping", "dephasing": "Dephasing"}


def apply_style() -> None:
    """Set global rcParams for consistent, print-legible, chartjunk-free figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "errorbar.capsize": 2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": False,
        "pdf.fonttype": 42,  # embed TrueType so text stays editable/searchable
        "ps.fonttype": 42,
    })


def save_figure(fig, name: str) -> list[Path]:
    """Save ``fig`` as both PDF (vector) and PNG (300 dpi); return the paths."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def write_csv(name: str, header: list[str], rows: list[list]) -> Path:
    """Write the exact plotted data as a tidy CSV (one row per point)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / f"{name}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return p
