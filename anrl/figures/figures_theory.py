"""Paper-one (theory) figures: statewise validation, pilot estimation, weight structure.

Built from the saved PASS 47/48 artifacts; no experiment is re-run here.  Same conventions as
:mod:`anrl.figures.figures` -- each builder returns ``(fig, csv_header, csv_rows, caption)``, is
registered in :data:`THEORY_FIGURES`, and is rendered to PDF + PNG + a tidy CSV of the exact
plotted data by ``experiments/pass48_make_theory_figures.py``.

* ``fig7_statewise_validation``  -- predicted vs measured per-state RMSE, varying-estimand
  ensemble against the fixed-estimand negative control.
* ``fig8_pilot_convergence``     -- relative error of the pilot-estimated ``M*`` versus budget.
* ``fig9_pilot_over_mstar``      -- the pilot cost relative to the threshold it locates, vs n.
* ``fig10_weight_truncation``    -- the two projection variances sit at opposite ends of the
  Pauli weight spectrum.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import OKABE_ITO

R = Path(__file__).resolve().parent.parent.parent / "results"

C_VARY = OKABE_ITO["blue"]          # varying-estimand ensemble (the informative one)
C_FIXED = OKABE_ITO["vermillion"]   # fixed-estimand negative control
C_Z1 = OKABE_ITO["vermillion"]      # zeta_1, the hard / high-weight factor
C_Z2 = OKABE_ITO["blue"]            # zeta_2, the easy / low-weight factor
C_REF = OKABE_ITO["black"]
_N_SHADES = (OKABE_ITO["sky"], OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["yellow"],
             OKABE_ITO["orange"], OKABE_ITO["vermillion"], OKABE_ITO["purple"])


def _load(name: str) -> dict:
    return json.loads((R / name).read_text())


def _clean_log_ticks(ax, which: str = "both") -> None:
    """Drop minor-tick labels on log axes so decade labels do not collide."""
    import matplotlib.ticker as mt
    for axis in ((ax.xaxis,) if which == "x" else (ax.yaxis,) if which == "y"
                 else (ax.xaxis, ax.yaxis)):
        axis.set_major_locator(mt.LogLocator(base=10.0))
        axis.set_minor_formatter(mt.NullFormatter())


# --------------------------------------------------------------------------- figure 7
def make_fig7():
    """Predicted vs measured per-state RMSE: varying ensemble vs the fixed-estimand control."""
    d = _load("pass47_statewise_ranking.json")
    units, cfg = d["units"], d["config"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharex=True, sharey=True)
    rows = []
    panels = [("variable_rank", "varying estimand (variable-rank Ginibre)", C_VARY, axes[0]),
              ("noisy_pure_q0.1", "fixed estimand (noisy-pure, control)", C_FIXED, axes[1])]
    series = {}
    for ens, _title, _color, _ax in panels:
        pr, me, se = [], [], []
        for n in cfg["sizes"]:
            for b in cfg["budgets"]:
                for s in range(cfg["n_states"]):
                    u = units[f"{ens}|{n}|{s}"]
                    pr.append(u[f"predicted_{b}"])
                    me.append(u[f"measured_{b}"])
                    se.append(u[f"measured_se_{b}"])
                    rows.append([ens, n, b, s, u["m_star"], u[f"predicted_{b}"],
                                 u[f"measured_{b}"], u[f"measured_se_{b}"]])
        series[ens] = (np.array(pr), np.array(me), np.array(se))
    # A shared range: both panels plot the same quantity at the same sizes and budgets, so the
    # comparison of scatter about y = x is only fair on one scale.
    allv = np.concatenate([np.concatenate(v[:2]) for v in series.values()])
    lo, hi = allv.min() * 0.78, allv.max() * 1.28
    for ens, title, color, ax in panels:
        pr, me, se = series[ens]
        ax.plot([lo, hi], [lo, hi], "--", color=C_REF, lw=0.9, zorder=1, label="y = x")
        ax.errorbar(pr, me, yerr=2 * se, fmt="o", color=color, ms=3.4, lw=0.7,
                    alpha=0.85, zorder=3, label="states ($\\pm 2$ SE)")
        cells = [c for c in d["cells"] if c["ensemble"] == ens]
        rho = float(np.mean([c["spearman"] for c in cells]))
        slope = float(np.mean([c["slope_measured_on_predicted"] for c in cells]))
        spread = float(np.mean([c["predicted_rel_spread"] for c in cells]))
        noise = float(np.mean([c["measurement_rel_noise"] for c in cells]))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        _clean_log_ticks(ax)
        ax.set_xlabel("predicted RMSE (exact statewise law)")
        ax.set_title(title)
        ax.text(0.04, 0.955,
                f"Spearman $\\rho = {rho:+.2f}$\nslope $= {slope:.2f}$\n"
                f"spread/noise $= {spread/noise:.2f}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6))
        ax.legend(loc="lower right")
    axes[0].set_ylabel("measured RMSE (forward simulation)")
    fig.tight_layout()
    caption = (
        "Figure 7. Statewise validation of the exact projection-variance law. Predicted versus measured "
        "single-copy RMSE for individual states at $k=2$, $n=3,4,5$ and two budgets, "
        f"{cfg['n_states']} states per cell, {cfg['n_trials']} forward-simulation trials each. "
        "\\emph{The predictions are computed from the exact statewise $\\zeta_1,\\zeta_2$ of each "
        "state with nothing fitted to the plotted points.} Left: a varying-estimand ensemble, "
        "where the statewise threshold spans an order of magnitude, so the law is tested "
        "state by state. Right: the fixed-estimand noisy-pure family as a negative control -- "
        "its statewise spread is smaller than the measurement noise, so no ordering can be "
        "resolved there and a law reading only the ensemble mean would score the same.")
    return fig, ["ensemble", "n", "budget", "state", "m_star", "predicted_rmse",
                 "measured_rmse", "measured_se"], rows, caption


# --------------------------------------------------------------------------- figure 8
def _pilot_series() -> dict:
    """Relative error of the pilot-estimated M* by n, merging PASS 47 and the 48.2 extension."""
    out: dict[int, dict] = {}
    d47 = _load("pass47_pilot_estimator.json")
    for n in (2, 3, 4, 5, 6):
        key = f"noisy_pure_q0.1|n{n}"
        if key in d47["summary"]:
            s = d47["summary"][key]
            out[n] = {"budgets": list(d47["config"]["pilots"]),
                      "mad": list(s["m_star_rel_mad"]),
                      "m_star": s["m_star_exact_median"], "source": "pass47"}
    p48 = R / "pass48_pilot_extension.json"
    if p48.exists():
        d48 = json.loads(p48.read_text())
        for key, s in d48["summary"].items():
            ens, nn = key.split("|n")
            if ens != "noisy_pure_q0.1":
                continue
            n = int(nn)
            budgets = next(list(b) for (e, m, b, _ns, _r) in
                           [tuple(c) for c in d48["config"]["cells"]] if e == ens and m == n)
            out[n] = {"budgets": budgets, "mad": list(s["m_star_rel_mad"]),
                      "m_star": s["m_star_exact_median"],
                      "source": "pass48" if n >= 7 else "pass48 (gate replay)"}
    return dict(sorted(out.items()))


def make_fig8():
    """Relative error of the pilot-estimated M* versus pilot budget, one curve per n."""
    series = _pilot_series()
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    rows = []
    for i, (n, s) in enumerate(series.items()):
        b = np.array(s["budgets"], float)
        v = np.array([np.nan if x is None else x for x in s["mad"]], float)
        ax.plot(b, v * 100, "o-", color=_N_SHADES[i % len(_N_SHADES)], ms=3.2, lw=1.2,
                label=f"$n={n}$")
        for bb, vv in zip(s["budgets"], s["mad"]):
            rows.append([n, bb, None if vv is None else vv * 100, s["m_star"], s["source"]])
    ax.axhline(10.0, ls="--", color=C_REF, lw=0.9)
    ax.text(0.98, 11.0, "10%", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color=C_REF)
    ax.set_xscale("log"); ax.set_yscale("log")
    _clean_log_ticks(ax)
    ax.set_xlabel("pilot budget M (snapshots)")
    ax.set_ylabel("relative error of $\\widehat{M^*}$ (%)")
    ax.legend(ncol=2, loc="lower left")
    fig.tight_layout()
    caption = (
        "Figure 8. Estimating the threshold from a pilot budget. Median relative error of the "
        "pilot-estimated $\\widehat{M^*}$ against the exact statewise $M^*$ for the same state, "
        "versus pilot budget, on the noisy-pure ensemble. The estimator uses only the snapshots: "
        "$\\zeta_2$ from the sample variance of $\\mathrm{Tr}(G_iG_j)$ over disjoint pairs and "
        "$\\zeta_1$ from a four-block construction, both unbiased. Error falls close to "
        "$M^{-1/2}$; the dashed line marks $10\\%$ accuracy.")
    return fig, ["n", "pilot_budget", "m_star_rel_error_pct", "m_star_exact", "source"], rows, caption


# --------------------------------------------------------------------------- figure 9
def make_fig9():
    """Pilot cost relative to the threshold it locates, versus n -- measured where available."""
    series = _pilot_series()
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    rows = []
    ns, ratio_meas, ratio_fit, kinds = [], [], [], []
    for n, s in series.items():
        b = np.array(s["budgets"], float)
        v = np.array([np.nan if x is None else x for x in s["mad"]], float)
        under = b[np.isfinite(v) & (v < 0.10)]
        first = float(under.min()) if under.size else np.nan
        ok = np.isfinite(v) & (v < 0.60)
        fit = np.nan
        if ok.sum() >= 2:
            c = np.polyfit(np.log(b[ok]), np.log(v[ok]), 1)
            fit = float(np.exp((np.log(0.10) - c[1]) / c[0]))
        ns.append(n)
        ratio_meas.append(first / s["m_star"])
        ratio_fit.append(fit / s["m_star"])
        kinds.append(s["source"])
        rows.append([n, s["m_star"], first, fit, first / s["m_star"], fit / s["m_star"], s["source"]])
    ns = np.array(ns, float)
    rm, rf = np.array(ratio_meas), np.array(ratio_fit)
    ax.plot(ns, rf, "s--", color=C_FIXED, ms=3.4, lw=1.1, alpha=0.85,
            label="fitted 10% budget / $M^*$")
    ax.plot(ns, rm, "o-", color=C_VARY, ms=4.2, lw=1.4,
            label="measured 10% budget / $M^*$")
    ax.axhline(1.0, ls="--", color=C_REF, lw=0.9)
    ax.text(ns.min() + 0.05, 1.12, "pilot $=M^*$", fontsize=7, color=C_REF)
    # Locate the crossing from the two measured points that BRACKET ratio 1, not from a global
    # fit: at small n the swept budget grid is coarse, so those ratios are upper bounds and a
    # global fit through them is dragged by grid granularity rather than by the trend.
    ok = np.isfinite(rm)
    above = [i for i in range(len(ns)) if ok[i] and rm[i] > 1.0]
    below = [i for i in range(len(ns)) if ok[i] and rm[i] <= 1.0]
    if above and below:
        i, j = max(above), min(below)
        c = np.polyfit([ns[i], ns[j]], [np.log(rm[i]), np.log(rm[j])], 1)
        cross = -c[1] / c[0]
        ax.plot([cross], [1.0], "*", color=C_VARY, ms=12, zorder=5,
                label=f"crossing at $n \\approx {cross:.1f}$")
        ax.annotate("", xy=(ns[j], rm[j]), xytext=(ns[i], rm[i]),
                    arrowprops=dict(arrowstyle="-", color=C_VARY, lw=2.4, alpha=0.35))
    ax.set_yscale("log")
    _clean_log_ticks(ax, "y")
    ax.set_xlabel("system size n (qubits)")
    ax.set_ylabel("pilot budget for 10% accuracy / $M^*$")
    ax.set_xticks([int(x) for x in ns])
    ax.legend(loc="upper right")
    fig.tight_layout()
    caption = (
        "Figure 9. The pilot overhead becomes relatively cheaper with size. Ratio of the pilot budget "
        "needed to pin $M^*$ to $10\\%$ to the threshold $M^*$ itself, versus $n$. Circles are "
        "the smallest swept budget that achieved $10\\%$, with no extrapolation, so they are "
        "upper bounds set by the granularity of the swept grid -- which is why the first three "
        "sit at the same budget; squares are the log-log fit through the usable range. The star "
        "marks where the measured ratio crosses unity, interpolated between the two measured "
        "sizes that bracket it. Below the dashed line the pilot costs less than the threshold it "
        "locates: the pilot budget grows about twofold per qubit against $M^*$'s fivefold, so "
        "the overhead becomes relatively cheaper exactly where the criterion matters.")
    return fig, ["n", "m_star_exact", "measured_10pct_budget", "fitted_10pct_budget",
                 "measured_ratio", "fitted_ratio", "source"], rows, caption


# -------------------------------------------------------------------------- figure 10
def make_fig10():
    """The two projection variances sit at opposite ends of the Pauli weight spectrum."""
    d = _load("pass47_statewise_mstar.json")["input_requirements_47_2b"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    rows = []
    for key, color, marker, label in (("noisy_pure_q0.1|n6", C_Z2, "o", None),
                                      ("haar_pure|n6", C_Z2, "s", None)):
        # w = n is exact by construction (nothing is truncated), so its error is identically
        # zero; plotting it on a log axis would invent a data point. Keep w < n only.
        t = [r for r in d[key]["truncation"] if r["max_weight"] < 6]
        w = [r["max_weight"] for r in t]
        ax.plot(w, [r["zeta2_rel_err"] * 100 for r in t], marker + "-",
                color=color, ms=3.2, lw=1.2, alpha=0.9,
                label="$\\zeta_2$ (kernel $14^{-|P|}$)" if key.startswith("noisy") else None)
        ax.plot(w, [r["zeta1_diag_rel_err"] * 100 for r in t], marker + "-",
                color=C_Z1, ms=3.2, lw=1.2, alpha=0.9,
                label="$\\zeta_1$ diagonal (kernel $3^{|P|}$)" if key.startswith("noisy") else None)
        for r in d[key]["truncation"]:
            rows.append([key, r["max_weight"], r["n_terms"], r["frac_of_4n"],
                         r["zeta2_rel_err"] * 100, r["zeta1_diag_rel_err"] * 100])
    ax.set_yscale("log")
    _clean_log_ticks(ax, "y")
    ax.set_xlabel("Pauli weight cutoff w (terms with $|P| \\leq w$ kept)")
    ax.set_ylabel("relative error of the truncated sum (%)")
    ax.set_xticks(range(0, 6))
    ax.legend(loc="center left")
    fig.tight_layout()
    caption = (
        "Figure 10. The two projection variances read opposite ends of the Pauli spectrum. Relative error "
        "of each functional when the sum over Pauli strings is truncated to weight $|P| \\leq w$, "
        "at $n = 6$ (circles noisy-pure, squares Haar-pure). $\\zeta_2$ carries the kernel "
        "$14^{-|P|}$, which suppresses high weight: keeping $3.8\\%$ of the strings already gives "
        "it to $0.3\\%$. The $\\zeta_1$ diagonal carries $3^{|P|}$, which amplifies high weight: "
        "keeping $82\\%$ of the strings still leaves $54\\%$ error, so the top weight shell alone "
        "carries more than half. High-weight expectations are exactly what local shadows estimate "
        "worst, which is why $\\zeta_1$ is the hard factor -- and why the pilot estimator of "
        "Fig.~8, which never forms the spectrum, is the practical route.")
    return fig, ["case", "max_weight", "n_terms", "frac_of_4n", "zeta2_rel_err_pct",
                 "zeta1_diag_rel_err_pct"], rows, caption


THEORY_FIGURES = {
    "fig7_statewise_validation": make_fig7,
    "fig8_pilot_convergence": make_fig8,
    "fig9_pilot_over_mstar": make_fig9,
    "fig10_weight_truncation": make_fig10,
}
