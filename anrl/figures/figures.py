"""The five publication figures.  Each ``make_figN`` returns ``(fig, header, rows,
caption)``: the Matplotlib figure, and the exact plotted data as a tidy CSV table.

Measured numbers are read from the saved results JSON; theory curves are the
parameter-free predictions recomputed from :mod:`anrl.theory` (never fit to these
points).  All figures share :mod:`anrl.figures.style` for colors and fonts.
"""

from __future__ import annotations

import numpy as np

from . import data as D
from .style import (
    C_COLL,
    C_ENS,
    C_K,
    C_SINGLE,
    C_TRUE,
    NOISE_LABEL,
    OKABE_ITO,
)
import matplotlib.pyplot as plt

NOISE_ORDER = ("depolarizing", "amplitude_damping", "dephasing")
RATE_ORDER = (0.05, 0.1)


def _err(lo: float, hi: float, y: float) -> list[float]:
    """Asymmetric error-bar half-widths [lower, upper] from a CI and the point."""
    return [max(0.0, y - lo), max(0.0, hi - y)]


def _clean_log_ticks(ax, which: str = "both") -> None:
    """Explicit, plainly-formatted log ticks (avoid mangled auto minor labels)."""
    import matplotlib.ticker as mticker

    ticks = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15]

    def fmt(v, _):
        return f"{v:g}"

    for axis, setter in (("x", ax.xaxis), ("y", ax.yaxis)):
        if which in (axis, "both"):
            setter.set_major_locator(mticker.FixedLocator(ticks))
            setter.set_minor_locator(mticker.NullLocator())
            setter.set_major_formatter(mticker.FuncFormatter(fmt))


# ===========================================================================
# Figure 1 — the crossover map (small multiples: 3 noise x 2 rate, k=2)
# ===========================================================================
def make_fig1():
    bs = D.load("budget_scaling.json")
    comps = D.zeta_components()
    xover = {(e["noise_model"], e["rate"]): e["crossover_n"]
             for e in bs["crossover_table"] if e["k"] == 2 and e["budget"] == 2000}
    budget = 2000
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.6), sharex=True, sharey=True)
    rows = []
    for i, nm in enumerate(NOISE_ORDER):
        for j, g in enumerate(RATE_ORDER):
            ax = axes[i, j]
            cells = sorted([r for r in bs["rows"] if r["k"] == 2 and r["budget"] == budget
                            and r["noise_model"] == nm and r["rate"] == g], key=lambda r: r["n"])
            ns = [r["n"] for r in cells]
            # shaded "collective wins" region
            xc = xover.get((nm, g))
            if xc is not None:
                ax.axvspan(xc - 0.5, max(ns) + 0.5, color=C_COLL, alpha=0.08, lw=0)
                ax.axvline(xc - 0.5, color=C_COLL, lw=0.8, ls=":", alpha=0.7)
            # measured single-copy + collective (points w/ CI)
            for series, color, key, cikey in (("single", C_SINGLE, "single_rmse", "single_rmse_ci68"),
                                              ("collective", C_COLL, "collective_rmse", "collective_rmse_ci68")):
                y = [r[key] for r in cells]
                yerr = np.array([_err(r[cikey][0], r[cikey][1], r[key]) for r in cells]).T
                ax.errorbar(ns, y, yerr=yerr, fmt="o", color=color, ms=3.4, lw=1.0, zorder=3)
                for r in cells:
                    rows.append([nm, g, 2, r["n"], series, "measured", r[key], r[cikey][0], r[cikey][1]])
            # theory curves (parameter-free)
            th_n = [n for n in ns if (n, 2) in comps]
            ts = [D.theory_single_rmse(n, 2, budget, comps) for n in th_n]
            tc = [D.theory_collective_rmse(n, 2, nm, g, budget) for n in th_n]
            ax.plot(th_n, ts, "-", color=C_SINGLE, lw=1.3, zorder=2)
            ax.plot(th_n, tc, "-", color=C_COLL, lw=1.3, zorder=2)
            for n, s, c in zip(th_n, ts, tc):
                rows.append([nm, g, 2, n, "single", "theory", s, "", ""])
                rows.append([nm, g, 2, n, "collective", "theory", c, "", ""])
            ax.set_yscale("log")
            ax.set_xticks(ns)
            if i == 0:
                ax.set_title(f"noise rate g = {g}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{NOISE_LABEL[nm]}\nRMSE", fontsize=8.5)
            if i == 2:
                ax.set_xlabel("system size n (qubits)")
    # one shared legend
    handles = [
        plt.Line2D([], [], color=C_SINGLE, marker="o", ls="-", ms=3.4, label="single-copy (points: measured, line: theory)"),
        plt.Line2D([], [], color=C_COLL, marker="o", ls="-", ms=3.4, label="collective (points: measured, line: theory)"),
        plt.Line2D([], [], color=C_COLL, lw=6, alpha=0.15, label="collective wins (n ≥ n*)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=7.2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    caption = (
        "Figure 1. The crossover, k=2 (purity). Single-copy classical-shadow RMSE (vermillion) "
        "and collective 2-copy RMSE (green) versus system size n, at a fixed 2000-copy budget, for "
        "three noise channels (rows) and two rates (columns). Points are measured (48 states x 8 "
        "trials, 68% bootstrap CIs); solid lines are the PARAMETER-FREE theory (exact bias law + "
        "exact-Hoeffding single-copy variance), not fit to these points. Single-copy error grows "
        "exponentially while the collective error saturates at a budget-independent bias floor; "
        "shading marks where collective wins (n >= n*). The reader should conclude that a genuine "
        "crossover exists and that its location is predicted, not fitted.")
    header = ["noise_model", "rate", "k", "n", "series", "kind", "rmse", "ci_lo", "ci_hi"]
    return fig, header, rows, caption


# ===========================================================================
# Figure 2 — the crossover boundary (predicted vs measured n*)
# ===========================================================================
def make_fig2():
    ct = D.load("crossover_theory.json")["comparison"]
    st4 = D.load("stress_test.json")["part4"]
    pts = []  # (measured, predicted, ensemble, label)
    rows = []
    for c in ct:
        if c["measured_n"] is not None and c["predicted_n_exact"] is not None:
            pts.append((c["measured_n"], c["predicted_n_exact"], "noisy_pure"))
            rows.append(["noisy_pure(dev)", "noisy_pure", c["k"], c["noise_model"], c["rate"],
                         c["budget"], c["measured_n"], c["predicted_n_exact"]])
    for c in st4:
        if c["measured_n"] is not None and c["predicted_n"] is not None:
            pts.append((c["measured_n"], c["predicted_n"], c["ensemble"]))
            rows.append(["stress", c["ensemble"], 2, c["noise"], c["rate"], 2000,
                         c["measured_n"], c["predicted_n"]])
    meas = np.array([p[0] for p in pts]); pred = np.array([p[1] for p in pts])
    within1 = np.mean(np.abs(pred - meas) <= 1); exact = np.mean(pred == meas)

    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    lo, hi = 1.5, max(meas.max(), pred.max()) + 0.5
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color=OKABE_ITO["sky"], alpha=0.15, lw=0,
                    label="within ±1 qubit")
    ax.plot([lo, hi], [lo, hi], color=C_TRUE, lw=1.0, ls="--", label="perfect (y = x)")
    rng = np.random.default_rng(0)
    for ens, color in [("noisy_pure", OKABE_ITO["black"]), ("haar_pure", C_ENS["haar_pure"]),
                       ("low_rank", C_ENS["low_rank"]), ("ghz_noisy", C_ENS["ghz_noisy"])]:
        m = [p[0] for p in pts if p[2] == ens]; pr = [p[1] for p in pts if p[2] == ens]
        if not m:
            continue
        jx = rng.uniform(-0.13, 0.13, len(m)); jy = rng.uniform(-0.13, 0.13, len(m))
        lab = "noisy-pure (dev)" if ens == "noisy_pure" else ens
        ax.scatter(np.array(m) + jx, np.array(pr) + jy, s=16, color=color, alpha=0.8,
                   edgecolors="white", linewidths=0.3, label=lab, zorder=3)
    ax.set_xlabel("measured crossover n*")
    ax.set_ylabel("predicted crossover n*")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks(range(2, int(hi) + 1)); ax.set_yticks(range(2, int(hi) + 1))
    ax.text(0.03, 0.97, f"N = {len(pts)} cells\nwithin ±1: {within1:.0%}\nexact: {exact:.0%}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    ax.legend(loc="lower right", fontsize=6.3, handletextpad=0.4)
    fig.tight_layout()
    caption = (
        "Figure 2. The crossover boundary. Predicted versus measured crossover size n* for every "
        f"resolved cell we have (N = {len(pts)}): all moment orders k, all three noise channels, "
        "all rates, all copy budgets, and all four ensembles (points jittered; integer n*). The "
        "prediction is the parameter-free theory (exact Hoeffding variance vs the exact bias floor). "
        f"{within1:.0%} of cells fall within +/-1 qubit of the diagonal and {exact:.0%} are exact. "
        "The reader should conclude that the law locates the crossover across the entire study, "
        "including states the theory never saw.")
    header = ["source", "ensemble", "k", "noise_model", "rate", "budget", "measured_n", "predicted_n"]
    return fig, header, rows, caption


# ===========================================================================
# Figure 3 — the alpha transition (the single-copy mechanism)
# ===========================================================================
def _mstar_crosses(mstar: dict, k: int, target: float) -> float:
    """Interpolate (log-linear in n) the n at which ``M*(n, k) == target``."""
    pts = sorted((n, mstar[(n, kk)]) for (n, kk) in mstar if kk == k)
    ns = np.array([n for n, _ in pts], float)
    ly = np.log(np.array([v for _, v in pts]))
    return float(np.interp(np.log(target), ly, ns))


def make_fig3():
    bs = D.load("budget_scaling.json")
    comps = D.zeta_components()
    mstar = D.m_star_table()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    rows = []
    # M* transition band for k=2: where M*(n) spans the budget window (2000..128000).
    nb_lo, nb_hi = _mstar_crosses(mstar, 2, 2000), _mstar_crosses(mstar, 2, 128000)
    ax.axvspan(nb_lo, nb_hi, color="0.85", alpha=0.6, lw=0, zorder=0)
    for k in (2, 3, 4):
        fits = sorted([a for a in bs["alpha_fits"] if a["k"] == k], key=lambda a: a["n"])
        ns = [a["n"] for a in fits]
        ax.errorbar(ns, [a["alpha"] for a in fits], yerr=[a["alpha_se"] for a in fits],
                    fmt="o", color=C_K[k], ms=3.6, lw=1.0, label=f"k = {k}", zorder=3)
        th_n = [a["n"] for a in fits if (a["n"], k) in comps]
        th = [D.theory_alpha(a["budgets"], k, comps, a["n"]) for a in fits if (a["n"], k) in comps]
        ax.plot(th_n, th, "-", color=C_K[k], lw=1.3, alpha=0.9, zorder=2)
        for a in fits:
            rows.append([k, a["n"], "measured", a["alpha"], a["alpha_se"]])
            if (a["n"], k) in comps:
                rows.append([k, a["n"], "theory", D.theory_alpha(a["budgets"], k, comps, a["n"]), ""])
    for yv, lab in [(0.5, r"$\alpha=0.5$  (1/$\sqrt{M}$)"), (1.0, r"$\alpha=1.0$  (1/$M$)")]:
        ax.axhline(yv, color="0.55", lw=0.8, ls=":", zorder=1)
        ax.text(2.05, yv + 0.018, lab, fontsize=6.6, color="0.4", va="bottom", ha="left")
    ax.text((nb_lo + nb_hi) / 2, 1.235, r"$M^{*}(k{=}2)\approx$ budget", fontsize=6.4, color="0.4",
            ha="center", va="top")
    ax.set_xlabel("system size n (qubits)")
    ax.set_ylabel(r"budget-scaling exponent $\alpha$")
    ax.set_xticks(range(2, 10))
    ax.set_ylim(0.42, 1.25)
    ax.legend(loc="lower right", fontsize=7.0, handletextpad=0.4, title="measured (pts) / theory (line)",
              title_fontsize=6.6)
    fig.tight_layout()
    caption = (
        "Figure 3. The single-copy mechanism: the budget-scaling exponent alpha. Measured alpha "
        "(single-copy RMSE ~ M^-alpha, fit over budgets 2000-128000; points with bootstrap SE) "
        "versus system size n for moment orders k=2,3,4, with the parameter-free theory overlaid "
        "(solid lines). alpha migrates from 0.5 (variance-limited, 1/sqrt(M)) to 1.0 "
        "(higher-order-limited, 1/M) as n grows. The grey band marks where the budget threshold "
        "M*(n) for k=2 spans the copy budget (2000-128000); the transition coincides with M* "
        "overtaking the budget, and shifts to larger n for higher k (M* grows faster with k). The "
        "reader should conclude that we understand WHY single-copy fails, not merely that it does.")
    header = ["k", "n", "kind", "alpha_or_value", "alpha_se"]
    return fig, header, rows, caption


# ===========================================================================
# Figure 4 — out-of-ensemble validation (the stress test)
# ===========================================================================
def make_fig4():
    p2 = D.load("stress_test.json")["part2"]
    med_rel = float(np.median([c["rel_err"] for c in p2]))
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    lo = 0.9 * min(min(c["measured"], c["predicted"]) for c in p2)
    hi = 1.1 * max(max(c["measured"], c["predicted"]) for c in p2)
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1], color=OKABE_ITO["sky"],
                    alpha=0.15, lw=0, label="±10%")
    ax.plot([lo, hi], [lo, hi], color=C_TRUE, lw=1.0, ls="--", label="perfect (y = x)")
    rows = []
    for ens, color in C_ENS.items():
        cells = [c for c in p2 if c["ensemble"] == ens]
        xerr = np.array([_err(c["ci"][0], c["ci"][1], c["measured"]) for c in cells]).T
        ax.errorbar([c["measured"] for c in cells], [c["predicted"] for c in cells], xerr=xerr,
                    fmt="o", color=color, ms=4, lw=0.8, alpha=0.85, label=ens.replace("_", "-"), zorder=3)
        for c in cells:
            rows.append([ens, c["n"], c["k"], c["budget"], c["measured"], c["ci"][0], c["ci"][1],
                         c["predicted"], c["rel_err"]])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    _clean_log_ticks(ax, "both")
    ax.set_xlabel("measured single-copy RMSE")
    ax.set_ylabel("predicted RMSE (theory)")
    ax.text(0.03, 0.97, f"N = {len(p2)} cells (unseen states)\nmedian rel. error: {med_rel:.1%}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.3,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    ax.legend(loc="lower right", fontsize=6.6, handletextpad=0.4)
    fig.tight_layout()
    caption = (
        "Figure 4. Out-of-ensemble validation (the stress test). Parameter-free predicted single-copy "
        "RMSE versus measured RMSE (68% CIs on x) for three ensembles the theory was never developed "
        f"on -- Haar-pure, rank-2 mixed, and depolarized-GHZ -- across n, k, and budget (N = {len(p2)}). "
        f"The median relative error is {med_rel:.1%}; the shaded band is +/-10%. Points cluster on the "
        "diagonal, showing the theory transfers to unseen states; the residual scatter is shown "
        "honestly rather than hidden, marking the ~10% systematic-error boundary of validity.")
    header = ["ensemble", "n", "k", "budget", "measured_rmse", "ci_lo", "ci_hi", "predicted_rmse", "rel_err"]
    return fig, header, rows, caption


# ===========================================================================
# Figure 5 — the exponential wall
# ===========================================================================
def make_fig5():
    sh = D.load("scaling_hardened.json")["rows"]
    comps = D.zeta_components()
    # single-copy is noise-independent for noisy_pure; take one (noise, rate) per n.
    single = {}
    coll = {}
    true_val = None
    for r in sh:
        if r["ensemble"] != "noisy_pure":
            continue
        single.setdefault(r["n"], (r["single_rmse"], r["single_rmse_ci68"]))
        if r["noise_model"] == "dephasing" and r["rate"] == 0.05:
            coll[r["n"]] = (r["collective_rmse"], r["collective_rmse_ci68"])
        true_val = r["mean_true_purity"]
    ns = sorted(single)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    rows = []
    # true value line
    ax.axhline(true_val, color=C_TRUE, lw=1.1, ls="-")
    ax.text(ns[0], true_val * 1.15, r"true value  Tr($\rho^2$) $\approx$ %.2f" % true_val,
            fontsize=6.8, color=C_TRUE, va="bottom")
    rows.append(["", "true_value", true_val, "", ""])
    # single-copy measured
    ys = [single[n][0] for n in ns]
    yerr = np.array([_err(single[n][1][0], single[n][1][1], single[n][0]) for n in ns]).T
    ax.errorbar(ns, ys, yerr=yerr, fmt="o", color=C_SINGLE, ms=4, lw=1.0, label="single-copy (measured)", zorder=3)
    for n in ns:
        rows.append([n, "single_measured", single[n][0], single[n][1][0], single[n][1][1]])
    # single-copy theory (where components exist, n<=9)
    th_n = [n for n in ns if (n, 2) in comps]
    th = [D.theory_single_rmse(n, 2, 2000, comps) for n in th_n]
    ax.plot(th_n, th, "-", color=C_SINGLE, lw=1.2, alpha=0.8, label="single-copy (theory)", zorder=2)
    for n, v in zip(th_n, th):
        rows.append([n, "single_theory", v, "", ""])
    # collective (bounded)
    cn = sorted(coll)
    ax.errorbar(cn, [coll[n][0] for n in cn], fmt="s", color=C_COLL, ms=3.4, lw=1.0,
                label="collective (dephasing, g=0.05)", zorder=3)
    ax.plot(cn, [coll[n][0] for n in cn], "-", color=C_COLL, lw=1.0, alpha=0.7)
    for n in cn:
        rows.append([n, "collective_measured", coll[n][0], coll[n][1][0], coll[n][1][1]])
    ax.set_yscale("log")
    ax.set_xlabel("system size n (qubits)")
    ax.set_ylabel(r"purity RMSE (log scale)")
    ax.set_xticks(ns)
    # annotate the n=10 blow-up
    n10 = single[10][0]
    ax.annotate(f"n=10: error {n10:.0f}\n= {n10 / true_val:.0f}× the true value",
                xy=(10, n10), xytext=(7.2, n10 * 0.9), fontsize=6.8, color=C_SINGLE, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C_SINGLE, lw=0.8))
    ax.legend(loc="upper left", fontsize=6.6, handletextpad=0.4)
    fig.tight_layout()
    caption = (
        "Figure 5. The exponential wall. Single-copy purity RMSE versus system size n out to n=10 "
        "(log scale; measured points with 68% CIs, theory line), with the true quantity Tr(rho^2) "
        f"~= {true_val:.2f} marked (black). Single-copy error grows exponentially and by n=10 reaches "
        f"~{n10:.0f} -- about {n10 / true_val:.0f}x the quantity being estimated, so the estimate is "
        "meaningless -- while the collective error (green) stays bounded. The reader should conclude "
        "that single-copy shadow estimation of purity hits a hard exponential wall that the collective "
        "route does not.")
    header = ["n", "series", "rmse", "ci_lo", "ci_hi"]
    return fig, header, rows, caption


ALL_FIGURES = {
    "fig1_crossover_map": make_fig1,
    "fig2_crossover_boundary": make_fig2,
    "fig3_alpha_transition": make_fig3,
    "fig4_out_of_ensemble": make_fig4,
    "fig5_exponential_wall": make_fig5,
}
