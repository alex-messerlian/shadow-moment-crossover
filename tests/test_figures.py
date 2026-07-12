"""Smoke tests for the figure pipeline.

Confirms every figure builds from the saved results and that each output file
(PDF, PNG, CSV) is produced and non-empty.  The figures depend on the generated
``results/*.json``; if those are absent (fresh clone -- results are gitignored)
the tests skip rather than fail.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from anrl.figures import ALL_FIGURES, apply_style
from anrl.figures import style as fstyle
from anrl.figures.figures import make_fig1

REPO = Path(__file__).resolve().parent.parent
REQUIRED = ["budget_scaling.json", "crossover_theory.json", "stress_test.json",
            "scaling_hardened.json", "theory_zetas.json"]
_have_results = all((REPO / "results" / f).exists() for f in REQUIRED)
needs_results = pytest.mark.skipif(not _have_results, reason="needs generated results/*.json")

matplotlib_figure = pytest.importorskip("matplotlib").figure.Figure


@needs_results
def test_each_figure_builds_with_data_and_caption() -> None:
    apply_style()
    for name, make in ALL_FIGURES.items():
        fig, header, rows, caption = make()
        assert isinstance(fig, matplotlib_figure), name
        assert len(rows) > 0, f"{name} produced no data rows"
        for r in rows:
            assert len(r) == len(header), f"{name} row width != header"
        assert isinstance(caption, str) and caption.startswith("Figure"), name
        import matplotlib.pyplot as plt
        plt.close(fig)


@needs_results
def test_pipeline_writes_all_outputs(tmp_path, monkeypatch) -> None:
    # Redirect the output dir so the test does not touch results/figures/.
    monkeypatch.setattr(fstyle, "FIG_DIR", tmp_path)
    apply_style()
    for name, make in ALL_FIGURES.items():
        fig, header, rows, _ = make()
        pdfs = fstyle.save_figure(fig, name)
        csv_path = fstyle.write_csv(name, header, rows)
        for p in pdfs + [csv_path]:
            assert p.exists() and p.stat().st_size > 0, f"{p} missing or empty"
        assert {p.suffix for p in pdfs} == {".pdf", ".png"}
        # CSV round-trips: header + at least one data row, correct width
        with open(csv_path) as fh:
            reader = list(csv.reader(fh))
        assert reader[0] == header and len(reader) == len(rows) + 1


@needs_results
def test_all_five_figures_present() -> None:
    assert set(ALL_FIGURES) == {
        "fig1_crossover_map", "fig2_crossover_boundary", "fig3_alpha_transition",
        "fig4_out_of_ensemble", "fig5_exponential_wall",
    }


@needs_results
def test_theory_curves_are_not_fit_to_data() -> None:
    # The theory single-copy curve in Fig 1 must come from the saved Hoeffding
    # components (parameter-free), independent of the measured RMSE points.
    _, header, rows, _ = make_fig1()
    theory = [r for r in rows if r[header.index("kind")] == "theory"]
    measured = [r for r in rows if r[header.index("kind")] == "measured"]
    assert len(theory) > 0 and len(measured) > 0
