"""PASS 48.4: render paper one's four new theory figures.

    PYTHONPATH=. .venv/bin/python experiments/pass48_make_theory_figures.py

Reads the saved PASS 47/48 artifacts (never re-runs the science) and writes each figure as a
vector PDF, a 300-dpi PNG, and a tidy CSV of the exact plotted data, into ``results/figures/``,
in the same style as the existing six.  The existing figures are not touched.
"""

from __future__ import annotations

import time
from pathlib import Path

from anrl.figures import apply_style, save_figure, write_csv
from anrl.figures.figures_theory import THEORY_FIGURES

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    apply_style()
    t0 = time.time()
    produced, captions = [], {}
    for name, make in THEORY_FIGURES.items():
        fig, header, rows, caption = make()
        produced += save_figure(fig, name)
        produced.append(write_csv(name, header, rows))
        captions[name] = caption
        print(f"  built {name}: {len(rows)} data rows")

    print(f"\nDone in {time.time()-t0:.1f}s. Produced {len(produced)} files:")
    for p in produced:
        print(f"  {p.relative_to(REPO)}  ({p.stat().st_size} bytes)")

    print("\n" + "=" * 78 + "\nDRAFT CAPTIONS\n" + "=" * 78)
    for name, cap in captions.items():
        print(f"\n[{name}]\n{cap}")


if __name__ == "__main__":
    main()
