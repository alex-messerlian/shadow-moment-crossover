"""Render all publication figures from the saved results.

    .venv/bin/python experiments/make_figures.py

Reads ``results/*.json`` (never re-runs the science), builds the five figures, and
writes each as a vector PDF (paper), a 300-dpi PNG (slides/web), and a tidy CSV of
the exact plotted data (for the interactive web demo) into ``results/figures/``.
Prints the file list and the draft captions.
"""

from __future__ import annotations

import time
from pathlib import Path

from anrl.figures import ALL_FIGURES, apply_style, save_figure, write_csv

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    apply_style()
    start = time.time()
    produced = []
    captions = {}
    for name, make in ALL_FIGURES.items():
        fig, header, rows, caption = make()
        produced += save_figure(fig, name)
        produced.append(write_csv(name, header, rows))
        captions[name] = caption
        print(f"  built {name}: {len(rows)} data rows")
    wall = time.time() - start

    print(f"\nDone in {wall:.1f}s. Produced {len(produced)} files in results/figures/:")
    for p in produced:
        print(f"  {p.relative_to(REPO)}  ({p.stat().st_size} bytes)")

    print("\n" + "=" * 78 + "\nDRAFT CAPTIONS\n" + "=" * 78)
    for name, cap in captions.items():
        print(f"\n[{name}]\n{cap}")


if __name__ == "__main__":
    main()
