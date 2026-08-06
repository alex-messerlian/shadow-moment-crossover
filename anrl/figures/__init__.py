"""Publication figures built from the saved results (no experiment is re-run).

* :mod:`~anrl.figures.style`, colorblind-safe palette, print-legible fonts, savers.
* :mod:`~anrl.figures.data`, loaders for ``results/*.json`` + parameter-free theory curves.
* :mod:`~anrl.figures.figures`; the five ``make_figN`` builders.
"""

from __future__ import annotations

from .figures import ALL_FIGURES
from .style import apply_style, save_figure, write_csv

__all__ = ["ALL_FIGURES", "apply_style", "save_figure", "write_csv"]
