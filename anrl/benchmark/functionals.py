"""Nonlinear state functionals to benchmark.

Currently a single target, purity ``Tr(rho^2)``, re-exported from the physics
core as the exact ground truth.  Additional functionals (e.g. higher moments,
Renyi entropies) would be added here.
"""

from __future__ import annotations

from anrl.physics import purity

__all__ = ["purity"]
