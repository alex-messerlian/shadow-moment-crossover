"""Shot-budget and credit arithmetic for the purity experiment.

The destructive SWAP test returns, per shot, a sign value ``s in {+1, -1}`` with
``E[s]`` equal to the measured purity ``mu``.  Since ``s`` is bounded,
``Var[s] = 1 - mu^2`` exactly, so the standard error of the purity estimate over
``N`` shots is ``sqrt((1 - mu^2) / N)`` — no simulation needed for the collective
route's error bars.  We size ``N`` so that this SE sits well below the collective
bias we are trying to resolve.

Credits: Open Quantum charges 26 credits per 100,000 shots on Rigetti Public
Compute.  We have ~45 credits.
"""

from __future__ import annotations

import math

CREDITS_PER_100K_SHOTS = 26.0
CREDIT_BUDGET = 45.0
MAX_SHOTS_PER_CIRCUIT = 50_000  # device limit (from cepheus_metadata limits)


def swap_shot_se(measured: float, n_shots: int) -> float:
    """Standard error of the SWAP-test purity estimate: ``sqrt((1 - mu^2) / N)``."""
    return math.sqrt(max(0.0, 1.0 - measured * measured) / n_shots)


def swap_shots_for_se(measured: float, target_se: float) -> int:
    """Minimum shots so the SWAP-test SE is at most ``target_se`` (ceil)."""
    if target_se <= 0:
        raise ValueError("target_se must be > 0")
    return int(math.ceil((1.0 - measured * measured) / (target_se * target_se)))


def shots_to_credits(shots: int) -> float:
    """Open Quantum credits for a shot count (26 credits / 100k shots)."""
    return shots / 100_000.0 * CREDITS_PER_100K_SHOTS


def credits_to_shots(credits: float) -> int:
    """Shots affordable for a credit amount."""
    return int(credits / CREDITS_PER_100K_SHOTS * 100_000)


def shots_for_bias_resolution(measured: float, bias: float, sigma: float = 5.0) -> int:
    """Shots so the SWAP-test SE is ``bias / sigma`` — i.e. resolve the bias at ``sigma``-σ."""
    target_se = bias / sigma
    return swap_shots_for_se(measured, target_se)
