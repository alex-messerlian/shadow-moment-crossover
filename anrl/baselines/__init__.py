"""Baseline measurement policies and evaluation harness.

The four baselines the RL agent is benchmarked against
(:class:`FixedPolicy`, :class:`RandomPolicy`, :class:`GreedyPolicy`,
:class:`OraclePolicy`) plus :func:`evaluate_policy`.  No agent or training code.
"""

from __future__ import annotations

from .evaluation import evaluate_policy
from .policies import (
    FixedPolicy,
    GreedyPolicy,
    OraclePolicy,
    Policy,
    RandomPolicy,
)

__all__ = [
    "Policy",
    "FixedPolicy",
    "RandomPolicy",
    "GreedyPolicy",
    "OraclePolicy",
    "evaluate_policy",
]
