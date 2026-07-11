"""Baseline measurement policies for the adaptive-negativity environment.

A *policy* maps the environment's current observation (and, for the oracle,
privileged access to the true state) to the next action — an index into the
``3**n`` local Pauli measurement settings.  Four baselines are provided:

* :class:`FixedPolicy`  — a preset cyclic schedule (``t mod 3**n``).
* :class:`RandomPolicy` — a uniformly random setting each step (reproducible).
* :class:`GreedyPolicy` — myopic sensitivity x uncertainty sampling from the
  *estimated* state.
* :class:`OraclePolicy`— the same scoring computed from the *true* state (the
  ceiling on what an adaptive policy could achieve).

No agent or training logic lives here.  Policies read from the environment's
public accessors (``accumulated_counts``, ``density_matrix``, ``settings``,
``dA``/``dB``); the oracle's use of ``density_matrix`` is the privileged channel.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from anrl.physics import (
    estimate_pauli_expectations,
    reconstruct,
    witness_weights,
)

# Unmeasured settings get the maximal per-shot standard error of a +/-1 mean
# (achieved at 1 shot with a zero-mean correlator), so they are always explored
# before an already-measured setting of comparable sensitivity.
_MAX_UNCERTAINTY = 1.0

# Small exploration floor added to the sensitivity so the uncertainty term is
# never fully masked.  A state reconstructed from a single setting's data is PPT
# (all its Pauli terms are transpose-symmetric), so its witness weights — and
# hence every sensitivity — are exactly zero; without this floor the
# sensitivity x uncertainty product would be uniformly zero and GREEDY would
# never leave setting 0.  The floor lets the maximal uncertainty of unmeasured
# settings drive exploration at the cold start, while real sensitivities
# (~0.05-0.5 once the reconstruction develops negativity) dominate it by orders.
#
# ONLY greedy uses it.  Greedy's zero sensitivity is a data-starvation artifact
# (its estimated state is PPT until it measures diversely) — it must explore to
# discover negativity.  The oracle scores off the TRUE state, where a zero
# sensitivity genuinely means the state is separable (N=0); there the optimal
# move is to NOT diversify (concentrated measurement keeps the reconstruction
# PPT, so the estimate stays at the correct 0), and an exploration floor would
# only inject reconstruction noise as spurious negativity.  So the oracle passes
# floor 0, keeping it a true ceiling.
_EXPLORATION_FLOOR = 1e-9


class Policy:
    """Base policy interface: ``reset(rng)`` per episode, then ``policy(obs, env)``."""

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Prepare for a new episode.  ``rng`` is a per-episode generator."""

    def __call__(self, observation, env) -> int:  # noqa: D401 - callable protocol
        raise NotImplementedError


class FixedPolicy(Policy):
    """Preset cyclic schedule: step ``t`` measures setting ``t mod 3**n``."""

    def __init__(self, n_settings: int) -> None:
        self._n_settings = n_settings
        self._t = 0

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._t = 0

    def __call__(self, observation, env) -> int:
        action = self._t % self._n_settings
        self._t += 1
        return action


class RandomPolicy(Policy):
    """Uniformly random setting each step; reproducible for a fixed seed."""

    def __init__(self, n_settings: int, seed: int | None = None) -> None:
        self._n_settings = n_settings
        self._rng = np.random.default_rng(seed)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        # The harness supplies a per-episode generator for reproducible,
        # state-paired evaluation; standalone use keeps the construction rng.
        if rng is not None:
            self._rng = rng

    def __call__(self, observation, env) -> int:
        return int(self._rng.integers(self._n_settings))


def _parity_signs(n: int) -> np.ndarray:
    """``(-1)**popcount(b)`` for each outcome ``b`` — the full-weight eigenvalue."""
    outcomes = np.arange(2 ** n)
    popcount = np.array([int(b).bit_count() for b in outcomes])
    return 1 - 2 * (popcount % 2)


def _setting_scores(
    state: np.ndarray,
    counts_by_setting: dict,
    settings: Sequence[tuple],
    n: int,
    dA: int | None,
    dB: int | None,
    exploration_floor: float,
) -> np.ndarray:
    """Sensitivity x uncertainty score for every setting.

    * sensitivity: ``|w_{P_s}(state)|`` — the witness weight of the setting's
      full-weight Pauli correlator, i.e. the gradient of the witness-estimator
      negativity w.r.t. that correlator (via the negative-eigenvector projector).
    * uncertainty: the standard error of that correlator's current estimate,
      ``sqrt((1 - m^2) / n_shots)``; unmeasured settings get ``_MAX_UNCERTAINTY``.

    ``exploration_floor`` is added to the sensitivity; see the module note on
    :data:`_EXPLORATION_FLOOR`.  Greedy passes a small positive floor (its
    reconstructed witness is uninformative until it has gathered diverse data);
    the oracle passes 0 (its true witness is never data-starved).
    """
    weights = witness_weights(state, dA, dB)
    signs = _parity_signs(n)
    scores = np.empty(len(settings), dtype=np.float64)
    for i, setting in enumerate(settings):
        sensitivity = abs(weights[tuple(setting)])
        counts = counts_by_setting[setting]
        shots = int(counts.sum())
        if shots == 0:
            uncertainty = _MAX_UNCERTAINTY
        else:
            mean = float(counts @ signs) / shots
            uncertainty = np.sqrt(max(0.0, 1.0 - mean * mean) / shots)
        scores[i] = (sensitivity + exploration_floor) * uncertainty
    return scores


class GreedyPolicy(Policy):
    """Myopic sensitivity x uncertainty sampling from the *estimated* state.

    Reconstructs ``rho_hat`` from all accumulated data, scores each setting by
    the sensitivity of the witness negativity to its correlator times that
    correlator's standard error, and picks the highest-scoring setting.
    """

    def __call__(self, observation, env) -> int:
        counts = env.accumulated_counts()
        expectations = estimate_pauli_expectations(counts, env.n)
        rho_hat = reconstruct(expectations, env.n)
        scores = _setting_scores(
            rho_hat, counts, env.settings, env.n, env.dA, env.dB,
            exploration_floor=_EXPLORATION_FLOOR,
        )
        return int(np.argmax(scores))


class OraclePolicy(Policy):
    """The ceiling: same scoring as greedy but computed from the *true* state.

    Uses the environment's privileged ``density_matrix`` for the sensitivity
    (true witness direction) while still using the accumulated counts for the
    uncertainty.  Upper-bounds what an adaptive policy could achieve.
    """

    def __call__(self, observation, env) -> int:
        counts = env.accumulated_counts()
        rho_true = env.density_matrix  # privileged access to the hidden state
        # No exploration floor: a zero true sensitivity means N=0, where staying
        # concentrated (est=0) is optimal — see the _EXPLORATION_FLOOR note.
        scores = _setting_scores(
            rho_true, counts, env.settings, env.n, env.dA, env.dB,
            exploration_floor=0.0,
        )
        return int(np.argmax(scores))
