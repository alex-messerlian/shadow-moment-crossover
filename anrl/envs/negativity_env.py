"""Adaptive single-copy negativity-estimation environment (TorchRL).

A sequential measurement problem: a hidden ``n``-qubit noisy state is fixed at
reset, and the agent spends a shot budget by choosing, at each step, one of the
``3**n`` local Pauli measurement settings.  After each measurement the
environment recomputes the realizable witness estimate of the state's negativity
from *all* accumulated data, and rewards the reduction in estimation error.  The
true negativity is hidden ground truth used only to shape the reward during
simulation.

Built on the Phase 1 physics core (state sampling + local-Pauli measurement) and
the Phase 2 witness estimator.  This module contains no agent, training, or
baseline-policy logic — only the environment.

TorchRL API (torchrl 0.13.2)
----------------------------
* Base class: :class:`torchrl.envs.EnvBase` (override ``_reset``, ``_step``,
  ``_set_seed``).
* Specs (``torchrl.data``): :class:`Composite` (observation), :class:`Unbounded`
  (observation vector float32, reward float64), :class:`Categorical` (discrete
  action over the ``3**n`` settings).  ``done``/``terminated`` specs are the
  auto-created boolean defaults.
* Validated with :func:`torchrl.envs.utils.check_env_specs`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from anrl.physics import (
    all_local_pauli_settings,
    depolarize,
    estimate_pauli_expectations,
    negativity,
    negativity_witness_estimator,
    random_density,
    sample_counts,
)

Setting = Tuple[str, ...]


class AdaptiveNegativityEnv(EnvBase):
    """Sequential single-copy measurement environment for negativity estimation.

    Parameters
    ----------
    n:
        Number of qubits (Hilbert-space dimension ``2**n``).  Default 2.
    n_steps:
        Number of measurement steps per episode.  Default 24.
    shots_per_step:
        Shots committed to the chosen setting on each step.  Default 8.
    noise_range:
        Inclusive range ``(low, high)`` for the depolarizing rate ``q`` drawn
        uniformly at reset.  Default ``(0.0, 0.5)``.
    dA, dB:
        Subsystem dimensions of the A|B bipartition.  Default ``None`` selects
        the balanced split (first ``ceil(n/2)`` qubits are A), matching the
        physics core.
    seed:
        Seed for the environment's internal ``numpy`` generator (state sampling
        and measurement outcomes).  Reproducible across episodes.
    device:
        Torch device for the tensordicts.  Default ``"cpu"``.

    Observation (flat float32 vector, length ``2 * 3**n + 2``)
        For each setting, in the canonical
        :func:`~anrl.physics.all_local_pauli_settings` order,
        ``[fraction of the total budget spent on it, current estimate of that
        setting's n-qubit Pauli correlator]``; followed by
        ``[current witness negativity estimate, fraction of budget remaining]``.
    """

    # Single-copy env holding a hidden numpy state: batch-locked (accepts only
    # tensordicts matching its own scalar batch_size); rollouts run one episode
    # at a time rather than a vectorized batch.
    batch_locked = True

    def __init__(
        self,
        n: int = 2,
        n_steps: int = 24,
        shots_per_step: int = 8,
        noise_range: Tuple[float, float] = (0.0, 0.5),
        dA: int | None = None,
        dB: int | None = None,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([]))

        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if n_steps < 1 or shots_per_step < 1:
            raise ValueError("n_steps and shots_per_step must be >= 1")
        low, high = noise_range
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError(f"noise_range must satisfy 0 <= low <= high <= 1, got {noise_range}")

        self._n = n
        self._dim = 2 ** n
        self._n_steps = n_steps
        self._shots_per_step = shots_per_step
        self._noise_low = float(low)
        self._noise_high = float(high)
        self._dA = dA
        self._dB = dB
        self._total_budget = n_steps * shots_per_step

        self._settings: list[Setting] = all_local_pauli_settings(n)
        self._n_settings = len(self._settings)  # 3**n
        self._obs_dim = 2 * self._n_settings + 2

        # Specs.  Observation is float32; reward is float64 so the potential-based
        # reward telescopes to the true accuracy improvement without float32 drift.
        self.observation_spec = Composite(
            observation=Unbounded(shape=(self._obs_dim,), dtype=torch.float32),
            shape=torch.Size([]),
        )
        self.action_spec = Categorical(n=self._n_settings)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float64)

        # Internal (episode) state.
        self._np_rng = np.random.default_rng(seed)
        self._rho: np.ndarray | None = None
        self._true_negativity: float = 0.0
        self._counts_by_setting: Dict[Setting, np.ndarray] = {}
        self._step_idx: int = 0
        self._current_estimate: float = 0.0

    # -- introspection helpers (used by tests; read-only views) --------------
    @property
    def n(self) -> int:
        return self._n

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def settings(self) -> list[Setting]:
        return list(self._settings)

    @property
    def dA(self) -> int | None:
        """Configured subsystem-A dimension (``None`` = balanced default)."""
        return self._dA

    @property
    def dB(self) -> int | None:
        """Configured subsystem-B dimension (``None`` = balanced default)."""
        return self._dB

    @property
    def true_negativity(self) -> float:
        """Exact negativity of the current hidden state (ground truth)."""
        return self._true_negativity

    @property
    def current_estimate(self) -> float:
        """The environment's current witness negativity estimate (float64)."""
        return self._current_estimate

    @property
    def density_matrix(self) -> np.ndarray:
        """Copy of the current hidden density matrix."""
        if self._rho is None:
            raise RuntimeError("environment has not been reset yet")
        return self._rho.copy()

    def accumulated_counts(self) -> Dict[Setting, np.ndarray]:
        """Copy of the accumulated per-setting measurement counts."""
        return {s: c.copy() for s, c in self._counts_by_setting.items()}

    # -- observation ---------------------------------------------------------
    def _observation(self, expectations: Dict[Setting, float]) -> torch.Tensor:
        """Assemble the flat float32 observation from the current episode state."""
        feats = np.empty(self._obs_dim, dtype=np.float32)
        for idx, setting in enumerate(self._settings):
            spent = float(self._counts_by_setting[setting].sum())
            feats[2 * idx] = spent / self._total_budget
            # Full-weight correlator <P_setting>: only this setting resolves it,
            # so it is exactly that setting's shot-weighted estimate (0 if unmeasured).
            feats[2 * idx + 1] = expectations.get(setting, 0.0)
        feats[-2] = self._current_estimate
        feats[-1] = 1.0 - self._step_idx / self._n_steps
        return torch.from_numpy(feats).to(self.device)

    def _zero_counts(self) -> Dict[Setting, np.ndarray]:
        return {s: np.zeros(self._dim, dtype=np.int64) for s in self._settings}

    # -- EnvBase overrides ---------------------------------------------------
    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        # Sample a fresh hidden noisy state: Ginibre random density with random
        # rank, then a depolarizing channel with a uniformly-drawn rate.
        rank = int(self._np_rng.integers(1, self._dim + 1))
        rho = random_density(self._dim, rank, self._np_rng)
        q = float(self._np_rng.uniform(self._noise_low, self._noise_high))
        self._rho = depolarize(rho, q)
        self._true_negativity = float(negativity(self._rho, self._dA, self._dB))

        self._counts_by_setting = self._zero_counts()
        self._step_idx = 0
        self._current_estimate = 0.0  # estimate on no data

        obs = self._observation(expectations={})
        return TensorDict(
            {
                "observation": obs,
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = int(tensordict["action"].item())
        setting = self._settings[action]

        # Error of the estimate *before* this measurement.
        prev_error = abs(self._current_estimate - self._true_negativity)

        # Commit shots to the chosen setting and accumulate counts.
        new_counts = sample_counts(self._rho, setting, self._shots_per_step, self._np_rng)
        self._counts_by_setting[setting] = self._counts_by_setting[setting] + new_counts
        self._step_idx += 1

        # Recompute the realizable witness estimate from ALL accumulated data.
        expectations = estimate_pauli_expectations(self._counts_by_setting, self._n)
        self._current_estimate = float(
            negativity_witness_estimator(expectations, self._n, self._dA, self._dB)
        )
        new_error = abs(self._current_estimate - self._true_negativity)

        # Potential-based reward: reduction in estimation error.
        reward = prev_error - new_error

        # Fixed-horizon episode ends after n_steps.  Per spec we signal this as
        # `terminated` (done); a future value-learning phase that distinguishes a
        # time-limit truncation from an absorbing terminal may revisit this.
        terminated = self._step_idx >= self._n_steps
        obs = self._observation(expectations)
        return TensorDict(
            {
                "observation": obs,
                "reward": torch.tensor([reward], dtype=torch.float64, device=self.device),
                "done": torch.tensor([terminated], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        self._np_rng = np.random.default_rng(seed)
