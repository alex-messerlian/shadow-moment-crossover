"""Tests for the TorchRL adaptive-negativity environment (Phase 3).

Pins: TorchRL spec conformance (check_env_specs), rollout shapes/dtypes, exact
reward telescoping, estimate consistency with the standalone estimator, and
seeded determinism.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchrl.envs.utils import check_env_specs

from anrl.envs import AdaptiveNegativityEnv
from anrl.physics import estimate_pauli_expectations, negativity_witness_estimator


def _round_robin_policy(n_settings: int):
    """Policy that deterministically cycles through every measurement setting."""
    state = {"i": 0}

    def policy(td):
        td.set("action", torch.tensor(state["i"] % n_settings, dtype=torch.int64))
        state["i"] += 1
        return td

    return policy


# ---------------------------------------------------------------------------
# 1 — TorchRL spec conformance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3])
def test_check_env_specs(n: int) -> None:
    env = AdaptiveNegativityEnv(n=n, seed=0)
    check_env_specs(env)  # raises if any spec is violated during a rollout


# ---------------------------------------------------------------------------
# 2 — full rollout completes with the specified shapes and dtypes
# ---------------------------------------------------------------------------
def test_full_rollout_shapes_and_dtypes() -> None:
    env = AdaptiveNegativityEnv(n=2, n_steps=24, shots_per_step=8, seed=1)
    obs_dim = 2 * len(env.settings) + 2
    assert obs_dim == 20  # 2*3^2 + 2

    rollout = env.rollout(env.n_steps, _round_robin_policy(len(env.settings)))

    assert rollout.shape == torch.Size([env.n_steps])
    assert rollout["observation"].shape == torch.Size([env.n_steps, obs_dim])
    assert rollout["observation"].dtype == torch.float32
    assert rollout["action"].shape == torch.Size([env.n_steps])
    assert rollout["action"].dtype == torch.int64
    assert rollout["next", "reward"].shape == torch.Size([env.n_steps, 1])
    assert rollout["next", "reward"].dtype == torch.float64
    # Episode terminates exactly on the last step.
    assert bool(rollout["next", "terminated"][-1].item()) is True
    assert bool(rollout["next", "terminated"][:-1].any().item()) is False


# ---------------------------------------------------------------------------
# 3 — reward telescoping (the key correctness pin)
# ---------------------------------------------------------------------------
def test_reward_telescoping() -> None:
    # Config chosen so the episode is non-degenerate: the estimate genuinely
    # moves, so the telescoping identity is exercised on non-trivial rewards.
    env = AdaptiveNegativityEnv(n=2, n_steps=24, shots_per_step=64, noise_range=(0.0, 0.3), seed=3)
    rollout = env.rollout(env.n_steps, _round_robin_policy(len(env.settings)))

    n_true = env.true_negativity
    est_final = env.current_estimate
    sum_rewards = rollout["next", "reward"].sum().item()
    expected = abs(0.0 - n_true) - abs(est_final - n_true)

    assert sum_rewards == pytest.approx(expected, abs=1e-9)
    # Non-vacuous: the episode actually improved a nonzero estimate.
    assert est_final > 1e-3
    assert abs(sum_rewards) > 1e-3


# ---------------------------------------------------------------------------
# 4 — the env's estimate matches the standalone estimator on the same counts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n, dA, dB", [(2, None, None), (3, 2, 4)]  # default balanced, and a non-default split
)
def test_estimate_consistency(n: int, dA: int | None, dB: int | None) -> None:
    env = AdaptiveNegativityEnv(
        n=n, n_steps=24, shots_per_step=64, noise_range=(0.0, 0.3), dA=dA, dB=dB, seed=3
    )
    env.rollout(env.n_steps, _round_robin_policy(len(env.settings)))

    counts = env.accumulated_counts()
    expectations = estimate_pauli_expectations(counts, env.n)
    # Recompute with the env's OWN bipartition so the check pins dA/dB propagation,
    # not merely the default.
    direct = negativity_witness_estimator(expectations, env.n, env.dA, env.dB)

    assert env.current_estimate == pytest.approx(direct, abs=1e-9)
    assert direct > 1e-3  # non-degenerate, so the check is meaningful


# ---------------------------------------------------------------------------
# 5 — determinism: same seed + same actions -> identical states/obs/rewards
# ---------------------------------------------------------------------------
def test_determinism() -> None:
    n_settings = len(AdaptiveNegativityEnv(seed=0).settings)

    env_a = AdaptiveNegativityEnv(n=2, n_steps=24, shots_per_step=16, seed=2024)
    roll_a = env_a.rollout(env_a.n_steps, _round_robin_policy(n_settings))

    env_b = AdaptiveNegativityEnv(n=2, n_steps=24, shots_per_step=16, seed=2024)
    roll_b = env_b.rollout(env_b.n_steps, _round_robin_policy(n_settings))

    assert np.array_equal(env_a.density_matrix, env_b.density_matrix)
    assert env_a.true_negativity == env_b.true_negativity
    assert torch.equal(roll_a["observation"], roll_b["observation"])
    assert torch.equal(roll_a["next", "reward"], roll_b["next", "reward"])
    assert torch.equal(roll_a["action"], roll_b["action"])


def test_resets_are_fresh_but_reproducible() -> None:
    # Consecutive resets on one instance must yield DIFFERENT episodes (the rng
    # advances; reset must not re-seed to a constant), while a same-seed instance
    # must reproduce the whole episode sequence.
    env = AdaptiveNegativityEnv(n=2, seed=99)
    env.reset()
    rho1 = env.density_matrix
    env.reset()
    rho2 = env.density_matrix
    assert not np.array_equal(rho1, rho2)  # fresh state each episode

    twin = AdaptiveNegativityEnv(n=2, seed=99)
    twin.reset()
    assert np.array_equal(twin.density_matrix, rho1)
    twin.reset()
    assert np.array_equal(twin.density_matrix, rho2)  # full sequence reproduced


# ---------------------------------------------------------------------------
# Sanity — observation structure matches the spec layout
# ---------------------------------------------------------------------------
def test_observation_layout_semantics() -> None:
    env = AdaptiveNegativityEnv(n=2, n_steps=10, shots_per_step=8, seed=5)
    reset_td = env.reset()
    obs0 = reset_td["observation"]
    # On no data: all per-setting fractions/correlators are 0, estimate 0,
    # remaining budget fraction 1.
    assert torch.allclose(obs0[:-2], torch.zeros(obs0.shape[0] - 2))
    assert obs0[-2].item() == pytest.approx(0.0)
    assert obs0[-1].item() == pytest.approx(1.0)

    # After a full episode the spent fractions sum to 1 and remaining is 0.
    rollout = env.rollout(env.n_steps, _round_robin_policy(len(env.settings)))
    last = rollout["next", "observation"][-1]
    spent_fractions = last[0:-2:2]  # even indices are the per-setting spent fractions
    assert spent_fractions.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert last[-1].item() == pytest.approx(0.0, abs=1e-6)
