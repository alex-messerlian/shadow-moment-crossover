"""Tests for the baseline policies and the evaluation harness (Phase 4).

Pins: valid action indices, the fixed cyclic schedule, random reproducibility,
paired (state-aligned) evaluation, and the n=2 ordering sanity anchor — greedy
does not meaningfully beat uniform, and the oracle is no worse than uniform.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchrl.envs.utils import step_mdp

from anrl.baselines import (
    FixedPolicy,
    GreedyPolicy,
    OraclePolicy,
    RandomPolicy,
    evaluate_policy,
)
from anrl.envs import AdaptiveNegativityEnv

N_SETTINGS_N2 = 9  # 3**2


def _run_episode_actions(policy, env, rng=None):
    """Run one episode; return the list of actions taken."""
    td = env.reset()
    policy.reset(rng)
    actions = []
    for _ in range(env.n_steps):
        action = policy(td["observation"], env)
        actions.append(action)
        td.set("action", torch.tensor(int(action), dtype=torch.int64))
        td = step_mdp(env.step(td))
    return actions


# ---------------------------------------------------------------------------
# 1 — every policy emits valid action indices throughout an episode
# ---------------------------------------------------------------------------
def test_policies_return_valid_actions() -> None:
    rng = np.random.default_rng(0)
    policies = [
        FixedPolicy(N_SETTINGS_N2),
        RandomPolicy(N_SETTINGS_N2, seed=0),
        GreedyPolicy(),
        OraclePolicy(),
    ]
    for policy in policies:
        env = AdaptiveNegativityEnv(n=2, seed=1)
        actions = _run_episode_actions(policy, env, rng)
        assert len(actions) == env.n_steps
        assert all(isinstance(a, int) for a in actions)
        assert all(0 <= a < N_SETTINGS_N2 for a in actions)


def test_greedy_explores_not_stuck() -> None:
    # Regression guard: a state reconstructed from a single setting is PPT, so
    # its witness weights (and every sensitivity) are zero; without an
    # exploration bootstrap the sensitivity*uncertainty product is uniformly
    # zero and greedy would freeze on setting 0 forever, always estimating 0.
    for seed in range(6):
        env = AdaptiveNegativityEnv(n=2, seed=seed)
        actions = _run_episode_actions(GreedyPolicy(), env, np.random.default_rng(0))
        assert len(set(actions)) >= 3, f"greedy stuck on {set(actions)} at seed {seed}"
    # On a clearly entangled state greedy must actually discover nonzero negativity.
    env = AdaptiveNegativityEnv(n=2, seed=3)
    _run_episode_actions(GreedyPolicy(), env, np.random.default_rng(0))
    assert env.true_negativity > 0.1
    assert env.current_estimate > 0.0
    # The oracle likewise explores when the true state is entangled.
    env = AdaptiveNegativityEnv(n=2, seed=3)
    oracle_actions = _run_episode_actions(OraclePolicy(), env, np.random.default_rng(0))
    assert len(set(oracle_actions)) >= 3


# ---------------------------------------------------------------------------
# 2 — fixed is the exact cyclic schedule; random is reproducible
# ---------------------------------------------------------------------------
def test_fixed_is_cyclic_schedule() -> None:
    env = AdaptiveNegativityEnv(n=2, n_steps=24, seed=3)
    actions = _run_episode_actions(FixedPolicy(N_SETTINGS_N2), env)
    assert actions == [t % N_SETTINGS_N2 for t in range(env.n_steps)]


def test_random_is_reproducible() -> None:
    a = RandomPolicy(N_SETTINGS_N2, seed=123)
    b = RandomPolicy(N_SETTINGS_N2, seed=123)
    seq_a = [a(None, None) for _ in range(50)]
    seq_b = [b(None, None) for _ in range(50)]
    assert seq_a == seq_b
    assert all(0 <= x < N_SETTINGS_N2 for x in seq_a)
    # A different seed gives a different sequence.
    c = RandomPolicy(N_SETTINGS_N2, seed=456)
    assert [c(None, None) for _ in range(50)] != seq_a


# ---------------------------------------------------------------------------
# 3 — paired evaluation: same seed -> same states across policies
# ---------------------------------------------------------------------------
def test_paired_evaluation_shares_states() -> None:
    cfg = dict(n=2, n_steps=24, shots_per_step=8)
    res_fixed = evaluate_policy(FixedPolicy(N_SETTINGS_N2), cfg, n_episodes=12, seed=2024)
    res_greedy = evaluate_policy(GreedyPolicy(), cfg, n_episodes=12, seed=2024)

    # Same seed => identical per-episode true negativities (same states),
    # aligned index-by-index so per-episode errors are directly comparable.
    assert np.allclose(res_fixed["true_negativities"], res_greedy["true_negativities"])
    assert res_fixed["errors"].shape == (12,)
    assert res_greedy["errors"].shape == (12,)
    # A different seed yields a different state sequence.
    res_other = evaluate_policy(FixedPolicy(N_SETTINGS_N2), cfg, n_episodes=12, seed=99)
    assert not np.allclose(res_fixed["true_negativities"], res_other["true_negativities"])


def test_harness_rejects_seed_in_config() -> None:
    with pytest.raises(ValueError):
        evaluate_policy(FixedPolicy(N_SETTINGS_N2), {"n": 2, "seed": 0}, n_episodes=2, seed=0)


# ---------------------------------------------------------------------------
# 4 — ordering sanity at n=2 (the anchor to our known results)
# ---------------------------------------------------------------------------
# Pooled over several seeds for a representative, tight-error-bar result (a
# single seed's random baseline is noisy).  Config is the modest budget from
# the spec: 24 steps x 8 shots.
_ORDERING_SEEDS = [2024, 77, 1, 42, 7]
_ORDERING_N_PER_SEED = 100
_MEANINGFUL_TOL = 0.01  # a "small tolerance" on the ~0.066 error scale


def _pooled_errors(make_policy, cfg):
    parts = [
        evaluate_policy(make_policy(), cfg, _ORDERING_N_PER_SEED, seed)["errors"]
        for seed in _ORDERING_SEEDS
    ]
    return np.concatenate(parts)


def test_ordering_sanity_n2() -> None:
    cfg = dict(n=2, n_steps=24, shots_per_step=8, noise_range=(0.0, 0.5))

    fixed = _pooled_errors(lambda: FixedPolicy(N_SETTINGS_N2), cfg)
    random = _pooled_errors(lambda: RandomPolicy(N_SETTINGS_N2), cfg)
    greedy = _pooled_errors(lambda: GreedyPolicy(), cfg)
    oracle = _pooled_errors(lambda: OraclePolicy(), cfg)

    def stats(e):
        return float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(e)))

    fm, fs = stats(fixed)
    rm, rs = stats(random)
    gm, gs = stats(greedy)
    om, os = stats(oracle)

    n_ep = len(fixed)
    print(f"\nn=2 ordering ({n_ep} episodes, 24x8 shots):")
    print(f"  fixed  {fm:.4f} +/- {fs:.4f}")
    print(f"  random {rm:.4f} +/- {rs:.4f}")
    print(f"  greedy {gm:.4f} +/- {gs:.4f}")
    print(f"  oracle {om:.4f} +/- {os:.4f}")
    print(f"  paired greedy-fixed = {(greedy - fixed).mean():+.4f}")
    print(f"  paired oracle-fixed = {(oracle - fixed).mean():+.4f}")

    best_uniform = min(fm, rm)  # fixed is the stronger uniform baseline
    # Greedy must NOT meaningfully beat the best uniform schedule: at n=2 full-
    # information estimation leaves no adaptive room, so myopic concentration
    # gives no advantage.
    assert gm >= best_uniform - _MEANINGFUL_TOL
    # Statistically (paired on shared states) greedy is not better than fixed.
    assert (greedy - fixed).mean() >= -_MEANINGFUL_TOL
    # The oracle is the ceiling: no worse than the best uniform baseline.
    assert om <= best_uniform + _MEANINGFUL_TOL
