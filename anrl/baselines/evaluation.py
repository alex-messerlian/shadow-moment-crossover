"""Evaluation harness: run a policy over many episodes and report the error.

``evaluate_policy`` runs a measurement policy over ``n_episodes`` freshly sampled
states of :class:`~anrl.envs.AdaptiveNegativityEnv` and reports the final
absolute negativity-estimation error ``|estimate_final - N_true|``.

Paired comparison: the episode *states* are derived from ``seed`` independently
of the policy (each episode uses a fresh env seeded from a policy-independent
child seed, and the state is fixed at ``reset`` before any action).  Two calls
with the same ``env_config`` and ``seed`` therefore evaluate different policies
on the *same* sequence of states, so their per-episode errors are aligned by
state and directly comparable.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torchrl.envs.utils import step_mdp

from anrl.envs import AdaptiveNegativityEnv
from anrl.baselines.policies import Policy


def evaluate_policy(
    policy: Policy,
    env_config: Mapping[str, Any],
    n_episodes: int,
    seed: int,
) -> dict:
    """Evaluate ``policy`` over ``n_episodes`` and return error statistics.

    Parameters
    ----------
    policy:
        A :class:`~anrl.baselines.policies.Policy`.  ``policy.reset(rng)`` is
        called at the start of each episode with a per-episode generator.
    env_config:
        Keyword arguments for :class:`AdaptiveNegativityEnv` (must not contain
        ``seed``; the harness controls seeding).
    n_episodes:
        Number of episodes / freshly sampled states.
    seed:
        Master seed.  Determines the shared state sequence (paired across
        policies) and each policy's per-episode randomness.

    Returns
    -------
    dict with keys:
        ``mean``               — mean final error over episodes.
        ``sem``                — standard error of that mean.
        ``errors``             — ``(n_episodes,)`` per-episode final errors.
        ``true_negativities``  — ``(n_episodes,)`` per-episode true negativity
                                 (identical across policies for a shared seed).
    """
    if "seed" in env_config:
        raise ValueError("env_config must not set 'seed'; the harness controls seeding")
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    # Two independent, reproducible seed streams: one for the environment/state
    # sequence, one for the policy's per-episode randomness.
    env_ss, policy_ss = np.random.SeedSequence(seed).spawn(2)
    env_seeds = env_ss.spawn(n_episodes)
    policy_seeds = policy_ss.spawn(n_episodes)

    errors = np.empty(n_episodes, dtype=np.float64)
    true_negativities = np.empty(n_episodes, dtype=np.float64)

    for i in range(n_episodes):
        env = AdaptiveNegativityEnv(seed=env_seeds[i], **env_config)
        td = env.reset()
        policy.reset(np.random.default_rng(policy_seeds[i]))

        for _ in range(env.n_steps):
            action = policy(td["observation"], env)
            td.set("action", torch.tensor(int(action), dtype=torch.int64))
            td = step_mdp(env.step(td))

        true_negativities[i] = env.true_negativity
        errors[i] = abs(env.current_estimate - env.true_negativity)

    mean = float(errors.mean())
    sem = float(errors.std(ddof=1) / np.sqrt(n_episodes)) if n_episodes > 1 else 0.0
    return {
        "mean": mean,
        "sem": sem,
        "errors": errors,
        "true_negativities": true_negativities,
    }
