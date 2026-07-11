"""Train a PPO agent on AdaptiveNegativityEnv (n=2) and benchmark it.

Reproducible entry point for the Phase 5 validation checkoff:

    .venv/bin/python experiments/train_ppo.py

Trains PPO with truncation-aware GAE bootstrapping, checkpoints the policy and
logs the training curve to ``results/``, then evaluates the trained policy
against all four Phase 4 baselines on the same paired states.  Both the sampled
(the meaningful metric at n=2, where the optimum is near-uniform) and the argmax
(degenerate mode) evaluations are reported.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from anrl.agents import PPOAgentPolicy, PPOConfig, build_actor_critic, train_ppo
from anrl.baselines import (
    FixedPolicy,
    GreedyPolicy,
    OraclePolicy,
    RandomPolicy,
    evaluate_policy,
)

import torch

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CHECKPOINT = RESULTS / "ppo_policy.pt"
LOG = RESULTS / "ppo_training_log.jsonl"

SEED = 0
# Paired evaluation protocol shared with the Phase 4 baselines (500 episodes).
EVAL_SEEDS = [2024, 77, 1, 42, 7]
EVAL_N_PER_SEED = 100


def _pooled(make_policy, env_config) -> tuple[float, float]:
    errors = np.concatenate(
        [evaluate_policy(make_policy(), env_config, EVAL_N_PER_SEED, s)["errors"] for s in EVAL_SEEDS]
    )
    return float(errors.mean()), float(errors.std(ddof=1) / np.sqrt(len(errors)))


def main() -> None:
    config = PPOConfig()
    n_actions = 3 ** config.n
    env_config = {
        "n": config.n,
        "n_steps": config.n_steps,
        "shots_per_step": config.shots_per_step,
        "noise_range": config.noise_range,
    }

    print(f"Training PPO on AdaptiveNegativityEnv (n={config.n}), "
          f"{config.total_frames} frames, device={config.device} ...")
    start = time.time()
    history = train_ppo(config, seed=SEED, checkpoint_path=CHECKPOINT, log_path=LOG)
    wall = time.time() - start
    print(f"Training done in {wall:.1f}s ({wall / 60:.1f} min). "
          f"Checkpoint: {CHECKPOINT.relative_to(REPO)}  Log: {LOG.relative_to(REPO)}")

    # Training curve summary.
    iters = history["iterations"]
    first_ret = iters[0]["mean_episode_return"]
    last_ret = iters[-1]["mean_episode_return"]
    evals = [(r["frames"], r["eval_mean_error"]) for r in iters if "eval_mean_error" in r]
    print(f"\nTraining curve: mean episode return {first_ret:+.4f} -> {last_ret:+.4f}")
    print(f"                eval final error {evals[0][1]:.4f} -> {evals[-1][1]:.4f} "
          f"over {len(evals)} eval points")

    # Reload the checkpointed policy and benchmark it against the baselines.
    _, _, policy_module = build_actor_critic(
        2 * n_actions + 2, n_actions, config.num_cells, config.device
    )
    policy_module.load_state_dict(torch.load(CHECKPOINT))
    agent_argmax = PPOAgentPolicy(policy_module, config.device, deterministic=True)
    agent_sample = PPOAgentPolicy(policy_module, config.device, deterministic=False)

    print(f"\nPaired evaluation ({len(EVAL_SEEDS) * EVAL_N_PER_SEED} episodes, "
          f"{config.n_steps}x{config.shots_per_step} shots):")
    rows = {
        "PPO agent (sampled)": _pooled(lambda: agent_sample, env_config),
        "oracle": _pooled(lambda: OraclePolicy(), env_config),
        "greedy": _pooled(lambda: GreedyPolicy(), env_config),
        "fixed (uniform)": _pooled(lambda: FixedPolicy(n_actions), env_config),
        "random": _pooled(lambda: RandomPolicy(n_actions), env_config),
        "PPO agent (argmax)": _pooled(lambda: agent_argmax, env_config),
    }
    for name, (mean, sem) in sorted(rows.items(), key=lambda kv: kv[1][0]):
        print(f"  {name:20s} {mean:.4f} +/- {sem:.4f}")

    fixed_mean = rows["fixed (uniform)"][0]
    greedy_mean = rows["greedy"][0]
    sample_mean = rows["PPO agent (sampled)"][0]
    argmax_mean = rows["PPO agent (argmax)"][0]
    print(f"\nAt n=2 the optimal policy is near-uniform, so the trained policy is "
          f"stochastic; its argmax collapses to one setting (a degenerate mode).")
    print(f"Learned (sampled) agent vs uniform (fixed): {sample_mean - fixed_mean:+.4f} "
          f"| vs greedy: {sample_mean - greedy_mean:+.4f}  -> "
          f"{'parity or better (expected)' if sample_mean <= fixed_mean + 0.01 else 'worse than uniform'}.")
    print(f"Argmax agent (degenerate mode): {argmax_mean:.4f}.")


if __name__ == "__main__":
    main()
