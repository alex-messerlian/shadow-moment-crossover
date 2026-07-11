"""Tests for the PPO agent (Phase 5).

Pins the truncation-vs-termination correctness requirement and the GAE
bootstrapping wiring, plus a smoke training run and its determinism.
"""

from __future__ import annotations

import json

import torch
from tensordict import TensorDict
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from anrl.agents import PPOConfig, build_actor_critic, make_rl_env, train_ppo


def _smoke_config() -> PPOConfig:
    return PPOConfig(
        frames_per_batch=240,
        total_frames=480,
        num_epochs=1,
        minibatch_size=120,
        eval_every=1,
        eval_episodes=20,
    )


# ---------------------------------------------------------------------------
# 1 — the horizon is a TRUNCATION and GAE bootstraps on it
# ---------------------------------------------------------------------------
def test_horizon_is_truncation_not_termination() -> None:
    config = PPOConfig(n_steps=24)
    env = make_rl_env(config, seed=0)
    rollout = env.rollout(40)  # stops at the horizon

    assert rollout.shape[0] == config.n_steps  # truncated exactly at the budget
    last = rollout[-1]["next"]
    assert bool(last["truncated"].item()) is True
    assert bool(last["terminated"].item()) is False
    assert bool(last["done"].item()) is True
    # Mid-episode nothing is set.
    mid = rollout[config.n_steps // 2]["next"]
    assert bool(mid["truncated"].item()) is False
    assert bool(mid["terminated"].item()) is False


def test_gae_bootstraps_on_truncation() -> None:
    config = PPOConfig()
    obs_dim, n_actions = 2 * (3 ** config.n) + 2, 3 ** config.n
    actor, critic, _ = build_actor_critic(obs_dim, n_actions)

    loss = ClipPPOLoss(actor, critic, entropy_bonus=True)
    loss.make_value_estimator(ValueEstimators.GAE, gamma=config.gamma, lmbda=config.lmbda)
    gae = loss.value_estimator
    # GAE decides bootstrapping from the `terminated` key, not `done`.
    assert gae.tensor_keys.terminated == "terminated"

    def transition(terminated: bool, truncated: bool) -> TensorDict:
        obs = torch.zeros(1, 1, obs_dim)
        next_obs = torch.ones(1, 1, obs_dim)
        return TensorDict(
            {
                "observation": obs,
                "next": TensorDict(
                    {
                        "observation": next_obs,
                        "reward": torch.zeros(1, 1, 1, dtype=torch.float64),
                        "done": torch.ones(1, 1, 1, dtype=torch.bool),
                        "terminated": torch.full((1, 1, 1), terminated, dtype=torch.bool),
                        "truncated": torch.full((1, 1, 1), truncated, dtype=torch.bool),
                    },
                    [1, 1],
                ),
            },
            [1, 1],
        )

    with torch.no_grad():
        v_next = float(critic(TensorDict({"observation": torch.ones(1, 1, obs_dim)}, [1, 1]))["state_value"])
        vt_term = float(gae(transition(True, False))["value_target"])
        vt_trunc = float(gae(transition(False, True))["value_target"])

    # reward is 0, so: terminated -> value_target ~ 0 (no bootstrap);
    # truncated  -> value_target ~ gamma * V(next_obs) (bootstrap).
    assert abs(vt_term) < 1e-5
    assert abs(vt_trunc - config.gamma * v_next) < 1e-4
    # And they genuinely differ (unless V(next) is coincidentally ~0).
    assert abs(vt_trunc - vt_term) > 1e-6 or abs(v_next) < 1e-6


# ---------------------------------------------------------------------------
# 2 — the training script runs end to end and produces a checkpoint + log
# ---------------------------------------------------------------------------
def test_training_smoke_runs(tmp_path) -> None:
    ckpt = tmp_path / "policy.pt"
    log = tmp_path / "log.jsonl"
    history = train_ppo(_smoke_config(), seed=0, checkpoint_path=ckpt, log_path=log)

    assert ckpt.exists()
    assert log.exists()
    assert len(history["iterations"]) >= 1
    # The log is valid JSON Lines with the expected fields.
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == len(history["iterations"])
    assert all("mean_episode_return" in r for r in lines)
    assert any("eval_mean_error" in r for r in lines)
    # The checkpoint reloads into a fresh policy module.
    _, _, fresh = build_actor_critic(20, 9)
    fresh.load_state_dict(torch.load(ckpt))


# ---------------------------------------------------------------------------
# 3 — determinism: same seed reproduces the same training curve
# ---------------------------------------------------------------------------
def test_training_is_deterministic(tmp_path) -> None:
    def run(tag: str):
        h = train_ppo(
            _smoke_config(),
            seed=123,
            checkpoint_path=tmp_path / f"{tag}.pt",
            log_path=tmp_path / f"{tag}.jsonl",
        )
        return [
            (r["mean_episode_return"], r.get("eval_mean_error")) for r in h["iterations"]
        ]

    assert run("a") == run("b")
