"""PPO training for the adaptive-negativity environment (torchrl 0.13.2).

Components (torchrl current API):
* :class:`torchrl.collectors.Collector` — synchronous data collection.
* :class:`torchrl.objectives.ClipPPOLoss` — clipped PPO objective.
* GAE via ``loss.make_value_estimator(ValueEstimators.GAE, ...)``.
* :class:`torchrl.envs.StepCounter` — imposes the fixed budget as a *truncation*.

Truncation bootstrapping (the correctness requirement): the RL env is built with
``terminate_at_horizon=False`` so it never self-terminates, and a
``StepCounter(max_steps=n_steps)`` transform ends the episode with
``truncated=True`` while ``terminated`` stays ``False``.  GAE reads the
``terminated`` key (default) to decide bootstrapping — it masks the next value
only at ``terminated`` steps — so at the horizon it bootstraps ``V(next_obs)``
instead of assuming zero future value.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import Compose, InitTracker, StepCounter, TransformedEnv
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from anrl.agents.networks import PPOAgentPolicy, build_actor_critic
from anrl.baselines import evaluate_policy
from anrl.envs import AdaptiveNegativityEnv


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters and run configuration for PPO training."""

    # environment
    n: int = 2
    n_steps: int = 24
    shots_per_step: int = 8
    noise_range: tuple[float, float] = (0.0, 0.5)
    # network
    num_cells: tuple[int, ...] = (64, 64)
    # PPO
    gamma: float = 0.99
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 1.0
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    # collection / optimization
    frames_per_batch: int = 2400  # multiple of n_steps -> whole episodes
    total_frames: int = 150_000
    num_epochs: int = 8
    minibatch_size: int = 256
    # evaluation / logging
    eval_every: int = 5
    eval_episodes: int = 200
    eval_seed: int = 2024
    device: str = "cpu"


def make_rl_env(config: PPOConfig, seed: int | None) -> TransformedEnv:
    """Build the truncation-aware training environment.

    ``terminate_at_horizon=False`` + ``StepCounter(max_steps=n_steps)`` => the
    horizon is a truncation (``truncated=True, terminated=False``).
    """
    base = AdaptiveNegativityEnv(
        n=config.n,
        n_steps=config.n_steps,
        shots_per_step=config.shots_per_step,
        noise_range=config.noise_range,
        seed=seed,
        device=config.device,
        terminate_at_horizon=False,
    )
    return TransformedEnv(base, Compose(StepCounter(max_steps=config.n_steps), InitTracker()))


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_ppo(
    config: PPOConfig,
    seed: int,
    checkpoint_path: str | Path,
    log_path: str | Path,
) -> dict:
    """Train a PPO agent; checkpoint the policy and log the training curve.

    Returns a history dict with per-iteration ``mean episode return`` and, at the
    evaluation cadence, ``mean final negativity error`` (± SE).  The log is also
    written to ``log_path`` as JSON Lines and the policy weights to
    ``checkpoint_path``.
    """
    _seed_everything(seed)
    device = config.device
    obs_dim = 2 * (3 ** config.n) + 2
    n_actions = 3 ** config.n

    actor, critic, policy_module = build_actor_critic(
        obs_dim, n_actions, config.num_cells, device
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=config.entropy_coeff,
        critic_coeff=config.critic_coeff,
        normalize_advantage=True,
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=config.gamma, lmbda=config.lmbda
    )
    adv_module = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

    collector = Collector(
        lambda: make_rl_env(config, seed),
        actor,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
        device=device,
        auto_register_policy_transforms=True,
    )

    replay = ReplayBuffer(
        storage=LazyTensorStorage(config.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=config.minibatch_size,
    )

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict = {"config": asdict(config), "seed": seed, "iterations": []}
    n_minibatches = max(1, config.frames_per_batch // config.minibatch_size)
    eval_config = {
        "n": config.n,
        "n_steps": config.n_steps,
        "shots_per_step": config.shots_per_step,
        "noise_range": config.noise_range,
    }

    with open(log_path, "w") as log_file:
        frames_seen = 0
        for it, batch in enumerate(collector):
            frames_seen += batch.numel()

            # Advantage + value target (GAE bootstraps on truncation).
            with torch.no_grad():
                adv_module(batch)

            data = batch.reshape(-1)
            replay.empty()
            replay.extend(data)

            for _ in range(config.num_epochs):
                for _ in range(n_minibatches):
                    minibatch = replay.sample()
                    losses = loss_module(minibatch)
                    total_loss = (
                        losses["loss_objective"]
                        + losses["loss_critic"]
                        + losses["loss_entropy"]
                    )
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config.max_grad_norm)
                    optim.step()
                    optim.zero_grad()

            # Mean episode return: rewards summed per completed episode.
            rewards = batch["next", "reward"]
            n_done = int(batch["next", "done"].sum().item())
            mean_return = float(rewards.sum().item() / max(1, n_done))

            record = {
                "iteration": it,
                "frames": frames_seen,
                "mean_episode_return": mean_return,
            }

            is_last = frames_seen >= config.total_frames
            if it % config.eval_every == 0 or is_last:
                # Evaluate the actual (stochastic) learned policy: at n=2 the
                # optimum is near-uniform, so the argmax mode is degenerate and
                # the sampled policy is the meaningful metric.
                eval_res = evaluate_policy(
                    PPOAgentPolicy(policy_module, device, deterministic=False),
                    eval_config,
                    config.eval_episodes,
                    config.eval_seed,
                )
                record["eval_mean_error"] = eval_res["mean"]
                record["eval_sem"] = eval_res["sem"]

            history["iterations"].append(record)
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

    collector.shutdown()
    torch.save(policy_module.state_dict(), checkpoint_path)
    history["checkpoint_path"] = str(checkpoint_path)
    return history
