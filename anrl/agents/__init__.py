"""PPO reinforcement-learning agent for the adaptive-negativity environment.

Exposes the network builder, the evaluation adapter, and the PPO training entry
point.  Truncation-aware: the fixed budget is a truncation and the value
estimator bootstraps at the horizon (see :mod:`anrl.agents.ppo`).
"""

from __future__ import annotations

from .networks import PPOAgentPolicy, build_actor_critic
from .ppo import PPOConfig, make_rl_env, train_ppo

__all__ = [
    "build_actor_critic",
    "PPOAgentPolicy",
    "PPOConfig",
    "make_rl_env",
    "train_ppo",
]
