"""RL environments for adaptive negativity estimation (TorchRL).

Exposes :class:`~anrl.envs.negativity_env.AdaptiveNegativityEnv`, a TorchRL
``EnvBase`` for the sequential single-copy measurement problem.  No agent,
training, or baseline-policy code lives here.
"""

from __future__ import annotations

from .negativity_env import AdaptiveNegativityEnv

__all__ = ["AdaptiveNegativityEnv"]
