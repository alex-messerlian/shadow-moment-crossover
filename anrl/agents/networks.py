"""Policy-value network and evaluation adapter for the PPO agent.

A small MLP maps the 20-dim observation to a categorical policy over the 9
measurement settings and a scalar state value.  Policy and value use separate
MLP trunks (simplest correct choice for this small problem).
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import Categorical
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator

from anrl.baselines.policies import Policy


def build_actor_critic(
    obs_dim: int,
    n_actions: int,
    num_cells: tuple[int, ...] = (64, 64),
    device: str | torch.device = "cpu",
) -> tuple[ProbabilisticActor, ValueOperator, TensorDictModule]:
    """Build the categorical actor, the value critic, and the raw logits module.

    Returns
    -------
    (actor, critic, policy_module)
        ``actor`` samples an integer action from a Categorical over the logits;
        ``critic`` maps the observation to ``state_value``; ``policy_module`` is
        the deterministic observation -> logits map (used for argmax evaluation).
    """
    policy_net = MLP(
        in_features=obs_dim,
        out_features=n_actions,
        num_cells=list(num_cells),
        activation_class=torch.nn.Tanh,
        device=device,
    )
    policy_module = TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        spec=Categorical(n=n_actions),
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    value_net = MLP(
        in_features=obs_dim,
        out_features=1,
        num_cells=list(num_cells),
        activation_class=torch.nn.Tanh,
        device=device,
    )
    critic = ValueOperator(value_net, in_keys=["observation"], out_keys=["state_value"])
    return actor, critic, policy_module


class PPOAgentPolicy(Policy):
    """Adapter exposing a trained policy as a Phase 4 :class:`Policy`.

    Scored with the same :func:`~anrl.baselines.evaluate_policy` harness as the
    baselines.  ``deterministic=True`` (default) acts greedily (argmax of the
    logits); ``deterministic=False`` samples from the categorical policy, using
    the per-episode generator from :meth:`reset` for reproducibility.

    Note: at n=2 the optimal policy is near-uniform, so its argmax collapses onto
    a single setting (no meaningful mode); the sampled policy is the meaningful
    evaluation of what PPO learned.
    """

    def __init__(
        self,
        policy_module: TensorDictModule,
        device: str | torch.device = "cpu",
        deterministic: bool = True,
    ) -> None:
        self._policy_module = policy_module
        self._device = device
        self._deterministic = deterministic
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, observation, env) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self._device)
            td = TensorDict({"observation": obs}, batch_size=[])
            logits = self._policy_module(td)["logits"]
            if self._deterministic:
                return int(torch.argmax(logits).item())
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            rng = self._rng if self._rng is not None else np.random.default_rng()
            return int(rng.choice(probs.shape[-1], p=probs))
