# =============================================================================
# @file   ddpg_agent.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Deep Deterministic Policy Gradient agent module."""
from typing import Any, Mapping, Optional, Tuple, Union

import torch as th
from src.agent.base_agent import BaseAgent
from src.critic.ddpg_critic import DDPGCritic
from src.nn import BaseNN
from src.policy.ddpg_policy import DDPGPolicy
from torch import Tensor, optim


class DDPGAgent(BaseAgent):

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 device: Optional[th.device] = None,
                 policy_net: Optional[Union[str, BaseNN]] = "mlp",
                 policy_net_kwargs: Optional[Mapping[str, Any]] = None,
                 critic_net: Optional[Union[str, BaseNN]] = "mlp",
                 critic_net_kwargs: Optional[Mapping[str, Any]] = None,
                 lr: Optional[float] = 1e-4,
                 policy_lr: Optional[float] = None,
                 critic_lr: Optional[float] = None,
                 discount: Optional[float] = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: Optional[float] = 0.9,
                 **kwargs) -> None:
        super().__init__()

        self.policy = DDPGPolicy(
            observation_size,
            action_size,
            policy_net,
            policy_net_kwargs,
            device,
            policy_lr or lr,
        )
        self.policy_opt = optim.Adam(
            self.policy.policy_net.parameters(),
            lr=policy_lr if policy_lr else lr
        )

        self.critic = DDPGCritic(
            observation_size,
            action_size,
            critic_net,
            critic_net_kwargs,
            device,
            discount,
            critic_lr,
            soft_update_tau,
            grad_clip
        )

        self.training_step = 0

    def train_one_step(self) -> None:
        return super().train_one_step()

    def save(self, filepath) -> None:
        return super().save(filepath)

    def _calc_bellman_target(self) -> Tensor:
        pass
