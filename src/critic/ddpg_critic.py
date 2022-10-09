# =============================================================================
# @file   ddpg_critic.py
# @author Juanwu Lu
# @date   Oct-9-22
# =============================================================================
"""Deep Deterministic Policy Gradient Q-function module"""
import torch as th
from torch import Tensor
from torch.nn import Module
from typing import Optional

from src.critic.base_critic import BaseCritic


class DDPGCritic(BaseCritic):
    q_net: Module
    target_q_net: Module

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 device: th.device = th.device('cpu'),
                 discount: float = 0.99,
                 soft_update_tau: Optional[float] = None,
                 grad_clip: Optional[float] = None) -> None:
        super().__init__()

    def forward(self, obs: Tensor, action: Optional[Tensor]) -> Tensor:
        pass

    def sync(self) -> None:
        if self.soft_update_tau is None:
            # Hard update
            with th.no_grad():
                for param, target_param in zip(
                    self.q_net.parameters(),
                    self.target_q_net.parameters()
                ):
                    target_param.data.copy_(param.data)

    def update(self, obs, act, next_obs, rewards, dones) -> None:
        pass
