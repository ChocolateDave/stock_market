# =============================================================================
# @file   base_agent.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base reinforcement learning agent module."""
import os
from typing import Optional, Union

from src.critic.base_critic import BaseCritic
from src.policy.base_policy import BasePolicy
from torch import Tensor, nn

# Type Alias
# =========================================
_PathLike = Union[str, 'os.PathLike[str]']


class BaseAgent:
    critic: BaseCritic
    policy: BasePolicy
    training_step: int

    def train_one_step(self) -> None:
        raise NotImplementedError

    def update_policy(self,
                      obs: Optional[Tensor] = None,
                      action: Optional[Tensor] = None) -> None:
        raise NotImplementedError

    def update_critic(self,
                      obs: Tensor,
                      action: Tensor,
                      next_obs: Tensor,
                      rewards: Tensor,
                      dones: Tensor) -> None:
        raise NotImplementedError

    def save(self, filepath: _PathLike) -> None:
        pass

    def train(self) -> None:
        for module in dir(self):
            if isinstance(module, nn.Module):
                module.train()

    def eval(self) -> None:
        for module in dir(self):
            if isinstance(module, nn.Module):
                module.eval()
