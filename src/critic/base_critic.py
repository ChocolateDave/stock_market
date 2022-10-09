# =============================================================================
# @file   base_critic.py
# @author Juanwu Lu
# @date   Oct-9-22
# =============================================================================
"""Base critic function approximator class"""
from torch import Tensor
from torch.nn import Module
from typing import Optional


class BaseCritic(Module):

    def forward(sef, obs: Tensor, action: Optional[Tensor]) -> Tensor:
        raise NotImplementedError

    def update(self,
               obs: Tensor,
               act: Tensor,
               next_obs: Tensor,
               rewards: Tensor,
               dones: Tensor) -> None:
        raise NotImplementedError
