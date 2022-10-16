# =============================================================================
# @file   base_critic.py
# @author Juanwu Lu
# @date   Oct-9-22
# =============================================================================
"""Base critic function approximator class"""
from typing import Optional

from torch import Tensor
from torch.nn import Module


class BaseCritic(Module):

    def forward(self, obs: Tensor, action: Optional[Tensor]) -> Tensor:
        raise NotImplementedError

    def sync(self, non_blocking: bool = False) -> None:
        raise NotImplementedError
