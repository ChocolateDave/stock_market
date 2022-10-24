# =============================================================================
# @file   base_critic.py
# @author Juanwu Lu
# @date   Oct-9-22
# =============================================================================
"""Base critic function approximator class"""
from __future__ import annotations

from typing import Optional

from src.nn import BaseNN
from torch import Tensor, nn


class BaseCritic(nn.Module):
    critic_net: BaseNN
    target_critic_net: BaseNN

    def forward(self,
                obs: Tensor,
                action: Optional[Tensor] = None,
                target: bool = False) -> Tensor:
        raise NotImplementedError

    def sync(self, non_blocking: bool = False) -> None:
        raise NotImplementedError
