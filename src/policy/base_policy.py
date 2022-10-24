# =============================================================================
# @file   base_policy.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base policy function module"""
from __future__ import annotations

from typing import Sequence

from src.nn import BaseNN
from torch import Tensor, nn
from torch.distributions import Distribution


class BasePolicy(nn.Module):
    policy_net: BaseNN
    target_policy_net: BaseNN

    def forward(self, obs: Tensor, target: bool = False) -> Distribution:
        raise NotImplementedError

    def get_action(self,
                   obs: Tensor,
                   explore: bool = True,
                   target: bool = False) -> Sequence:
        raise NotImplementedError

    def sync(self) -> None:
        raise NotImplementedError
