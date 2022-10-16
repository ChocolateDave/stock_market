# =============================================================================
# @file   base_policy.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base policy function module"""
import os
from typing import Sequence, Union

from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.nn import Module

# Type Alias
# =========================================
_PathLike = Union[str, 'os.PathLike[str]']


class BasePolicy(Module):

    def forward(self, obs: Tensor) -> Union[Tensor, Distribution]:
        raise NotImplementedError

    def get_action(self, obs: Sequence) -> Sequence:
        raise NotImplementedError

    def save(self, filepath: _PathLike) -> None:
        raise NotImplementedError

    def sync(self) -> None:
        raise NotImplementedError
