# =============================================================================
# @file   base_policy.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base policy function module"""
import os
from typing import Sequence, Union

# Type Alias
# =========================================
_PathLike = Union[str, 'os.PathLike[str]']


class BasePolicy:
    def get_action(self, obs: Sequence) -> Sequence:
        raise NotImplementedError

    def update(self, obs: Sequence, act: Sequence) -> None:
        raise NotImplementedError

    def save(self, filepath: _PathLike) -> None:
        raise NotImplementedError
