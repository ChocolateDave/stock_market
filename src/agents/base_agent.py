# =============================================================================
# @file   base_agent.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base reinforcement learning agent module."""
import os
from typing import Union

# Type Alias
# =========================================
_PathLike = Union[str, 'os.PathLike[str]']


class BaseAgent:
    def train_one_epoch(self,) -> None:
        raise NotImplementedError

    def add_to_replay_buffer(self,) -> None:
        return None

    def sample(self,) -> None:
        return None

    def save(self, filepath: _PathLike) -> None:
        pass
