# =============================================================================
# @file   base_replay_buffer.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""The Base Replay Buffer data structure."""
from dataclasses import dataclass
from numpy import ndarray
from typing import List, Sequence, Tuple, Union


# Trajectory type
# =========================================
@dataclass
class Path:
    observation: ndarray = None
    action: ndarray = None
    next_observation: ndarray = None
    reward: ndarray = None
    done: ndarray = None

    @property
    def length(self) -> int:
        return len(self.reward)


@dataclass
class BaseBuffer:
    max_size: int = 100000
    paths: List[Path] = None
    observations: Union[Sequence[ndarray], ndarray] = None
    actions: Union[Sequence[ndarray], ndarray] = None
    next_observations: Union[Sequence[ndarray], ndarray] = None
    rewards: Union[Sequence[ndarray], ndarray] = None
    dones: Union[Sequence[ndarray], ndarray] = None

    def __len__(self) -> int:
        if self.observations is None:
            return 0
        else:
            return self.observations.shape[0]

    def add(self, paths: Sequence[Path], noised: bool = False) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int, random: bool = False) -> Tuple:
        raise NotImplementedError
