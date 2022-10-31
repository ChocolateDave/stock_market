# =============================================================================
# @file   base_replay_buffer.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""The Base Replay Buffer data structure."""
from dataclasses import dataclass, field
from typing import List, Mapping, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray


# Trajectory type
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
    paths: List[Path] = field(default_factory=lambda: [])
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

    def add_paths(self, paths: Sequence[Path], noised: bool = False) -> None:
        raise NotImplementedError

    def add_transition(self,
                       ob: ndarray,
                       ac: ndarray,
                       next_ob: ndarray,
                       rew: ndarray,
                       done: ndarray,
                       noised: bool = False) -> None:

        if isinstance(ac, int):
            action = np.asarray([ac], dtype='float32')
        elif isinstance(ac, ndarray) and len(ac.shape) == 1:
            action = ac[None, ...]
        else:
            action = ac

        path = Path(observation=np.asarray([ob], dtype='float32'),
                    action=action,
                    next_observation=np.asarray([next_ob], dtype='float32'),
                    reward=np.asarray([rew], dtype='float32'),
                    done=np.asarray([done], dtype='int64'))
        self.add_paths([path], noised=noised)

    def sample(self, batch_size: int, random: bool = False) -> Tuple:
        raise NotImplementedError


@dataclass
class MultiBuffer:
    buffers: Mapping[str, BaseBuffer] = field(default_factory=lambda: [])

    def __len__(self) -> int:
        for _, buffer in self.buffer.items():
            return len(buffer)
