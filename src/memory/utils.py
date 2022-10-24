# =============================================================================
# @file   utils.py
# @author Juanwu Lu
# @date   Oct-9-22
# =============================================================================
"""Memory buffer utility functions"""
import numpy as np
from copy import deepcopy
from typing import Sequence, Tuple

from src.memory.base_buffer import Path


def add_noise(data: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
    data = deepcopy(data)
    data_mean = np.mean(data, axis=0)
    data_mean[data_mean == 0] = 1e-6
    data_std = data_mean * noise_scale
    for i, _ in enumerate(data_mean):
        data[:, i] = np.copy(
            data[:, i] + np.random.normal(
                loc=0.0,
                scale=np.absolute(data_std[i]),
                size=(data.shape[0], )
            )
        )


def convert_sequence_of_paths(paths: Sequence[Path]) -> Tuple:
    states = np.concatenate([path.observation for path in paths])
    actions = np.concatenate([path.action for path in paths])
    next_states = np.concatenate([path.next_observation for path in paths])
    rewards = np.concatenate([path.reward for path in paths])
    dones = np.concatenate([path.done for path in paths])

    return (states, actions, next_states, rewards, dones)
