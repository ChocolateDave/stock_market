# =============================================================================
# @file   replay_buffer.py
# @author Juanwu Lu
# @date   Oct-9-22
# =============================================================================
"""Vanilla Replay Buffer Module"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from src.memory.base_buffer import BaseBuffer, Path
from src.memory.utils import add_noise, convert_sequence_of_paths


class ReplayBuffer(BaseBuffer):

    def add_paths(self, paths: Sequence[Path], noised: bool = False) -> None:
        self.paths += list(paths)

        # Convert a sequence of rollouts
        obs, act, next_obs, r, d = convert_sequence_of_paths(paths)

        # Add noise to observations
        if noised:
            obs = add_noise(obs)
            next_obs = add_noise(next_obs)

        # Update data pointer
        if self.observations is None:
            self.observations = obs[-self.max_size:]
            self.actions = act[-self.max_size:]
            self.next_observations = next_obs[-self.max_size:]
            self.rewards = r[-self.max_size:]
            self.dones = d[-self.max_size:]
        else:
            self.observations = np.concatenate(
                [self.observations, obs])[-self.max_size:]
            self.actions = np.concatenate([self.actions, act])[-self.max_size:]
            self.next_observations = np.concatenate(
                [self.next_observations, next_obs])[-self.max_size:]
            self.rewards = np.concatenate([self.rewards, r])[-self.max_size:]
            self.dones = np.concatenate([self.dones, d])[-self.max_size:]

    def sample(self, batch_size: int, random=False) -> Sequence[Path]:
        assert (
            self.observations.shape[0] ==
            self.actions.shape[0] ==
            self.next_observations.shape[0] ==
            self.rewards.shape[0] ==
            self.dones.shape[0]
        ), RuntimeError("Unmatched size MDP tuple elements found!")

        if random:
            # Randomly sample data from buffer
            rand_idcs = np.random.permutation(len(self))[:batch_size]
            return (
                self.observations[rand_idcs],
                self.actions[rand_idcs],
                self.next_observations[rand_idcs],
                self.rewards[rand_idcs],
                self.dones[rand_idcs]
            )
        else:
            # Sample from recent paths
            cntr: int = 0
            idx: int = -1
            while cntr <= batch_size:
                recent_sample = self.paths[idx]
                cntr += recent_sample.length
                idx -= 1
            return convert_sequence_of_paths(self.paths[idx:])
