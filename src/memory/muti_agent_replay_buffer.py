# =============================================================================
# @file   multi_agent_replay_buffer.py
# @author Juanwu Lu
# @date   Oct-15-22
# =============================================================================
"""Replay Buffer Class for Multi-agent Reinforcement Learning"""
from typing import Sequence

import numpy as np
from src.memory.base_buffer import BaseBuffer, Path
from src.memory.utils import add_noise, convert_sequence_of_paths


class MultiAgentReplayBuffer(BaseBuffer):

    def __init__(self,
                 max_size: int,
                 obs_sizes: Sequence[int],
                 ac_sizes: Sequence[int]) -> None:
        assert len(obs_sizes) == len(ac_sizes), ValueError(
            'Number of agents not match. '
            f'Got {len(obs_sizes)} and {len(ac_sizes)}.'
        )
        self.num_agents = len(ac_sizes)
        super().__init__(max_size=max_size,
                         observations=[
                             np.empty((0, ob_sz)) for ob_sz in obs_sizes],
                         actions=[
                             np.empty((0, ac_sz)) for ac_sz in ac_sizes],
                         next_observations=[
                             np.empty((0, ob_sz)) for ob_sz in ac_sizes],
                         rewards=[np.empty((0, 1)) for _ in ac_sizes],
                         dones=[
                             np.empty((0, 1), dtype='bool') for _ in ac_sizes])

    def __len__(self) -> int:
        return self.observations[0].shape[0]

    def add(self,
            paths: Sequence[Sequence[Path]],
            noised: bool = False) -> None:
        """Add to buffer.

        Args:
            paths: A sequence of sequence of paths, shape = `[num_agents, *]`.
            noised: A boolean flag if add noise to observations.
        """

        assert len(paths) == self.num_agents, ValueError(
            'Number of agents not match. '
            f'Expected {self.num_agents:d}, but got {len(paths):d}.'
        )
        self.paths += list(paths)

        # Traverse all the agents
        # =========================================
        for i, path_i in enumerate(paths):
            # Convert a sequence of rollouts
            obs, act, next_obs, r, d = convert_sequence_of_paths(path_i)

            # Add noise to observations
            if noised:
                obs = add_noise(obs)
                next_obs = add_noise(next_obs)

            self.observations[i] = np.concatenate(
                [self.observations[i], obs])[-self.max_size:]
            self.actions = np.concatenate([self.actions, act])[-self.max_size]
            self.next_observations = np.concatenate(
                [self.next_observations[i], next_obs])[-self.max_size:]
            self.rewards = np.concatenate(
                [self.rewards[i], r])[-self.max_size:]
            self.dones = np.concatenate([self.dones[i], d])[-self.max_size:]

    def sample(self, batch_size: int, random: bool = False) -> Sequence[Path]:
        if random:
            # Randomly sample data from buffer
            rand_idcs = np.random.permutation(len(self))[:batch_size]
            return [
                (
                    self.observations[i][rand_idcs],
                    self.actions[i][rand_idcs],
                    self.next_observations[i][rand_idcs],
                    self.rewards[i][rand_idcs],
                    self.dones[i][rand_idcs]
                ) for i in range(len(self.paths[0]))
            ]
        else:
            # Sample from recent paths
            cntr: int = 0
            idx: int = -1
            while cntr <= batch_size:
                recent_sample = self.paths[idx]
                cntr += recent_sample[0].length
                idx -= 1
            return [
                convert_sequence_of_paths(self.paths[idx:][i])
                for i in range(len(self.paths[0]))
            ]
