# =============================================================================
# @file   multi_replay_buffer.py
# @author Juanwu Lu
# @date   Oct-15-22
# =============================================================================
"""Replay Buffer Class for Multi-agent Reinforcement Learning"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch as th
from src.memory.base_buffer import Path
from src.memory.replay_buffer import ReplayBuffer
from src.memory.utils import add_noise, convert_sequence_of_paths


class MultiAgentReplayBuffer:

    def __init__(self,
                 agents: Sequence[str],
                 max_size: int = 10000) -> None:
        self._storage: Dict[str, ReplayBuffer] = {}
        for agent in agents:
            self._storage[agent] = ReplayBuffer(max_size=max_size)

    def __len__(self) -> int:
        return max([len(buffer) for buffer in self._storage.values()])

    def add_transition(self,
                       ob_n: Dict[str, np.ndarray],
                       ac_n: Dict[str, np.ndarray],
                       next_ob_n: Dict[str, np.ndarray],
                       rew_n: Dict[str, np.ndarray],
                       done_n: Dict[str, np.ndarray],
                       noised: bool = False) -> None:
        assert ob_n.keys() == ac_n.keys() == next_ob_n.keys() == \
            rew_n.keys() == done_n.keys(), ValueError("Mismatch agents!")
        for agent in ob_n.keys():
            self._storage[agent].add_transition(
                ob=ob_n[agent],
                ac=ac_n[agent],
                next_ob=next_ob_n[agent],
                rew=rew_n[agent],
                done=done_n[agent],
                noised=noised
            )

    def sample(self,
               batch_size: int,
               device: th.device = th.device('cpu'),
               random: bool = True,
               ) -> Tuple[Dict[str, np.ndarray], ...]:
        obs_n, acs_n, next_obs_n, rews_n, dones_n = {}, {}, {}, {}, {}
        for agent, buffer in self._storage.items():
            obs, acs, next_obs, rews, dones = \
                buffer.sample(batch_size, random)
            assert obs.shape[0] == acs.shape[0] == next_obs.shape[0]
            obs_n[agent] = th.from_numpy(obs).to(device)
            acs_n[agent] = th.from_numpy(acs).to(device)
            next_obs_n[agent] = th.from_numpy(next_obs).to(device)
            rews_n[agent] = th.from_numpy(rews).to(device)
            dones_n[agent] = th.from_numpy(dones).to(device)

        return (obs_n, acs_n, next_obs_n, rews_n, dones_n)
