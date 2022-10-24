# =============================================================================
# @file   base_trainer.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base Reinforcement Learning trainer class."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from os import path as osp
from typing import Any, Mapping, Sequence, Union

import numpy as np
from gym.core import Env
from src.agent.base_agent import BaseAgent
from src.memory.base_buffer import BaseBuffer, Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_PathLike = Union[str, 'os.PathLike[str]']
_logger = logging.getLogger(__name__)


class BaseTrainer:
    agents: Union[BaseAgent, Sequence[BaseAgent]]
    batch_size: int
    buffer: BaseBuffer
    env: Env

    def train_one_epoch(self, epoch: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def exec_one_epoch(self, epoch: int = -1) -> Mapping[str, Any]:
        raise NotImplementedError

    def __init__(self,
                 log_dir: _PathLike,
                 num_epochs: int,
                 name: str = '') -> None:
        self.num_epochs = num_epochs
        now = datetime.now().strftime('%m-%d-%d_%H-%M-%S')
        log_dir = osp.join(log_dir, f'{name:s}_{now:s}')
        self.writer = SummaryWriter(log_dir)

    def train(self, execution: bool = False) -> Any:
        self.set_train()

        # Initialize and fill
        while len(self.buffer) < self.batch_size:
            obs, acs, next_obs, rews, dones = [], [], [], [], []
            ob = self.env.reset()
            done = False
            while not done:
                ac = self.env.action_space.sample()
                next_ob, rew, done, _, _ = self.env.step(ac)

                obs.append(ob)
                acs.append(ac)
                next_obs.append(next_ob)
                rews.append(rew)
                dones.append(done)
            path = Path(observation=np.asarray(obs, dtype="float32"),
                        action=np.asarray(acs, dtype="float32"),
                        next_observation=np.asarray(next_obs, dtype="float32"),
                        reward=np.asarray(rews, dtype="float32"),
                        done=np.asarray(dones, dtype="int64"))
            self.buffer.add([path], noised=False)

        for epoch in tqdm(range(1, self.num_epochs + 1),
                          desc='Training Progress',
                          position=0,
                          leave=False):
            log = self.train_one_epoch(epoch)
            for key, val in log.items():
                key = 'Train/' + key
                self.writer.add_scalar(key, val, epoch)

            if execution:
                log = self.exec_one_epoch(epoch)
                for key, val in log.items():
                    key = 'Execution/' + key
                    self.writer.add_scalar(key, val, epoch)

    def set_train(self) -> None:
        if isinstance(self.agents, Sequence):
            for agent in self.agents:
                agent.train()
        else:
            self.agents.train()

    def set_eval(self) -> None:
        if isinstance(self.agents, Sequence):
            for agent in self.agents:
                agent.eval()
        else:
            self.agents.eval()
