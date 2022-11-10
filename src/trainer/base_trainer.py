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
from typing import Any, Mapping, Optional, Union

from gym.core import Env
from src.agent.base_agent import BaseAgent
from src.memory.base_buffer import BaseBuffer
from src.utils import AverageMeterGroup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_PathLike = Union[str, 'os.PathLike[str]']
_logger = logging.getLogger(__name__)


class BaseTrainer:
    agent: BaseAgent
    buffer: BaseBuffer
    env: Env

    def train_one_episode(self, epoch: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def exec_one_episode(self, epoch: int = -1) -> Mapping[str, Any]:
        raise NotImplementedError

    def __init__(self,
                 log_dir: _PathLike,
                 num_episodes: int,
                 name: str = '',
                 max_episode_steps: Optional[int] = None,
                 eval_frequency: Optional[int] = 10) -> None:

        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.eval_frequency = eval_frequency
        self.steps_so_far = 0

        now = datetime.now().strftime('%m-%d-%d_%H-%M-%S')
        log_dir = osp.join(log_dir, f'{name:s}_{now:s}')
        self.writer = SummaryWriter(log_dir)

    def train(self, execution: bool = False) -> Any:
        meter = AverageMeterGroup()
        for episode in tqdm(range(1, self.num_episodes + 1),
                            desc='Training Progress',
                            position=0,
                            leave=False):
            self.set_train()

            log = self.train_one_episode(episode)
            # Update episodic tracker
            meter.update(log)
            for key, val in meter.items():
                key = 'Train/' + key
                self.writer.add_scalar(key, val, episode)
            #print('steps:', self.steps_so_far)
            if episode % self.eval_frequency == 0 and execution:
                self.set_eval()
                mean_reward = 0.
                for i in range(20):
                    log = self.exec_one_epoch(episode)
                    mean_reward += log['eval_returns']
                log = {'mean_reward': mean_reward / 20}
                #print('log:', log)
                for key, val in log.items():
                    key = 'Execution/' + key
                    self.writer.add_scalar(key, val, episode)

    def set_train(self) -> None:
        self.agent.set_train()

    def set_eval(self) -> None:
        self.agent.set_eval()
