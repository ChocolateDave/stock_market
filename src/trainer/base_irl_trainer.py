# =============================================================================
# @file   base_irl_trainer.py
# @author Maverick Zhang
# @date   Nov-22-22
# =============================================================================
"""Base Inverse Reinforcement Learning trainer class."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from os import path as osp
from typing import Any, Mapping, Optional, Union
from src.trainer.base_trainer import BaseTrainer

from src.utils import AverageMeterGroup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
import pickle as pkl

_PathLike = Union[str, 'os.PathLike[str]']
_logger = logging.getLogger(__name__)

class BaseIRLTrainer(BaseTrainer):
    def explore(self) -> None:
        raise NotImplementedError

    def train_one_episode(self, episode: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def exec_one_episode(self, episode: int = -1) -> Mapping[str, Any]:
        raise NotImplementedError

    def train_offline(self, episode: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def __init__(self,
                 log_dir: _PathLike = 'logs/',
                 max_episode_steps: Optional[int] = None,
                 num_episodes: int = 1,
                 num_warm_up_steps: Optional[int] = None,
                 name: str = '',
                 eval_frequency: Optional[int] = 10,
                 save: bool = False,
                 expert_behavior: _PathLike = 'Pendulum-v1_11-22-22_05-36-40.pkl') -> None:

        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes or 1
        self.num_warm_up_steps = num_warm_up_steps
        self.train_step: int = 0
        self.eval_step: int = 0
        self.eval_frequency = eval_frequency

        expert_file = osp.join('src/irl/expert_behavior/', f'{expert_behavior:s}')
        with open(expert_file, 'rb') as f:
            self.expert_behavior = pkl.load(f)

        now = datetime.now().strftime('%m-%d-%d_%H-%M-%S')
        log_dir = osp.join(log_dir, f'{name:s}_{now:s}')
        self.writer = SummaryWriter(log_dir)

    def train(self, execution: bool = False, save: bool = False) -> Any:
        meter = AverageMeterGroup()
        # Warm-up exploration before training
        #while self.train_step < self.num_warm_up_steps:
        #    self.explore()

        for episode in tqdm(range(1, self.num_episodes + 1),
                            desc='Training Progress',
                            position=0,
                            leave=False):
            for i in range(200):
                log = self.train_offline(episode)
            # Update episodic tracker
            meter.update(log)
            for key, val in meter.items():
                key = 'Train/' + key
                self.writer.add_scalar(key, val, episode)
            # print('steps:', self.steps_so_far)
            if (save and episode == self.num_episodes) or \
                (episode % self.eval_frequency == 0 and execution):
                mean_reward = 0.

                for i in range(20):
                    log = self.exec_one_episode(episode, save=False)
                    mean_reward += log['eval_returns']
                log = {'mean_reward': mean_reward / 20}
                # print('log:', log)
                for key, val in log.items():
                    key = 'Execution/' + key
                    self.writer.add_scalar(key, val, episode)


