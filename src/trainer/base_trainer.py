# =============================================================================
# @file   base_trainer.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base Reinforcement Learning trainer class."""
from __future__ import annotations

import logging
import os
import time
from os import path as osp
from pathlib import Path
from typing import Any, Mapping, Optional

from src.types import OptInt, PathLike
from src.utils import AverageMeterGroup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Global Variables
CWD_DEFAULT = Path(osp.abspath(__file__)).parents[2].joinpath('run_logs')
LOGGER = logging.getLogger(__name__)


class BaseTrainer:
    def explore(self) -> None:
        raise NotImplementedError

    def train_one_episode(self, episode: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def exec_one_episode(self, episode: int = -1) -> Mapping[str, Any]:
        raise NotImplementedError

    def __init__(self,
                 max_episode_steps: OptInt = None,
                 num_episodes: int = 1,
                 num_warm_up_steps: OptInt = None,
                 exp_name: str = 'default',
                 work_dir: Optional[PathLike] = None,
                 eval_frequency: OptInt = None) -> None:

        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes or 1
        self.num_warm_up_steps = num_warm_up_steps
        self.exp_name = exp_name
        self.train_step: int = 1
        self.eval_step: int = 1
        self.eval_frequency = eval_frequency or -1

        self.work_dir = work_dir or CWD_DEFAULT
        dir_name = time.strftime("%d-%m-%Y_%H-%M-%S") + '_' + exp_name
        self.log_dir = osp.join(self.work_dir, dir_name, 'logs')
        if not osp.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.ckpt_dir = osp.join(self.work_dir, dir_name, 'checkpoint')
        if not osp.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.img_dir = osp.join(self.work_dir, dir_name, 'img')
        if not osp.isdir(self.img_dir):
            os.makedirs(self.img_dir)
        self.writer = SummaryWriter(dir_name)

    def train(self) -> Any:
        meter = AverageMeterGroup()
        # Warm-up exploration before training
        while self.train_step < self.num_warm_up_steps:
            self.explore()

        for episode in tqdm(range(1, self.num_episodes + 1),
                            desc='Training Progress',
                            position=0,
                            leave=False):

            log = self.train_one_episode(episode)
            # Update episodic tracker
            meter.update(log)
            for key, val in meter.items():
                key = 'Train/' + key
                self.writer.add_scalar(key, val, episode)

            if self.eval_frequency > 0:
                if self.train_step % self.eval_frequency == 0:
                    self.set_eval()
                    mean_reward = 0.
                    for i in range(20):
                        log = self.exec_one_epoch(episode)
                        mean_reward += log['eval_returns']
                    log = {'mean_reward': mean_reward / 20}
                    # print('log:', log)
                    for key, val in log.items():
                        key = 'Execution/' + key
                        self.writer.add_scalar(key, val, episode)
