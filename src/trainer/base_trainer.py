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

from src.utils import AverageMeterGroup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
import pickle as pkl

_PathLike = Union[str, 'os.PathLike[str]']
_logger = logging.getLogger(__name__)

class BaseTrainer:
    def explore(self) -> None:
        raise NotImplementedError

    def train_one_episode(self, episode: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def exec_one_episode(self, episode: int = -1) -> Mapping[str, Any]:
        raise NotImplementedError

    def __init__(self,
                 log_dir: _PathLike = 'logs/',
                 max_episode_steps: Optional[int] = None,
                 num_episodes: int = 1,
                 num_warm_up_steps: Optional[int] = None,
                 name: str = '',
                 eval_frequency: Optional[int] = 10,
                 save: bool = False) -> None:

        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes or 1
        self.num_warm_up_steps = num_warm_up_steps
        self.train_step: int = 0
        self.eval_step: int = 0
        self.eval_frequency = eval_frequency

        now = datetime.now().strftime('%m-%d-%d_%H-%M-%S')
        log_dir = osp.join(log_dir, f'{name:s}_{now:s}')
        self.writer = SummaryWriter(log_dir)
        if save:
            self.save_dir = osp.join('src/irl/expert_behavior/', f'{name:s}_{now:s}.pkl')

    def train(self, execution: bool = False, save: bool = False) -> Any:
        meter = AverageMeterGroup()
        # Warm-up exploration before training
        #while self.train_step < self.num_warm_up_steps:
        #    self.explore()
        if save:
            paths_to_save = {'observations': [],
                             'actions': [],
                             'next_observations': [],
                             'rewards': [],
                             'dones': []}

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
            # print('steps:', self.steps_so_far)
            if (save and episode == self.num_episodes) or \
                (episode % self.eval_frequency == 0 and execution):
                mean_reward = 0.

                for i in range(100):
                    if save and episode == self.num_episodes:
                        log = self.exec_one_episode(episode, save=True)
                        paths_to_save['observations'].append(log['observations'])
                        paths_to_save['actions'].append(log['actions'])
                        paths_to_save['next_observations'].append(log['next_observations'])
                        paths_to_save['rewards'].append(log['rewards'])
                        paths_to_save['dones'].append(log['dones'])
                    else:
                        log = self.exec_one_episode(episode, save=False)
                    mean_reward += log['eval_returns']
                log = {'mean_reward': mean_reward / 100}
                # print('log:', log)
                for key, val in log.items():
                    key = 'Execution/' + key
                    self.writer.add_scalar(key, val, episode)
        if save:
            paths_to_save['observations'] = np.concatenate(paths_to_save['observations'])
            paths_to_save['actions'] = np.concatenate(paths_to_save['actions'])
            paths_to_save['next_observations'] = np.concatenate(paths_to_save['next_observations'])
            paths_to_save['rewards'] = np.concatenate(paths_to_save['rewards'])
            paths_to_save['dones'] = np.concatenate(paths_to_save['dones'])
            
            with open(self.save_dir, 'wb') as f:
                pkl.dump(paths_to_save, f)


