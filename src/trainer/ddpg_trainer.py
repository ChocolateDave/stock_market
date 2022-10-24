# =============================================================================
# @file   ddpg_trainer.py
# @author Juanwu Lu
# @date   Oct-23-22
# =============================================================================
"""Trainer Class for Deep Deterministic Policy Gradient"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

import torch as th
from gym.core import Env
from gym.spaces import Discrete
from src.agent.ddpg_agent import DDPGAgent
from src.memory.replay_buffer import ReplayBuffer
from src.nn.base_nn import BaseNN
from src.trainer.base_trainer import BaseTrainer


class DDPGTrainer(BaseTrainer):

    def __init__(self,
                 env: Env,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 device: th.device = th.device('cpu'),
                 log_dir: str = 'logs/',
                 num_epochs: int = 20000,
                 name: str = '',
                 policy_net: Optional[Union[str, BaseNN]] = 'MLP',
                 policy_net_kwargs: Optional[Mapping[str, Any]] = None,
                 critic_net: Optional[Union[str, BaseNN]] = 'MLP',
                 critic_net_kwargs: Optional[Mapping[str, Any]] = None,
                 learning_rate: Optional[float] = 1e-4,
                 policy_lr: Optional[float] = None,
                 critic_lr: Optional[float] = None,
                 discount: Optional[float] = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: Optional[float] = 0.9) -> None:
        super().__init__(log_dir, num_epochs, name)

        # Retreive observation and action size
        if isinstance(env.observation_space, Discrete):
            observation_size = 1
        else:
            observation_size = env.observation_space.shape[0]
        if isinstance(env.action_space, Discrete):
            action_size = 1
        else:
            action_size = env.action_space.shape[0]

        self.agents = DDPGAgent(observation_size=observation_size,
                                action_size=action_size,
                                device=device,
                                policy_net=policy_net,
                                policy_net_kwargs=policy_net_kwargs,
                                policy_lr=policy_lr,
                                critic_net=critic_net,
                                critic_net_kwargs=critic_net_kwargs,
                                learning_rate=learning_rate,
                                critic_lr=critic_lr,
                                discount=discount,
                                grad_clip=grad_clip,
                                soft_update_tau=soft_update_tau)
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.batch_size = batch_size
        self.env = env

    def train_one_epoch(self, epoch: int) -> Any:
        # Main training loop
        obs, acs, next_obs, rews, dones = self.buffer.sample(self.batch_size)
        log = self.agents.train_one_step(obs, acs, next_obs, rews, dones)

        return log

    def exec_one_epoch(self, epoch: int = -1) -> Any:
        raise NotImplementedError
