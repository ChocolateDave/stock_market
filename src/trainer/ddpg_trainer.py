# =============================================================================
# @file   ddpg_trainer.py
# @author Juanwu Lu
# @date   Oct-23-22
# =============================================================================
"""Trainer Class for Deep Deterministic Policy Gradient"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional, Tuple

import torch as th
from gym.core import Env
from gym.spaces import Discrete
from src.agent.ddpg_agent import DDPGAgent
from src.memory.replay_buffer import ReplayBuffer
from src.trainer.base_trainer import BaseTrainer


class DDPGTrainer(BaseTrainer):

    def __init__(self,
                 env: Env,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 device: th.device = th.device('cpu'),
                 log_dir: str = 'logs/',
                 eval_frequency: int = 1000,
                 num_episodes: int = 20000,
                 num_warm_up_steps: int = 1000,
                 name: str = '',
                 learning_rate: Optional[float] = 1e-4,
                 policy_lr: Optional[float] = None,
                 critic_lr: Optional[float] = None,
                 discount: Optional[float] = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: Optional[float] = 0.9,
                 max_episode_steps: Optional[int] = None,
                 seed: int = 42,
                 **kwargs) -> None:
        super().__init__(log_dir, max_episode_steps, num_episodes,
                         num_warm_up_steps, name, eval_frequency)

        # Retreive observation and action size
        if len(env.observation_space.shape) > 2:
            observation_size = env.observation_space
        else:
            observation_size = env.observation_space.shape[0]
        if isinstance(env.action_space, Discrete):
            action_size = env.action_space.n
            self.discrete_action = True
        else:
            action_size = env.action_space.shape[0]
            self.discrete_action = False

        self.agent: DDPGAgent = DDPGAgent(observation_size=observation_size,
                                          action_size=action_size,
                                          discrete_action=self.discrete_action,
                                          device=device,
                                          policy_lr=policy_lr,
                                          learning_rate=learning_rate,
                                          critic_lr=critic_lr,
                                          discount=discount,
                                          grad_clip=grad_clip,
                                          soft_update_tau=soft_update_tau)
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.env = env
        self.seed = seed
        self.env.reset(seed=self.seed)

    def train_one_episode(self, epoch: int, seed: Optional[int] = None) -> Any:
        seed = seed or self.seed
        log = defaultdict(list)

        # Initialize random process
        # self.agent.reset_noise()

        # Receive the initial state
        steps: int = 0
        ob = self.env.reset()

        while True:
            with th.no_grad():
                self.agent.eval_mode()

                ac = self.agent.get_action(ob, explore=True, target=False)
                if self.discrete_action:
                    # convert one-hot to integer
                    ac_loc = ac.max(1, keepdim=False)[1].cpu().numpy()
                else:
                    ac_loc = ac.float().cpu().numpy()

                next_ob, rew, done, _, _ = self.env.step(ac_loc[0])
                self.buffer.add_transition(
                    ob, ac.cpu().numpy()[0], next_ob, rew, done
                )
                log['episode_returns'].append(rew)
                steps += 1

            if len(self.buffer) > self.batch_size:
                self.agent.train_mode()

                obs, acs, next_obs, rews, dones = self.buffer.sample(
                    self.batch_size, random=True
                )
                obs = th.from_numpy(obs).to(self.agent.device)
                acs = th.from_numpy(acs).to(self.agent.device)
                next_obs = th.from_numpy(next_obs).to(self.agent.device)
                rews = th.from_numpy(rews).to(self.agent.device)
                dones = th.from_numpy(dones).to(self.agent.device)

                # Update critic network
                critic_loss = self.agent.update_critic(
                    obs, acs, next_obs, rews, dones)
                log['critic_loss'].append(critic_loss)

                # Update policy network
                policy_loss = self.agent.update_policy(obs)
                log['policy_loss'].append(policy_loss)

                # Update target networks
                self.agent.update_target()

            if self.max_episode_steps:
                done = steps > self.max_episode_steps or done

            if done:
                return {key: sum(value)
                        if key == 'episode_returns'
                        else sum(value) / steps
                        for key, value in log.items()}

    def exec_one_episode(self,
                         episode: int = -1,
                         seed: Optional[int] = None) -> Any:
        seed = seed or self.seed
        log = defaultdict(list)

        # Receive the initial state
        steps: int = 0
        ob = self.env.reset()

        self.agent.eval_mode()

        while True:
            with th.no_grad():
                ac = self.agent.get_action(ob, explore=False, target=False)
                if self.discrete_action:
                    # convert one-hot to integer
                    ac_loc = ac.max(1, keepdim=False)[1].cpu().numpy()
                else:
                    ac_loc = ac.float().cpu().numpy()
                ob, rew, done, _, _ = self.env.step(ac_loc[0])

                log['eval_returns'].append(rew)
                steps += 1
                print(ac_loc[0])

            if self.max_episode_steps:
                done = steps > self.max_episode_steps or done

            if done:
                return {key: sum(value)
                        if key == 'eval_returns'
                        else sum(value) / steps
                        for key, value in log.items()}
