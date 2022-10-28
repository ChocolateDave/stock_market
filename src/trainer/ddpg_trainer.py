# =============================================================================
# @file   ddpg_trainer.py
# @author Juanwu Lu
# @date   Oct-23-22
# =============================================================================
"""Trainer Class for Deep Deterministic Policy Gradient"""
from __future__ import annotations

from collections import defaultdict
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
                 num_episodes: int = 20000,
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
                 soft_update_tau: Optional[float] = 0.9,
                 max_episode_steps: Optional[int] = None,
                 policy_update_num_steps: Optional[int] = 1,
                 critic_update_frequency: Optional[int] = 1,
                 target_update_frequency: Optional[int] = 1,
                 seed: int = 42) -> None:
        super().__init__(log_dir, num_episodes, name, max_episode_steps)

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
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.env = env
        self.policy_update_num_steps = policy_update_num_steps
        self.critic_update_frequency = critic_update_frequency
        self.target_update_frequency = target_update_frequency
        self.seed = seed

    def train_one_episode(self, epoch: int, seed: Optional[int] = None) -> Any:
        seed = seed or self.seed
        log = defaultdict(list)

        # Initialize random process
        self.agent.reset_noise()

        # Receive the initial state
        steps: int = 0
        ob = self.env.reset(seed=seed)


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
                #if epoch % self.critic_update_frequency:
                critic_loss = self.agent.update_critic(
                    obs, acs, next_obs, rews, dones)
                log['critic_loss'].append(critic_loss)

                # Update policy network
                #for policy_step in range(self.policy_update_num_steps):
                policy_loss = self.agent.update_policy(obs)
                log['policy_loss'].append(policy_loss)

                # Update target networks
                #if epoch % self.target_update_frequency:
                self.agent.update_target()

            if self.max_episode_steps:
                done = steps > self.max_episode_steps or done

            if done:
                return {key: sum(value)
                        if key == 'episode_returns'
                        else sum(value) / steps
                        for key, value in log.items()}

    def exec_one_epoch(self, epoch: int = -1, seed: Optional[int] = None) -> Any:
        seed = seed or self.seed
        log = defaultdict(list)

        # Receive the initial state
        steps: int = 0
        ob = self.env.reset(seed=seed)
        actions_taken = []

        self.agent.eval_mode()

        while True:
            with th.no_grad():
                ac = self.agent.get_action(ob, explore=False, target=False)
                if self.discrete_action:
                    # convert one-hot to integer
                    ac_loc = ac.max(1, keepdim=False)[1].cpu().numpy()
                else:
                    ac_loc = ac.float().cpu().numpy()
                actions_taken.append(ac_loc[0])
                ob, rew, done, _, _ = self.env.step(ac_loc[0])
                
                log['eval_returns'].append(rew)
                steps += 1
                print(actions_taken)

            if self.max_episode_steps:
                done = steps > self.max_episode_steps or done

            if done:
                return {key: sum(value)
                        if key == 'eval_returns'
                        else sum(value) / steps
                        for key, value in log.items()}
