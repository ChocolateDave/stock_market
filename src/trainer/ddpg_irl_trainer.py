# =============================================================================
# @file   ddpg_irl_trainer.py
# @author Maverick Zhang
# @date   Nov-22-22
# =============================================================================
"""Trainer Class for Deep Deterministic Policy Gradient with IQ learn"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional, Tuple

import torch as th
from gym.core import Env
from gym.spaces import Discrete
from src.agent.ddpg_agent import DDPGAgent
from src.memory.replay_buffer import ReplayBuffer
from src.memory.base_buffer import Path
from src.trainer.base_irl_trainer import BaseIRLTrainer
from src.irl.iq_loss import iq_loss
from src.irl.utils import load_expert_behavior

import numpy as np

# TODO: 3 ways to do this:
# 1. Hard maxes <= currently this
# 2. Soft max on the OU noise
# 3. Output Gaussians and induce deterministic function from mean.
# 4. implement SAC and take rewards from Q function learned

# TODO: lots of different setups:
# 1. fully offline IRL and policy <= currently this
# 2. offline IRL and then train policy online
# 3. offline IRL, induce rewards, then train policy online
# 4. offline IRL, with online finetuning
# 5. mixed offline and online training (online IRL)
class DDPGIRLTrainer(BaseIRLTrainer):

    def __init__(self,
                 env: Env,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 device: th.device = th.device('cpu'),
                 log_dir: str = 'logs/',
                 eval_frequency: int = 10,
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
                 seed: int = 2,
                 save: bool = False,
                 expert_behavior: str = 'Pendulum-v1_11-22-22_05-36-40.pkl',
                 **kwargs) -> None:
        super().__init__(log_dir, max_episode_steps, num_episodes,
                         num_warm_up_steps, name, eval_frequency, save,
                         expert_behavior)

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
        self.steps_so_far = 0
        self.num_timesteps_before_training = 100

        expert_paths = load_expert_behavior(self.expert_behavior)
        self.expert_buffer = ReplayBuffer(max_size=buffer_size)
        self.expert_buffer.add_paths(expert_paths)
        self.num_offline_train_episodes = 10
        self.num_offline_train_iter = 1

    # just sample random batches, no need to cycle through
    # 
    def train_offline(self, episode: int) -> Any:
        #log = defaultdict(list)

        self.agent.train_mode()

        obs, acs, next_obs, rews, dones = self.expert_buffer.sample(
            self.batch_size, random=True
        )
        obs = th.from_numpy(obs).to(self.agent.device)
        acs = th.from_numpy(acs).to(self.agent.device)
        next_obs = th.from_numpy(next_obs).to(self.agent.device)
        rews = th.from_numpy(rews).to(self.agent.device)
        dones = th.from_numpy(dones).to(self.agent.device)
        is_expert = th.ones(rews.size()).to(th.bool).to(self.agent.device)
        # Update critic network
        critic_loss, log = iq_loss(self.agent, 
                self.agent.critic.discount,
                obs, 
                acs, 
                next_obs,
                0.,
                dones,
                is_expert,
                alpha = 1e-2)
        self.agent.critic_opt.zero_grad()
        critic_loss.backward()
        self.agent.critic_opt.step()
        #log['critic_loss'].append(critic_loss)

        # Update policy network
        policy_loss = self.agent.update_policy(obs, acs=None)
        log['policy_loss'] = policy_loss

        # Update target networks
        self.agent.update_target()

        return log
        

    def train_one_episode(self, epoch: int, seed: Optional[int] = None) -> Any:
        seed = seed or self.seed
        log = defaultdict(list)

        # Initialize random process
        self.agent.reset_noise()

        # Receive the initial state
        steps: int = 0
        ob = self.env.reset()

        while True:
            with th.no_grad():
                self.agent.eval_mode()

                if self.steps_so_far < self.num_timesteps_before_training:
                    ac = self.env.action_space.sample()
                    ac_loc = [ac]
                else:
                    ac = self.agent.get_action(ob, explore=True, target=False)
                    if self.discrete_action:
                        # convert one-hot to integer
                        ac_loc = ac.max(1, keepdim=False)[1].cpu().numpy()
                    else:
                        ac_loc = ac.float().cpu().numpy()
                    ac = ac.cpu().numpy()[0]

                next_ob, rew, done, truncated, _ = self.env.step(ac_loc[0])
                self.buffer.add_transition(
                    ob, ac, next_ob, rew, done
                )
                log['episode_returns'].append(rew)
                steps += 1
                self.steps_so_far += 1
                ob = next_ob
                done = done or truncated

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
                    obs=obs, acs=acs, next_obs=next_obs, next_acs=None, rews=rews, dones=dones)
                log['critic_loss'].append(critic_loss)

                # Update policy network
                policy_loss = self.agent.update_policy(obs, acs=None)
                log['policy_loss'].append(policy_loss)

                # Update target networks
                self.agent.update_target()

            if self.max_episode_steps:
                done = done or steps > self.max_episode_steps

            if done:
                return {key: sum(value)
                        if key == 'episode_returns'
                        else sum(value) / steps
                        for key, value in log.items()}

    def exec_one_episode(self,
                         episode: int = -1,
                         seed: Optional[int] = None,
                         save: bool = False) -> Any:
        seed = seed or self.seed
        log = defaultdict(list)

        if save:
            observations = []
            actions = []
            next_observations = []
            rewards = []
            dones = []

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
                next_ob, rew, done, truncated, _ = self.env.step(ac_loc[0])
                done = done or truncated

                if save:
                    observations.append(ob)
                    actions.append(ac_loc[0])
                    next_observations.append(next_ob)
                    rewards.append(rew)
                    dones.append(done)

                log['eval_returns'].append(rew)
                steps += 1
                ob = next_ob
                # print(ac_loc[0])

            if self.max_episode_steps:
                done = steps > self.max_episode_steps or done

            if done:
                ret = {key: sum(value)
                        if key == 'eval_returns'
                        else sum(value) / steps
                        for key, value in log.items()}
                if save:
                    ret['observations'] = np.stack(observations)
                    ret['actions'] = np.stack(actions)
                    ret['next_observations'] = np.stack(next_observations)
                    ret['rewards'] = np.stack(rewards)
                    ret['dones'] = np.stack(dones)
                return ret
