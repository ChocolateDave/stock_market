# =============================================================================
# @file   trainer.py
# @author Juanwu Lu
# @date   Nov-28-22
# =============================================================================
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th
from gymnasium import Env
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stock_market.ddpg import DDPGAgent
from stock_market.memory import MADDPGReplayBuffer, ReplayBuffer
from stock_market.types import LOG, OptFloat, OptInt, PathLike
from stock_market.utils import (AverageMeterGroup, get_agent_dims,
                                process_sample_ac, process_step_ac)

# Global Variables
CWD_DEFAULT = Path(os.path.abspath(__file__)).parents[1].joinpath('run_logs')
LOGGER = logging.getLogger(__name__)

th.autograd.set_detect_anomaly(True)


# Base Trainer Class
# ==================
class BaseTrainer:

    def explore(self) -> int:
        """Exploration before training."""
        raise NotImplementedError

    def train_one_episode(self, episode: int) -> LOG:
        """Train model for one episode."""
        raise NotImplementedError

    def exec_one_episode(self, episode: int = -1) -> LOG:
        """Execute model for one episode."""
        raise NotImplementedError

    def __init__(self,
                 max_episode_steps: OptInt = None,
                 num_episodes: int = 1,
                 num_warm_up_steps: OptInt = None,
                 exp_name: str = 'default',
                 work_dir: Optional[PathLike] = None,
                 eval_frequency: OptInt = None,
                 save_frequency: OptInt = None) -> None:

        self.buffer = None
        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes or 1
        self.num_warm_up_steps = num_warm_up_steps
        self.exp_name = exp_name
        self.eval_step: int = 1
        self.eval_frequency = eval_frequency or -1
        self.save_frequency = save_frequency or -1

        self.work_dir = work_dir or CWD_DEFAULT
        dir_name = time.strftime("%d-%m-%Y_%H-%M-%S") + '_' + exp_name
        self.log_dir = os.path.join(self.work_dir, dir_name, 'logs')
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.ckpt_dir = os.path.join(self.work_dir, dir_name, 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.img_dir = os.path.join(self.work_dir, dir_name, 'img')
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir)
        self.writer = SummaryWriter(self.log_dir)

    def train(self) -> Any:
        meter = AverageMeterGroup()

        # Warm-up exploration before training
        pbar = tqdm(desc='Exploring...', total=self.num_warm_up_steps)
        while len(self.buffer) < self.num_warm_up_steps:
            steps = self.explore()
            pbar.update(steps)
        pbar.close()

        for episode in tqdm(range(1, self.num_episodes + 1),
                            desc='Training Progress',
                            position=0,
                            leave=False):

            log: LOG = self.train_one_episode(episode)
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


# DDPG Trainer Class
# ==================
class DDPGTrainer(BaseTrainer):

    def __init__(self,
                 env: Env,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 device: th.device = th.device('cpu'),
                 log_dir: str = 'logs/',
                 eval_frequency: int = 1000,
                 save_frequency: int = 100,
                 num_episodes: int = 20000,
                 num_warm_up_steps: int = 1000,
                 exp_name: str = '',
                 learning_rate: OptFloat = 1e-4,
                 policy_lr: OptFloat = None,
                 critic_lr: OptFloat = None,
                 discount: OptFloat = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: OptFloat = 0.9,
                 max_episode_steps: OptInt = None,
                 seed: int = 42,
                 **kwargs) -> None:
        super().__init__(log_dir, max_episode_steps, num_episodes,
                         num_warm_up_steps, exp_name,
                         eval_frequency, save_frequency)

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

    def train_one_episode(self, episode: int) -> Any:
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
                    obs, acs, next_obs, rews, dones)
                log['critic_loss'].append(critic_loss)

                # Update policy network
                policy_loss = self.agent.update_policy(obs)
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
                         seed: OptInt = None) -> Any:
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
                ob, rew, done, truncated, _ = self.env.step(ac_loc[0])
                done = done or truncated

                log['eval_returns'].append(rew)
                steps += 1
                # print(ac_loc[0])

            if self.max_episode_steps:
                done = steps > self.max_episode_steps or done

            if done:
                return {key: sum(value)
                        if key == 'eval_returns'
                        else sum(value) / steps
                        for key, value in log.items()}


# MADDPG Trainer Class
# ====================
class MADDPGTrainer(BaseTrainer):

    def __init__(self,
                 env: ParallelEnv,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 device: th.device = th.device('cpu'),
                 max_episode_steps: OptInt = None,
                 num_episodes: int = 20000,
                 num_warm_up_steps: int = 1000,
                 exp_name: str = '',
                 work_dir: Optional[PathLike] = None,
                 eval_frequency: OptInt = 1000,
                 save_frequency: OptInt = 100,
                 learning_rate: OptFloat = 1e-4,
                 policy_lr: OptFloat = None,
                 critic_lr: OptFloat = None,
                 action_range: Optional[List[float]] = None,
                 discount: OptFloat = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: OptFloat = 0.01,
                 seed: int = 42,
                 **kwargs) -> None:
        super().__init__(
            max_episode_steps, num_episodes, num_warm_up_steps,
            exp_name, work_dir, eval_frequency, save_frequency
        )

        assert isinstance(env, ParallelEnv), TypeError(
            'Currently only support pettingzoo.ParallelEnv environment.'
        )
        self.env = env
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        agent_dims = get_agent_dims(self.env)

        self.agents: Dict[str, DDPGAgent] = {}
        if hasattr(self.env, 'state_space'):
            ob_n_dim = self.env.state_space.shape[0]
        else:
            ob_n_dim = sum(val[0] for val in agent_dims.values())
        ac_n_dim = sum(val[1] for val in agent_dims.values())
        for agent_id, (ob_dim, _) in agent_dims.items():
            self.agents[agent_id] = DDPGAgent(
                action_space=self.env.action_space(agent_id),
                critic_observation_size=ob_n_dim,
                policy_observation_size=ob_dim,
                critic_action_size=ac_n_dim,
                action_range=action_range,
                device=device,
                learning_rate=learning_rate,
                policy_lr=policy_lr,
                critic_lr=critic_lr,
                discount=discount,
                grad_clip=grad_clip,
                soft_update_tau=soft_update_tau)
        self.buffer = MADDPGReplayBuffer(self.agents, buffer_size)

    def explore(self) -> int:
        ob_n = self.env.reset()
        state = self.env.state()
        episode_done = False
        step: int = 0
        while self.env.agents and not episode_done:
            actions = {_id: self.env.action_space(_id).sample()
                       for _id in self.env.agents}
            ac_n = {
                _id: process_sample_ac(ac, self.env.action_space(_id))
                for _id, ac in actions.items()
            }
            # Step action and store transition
            next_ob_n, rew_n, done_n, truncated_n, _ = self.env.step(actions)
            episode_done = all(truncated_n.values())
            next_state: np.ndarray = self.env.state()
            self.buffer.add_transition(
                state, next_state, ob_n, ac_n, next_ob_n, rew_n, done_n
            )
            ob_n = next_ob_n
            state = next_state
            step += 1

        return step

    def train(self) -> Any:
        meters = {agent_id: AverageMeterGroup() for agent_id in self.agents}

        # Warm-up exploration before training
        self.num_warm_up_steps = self.num_warm_up_steps or self.batch_size
        pbar = tqdm(desc='Exploring...', total=self.num_warm_up_steps)
        while len(self.buffer) < self.num_warm_up_steps:
            steps = self.explore()
            pbar.update(steps)
        pbar.close()

        LOGGER.info('Training...')
        for episode in tqdm(range(1, self.num_episodes + 1),
                            desc='Training Progress',
                            position=0,
                            leave=False):
            logs = self.train_one_episode(episode)

            # Update episodic tracker
            for agent_id, meter in meters.items():
                meter.update(logs[agent_id])
                for key, val in meter.items():
                    key = '/'.join([agent_id, 'Train', key])
                    self.writer.add_scalar(key, val, episode)

            # Dumping state dictionary
            if self.save_frequency > 0:
                if episode % self.save_frequency == 0:
                    state_dict = {
                        agent_id: agent.state_dict
                        for agent_id, agent in self.agents.items()
                    }
                    th.save(
                        state_dict,
                        os.path.join(
                            self.ckpt_dir,
                            '_'.join([self.exp_name, str(episode)]) + '.pt'
                        )
                    )

            if self.eval_frequency > 0:
                if self.eval_step % self.eval_frequency == 0:
                    self.set_eval()
                    logs = self.exec_one_episode(episode)

                    # Update episodic tracker
                    for agent_id, meter in meters.items():
                        meter.update(logs[agent_id])
                        for key, val in meter.items():
                            key = '/'.join([agent_id, 'Eval', key])
                            self.writer.add_scalar(key, val, episode)

    def train_one_episode(self, episode: int = 0) -> LOG:
        logs = {agent_id: defaultdict(list) for agent_id in self.agents}
        episode_done: bool = False
        episode_steps: int = 0
        ob_n: Dict[str, np.ndarray] = self.env.reset()
        state: np.ndarray = self.env.state()
        for agent in self.agents.values():
            agent.set_train()

        while self.env.agents and not episode_done:
            # Run policy network
            actions, ac_n = {}, {}
            for _id, ob in ob_n.items():
                ob = th.from_numpy(ob).view(1, -1).float().to(self.device)
                ac = self.agents[_id].get_action(
                    ob, explore=True, target=False
                )
                actions[_id] = process_step_ac(
                    ac, self.env.action_space(_id)
                )
                ac_n[_id] = ac.detach().cpu().numpy()

            # Step action and store transition
            next_ob_n, rew_n, done_n, truncated_n, _ = self.env.step(actions)
            episode_done = all(truncated_n.values())
            next_state: np.ndarray = self.env.state()
            for agent_id, rew in rew_n.items():
                logs[agent_id]['episode_returns'].append(rew)
            self.buffer.add_transition(
                state, next_state, ob_n, ac_n, next_ob_n, rew_n, done_n
            )
            ob_n = next_ob_n
            state = next_state

            for agent_id, agent in self.agents.items():
                # Train the agent
                states, next_states, obs_n, acs_n, \
                    next_obs_n, rew_n, dones_n = \
                    self.buffer.sample(self.batch_size, self.device)

                # Derive centralized state
                states = th.from_numpy(states).to(self.device)
                next_states = th.from_numpy(next_states).to(self.device)
                actions = th.hstack(list(acs_n.values()))
                next_actions = th.hstack(
                    [self.agents[_id].get_action(next_obs_n[_id],
                                                 explore=False,
                                                 target=True).detach()
                        for _id in self.agents]
                )

                ob, ac, _, rew, done = \
                    obs_n[agent_id], acs_n[agent_id], \
                    next_ob_n[agent_id], rew_n[agent_id], dones_n[agent_id]

                # Update critic network and update actions
                critic_loss = agent.update_critic(
                    obs=states,
                    acs=th.hstack(list(acs_n.values())),
                    next_obs=next_states,
                    next_acs=next_actions,
                    rews=rew,
                    dones=done
                )
                logs[agent_id]['critic_loss'].append(critic_loss)

                # Update policy network and update target network
                acs_n[agent_id] = agent.get_action(
                    ob, explore=False, target=False
                )
                policy_loss = agent.update_policy(
                    obs=states,
                    acs=th.hstack(list(acs_n.values()))
                )
                logs[agent_id]['policy_loss'].append(policy_loss)

                # Update target networks
                agent.update_target()

            episode_steps += 1

            if self.max_episode_steps:
                episode_done = episode_steps > self.max_episode_steps

        # Step learning rate scheduler
        for agent in self.agents.values():
            agent.step_lr_scheduler()

        return {agent_id: {key: sum(value)
                           if key == 'episode_returns'
                           else sum(value) / len(value)
                           for key, value in log.items()}
                for agent_id, log in logs.items()}

    def exec_one_episode(self, episode: int = -1) -> Any:
        # TODO: Implement single epoch execution
        pass
