# =============================================================================
# @file   maddpg_trainer.py
# @author Juanwu Lu
# @date   Oct-15-22
# =============================================================================
"""Trainer for Multi-agent Deep Deterministic Policy Gradient."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium.spaces import Box, Discrete, Space
from gymnasium.spaces import Tuple as TupleSpace
from pettingzoo import ParallelEnv
from src.agent.ddpg_agent import DDPGAgent
from src.memory.multi_replay_buffer import MADDPGReplayBuffer
from src.trainer.base_trainer import BaseTrainer
from src.utils import AverageMeterGroup
from tqdm import tqdm


class MADDPGTrainer(BaseTrainer):

    def __init__(self,
                 env: ParallelEnv,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 device: th.device = th.device('cpu'),
                 log_dir: str = 'logs/',
                 max_episode_steps: Optional[int] = None,
                 num_episodes: int = 20000,
                 num_warm_up_steps: int = 1000,
                 name: str = '',
                 learning_rate: Optional[float] = 1e-4,
                 policy_lr: Optional[float] = None,
                 critic_lr: Optional[float] = None,
                 discount: Optional[float] = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: Optional[float] = 0.9,
                 seed: int = 42,
                 **kwargs) -> None:
        super().__init__(
            log_dir, max_episode_steps, num_episodes, num_warm_up_steps, name
        )

        assert isinstance(env, ParallelEnv), TypeError(
            'Currently only support pettingzoo.ParallelEnv environment.'
        )
        self.env = env
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        _agent_dim = self.__init_env()

        self.agents: Dict[str, DDPGAgent] = {}
        if hasattr(self.env, 'state_space'):
            ob_n_dim = self.env.state_space()
        else:
            ob_n_dim = sum(val[0] for val in _agent_dim.values())
        ac_n_dim = sum(val[1] for val in _agent_dim.values())
        for agent_id, (ob_dim, ac_dim) in _agent_dim.items():
            self.agents[agent_id] = DDPGAgent(critic_observation_size=ob_n_dim,
                                              policy_observation_size=ob_dim,
                                              critic_action_size=ac_n_dim,
                                              policy_action_size=ac_dim,
                                              discrete_action=True,
                                              device=device,
                                              learning_rate=learning_rate,
                                              policy_lr=policy_lr,
                                              critic_lr=critic_lr,
                                              discount=discount,
                                              grad_clip=grad_clip,
                                              soft_update_tau=soft_update_tau,)
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
                _id: self.process_sample_ac(ac, self.env.action_space(_id))
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

    def train(self, execution: bool = False) -> Any:
        meters = {agent_id: AverageMeterGroup() for agent_id in self.agents}

        # Warm-up exploration before training
        print('Exploring...')
        self.num_warm_up_steps = self.num_warm_up_steps or self.batch_size
        pbar = tqdm(total=self.num_warm_up_steps)
        while len(self.buffer) < self.num_warm_up_steps:
            steps = self.explore()
            pbar.update(steps)
        pbar.close()

        print('Training...')
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

            if execution:
                self.set_eval()
                logs = self.exec_one_episode(episode)

                # Update episodic tracker
                for agent_id, meter in meters.items():
                    meter.update(logs[agent_id])
                    for key, val in meter.items():
                        key = '/'.join([agent_id, 'Train', key])
                        self.writer.add_scalar(key, val, episode)

    def train_one_episode(self, episode: int) -> Any:
        logs = {agent_id: defaultdict(list) for agent_id in self.agents}
        episode_done: bool = False
        steps: int = 0
        ob_n: Dict[str, np.ndarray] = self.env.reset()
        state: np.ndarray = self.env.state()
        for agent in self.agents.values():
            agent.set_train()

        while self.env.agents and not episode_done:
            # Run policy network
            actions, ac_n = {}, {}
            for _id, ob in ob_n.items():
                ob = th.from_numpy(ob).view(1, -1).float().to(self.device)
                ac = self.agents[_id].get_action(ob, explore=True)
                actions[_id] = self.process_forward_ac(
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

            # Train the agent
            for agent_id, agent in self.agents.items():
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

                # Update critic network
                critic_loss = agent.update_critic(
                    obs=states,
                    acs=th.hstack(list(acs_n.values())),
                    next_obs=next_states,
                    next_acs=next_actions,
                    rews=rew,
                    dones=done
                )
                logs[agent_id]['critic_loss'].append(critic_loss)

                # Update actor network
                new_action = agent.get_action(ob, False, target=False)
                acs_n[agent_id] = new_action
                policy_loss = agent.update_policy(
                    obs=states,
                    acs=th.hstack(list(acs_n.values()))
                )
                logs[agent_id]['policy_loss'].append(policy_loss)

                # Update target networks
                agent.update_target()

            self.train_step += 1
            steps += 1

            if self.max_episode_steps:
                episode_done = steps > self.max_episode_steps

        return {agent_id: {key: sum(value)
                           if key == 'episode_returns'
                           else sum(value) / len(value)
                           for key, value in log.items()}
                for agent_id, log in logs.items()}

    def exec_one_episode(self, epoch: int = -1) -> Any:
        # TODO: Implement single epoch execution
        pass

    def __init_env(self) -> Mapping[str, Tuple[int, int]]:
        # Initialize the multi-agent environment and retrieve information
        _ = self.env.reset(seed=self.seed)

        _agent_dim = {}
        for _id in self.env.agents:
            if len(self.env.observation_space(_id).shape) > 2:
                raise RuntimeError('Image observation not supported')
            observation_size = self.env.observation_space(_id).shape[0]
            _action_space = self.env.action_space(_id)
            if isinstance(_action_space, TupleSpace):
                action_size = 0
                for _sub_space in _action_space:
                    if isinstance(_sub_space, Discrete):
                        if _sub_space.n > 10:
                            # NOTE: discrete space too large
                            action_size += 1
                        else:
                            action_size += _sub_space.n
                    elif isinstance(_sub_space, Box):
                        action_size += _sub_space.shape[0]
                    else:
                        raise TypeError('Invalid action space.')
            else:
                if isinstance(_action_space, Discrete):
                    action_size = _action_space.n
                elif isinstance(_action_space, Box):
                    action_size = _action_space.shape
                else:
                    raise TypeError('Invalid action space.')
            _agent_dim[_id] = [observation_size, action_size]

        return _agent_dim

    def process_forward_ac(self,
                           data: th.Tensor,
                           action_space: Space) -> Any:
        if isinstance(action_space, Discrete):
            if action_space.n > 10:
                return data.item()
            else:
                return data.argmax(-1).item()
        elif isinstance(action_space, Box):
            return data.view(-1).detach().cpu().numpy()
        elif isinstance(action_space, TupleSpace):
            assert data.shape[-1] == len(action_space), \
                ValueError('Not enough value to unpack.')
            return tuple([self.process_forward_ac(data[:, i:i+1], space)
                          for i, space in enumerate(action_space)])
        else:
            raise TypeError('Invalid action space.')

    def process_sample_ac(self,
                          data: Union[int, np.ndarray, Tuple],
                          action_space: Space) -> np.ndarray:
        if isinstance(action_space, Discrete):
            if action_space.n > 10:
                # NOTE: Handle large discrete space
                return data
            else:
                return self.to_one_hot(data, action_space.n)
        elif isinstance(action_space, Box):
            return data
        elif isinstance(action_space, TupleSpace):
            out = np.zeros(shape=(1, len(action_space)))
            for i, (ac, ac_space) in enumerate(zip(data, action_space)):
                out[0, i] = self.process_sample_ac(ac, ac_space)
            return out
        else:
            raise TypeError('Invalid action space')

    @staticmethod
    def to_one_hot(data: Union[int, np.ndarray],
                   num_classes: int) -> np.ndarray:
        if num_classes == -1:
            num_classes = int(max(data) + 1)

        if isinstance(data, int):
            output = np.eye(num_classes)[data]
        elif isinstance(data, np.ndarray):
            raise NotImplementedError
        else:
            raise TypeError('Only support int and np.ndarray.')

        return output
