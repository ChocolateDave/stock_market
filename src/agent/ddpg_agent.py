# =============================================================================
# @file   ddpg_agent.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Deep Deterministic Policy Gradient agent module."""
from typing import Any, Mapping, Optional, Tuple, Union

import torch as th
from src.agent.base_agent import BaseAgent
from src.critic.ddpg_critic import DDPGCritic
from src.nn import BaseNN
from src.nn.utils import from_numpy, to_numpy
from src.policy.ddpg_policy import DDPGPolicy
from src.memory.replay_buffer import ReplayBuffer
from src.memory.utils import convert_sequence_of_paths
from torch import Tensor, optim, nn


class DDPGAgent(BaseAgent):

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 device: Optional[th.device] = None,
                 policy_net: Optional[Union[str, BaseNN]] = "mlp",
                 policy_net_kwargs: Optional[Mapping[str, Any]] = None,
                 critic_net: Optional[Union[str, BaseNN]] = "mlp",
                 critic_net_kwargs: Optional[Mapping[str, Any]] = None,
                 lr: Optional[float] = 1e-4,
                 policy_lr: Optional[float] = None,
                 critic_lr: Optional[float] = None,
                 discount: Optional[float] = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: Optional[float] = 0.9,
                 batch_size: Optional[int] = 1000,
                 **kwargs) -> None:
        super().__init__()

        self.policy = DDPGPolicy(
            observation_size,
            action_size,
            policy_net,
            policy_net_kwargs,
            device,
            policy_lr or lr,
            soft_update_tau,
        )
        self.policy_opt = optim.Adam(
            self.policy.policy_net.parameters(),
            lr=policy_lr if policy_lr else lr
        )

        self.critic = DDPGCritic(
            observation_size,
            action_size,
            critic_net,
            critic_net_kwargs,
            device,
            discount,
            critic_lr,
            soft_update_tau,
            grad_clip
        )

        self.replay_buffer = ReplayBuffer(max_size=100000)

        self.training_step = 0

    def train_one_step(self) -> None:
        samples = self.replay_buffer.sample(batch_size = self.batch_size, random = True)
        states, actions, next_states, rewards, dones = convert_sequence_of_paths(samples)
        states = from_numpy(states)
        actions = from_numpy(actions)
        rewards = from_numpy(rewards)
        next_states = from_numpy(next_states)
        dones = from_numpy(dones)

        self.update_critic(states, actions, next_states, rewards, dones)
        self.update_policy(obs=states)
        self.sync(non_blocking=True) # I'm not sure what boolean needs to go in here?

    def save(self, filepath) -> None:
        return super().save(filepath)

    # I'm not using it, but if you want you can modify update_critic to use this one
    def _calc_bellman_target(self) -> Tensor:
        pass

    def update_critic(self, obs: Tensor, action: Tensor, next_obs: Tensor, rewards: Tensor, dones: Tensor) -> None:
        with th.no_grad():
            next_action = self.policy.get_target_action(next_obs)
            targets = rewards + self.critic.discount * (1. - dones) * self.critic.target_forward(obs, next_action)
        Q_vals = self.critic.forward(obs, action)
        critic_loss = self.critic.loss(Q_vals, targets)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

    def update_policy(self, obs: Optional[Tensor] = None, action: Optional[Tensor] = None) -> None:
        policy_action = self.policy.get_action(obs, explore=False)
        policy_loss = self.critic.forward(obs, policy_action).mean()

        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

    # Updates target critic and policy networks
    def sync(self, non_blocking: bool = False) -> None:
        self.policy.sync(non_blocking)
        self.critic.sync(non_blocking)

