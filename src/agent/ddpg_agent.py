# =============================================================================
# @file   ddpg_agent.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Deep Deterministic Policy Gradient agent module."""
import gym
import torch as th
from copy import deepcopy
from torch import optim
from typing import Any, Mapping, Optional, Union

from src.agents.base_agent import BaseAgent
from src.nn import BaseNN, network_resolver


class DDPGAgent(BaseAgent):

    def __init__(
        self,
        env: gym.Env,
        device: Optional[th.device] = None,
        policy_net: Optional[Union[str, BaseNN]] = "mlp",
        policy_net_kwargs: Optional[Mapping[str, Any]] = None,
        critic_net: Optional[Union[str, BaseNN]] = "mlp",
        critic_net_kwargs: Optional[Mapping[str, Any]] = None,
        lr: Optional[float] = 1e-4,
        policy_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        soft_update: bool = False,
        soft_tau: Optional[float] = 0.9,
        **kwargs
    ) -> None:

        self.env = env

        if isinstance(policy_net, BaseNN):
            self.online_policy = policy_net.to(device)
        else:
            self.online_policy = network_resolver(
                policy_net, **(policy_net_kwargs or {})
            )
        self.target_policy = deepcopy(self.online_policy).to(device)
        self.policy_opt = optim.Adam(
            self.online_policy.parameters(),
            lr=policy_lr if policy_lr else lr
        )

        if isinstance(critic_net, BaseNN):
            self.online_critic = critic_net.to(device)
        else:
            self.online_critic = network_resolver(
                critic_net, **(critic_net_kwargs or {})
            )
        self.target_critic = deepcopy(self.online_critic).to(device)
        self.critic_opt = optim.Adam(
            self.online_critic.parameters(),
            lr=critic_lr if critic_lr else lr
        )

        self.soft_update = soft_update
        self.soft_tau = soft_tau

    def train_one_step(self) -> None:
        return super().train_one_step()

    def add_to_replay_buffer(self) -> None:
        return super().add_to_replay_buffer()

    def sample(self) -> None:
        return super().sample()

    def save(self, filepath) -> None:
        return super().save(filepath)
