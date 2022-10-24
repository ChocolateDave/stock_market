# =============================================================================
# @file   ddpg_agent.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Deep Deterministic Policy Gradient agent module."""
from typing import Any, Mapping, Optional, Tuple, Union

import torch as th
from numpy import ndarray
from src.agent.base_agent import BaseAgent
from src.critic.ddpg_critic import DDPGCritic
from src.nn import BaseNN
from src.policy.ddpg_policy import DDPGPolicy
from torch import Tensor, optim


class DDPGAgent(BaseAgent):

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 device: Optional[th.device] = None,
                 policy_net: Optional[Union[str, BaseNN]] = "MLP",
                 policy_net_kwargs: Optional[Mapping[str, Any]] = None,
                 critic_net: Optional[Union[str, BaseNN]] = "MLP",
                 critic_net_kwargs: Optional[Mapping[str, Any]] = None,
                 learning_rate: Optional[float] = 1e-4,
                 policy_lr: Optional[float] = None,
                 critic_lr: Optional[float] = None,
                 discount: Optional[float] = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: Optional[float] = 0.9,
                 **kwargs) -> None:
        super().__init__()

        self.device = device
        self.policy: DDPGPolicy = DDPGPolicy(
            observation_size=observation_size,
            action_size=action_size,
            policy_net=policy_net,
            policy_net_kwargs=policy_net_kwargs,
            device=device,
            learning_rate=policy_lr or learning_rate,
            soft_update_tau=soft_update_tau,
        )
        self.policy_opt = optim.Adam(
            self.policy.policy_net.parameters(),
            lr=policy_lr or learning_rate
        )

        self.critic: DDPGCritic = DDPGCritic(
            observation_size=observation_size,
            action_size=action_size,
            critic_net=critic_net,
            critic_net_kwargs=critic_net_kwargs,
            device=device,
            discount=discount,
            learning_rate=critic_lr or learning_rate,
            soft_update_tau=soft_update_tau,
            grad_clip=grad_clip
        )

        self.critic_opt = optim.Adam(
            self.critic.q_net.parameters(),
            lr=critic_lr or learning_rate
        )
        self.training_step = 0

    def train_one_step(self,
                       obs: ndarray,
                       acs: ndarray,
                       next_obs: ndarray,
                       rews: ndarray,
                       dones: ndarray,
                       other_acs: Optional[ndarray] = None
                       ) -> Mapping[str, float]:
        obs = th.from_numpy(obs).to(self.device)
        acs = th.from_numpy(acs).to(self.device)
        rews = th.from_numpy(rews).to(self.device)
        next_obs = th.from_numpy(next_obs).to(self.device)
        dones = th.from_numpy(dones).to(self.device)
        if other_acs:
            other_acs = th.from_numpy(other_acs).to(self.device)

        # Resolve dimension issue
        if rews.dim() == 1:
            rews = rews.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)

        critic_loss = self.update_critic(obs, acs, next_obs, rews, dones)
        policy_loss = self.update_policy(obs, other_acs)
        # I'm not sure what boolean needs to go in here?
        # (Juanwu): non_blocking is used in case data is transferred in
        # between cpu and gpu without proper handling
        self.sync(non_blocking=True)

        return {
            "Critic/Loss": critic_loss,
            "Policy/Loss": policy_loss
        }

    def save(self, filepath) -> None:
        return super().save(filepath)

    def update_critic(self,
                      obs: Tensor,
                      acs: Tensor,
                      next_obs: Tensor,
                      rews: Tensor,
                      dones: Tensor) -> float:
        # Shape issue

        # Bellman error target
        with th.no_grad():
            next_acs = self.policy.get_action(next_obs,
                                              explore=False,
                                              target=True)
            target = rews + self.critic.discount * \
                (1. - dones) * self.critic.target_forward(next_obs, next_acs)
        Q_vals = self.critic.forward(obs, acs)
        critic_loss = self.critic.loss(Q_vals, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return critic_loss.item()

    def update_policy(self,
                      obs: Optional[Tensor] = None,
                      other_acs: Optional[Tensor] = None) -> float:
        acs = self.policy.get_action(obs, explore=False)
        if other_acs is not None:
            # For MADDPG, concatenate actions from other agents
            acs = th.hstack([acs, other_acs])

        policy_loss = self.critic.forward(obs, acs).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        return policy_loss.item()

    # Updates target critic and policy networks
    def sync(self, non_blocking: bool = False) -> None:
        self.policy.sync(non_blocking)
        self.critic.sync(non_blocking)


if __name__ == "__main__":
    import argparse
    import gym
    from src.utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='YAML configuration file path.')
    args = parser.parse_args()

    config = load_config(args.config)
    env = gym.make(**config["Env"])
    agent = DDPGAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        **config["Agent"]
    )
    print(f'Agent: {agent}.')
