# =============================================================================
# @file   ddpg_agent.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Deep Deterministic Policy Gradient agent module."""
from typing import Optional, Tuple

import torch as th
from src.agent.base_agent import BaseAgent
from src.critic.ddpg_critic import DDPGCritic
from src.policy.ddpg_policy import DDPGPolicy
from torch import Tensor, optim


class DDPGAgent(BaseAgent):

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 discrete_action: bool = False,
                 device: Optional[th.device] = None,
                 learning_rate: Optional[float] = 1e-4,
                 policy_lr: Optional[float] = None,
                 critic_lr: Optional[float] = None,
                 discount: Optional[float] = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: Optional[float] = 0.001,
                 **kwargs) -> None:
        super().__init__()

        self.device = device
        self.policy: DDPGPolicy = DDPGPolicy(
            observation_size=observation_size,
            action_size=action_size,
            discrete_action=discrete_action,
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
            device=device,
            discount=discount,
            learning_rate=critic_lr or learning_rate,
            soft_update_tau=soft_update_tau,
            grad_clip=grad_clip
        )

        self.critic_opt = optim.Adam(
            self.critic.critic_net.parameters(),
            lr=critic_lr or learning_rate,
            weight_decay=1e-6 # 0 sometimes works better
        )

        print(learning_rate, policy_lr, critic_lr, self.critic_opt.param_groups[0]['lr'], self.policy_opt.param_groups[0]['lr'])

        self.training_step = 0

    def reset_noise(self) -> None:
        self.policy.reset_noise()

    def update_critic(self,
                      obs: Tensor,
                      acs: Tensor,
                      next_obs: Tensor,
                      rews: Tensor,
                      dones: Tensor) -> float:
        # Shape issue
        if rews.dim() == 1:
            rews = rews.view(-1, 1)
        if dones.dim() == 1:
            dones = dones.view(-1, 1)

        # Bellman error target
        with th.no_grad():
            next_acs = self.policy.get_action(next_obs,
                                              explore=False,
                                              target=True)
            target = rews + self.critic.discount * (1. - dones) * \
                self.critic.forward(next_obs, next_acs, target=True)
        Q_vals = self.critic.forward(obs, acs, target=False)
        critic_loss = self.critic.loss(Q_vals, target.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return critic_loss.item()

    def update_policy(self, obs: Tensor) -> float:
        acs = self.policy.get_action(obs, explore=False, target=False)

        policy_loss = -self.critic.forward(obs, acs).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        return policy_loss.item()

    def update_target(self, non_blocking: bool = False) -> None:
        self.policy.sync(non_blocking)
        self.critic.sync(non_blocking)

    def train_mode(self):
        self.policy.policy_net.train()
        self.critic.critic_net.train()

    def eval_mode(self):
        self.policy.policy_net.eval()
        self.critic.critic_net.eval()


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
