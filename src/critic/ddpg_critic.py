# =============================================================================
# @file   ddpg_critic.py
# @author Maverick Zhang, Juanwu Lu
# @date   Oct-9-22
# =============================================================================
"""Deep Deterministic Policy Gradient Q-function module"""
from copy import deepcopy
from typing import Optional

import torch as th
from src.critic.base_critic import BaseCritic
from src.nn.ddpg_nn import CriticNet
from torch import Tensor, nn


class DDPGCritic(BaseCritic):

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 device: th.device = th.device('cpu'),
                 discount: float = 0.99,
                 learning_rate: float = 1e-4,
                 soft_update_tau: Optional[float] = None,
                 grad_clip: Optional[float] = None) -> None:
        super().__init__()

        self.obs_size = observation_size
        self.act_size = action_size
        self.discount = discount
        self.learning_rate = learning_rate
        self.soft_update_tau = soft_update_tau
        self.grad_clip = grad_clip
        self.loss = nn.MSELoss()

        self.critic_net = CriticNet(observation_size, action_size).to(device)
        self.target_critic_net = deepcopy(self.critic_net).to(device)

        self.target_critic_net.hard_update(self.critic_net, False)

    def forward(self,
                obs: Tensor,
                acs: Tensor,
                target: bool = False) -> Tensor:
        if acs is None:
            raise RuntimeError('DDPG state-action function requires actions.')

        if acs.dim() == 1:
            acs = acs.unsqueeze(1)

        if target:
            return self.target_critic_net.forward(obs, acs)
        else:
            return self.critic_net.forward(obs, acs)

    def sync(self, non_blocking: bool = False) -> None:
        if self.soft_update_tau is None:
            self.target_critic_net.hard_update(self.critic_net,
                                               non_blocking)
        else:
            self.target_critic_net.soft_update(self.critic_net,
                                               self.soft_update_tau,
                                               non_blocking)


if __name__ == "__main__":
    # Unit Test Cases
    # =========================================
    critic = DDPGCritic(
        observation_size=10,
        action_size=2,
        critic_net="mlp",
        critic_net_kwargs={
            "in_feature": 12,
            "hidden_size": 64,
            "out_feature": 1,
            "num_layers": 2
        }
    )
    print(f"Critic Function: {critic}.")
