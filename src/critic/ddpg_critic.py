# =============================================================================
# @file   ddpg_critic.py
# @author Maverick Zhang, Juanwu Lu
# @date   Oct-9-22
# =============================================================================
"""Deep Deterministic Policy Gradient Q-function module"""
from copy import deepcopy
from typing import Any, Mapping, Optional, Union

import torch as th
from src.critic.base_critic import BaseCritic
from src.nn.base_nn import BaseNN
from src.nn.utils import network_resolver
from torch import Tensor, nn


class DDPGCritic(BaseCritic):
    q_net: BaseNN
    target_q_net: BaseNN

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 critic_net: Optional[Union[str, BaseNN]] = "mlp",
                 critic_net_kwargs: Optional[Mapping[str, Any]] = None,
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

        critic_net_kwargs = critic_net_kwargs or {}
        if isinstance(critic_net, BaseNN):
            assert critic_net.in_feature == observation_size + action_size, \
                ValueError(
                    "Expect the Q-function to have an input feature of "
                    f"{observation_size + action_size:d}, "
                    f"but got {critic_net.in_feature:d}."
                )
            assert critic_net.out_feature == 1, ValueError(
                "Expect the Q-function to output a real value, "
                f"but got an output size of {critic_net.out_feature:d}."
            )
            self.q_net = critic_net.to(device)
        else:
            # Enforce input/ouput feature
            critic_net_kwargs["in_feature"] = observation_size + action_size
            critic_net_kwargs["out_feature"] = 1
            self.q_net = network_resolver(
                critic_net, **critic_net_kwargs
            ).to(device)
        self.target_q_net = deepcopy(self.q_net).to(device)

    def forward(self, obs: Tensor, action: Optional[Tensor]) -> Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(-1)

        inputs = th.cat((obs, action), dim=-1)
        return self.q_net.forward(inputs)

    def target_forward(self, obs: Tensor, action: Optional[Tensor]) -> Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(-1)

        inputs = th.cat((obs, action), dim=-1)
        return self.target_q_net.forward(inputs)

    def sync(self, non_blocking: bool = False) -> None:
        if self.soft_update_tau is None:
            # Hard update
            with th.no_grad():
                for param, target_param in zip(
                    self.q_net.parameters(),
                    self.target_q_net.parameters()
                ):
                    target_param.data.copy_(param.data)
        else:
            self.target_q_net.soft_update(
                self.q_net, self.soft_update_tau, non_blocking)


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
