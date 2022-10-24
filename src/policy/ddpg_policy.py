# =============================================================================
# @file   ddpg_policy.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Deep Deterministic Policy module."""
from copy import deepcopy
from typing import Any, Mapping, Optional, Union

import torch as th
from numpy import ndarray
from src.nn.base_nn import BaseNN
from src.nn.utils import network_resolver
from src.policy.base_policy import BasePolicy
from torch import Tensor, nn


# OUNoise for continous random exploration
# https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# =========================================================
class OUNoise(nn.Module):
    def __init__(self,
                 action_size: int,
                 mu: float = 0.0,
                 scale: float = 0.1,
                 theta: float = 0.15,
                 sigma: float = 0.2) -> None:
        super().__init__()

        self.action_size = action_size
        self.mu = mu
        self.scale = scale
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        device = x.device

        x = self.state.to(device)
        dx = self.theta * (self.mu - x) + \
            self.sigma * th.randn(size=(len(x),), device=device)
        self.state = x + dx
        noise = self.state * self.scale
        noise = noise.requires_grad_(False)

        return noise

    def reset(self) -> None:
        self.state = th.ones(self.action_size) * self.mu


class DDPGPolicy(BasePolicy, nn.Module):
    policy_net: BaseNN
    target_policy_net: BaseNN

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 discrete_action: bool = False,
                 policy_net: Optional[Union[str, BaseNN]] = "mlp",
                 policy_net_kwargs: Optional[Mapping[str, Any]] = None,
                 device: th.device = th.device('cpu'),
                 learning_rate: float = 1e-4,
                 soft_update_tau: Optional[float] = None,
                 optimizer_kwargs: Optional[Mapping[str, Any]] = None,
                 **kwargs) -> None:
        super().__init__()

        self.obs_size = observation_size
        self.act_size = action_size
        self.soft_update_tau = soft_update_tau

        policy_net_kwargs = policy_net_kwargs or {}
        if isinstance(policy_net, BaseNN):
            assert policy_net.in_feature == observation_size and \
                policy_net.out_feature == action_size, ValueError(
                    "Expect policy network to have `in_feature` == "
                    "`observation_size` and `out_feature` == "
                    "`action_size`. But got "
                    f"{policy_net.in_feature:d} and {policy_net.out_feature:d}"
                )
            self.policy_net = policy_net.to(device)
        else:
            # Enforce input/output feature
            policy_net_kwargs['in_feature'] = observation_size
            policy_net_kwargs['out_feature'] = action_size
            self.policy_net = network_resolver(
                policy_net, **policy_net_kwargs
            )
            self.policy_net = self.policy_net.to(device)
        self.target_policy_net = deepcopy(self.policy_net).to(device)

        self.optimizer = th.optim.Adam(self.policy_net.parameters(),
                                       lr=learning_rate,
                                       **(optimizer_kwargs or {}))

        # Exploration
        self.discrete_action = discrete_action
        if discrete_action:
            self.exploration = 0.9  # epsilon-greedy initial value
        else:
            self.exploration = OUNoise(action_size)

    def forward(self, obs: Tensor) -> Tensor:
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        return self.policy_net.forward(obs)

    def get_action(self, obs: Tensor, explore: bool = True) -> ndarray:
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        actions: Tensor = self.policy_net(obs).detach()  # not in comp. graph
        if self.discrete_action:
            if explore:
                # Random sample from discrete action space with gumbel noise
                # ==========================================================
                actions = nn.functional.gumbel_softmax(actions, hard=True)
            else:
                # Generate one-hot encoding of the max-policy actions
                # ===================================================
                actions = (actions == actions.max(1, keepdim=True)[0]).float()
        else:
            if explore:
                # Explore continous action space with OUNoise
                # ===========================================
                actions += self.exploration.forward(actions)
            actions.clamp(-1, 1)

        return actions.cpu().numpy()

    def get_target_action(self, obs: Tensor) -> ndarray:
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        with th.no_grad():
            actions: Tensor = self.target_policy_net(obs)
            if self.discrete_action:
                # Generate one-hot encoding of the max-policy actions
                # ===================================================
                actions = (actions == actions.max(1, keepdim=True)[0]).float()
            else:
                actions.clamp(-1, 1)

        return actions.cpu().numpy()

    def reset_noise(self) -> None:
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale: float) -> None:
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def sync(self, non_blocking: bool = False) -> None:
        if self.soft_update_tau is None:
            # Hard update
            with th.no_grad():
                for param, target_param in zip(
                    self.policy_net.parameters(),
                    self.target_policy_net.parameters()
                ):
                    target_param.data.copy_(param.data)
        else:
            self.target_policy_net.soft_update(
                self.policy_net, self.soft_update_tau, non_blocking)


class EnsenmbledDDPGPolicy(BasePolicy):
    # TODO (Juanwu): Ensembled DDPG policy
    pass


if __name__ == "__main__":
    # Unit Test Cases
    # =========================================
    policy = DDPGPolicy(
        observation_size=10,
        action_size=(5,),
        discrete_action=False,
        policy_net="mlp",
        policy_net_kwargs=dict(
            in_feature=10,
            hidden_size=64,
            out_feature=5,
            num_layers=2
        )
    )
    print(f"Policy Function: {policy}.")
