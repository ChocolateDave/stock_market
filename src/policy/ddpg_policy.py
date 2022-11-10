# =============================================================================
# @file   ddpg_policy.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Deep Deterministic Policy module."""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Optional

import numpy as np
import torch as th
from src.nn.ddpg_nn import PolicyNet
from src.policy.base_policy import BasePolicy
from torch import Tensor, nn
from torch.distributions import Distribution


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, theta=0.15, mu=0., sigma=0.2, dt=1e-2, x0=None, sigma_min=None, n_steps_annealing=1000):
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * \
            self.dt + self.current_sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
    
    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


# Ornstein-Uhlenbeck Noise for continous random exploration
# https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# =========================================================
class OUNoise:
    def __init__(self,
                 action_size: int,
                 mu: float = 0.0,
                 scale: float = 0.1,
                 theta: float = 0.15,
                 sigma: float = 0.2,
                 dt: float = 1e-2, # 1e-2
                 final_anneal_scale: Optional[float] = 0.02,
                 scale_timesteps: Optional[float] = 10000) -> None: 
        super().__init__()

        self.action_size = action_size
        self.mu = mu
        self.scale = scale
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.final_anneal_scale = final_anneal_scale
        self.scale_timesteps = scale_timesteps

        self.reset()

    def sample(self) -> np.ndarray:

        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + \
            math.sqrt(self.dt) * self.sigma * \
                np.random.normal(size=self.action_size)
        self.state = x + dx
        noise = self.state * self.scale * self.scheduled_scale()
        return noise

    def reset(self) -> None:
        self.state = np.ones(self.action_size) * self.mu
        self.i = 0

    def scheduled_scale(self) -> float:
        if self.final_anneal_scale:
            if self.i < self.scale_timesteps:
                p = self.i / self.scale_timesteps
                scheduled_scale = self.final_anneal_scale * p + (1. - p)
                self.i += 1
                return scheduled_scale
            else:
                self.i += 1
                return self.final_anneal_scale
        else:
            return 1.
        


class DDPGPolicy(BasePolicy, nn.Module):

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 discrete_action: bool = False,
                 device: th.device = th.device('cpu'),
                 learning_rate: float = 1e-4,
                 soft_update_tau: Optional[float] = None,
                 optimizer_kwargs: Optional[Mapping[str, Any]] = None,
                 random_timesteps: Optional[float] = 1000,
                 **kwargs) -> None:
        super().__init__()

        self.obs_size = observation_size
        self.act_size = action_size
        self.soft_update_tau = soft_update_tau

        self.policy_net = PolicyNet(observation_size, action_size).to(device)
        self.target_policy_net = deepcopy(self.policy_net).to(device)

        self.target_policy_net.hard_update(self.policy_net, False)

        self.random_timesteps = random_timesteps
        self.random_steps_elapsed = 0

        self.delta_eps = 1. / 50000
        self.eps = 1.0

        # Exploration
        self.discrete_action = discrete_action
        if discrete_action:
            self.exploration = 0.3  # epsilon-greedy initial value
        else:
            self.exploration = OrnsteinUhlenbeckProcess(action_size)

    def forward(self, obs: Tensor, target: bool = False) -> Distribution:
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        if target:
            return self.target_policy_net.forward(obs)
        else:
            return self.policy_net.forward(obs)

    def get_action(self,
                   obs: Tensor,
                   explore: bool = True,
                   target: bool = False) -> Tensor:
        acs: Tensor = self.forward(obs, target)

        if self.discrete_action:
            if explore:
                # Random sample from discrete action space with gumbel noise
                acs = nn.functional.gumbel_softmax(acs, hard=True)
            else:
                acs = (acs == acs.max(1, keepdim=True)[0]).float()

        else:
            if explore:
                # Explore continous action space with OUNoise
                acs += th.from_numpy(max(self.eps, 0) * 
                    self.exploration.sample()).to(acs.device)
                self.eps -= self.delta_eps
            acs = acs.clamp(min=-1.0, max=1.0)

        return acs

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
            self.target_policy_net.hard_update(self.policy_net,
                                               non_blocking=non_blocking)
        else:
            self.target_policy_net.soft_update(self.policy_net,
                                               self.soft_update_tau,
                                               non_blocking=non_blocking)


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
