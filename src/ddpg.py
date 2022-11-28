# =============================================================================
# @file   ddpg.py
# @author Juanwu Lu
# @date   Nov-28-22
# =============================================================================
from __future__ import annotations

import math
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import torch as th
from torch import nn, optim
from torch import distributions as D
from torch.nn import functional as F
from src.types import OptFloat, OptInt, OptTensor, PathLike


# Ornstein Uhlenbeck Process
# ==========================
class OUProcess:
    def __init__(self,
                 size,
                 theta: float = 0.15,
                 mu: float = 0.,
                 sigma: float = 0.2,
                 dt: float = 1e-2,
                 x0: OptFloat = None,
                 sigma_min: OptFloat = None,
                 n_steps_annealing: int = 1000) -> None:
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


# DDPG Policy Modules
# ===================
def fanin_init(size: Union[int, th.Size],
               fanin: OptInt = None) -> th.Tensor:
    fanin = fanin or size[0]
    v = 1.0 / math.sqrt(fanin)
    return th.Tensor(size).uniform_(-v, v)


class PolicyNet(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_hidden_1: int = 400,
                 num_hidden_2: int = 300,
                 init_w: float = 3e-3) -> None:
        super().__init__()

        self.fc_1 = nn.Linear(in_features, num_hidden_1)
        self.fc_2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.fc_3 = nn.Linear(num_hidden_2, out_features)
        self.reset_parameters(init_w)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs = obs.float()
        obs = F.relu(self.fc_1(obs))
        obs = F.relu(self.fc_2(obs))
        acs = th.tanh(self.fc_3(obs))

        return acs

    def reset_parameters(self, init_w: float = 3e-3) -> None:
        self.fc_1.weight.data = fanin_init(self.fc_1.weight.data.size())
        self.fc_2.weight.data = fanin_init(self.fc_2.weight.data.size())
        self.fc_3.weight.data.uniform_(-init_w, init_w)

    def hard_update(self,
                    target: nn.Module,
                    non_blocking: bool = False) -> None:
        for param, tgt_param in zip(self.parameters(), target.parameters()):
            param.data.copy_(tgt_param.data, non_blocking)

    def soft_update(self,
                    target: nn.Module,
                    tau: float,
                    non_blocking: bool = False) -> None:
        # Apply exponential moving average (EMA) updates
        for param, tgt_param in zip(self.parameters(), target.parameters()):
            param.data.copy_(
                param.data * tau + tgt_param.data * (1.0 - tau),
                non_blocking=non_blocking
            )


class DDPGPolicy(nn.Module):

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 action_range: Optional[List[float]] = None,
                 discrete_action: bool = False,
                 device: th.device = th.device('cpu'),
                 soft_update_tau: OptFloat = None,
                 random_timesteps: OptFloat = 1000,
                 **kwargs) -> None:
        super().__init__()

        self.obs_size = observation_size
        self.act_size = action_size
        self.act_rng = action_range
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
            self.exploration = OUProcess(action_size)

    def forward(self, obs: th.Tensor, target: bool = False) -> D.Distribution:
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        if target:
            return self.target_policy_net.forward(obs)
        else:
            return self.policy_net.forward(obs)

    def get_action(self,
                   obs: th.Tensor,
                   explore: bool = True,
                   target: bool = False) -> th.Tensor:
        acs: th.Tensor = self.forward(obs, target)

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
            if self.act_rng:
                acs = acs.clamp(*self.act_rng[:2])

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


# DDPG Critic Modules
# ===================
class CriticNet(nn.Module):

    def __init__(self,
                 obs_in_features: int,
                 acs_in_features: int,
                 num_hidden_1: int = 400,
                 num_hidden_2: int = 300,
                 init_w: float = 3e-3) -> None:
        super().__init__()

        self.obs_in_features = obs_in_features
        self.acs_in_features = acs_in_features
        self.fc_1 = nn.Linear(obs_in_features, num_hidden_1)
        self.fc_2 = nn.Linear(num_hidden_1 + acs_in_features, num_hidden_2)
        self.fc_act = nn.Linear(acs_in_features, num_hidden_2)
        self.fc_3 = nn.Linear(num_hidden_2, 1)
        self.reset_parameters(init_w)

    def forward(self, obs: th.Tensor, acs: th.Tensor) -> th.Tensor:
        obs, acs = obs.float(), acs.float()
        obs = F.relu(self.fc_1(obs))
        q_val = F.relu(self.fc_2(th.cat([obs, acs], 1)))
        q_val = self.fc_3(q_val)

        return q_val

    def reset_parameters(self, init_w: float = 3e-3) -> None:
        self.fc_1.weight.data = fanin_init(self.fc_1.weight.data.size())
        self.fc_2.weight.data = fanin_init(self.fc_2.weight.data.size())
        self.fc_act.weight.data = fanin_init(self.fc_act.weight.data.size())
        self.fc_3.weight.data.uniform_(-init_w, init_w)

    def hard_update(self,
                    target: nn.Module,
                    non_blocking: bool = False) -> None:
        for param, tgt_param in zip(self.parameters(), target.parameters()):
            param.data.copy_(tgt_param.data, non_blocking)

    def soft_update(self,
                    target: nn.Module,
                    tau: float,
                    non_blocking: bool = False) -> None:
        # Apply exponential moving average (EMA) updates
        for param, tgt_param in zip(self.parameters(), target.parameters()):
            param.data.copy_(
                param.data * tau + tgt_param.data * (1.0 - tau),
                non_blocking=non_blocking
            )


class DDPGCritic(nn.Module):

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 device: th.device = th.device('cpu'),
                 discount: float = 0.99,
                 learning_rate: float = 1e-4,
                 soft_update_tau: OptFloat = None,
                 grad_clip: OptFloat = None,
                 huber_loss: bool = False) -> None:
        super().__init__()

        self.obs_size = observation_size
        self.act_size = action_size
        self.discount = discount
        self.learning_rate = learning_rate
        self.soft_update_tau = soft_update_tau
        self.grad_clip = grad_clip
        if huber_loss:
            self.loss = nn.HuberLoss()
        else:
            self.loss = nn.MSELoss()

        self.critic_net = CriticNet(observation_size, action_size).to(device)
        self.target_critic_net = deepcopy(self.critic_net).to(device)

        self.target_critic_net.hard_update(self.critic_net, False)

    def forward(self,
                obs: th.Tensor,
                acs: th.Tensor,
                target: bool = False) -> th.Tensor:
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


# DDPG Agent Module
# =================
class DDPGAgent:

    def __init__(self,
                 observation_size: OptInt = None,
                 critic_observation_size: OptInt = None,
                 policy_observation_size: OptInt = None,
                 action_size: OptInt = None,
                 action_range: Optional[List[float]] = None,
                 critic_action_size: OptInt = None,
                 policy_action_size: OptInt = None,
                 discrete_action: bool = False,
                 device: Optional[th.device] = None,
                 learning_rate: OptFloat = None,
                 policy_lr: OptFloat = None,
                 critic_lr: OptFloat = None,
                 discount: OptFloat = 0.99,
                 grad_clip: OptFloat = None,
                 policy_grad_clip: OptFloat = None,
                 critic_grad_clip: OptFloat = None,
                 soft_update_tau: OptFloat = 0.001,
                 **kwargs) -> None:
        super().__init__()

        self.device = device
        self.policy: DDPGPolicy = DDPGPolicy(
            observation_size=observation_size or policy_observation_size,
            action_size=action_size or policy_action_size,
            action_range=action_range,
            discrete_action=discrete_action,
            device=device,
            learning_rate=policy_lr or learning_rate,
            soft_update_tau=soft_update_tau,
        )
        self.policy_opt = optim.Adam(self.policy.parameters(),
                                     lr=policy_lr or learning_rate)
        self.policy_grad_clip = policy_grad_clip or grad_clip
        self.policy_lr_scheduler = optim.lr_scheduler.StepLR(
            self.policy_opt, step_size=30, gamma=0.3
        )

        self.critic: DDPGCritic = DDPGCritic(
            observation_size=observation_size or critic_observation_size,
            action_size=action_size or critic_action_size,
            device=device,
            discount=discount,
            learning_rate=critic_lr or learning_rate,
            soft_update_tau=soft_update_tau,
            grad_clip=grad_clip
        )
        self.critic_opt = optim.Adam(self.critic.parameters(),
                                     lr=critic_lr or learning_rate,
                                     weight_decay=1e-2)
        self.critic_grad_clip = critic_grad_clip or grad_clip
        self.critic_lr_scheduler = optim.lr_scheduler.StepLR(
            self.critic_opt, step_size=30, gamma=0.3
        )

        self.training_step = 0

    def reset_noise(self) -> None:
        self.policy.reset_noise()

    def get_action(self,
                   obs: Union[np.ndarray, th.Tensor],
                   explore: bool = True,
                   target: bool = False) -> th.Tensor:
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        return self.policy.get_action(obs, explore, target)

    def update_critic(self,
                      obs: th.Tensor,
                      acs: th.Tensor,
                      next_obs: th.Tensor,
                      next_acs: OptTensor,
                      rews: th.Tensor,
                      dones: th.Tensor) -> float:
        # Shape issue
        if rews.dim() == 1:
            rews = rews.view(-1, 1)
        if dones.dim() == 1:
            dones = dones.view(-1, 1)

        # Bellman error target
        if next_acs is None:
            next_acs = self.policy.get_action(next_obs,
                                              explore=False,
                                              target=True)
        target = rews + self.critic.discount * (1. - dones) * \
            self.critic.forward(next_obs, next_acs, target=True).detach()
        Q_vals = self.critic.forward(obs, acs, target=False)
        critic_loss = self.critic.loss(Q_vals, target.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.critic_grad_clip:
            th.nn.utils.clip_grad_value_(self.critic.parameters(),
                                         self.critic_grad_clip)
        self.critic_opt.step()

        return critic_loss.item()

    def update_policy(self,
                      obs: th.Tensor,
                      acs: OptTensor = None) -> float:
        if acs is None:
            acs = self.policy.get_action(obs, explore=False, target=False)

        policy_loss = -self.critic.forward(obs, acs).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        if self.policy_grad_clip:
            th.nn.utils.clip_grad_value_(self.policy.parameters(),
                                         self.policy_grad_clip)
        self.policy_opt.step()

        return policy_loss.item()

    def update_target(self, non_blocking: bool = False) -> None:
        self.policy.sync(non_blocking)
        self.critic.sync(non_blocking)

    def load(self, filepath: PathLike) -> None:
        state_dict = th.load(filepath, map_location=self.device)
        self.critic.load_state_dict(state_dict['critic_state_dict'])
        self.critic_opt.load_state_dict(state_dict['critic_opt_state_dict'])
        if self.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.load_state_dict(
                state_dict['critic_lr_scheduler_state_dict']
            )
        self.policy.load_state_dict('policy_state_dict')
        self.policy_opt.load_state_dict('policy_opt_state_dict')
        if self.policy_lr_scheduler is not None:
            self.policy_lr_scheduler.load_state_dict(
                state_dict['policy_lr_scheduler_state_dict']
            )

    def save(self, filepath: PathLike) -> None:
        state_dict = {
            "critic_state_dict": self.critic.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "critic_lr_scheduler_state_dict": (
                self.critic_lr_scheduler.state_dict()
                if self.critic_lr_scheduler is not None else None
            ),
            "policy_state_dict": self.policy.state_dict(),
            "policy_opt_state_dict": self.policy_opt.state_dict(),
            "policy_lr_scheduler_state_dict": (
                self.policy_lr_scheduler.state_dict()
                if self.policy_lr_scheduler is not None else None
            )
        }
        th.save(state_dict, filepath)

    def set_train(self):
        self.policy.train()
        self.critic.train()

    def set_eval(self):
        self.policy.eval()
        self.critic.eval()

    def step_lr_scheduler(self) -> None:
        if self.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.step()
        if self.policy_lr_scheduler is not None:
            self.policy_lr_scheduler.step()
