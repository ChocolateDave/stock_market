# =============================================================================
# @file   ddpg_nn.py
# @author Juanwu Lu
# @date   Oct-26-22
# =============================================================================
"""Neural Network Structures for Deep Deterministic Policy Gradients"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch as th
from src.nn.base_nn import BaseNN
from torch import Tensor
from torch.nn import Linear, BatchNorm1d
from torch.nn import functional as F


def fanin_init(size: Union[int, th.Size],
               fanin: Optional[int] = None) -> Tensor:
    fanin = fanin or size[0]
    v = 1.0 / math.sqrt(fanin)
    return th.Tensor(size).uniform_(-v, v)


class PolicyNet(BaseNN):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_hidden_1: int = 400,
                 num_hidden_2: int = 300,
                 init_w: float = 3e-3,
                 batchnorm: bool = False) -> None:
        super().__init__()
        self.batchnorm = batchnorm
        self.bn_1 = BatchNorm1d(in_features)
        self.fc_1 = Linear(in_features, num_hidden_1)
        self.bn_2 = BatchNorm1d(num_hidden_1)
        self.fc_2 = Linear(num_hidden_1, num_hidden_2)
        self.bn_3 = BatchNorm1d(num_hidden_2)
        self.fc_3 = Linear(num_hidden_2, out_features)
        self.reset_parameters(init_w)

    def forward(self, obs: Tensor) -> Tensor:
        obs = obs.float()
        if self.batchnorm:
            obs = self.bn_1(obs)
        obs = F.relu(self.fc_1(obs))
        if self.batchnorm:
            obs = self.bn_2(obs)
        obs = F.relu(self.fc_2(obs))
        if self.batchnorm:
            obs = self.bn_3(obs)
        acs = th.tanh(self.fc_3(obs))

        return acs

    def reset_parameters(self, init_w: float = 3e-3) -> None:
        self.bn_1.weight.data = fanin_init(self.bn_1.weight.data.size())
        self.fc_1.weight.data = fanin_init(self.fc_1.weight.data.size())
        self.bn_2.weight.data = fanin_init(self.bn_2.weight.data.size())
        self.fc_2.weight.data = fanin_init(self.fc_2.weight.data.size())
        self.bn_3.weight.data.uniform_(-init_w, init_w)
        self.fc_3.weight.data.uniform_(-init_w, init_w)


class CriticNet(BaseNN):

    def __init__(self,
                 obs_in_features: int,
                 acs_in_features: int,
                 num_hidden_1: int = 64,  # 400
                 num_hidden_2: int = 64,  # 300
                 init_w: float = 3e-3,
                 batchnorm: bool = False) -> None:
        super().__init__()

        self.batchnorm = batchnorm
        self.obs_in_features = obs_in_features
        self.acs_in_features = acs_in_features
        self.bn_1 = BatchNorm1d(obs_in_features)
        self.fc_1 = Linear(obs_in_features, num_hidden_1)
        self.bn_2 = BatchNorm1d(num_hidden_1)
        self.fc_2 = Linear(num_hidden_1, num_hidden_2)
        self.fc_act = Linear(acs_in_features, num_hidden_2)
        self.fc_3 = Linear(num_hidden_2, 1)
        self.reset_parameters(init_w)

    def forward(self, obs: Tensor, acs: Tensor) -> Tensor:
        obs, acs = obs.float(), acs.float()
        if self.batchnorm:
            obs = self.bn_1(obs)
        obs = F.relu(self.fc_1(obs))
        if self.batchnorm:
            obs = self.bn_2(obs)
        obs = self.fc_2(obs)
        acs = self.fc_act(acs)
        q_val = F.relu(obs + acs)
        q_val = self.fc_3(q_val)

        return q_val

    def reset_parameters(self, init_w: float = 3e-3) -> None:
        self.bn_1.weight.data = fanin_init(self.bn_1.weight.data.size())
        self.fc_1.weight.data = fanin_init(self.fc_1.weight.data.size())
        self.bn_2.weight.data = fanin_init(self.bn_2.weight.data.size())
        self.fc_2.weight.data = fanin_init(self.fc_2.weight.data.size())
        self.fc_act.weight.data = fanin_init(self.fc_act.weight.data.size())
        self.fc_3.weight.data.uniform_(-init_w, init_w)
