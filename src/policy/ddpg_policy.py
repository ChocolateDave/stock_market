# =============================================================================
# @file   ddpg_policy.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Deep Deterministic Policy module."""
import itertools
import torch as th
from numpy import ndarray
from torch import nn, Tensor
from torch.distributions import Distribution, MultivariateNormal
from typing import Any, Mapping, Optional, Union

from src.nn.base_nn import BaseNN
from src.nn.utils import network_resolver
from src.policy.base_policy import BasePolicy


class DDPGPolicy(BasePolicy, nn.Module):

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 policy_net: Optional[Union[str, BaseNN]] = "mlp",
                 policy_net_kwargs: Optional[Mapping[str, Any]] = None,
                 device: th.device = th.device('cpu'),
                 learning_rate: float = 1e-4,
                 baseline: bool = False,
                 **kwargs) -> None:
        super().__init__()

        self.obs_size = observation_size
        self.act_size = action_size

        if isinstance(policy_net, BaseNN):
            self.policy_net = policy_net.to(device)
        else:
            self.policy_net = network_resolver(
                policy_net, **(policy_net_kwargs or {})
            ).to(device)

        self.policy_logstd = nn.Parameter(
            th.zeros(size=self.act_size, dtype=th.float32, device=device)
        ).to(device)

        self.opt = th.optim.Adam(
            itertools.chain(
                self.policy_net.parameters(), [self.policy_logstd]
            ),
            lr=learning_rate
        )

    def forward(self, obs: Tensor) -> Distribution:
        batch_loc = self.policy_net(obs)
        scale_tril = th.diag(th.exp(self.policy_logstd))
        batch_scale_tril = scale_tril.repeat(batch_loc.shape[0], 1, 1)
        action_dist = MultivariateNormal(
            loc=batch_loc, scale_tril=batch_scale_tril
        )
        return action_dist

    def get_action(self, obs: Tensor) -> ndarray:
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        with th.no_grad():
            distribution = self.forward(obs)
            actions = distribution.sample().cpu().numpy()

        return actions

    def update(self) -> None:
        pass


class EnsenmbledDDPGPolicy(BasePolicy):
    # TODO (Juanwu): Ensembled DDPG policy
    pass


if __name__ == "__main__":
    # Unit Test Cases
    # =========================================
    policy = DDPGPolicy(
        observation_size=10,
        action_size=(5,),
        policy_net="mlp",
        policy_net_kwargs=dict(
            in_feature=10,
            hidden_size=64,
            out_feature=5,
            num_layers=2
        )
    )
    print(f"Policy Function: {policy}.")
