# =============================================================================
# @file   iq_loss.py
# @author Maverick Zhang
# @date   Nov-17-22
# =============================================================================
import torch as th
from torch import Tensor
import torch.nn.functional as F

from typing import Any, Mapping, Optional, Union, Tuple

from src.agent.ddpg_agent import DDPGAgent

def iq_loss(agent: DDPGAgent,
            gamma: float, 
            obs: Tensor,
            acs: Tensor,
            next_obs: Tensor,
            rews: Tensor,
            dones: Tensor,
            is_expert: Tensor,
            offline: bool =  True,
            alpha: float = 1e-2) -> Tuple[Tensor, Mapping[str, Any]]: #TODO: tune alpha
    if offline:
        next_v = agent.get_V(next_obs, explore=False, target=False) # They say no target for offline
    else:
        next_v = agent.get_V(next_obs, explore=False, target=True)
    current_v = agent.get_V(obs, explore=False, target=False)
    q_vals = agent.critic.forward(obs, acs, target=False)
    loss_dict = {}

    # TODO: keep track of v0

    q_targets = (1. - dones) * gamma * next_v.detach()
    softq_loss = -(q_vals - q_targets)[is_expert].mean()
    loss_dict['softq_loss'] = softq_loss.item()

    value_loss = None
    if offline:
        # offline sampling strategy
        value_loss = (current_v - q_targets)[is_expert].mean()
    else:
        value_loss = (current_v - q_targets).mean()

    loss_dict['value_loss'] = value_loss.item()
    loss = softq_loss + value_loss

    chi2_loss = None
    if offline:
        softq = q_vals - q_targets
        chi2_loss = 1. / (4. * alpha) * (softq ** 2)[is_expert].mean() # alpha is basically the l2 coeff
        loss += chi2_loss
    else:
        softq = q_vals - q_targets
        chi2_loss = 1. / (4. * alpha) * (softq ** 2).mean()
        loss += chi2_loss
    loss_dict['chi2_loss'] = chi2_loss.item()

    return loss, loss_dict