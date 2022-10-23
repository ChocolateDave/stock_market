# =============================================================================
# @file   utils.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Neural network utilities."""
from typing import Any, Callable, Union

from src.nn.base_nn import BaseNN
from src.utils import resolver
import torch
from torch import Tensor, distributed


# Activation resolver
# =========================================
def _swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def activation_resolver(query: Union[str, Any] = "relu",
                        *args, **kwargs) -> Callable:
    import torch
    base_cls = torch.nn.Module
    base_cls_repr = "Act"
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [_swish]
    act_dict = {}
    return resolver(
        acts, act_dict, query, base_cls, base_cls_repr, *args, **kwargs
    )


# Normalization resolver
# =========================================
def normalization_resolver(query: Union[str, Any] = "layer_norm",
                           *args, **kwargs) -> Callable:
    import torch
    base_cls = torch.nn.Module
    base_cls_repr = "Norm"
    norms = [
        norm for norm in vars(torch.nn.modules.normalization).values()
        if isinstance(norm, type) and issubclass(norm, base_cls)
    ]
    norm_dict = {}
    return resolver(
        norms, norm_dict, query, base_cls, base_cls_repr, *args, **kwargs
    )


# Network Resolver
# =========================================
def network_resolver(query: Union[str, Any] = "mlp",
                     *args, **kwargs) -> Callable:
    import src.nn
    base_cls = src.nn.BaseNN
    base_cls_repr = "NN"
    networks = [
        nn for nn in vars(src.nn).values()
        if isinstance(nn, type) and issubclass(nn, base_cls)
    ]
    networks_dict = {}
    return resolver(
        networks, networks_dict, query, base_cls,
        base_cls_repr, *args, **kwargs
    )


# Gradient averaging function
# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
# =============================================================================
def average_gradients(model: BaseNN) -> None:
    size = float(distributed.get_world_size())
    for param in model.parameters():
        distributed.all_reduce(
            param.grad.data, op=distributed.reduce_op.SUM, group=0)
        param.grad.data /= size

device = None

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()