# =============================================================================
# @file   utils.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Neural network utilities."""
import inspect
from torch import Tensor
from typing import Any, Callable, Mapping, Sequence, Optional, Union


# Resolvers
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/resolver.py
# =========================================
def _normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def resolver(classes: Sequence[Any],
             class_dict: Mapping[str, Any],
             query: Union[str, Any],
             base_cls: Optional[Any],
             base_cls_repr: Optional[str],
             *args, **kwargs) -> Callable:

    if not isinstance(query, str):
        return query

    query_repr = _normalize_string(query)
    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls else ""
    base_cls_repr = _normalize_string(base_cls_repr)

    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    for cls in classes:
        cls_repr = _normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, "")]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f"Failed to resolve '{query:s}' among choices {choices}.")


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
def normalization_resolver(query: Union[str, Any] = "batch_norm",
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
