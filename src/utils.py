# =============================================================================
# @file   utils.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Utility functions for experiements."""
import inspect
import torch as th
from dataclasses import dataclass, field, Field
from numpy import ndarray
from torch import Tensor
from typing import (
    Any,
    Callable,
    ItemsView,
    Mapping,
    Sequence,
    Tuple,
    Optional,
    Union
)


@dataclass
class Pace:
    """A single step interaction transition data class."""

    observation: Tensor
    action: Tensor
    next_observation: Tensor
    reward: Tensor
    terminal: Tensor

    @property
    def _cls_fields(self) -> Tuple[Field, ...]:
        return field(self.__class__)

    def cpu(self) -> None:
        for _field in self._cls_fields:
            if isinstance(_field, Tensor):
                _field.cpu()

    def cuda(self) -> None:
        for _field in self._cls_fields:
            if isinstance(_field, Tensor):
                _field.cuda()

    def to(self,
           device: Union[str, int, th.device],
           non_blocking=False) -> None:
        for _field in self._cls_fields:
            if isinstance(_field, Tensor):
                _field.to(deivce=device, non_blocking=non_blocking)

    def pin_memory(self) -> None:
        for _field in self._cls_fields:
            if isinstance(_field, Tensor):
                _field.pin_memory()


# Metrics
# =========================================
class AverageMeter:
    """Average metric tracking meter class.

    Attributes:
        name: Name of the metric.
        count: Current total number of updates.
        val: Current value of the metric.
        sum: Current total value of the history metrics.
        avg: Current average value of the history metrics.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def __call__(self) -> float:
        return self.item()

    def __str__(self) -> str:
        fmt_str = '{name} {val:.4f} ({avg:.4f})'
        return fmt_str.format(**self.__dict__)

    def item(self) -> float:
        return self.avg

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0.0

    def update(self, value: float, n: int = 1) -> None:
        """Update meter value.

        Args:
            value: The floating-point value to track.
            n: Number of repeats.
        """
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterGroup(object):
    """Average meter group container.

    Attributes:
        meters: A mapping from string name to average meter class objects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def __getitem__(self, key: str) -> float:
        return self.meters[key].item()

    def __str__(self) -> str:
        return ', '.join(repr(meter) for meter in self.meters.values())

    def items(self) -> ItemsView[str, float]:
        for (key, meter) in self.meters.items():
            yield (key, meter.item())

    def reset(self) -> None:
        self.meters: Mapping[str, AverageMeter] = {}

    def update(self, dat: Mapping[str, Union[float, Sequence[float]]]) -> None:
        for (name, value) in dat.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter(name)
            if isinstance(value, Sequence):
                for val in value:
                    self.meters[name].update(val)
            else:
                self.meters[name].update(value)


# Common functions
# =========================================
def normalize(inputs: Union[ndarray, Tensor],
              mean: Union[ndarray, Tensor],
              std: Union[ndarray, Tensor],
              eps: float = 1e-6) -> Union[ndarray, Tensor]:
    return (inputs - mean) / (std + eps)


def unnormalize(inputs: Union[ndarray, Tensor],
                mean: Union[ndarray, Tensor],
                std: Union[ndarray, Tensor],
                eps: float = 1e-6) -> Union[ndarray, Tensor]:
    return inputs * (std + eps) + mean


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