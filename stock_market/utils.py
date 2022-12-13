# =============================================================================
# @file   utils.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
from __future__ import annotations

from typing import Any, Dict, ItemsView, List, Tuple, Union

import numpy as np
from gymnasium.spaces import Box, Discrete, Space
from gymnasium.spaces import Tuple as TupleSpace
from pettingzoo import AECEnv, ParallelEnv
from torch import Tensor


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
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, dat: Dict[str, Union[float, List[float]]]) -> None:
        for (name, value) in dat.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter(name)
            if isinstance(value, List):
                for val in value:
                    self.meters[name].update(val)
            else:
                self.meters[name].update(value)


# Tensor helper functions
# =========================================
def normalize(inputs: Union[np.ndarray, Tensor],
              mean: Union[np.ndarray, Tensor],
              std: Union[np.ndarray, Tensor],
              eps: float = 1e-6) -> Union[np.ndarray, Tensor]:
    return (inputs - mean) / (std + eps)


def unnormalize(inputs: Union[np.ndarray, Tensor],
                mean: Union[np.ndarray, Tensor],
                std: Union[np.ndarray, Tensor],
                eps: float = 1e-6) -> Union[np.ndarray, Tensor]:
    return inputs * (std + eps) + mean


def get_agent_dims(env: Union[AECEnv, ParallelEnv],
                   discrete_truncate: int = 3) -> Dict[str, Tuple[int, int]]:
    assert isinstance(env, (AECEnv, ParallelEnv)), TypeError(
        f'Expect multi-agent environment, but got {type(env):s}'
    )
    _ = env.reset()

    agent_dims = {}
    for agent_id in env.agents:
        if len(env.observation_space(agent_id).shape) > 2:
            raise NotImplementedError('Image observation not supported!')
        observation_size = env.observation_space(agent_id).shape[0]
        action_space = env.action_space(agent_id)
        if isinstance(action_space, TupleSpace):
            action_size = 0
            for sub_space in action_space:
                if isinstance(sub_space, Box):
                    action_size += np.prod(sub_space.shape)
                elif isinstance(sub_space, Discrete):
                    if sub_space.n > discrete_truncate:
                        # NOTE: use categorical variable if large space
                        action_size += 1
                    else:
                        action_size += sub_space.n
                else:
                    raise TypeError(
                        f'Unsupported action space: {action_space}!'
                    )
        else:
            if isinstance(action_space, Box):
                action_size = np.prod(action_space.shape)
            elif isinstance(action_space, Discrete):
                if action_space.n > discrete_truncate:
                    action_size = 1
                else:
                    action_size = action_space.n
            else:
                raise TypeError(f'Unsupported action space: {action_space}!')
        agent_dims[agent_id] = [observation_size, action_size]

    return agent_dims


def process_step_ac(data: Tensor,
                    action_space: Space,
                    discrete_truncate: int = 3) -> Any:
    if isinstance(action_space, Discrete):
        if action_space.n > discrete_truncate:
            # large discrete action space
            ac = action_space.start + int(action_space.n * data.item())
        else:
            # otherwise
            ac = data.argmax(-1).item()
        return ac
    elif isinstance(action_space, Box):
        ac = data.view(-1).detach().cpu().numpy()
        ac = np.exp(np.arctanh(np.clip(ac, -0.999999, 0.999999))) + 1.0
        return ac
    elif isinstance(action_space, TupleSpace):
        idx: int = 0
        ac = []
        for sub_space in action_space:
            if isinstance(sub_space, Box):
                ac.append(process_step_ac(
                    data[:, idx:idx+np.prod(sub_space.shape)],
                    sub_space
                ))
                idx += np.prod(sub_space.shape)
            if isinstance(sub_space, Discrete):
                if sub_space.n > discrete_truncate:
                    ac.append(process_step_ac(
                        data[:, idx:idx+1],
                        sub_space
                    ))
                else:
                    ac.append(process_step_ac(
                        data[:, idx:idx+sub_space.n],
                        sub_space
                    ))
        return tuple([process_step_ac(data[:, i:i+1], space)
                      for i, space in enumerate(action_space)])
    else:
        raise TypeError(f'Unsupported action space {action_space}.')


def process_sample_ac(data: Union[int, np.ndarray, Tuple],
                      action_space: Space,
                      discrete_truncate: int = 3) -> np.ndarray:
    if isinstance(action_space, Discrete):
        # Categorical variables
        if action_space.n > discrete_truncate:
            # large discrete action space
            ac = (data - action_space.start) / action_space.n
        else:
            ac = cat_to_one_hot(data, action_space.n)
        return ac
    elif isinstance(action_space, Box):
        ac = np.tanh(np.log(data - 1.0))
        return ac
    elif isinstance(action_space, TupleSpace):
        output = []
        for ac, ac_space in zip(data, action_space):
            output.append(process_sample_ac(ac, ac_space))
        output = np.hstack(output)[None, :]
        return output
    else:
        raise TypeError(f'Unsupported action space {action_space}.')


def cat_to_one_hot(data: Union[int, np.ndarray],
                   num_classes: int = -1) -> np.ndarray:
    if num_classes == -1:
        num_classes = int(max(data) + 1)

    if isinstance(data, (int, np.int16, np.int32, np.int64)):
        return np.eye(num_classes)[data]
    elif isinstance(data, np.ndarray):
        raise NotImplementedError(
            'Conversion from categorical to one hot array is not implemented.'
        )
    else:
        raise TypeError(f'Unsupproted data type {type(data)}.')
