# =============================================================================
# @file   types.py
# @author Juanwu Lu
# @date   Nov-26-22
# =============================================================================
from __future__ import annotations

import os
from typing import Any, Dict, Optional, TypeVar, Union

from torch import Tensor

LOG = TypeVar(name='LOG', bound=Dict[str, Any])
OptInt = TypeVar(name='OptInt', bound=Optional[int])
OptFloat = TypeVar(name='OptFloat', bound=Optional[float])
OptTensor = TypeVar(name='OptTensor', bound=Optional[Tensor])
PathLike = TypeVar(name='PathLike', bound=Union[str, 'os.PathLike[str]'])
StateDict = TypeVar(name='StateDict', bound=Dict[str, Any])
