# =============================================================================
# @file   types.py
# @author Juanwu Lu
# @date   Nov-26-22
# =============================================================================
from __future__ import annotations

import os
from typing import Any, Dict, Optional, TypeVar, Union


LOG = TypeVar(name='LOG', bound=Dict[str, Any])
OptInt = TypeVar(name='OptInt', bound=Optional[int])
OptFloat = TypeVar(name='OptFloat', bound=Optional[float])
PathLike = TypeVar(name='PathLike', bound=Union[str, 'os.PathLike[str]'])
