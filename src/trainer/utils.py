# =============================================================================
# @file   utils.py
# @author Juanwu Lu
# @date   Oct-24-22
# =============================================================================
"""Trainer utitilities."""
from __future__ import annotations
from typing import Any, Union

import src.trainer
from src.utils import resolver


# Trianer Resolver
# =========================================
def trainer_resolver(trainer_id: Union[str, Any] = 'DDPGTrainer',
                     *args, **kwargs) -> Any:
    base_cls = src.trainer.base_trainer.BaseTrainer
    base_cls_repr = "Trainer"
    trainers = [
        trainer for trainer in vars(src.trainer).values()
        if isinstance(trainer, type) and issubclass(trainer, base_cls)
    ]
    trainer_dict = {}
    return resolver(
        trainers, trainer_dict, trainer_id, base_cls,
        base_cls_repr, *args, **kwargs
    )
