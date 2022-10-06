# =============================================================================
# @file   base_trainer.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base Reinforcement Learning trainer class."""
from typing import Any, Optional, Sequence, Union

from src.agents.base_agent import BaseAgent


class Trainer:
    agents: Sequence[BaseAgent]

    def train_one_epoch(self, epoch: int) -> Any:
        raise NotImplementedError

    def exec_one_epoch(self, epoch: int = -1) -> Any:
        raise NotImplementedError

    def __init__(
        self,
        agents: Optional[Union[BaseAgent, Sequence[BaseAgent]]] = None,
    ) -> None:
        pass

    def train(self, execution: bool = False) -> Any:
        self._train()

    def _train(self) -> None:
        for agent in self.agents:
            agent.train()
