# =============================================================================
# @file   maddpg_trainer.py
# @author Juanwu Lu
# @date   Oct-15-22
# =============================================================================
"""Trainer for Multi-agent Deep Deterministic Policy Gradient."""
from typing import Any, Sequence

from gym.core import Env
from src.agent.base_agent import BaseAgent
from src.memory.multi_agent_replay_buffer import MultiAgentReplayBuffer
from src.trainer.base_trainer import BaseTrainer


class MADDPGTrainer(BaseTrainer):

    def __init__(self,
                 env: Env,
                 agents: Sequence[BaseAgent],
                 max_size: int = 10000) -> None:
        super().__init__()

        self.env = env
        self.agents = agents
        self.replay_buffer = MultiAgentReplayBuffer(max_size=max_size)

    def train_one_epoch(self, epoch: int) -> Any:
        # TODO: Implement single epoch training

        # Collect data for training

        pass

    def exec_one_epoch(self, epoch: int = -1) -> Any:
        # TODO: Implement single epoch execution
        pass
