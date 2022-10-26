# =============================================================================
# @file   base_nn.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base Neural Network Module."""
from torch import Tensor, nn


class BaseNN(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def hard_update(self,
                    target: nn.Module,
                    non_blocking: bool = False) -> None:
        for param, tgt_param in zip(self.parameters(), target.parameters()):
            param.data.copy_(tgt_param.data, non_blocking)

    def soft_update(self,
                    target: nn.Module,
                    tau: float,
                    non_blocking: bool = False) -> None:
        # Apply exponential moving average (EMA) update.
        for param, tgt_param in zip(self.parameters(), target.parameters()):
            param.data.copy_(
                param.data * tau + tgt_param.data * (1.0 - tau),
                non_blocking
            )
