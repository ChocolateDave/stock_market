# =============================================================================
# @file   base_nn.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Base Neural Network Module."""
from torch import nn, Tensor


class BaseNN(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def hard_update(self, src: nn.Module, non_blocking: bool = False) -> None:
        for param, src_param in zip(self.parameters(), src.parameters()):
            param.data.copy_(src_param.data, non_blocking)

    def soft_update(self,
                    src: nn.Module,
                    tau: float,
                    non_blocking: bool = False) -> None:
        for param, src_param in zip(self.parameters(), src.parameters()):
            param.data.copy_(
                src_param.data * (1.0 - tau) + param.data * tau,
                non_blocking
            )
