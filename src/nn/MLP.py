# =============================================================================
# @file   MLP.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
"""Multi Layer Perceptron Neural Network"""
from torch import nn, Tensor
from typing import Any, Callable, Mapping, Sequence, Optional, Union

from src.nn.utils import activation_resolver, normalization_resolver


class MLP(nn.Module):
    """Multi Layer Perceptron neural network module.

    Args:
        hidden_list (Sequence[int] or int, optional): A sequence of input,
        hidden, and output feature dimensions such that `len(hidden_list) - 1`
        denotes the number of layers of the MLP (default: :obj:`None`).
        in_feature (int, optional): Input feature dimension, will override the
        :attr:`hidden_list` (default: :obj:`None`).
        hidden_feature (int, optional): Hidden layer dimension, will override
        the :attr:`hidden_list` (default: :obj:`None`).
        out_feature (int, optional): Output feature dimension, will override
        the :attr:`hidden_list` (default: :obj:`None`).
        num_layers (int, optional): Number of hidden layers, will override the
        :attr:`num_layers` (default: :obj:`None`).
        dropout (int or Sequence[int], optional): Dropout probabilities of each
        hidden layer embedding. If a sequence is given, set the dropout prob
        per hidden layer (default: :obj:`0.0`).
        act (str or Callable, optional): The non-linear activation function to
        use (default: :obj:`"relu"`).
        act_first (bool): If set to :obj:`True`, activation is applied before
        the normalization.
        act_kwargs (Mapping[str, Any], optional): Arguments passed to the
        respective activation function (default: :obj:`None`).
        norm (str or Callable, optional): The normalization function to use
        (default: :obj:`"layer_norm"`).
        norm_kwargs (Mapping[str, Any], optional): Arguments passed to the
        respective normalization function (default: :obj:`None`).
        direct_last (bool): If set to :obj:`False`, apply non-linear, norm,
        and dropout function to the last-layer output (default: :obj:`False`).
        bias (bool or Sequence[bool], optional): If set to :obj:`False`, the
        linear layers will not train on additive biases. If given a sequence,
        apply bias settings to each layer (default: :obj:`True`).
        **kwargs (optional): Additional keyword arguments for the MLP module.
    """

    def __init__(self,
                 hidden_list: Optional[Union[Sequence[int], int]] = None,
                 in_feature: Optional[int] = None,
                 hidden_feature: Optional[int] = None,
                 out_feature: Optional[int] = None,
                 num_layers: Optional[int] = None,
                 dropout: Optional[Union[float, Sequence[float]]] = 0.0,
                 act: Optional[Union[str, Callable]] = "relu",
                 act_first: bool = False,
                 act_kwargs: Optional[Mapping[str, Any]] = None,
                 norm: Optional[Union[str, Callable]] = "layer_norm",
                 norm_kwargs: Optional[Mapping[str, Any]] = None,
                 direct_output: bool = True,
                 bias: Union[bool, Sequence[bool]] = True,
                 **kwargs) -> None:
        super().__init__()

        if isinstance(hidden_list, int):
            in_feature = hidden_list

        if in_feature is not None:
            assert num_layers >= 1
            hidden_list = [hidden_feature] * (num_layers - 1)
            hidden_list = [in_feature] + hidden_list + [out_feature]

        assert isinstance(hidden_list, Sequence)
        assert len(hidden_list) > 1
        self.hidden_list = hidden_list

        if isinstance(dropout, float):
            dropout = [dropout] * (len(hidden_list) - 1)
            if direct_output:
                dropout[-1] = 0.0
        if len(dropout) != len(hidden_list) - 1:
            raise ValueError(
                f"Number of dropout probabilities given ({len(dropout)}) does "
                f"not match the number of layers ({len(hidden_feature) - 1})."
            )
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(hidden_list) - 1)
        if len(bias) != len(hidden_list) - 1:
            raise ValueError(
                f"Number of bias booleans given ({len(bias)}) does not match "
                f"the number of hidden layers ({len(hidden_feature) - 1})."
            )

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.direct_output = direct_output

        # Initialize Linear layers
        # =========================================
        self.lins: nn.ModuleList[nn.Linear] = nn.ModuleList()
        iterator = zip(hidden_list[:-1], hidden_list[1:], bias)
        for in_feature, out_feature, bias in iterator:
            self.lins.append(nn.Linear(in_feature, out_feature, bias))

        # Initialize Normalization layers
        # =========================================
        self.norms: Sequence[nn.Module] = nn.ModuleList()
        iterator = hidden_list[1:-1] if direct_output else hidden_list[1:]
        for hidden_size in iterator:
            if norm is None:
                norm_layer = nn.Identity()
            else:
                norm_layer = normalization_resolver(
                    norm, hidden_size, **(norm_kwargs or {})
                )
            self.norms.append(norm_layer)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.hidden_list)[1:-1]})"

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()

        for i, (linear, norm) in enumerate(zip(self.lins, self.norms)):
            x = linear(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = nn.functional.dropout(
                x, p=self.dropout[i], training=self.training
            )

        # Output layer
        # =========================================
        if self.direct_output:
            output = self.lins[-1](x)
            output = nn.functional.dropout(
                output, p=self.dropout[-1], training=self.training
            )

        return output

    def reset_parameters(self) -> None:
        for lin in self.lins:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    @property
    def in_feature(self) -> int:
        return self.hidden_list[0]

    @property
    def out_feature(self) -> int:
        return self.hidden_list[-1]


if __name__ == "__main__":
    # Unit Test
    # =========================================
    import torch as th

    try:
        model = MLP(
            in_feature=10,
            hidden_feature=64,
            out_feature=30,
            num_layers=2,
        )
        print(f"Model: {model}")

        test_x = th.randn(size=(64, 10)).float()
        test_y = model(test_x)
        print(f"Success!\nInput {test_x},\nOutput {test_y}.")
    except Exception:
        import pdb
        pdb.set_trace()
