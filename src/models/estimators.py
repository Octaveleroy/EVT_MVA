"""
Concrete phi network implementations for parameter estimation.

The phi network maps the aggregated representation to the final
parameter estimates. It typically consists of a multi-layer perceptron.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from .base import BasePhiNetwork


class MLPEstimator(BasePhiNetwork):
    """
    Multi-Layer Perceptron for parameter estimation.

    Architecture:
        Linear → ReLU → Dropout → ... → Linear

    Maps the aggregated representation to parameter estimates.

    Args:
        input_dim: Input dimension (hidden_dim from aggregation)
        num_params: Number of parameters to estimate. Default: 4 for SDE params
        hidden_layers: List of hidden layer sizes. Default: [256, 128, 64]
        dropout: Dropout probability. Default: 0.1
    """

    def __init__(
        self,
        input_dim: int,
        num_params: int = 4,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._num_params = num_params

        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        layers = []
        in_features = input_dim

        # Build hidden layers
        for out_features in hidden_layers:
            layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_features = out_features

        # Final layer to num_params (no activation - unconstrained outputs)
        layers.append(nn.Linear(in_features, num_params))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Map aggregated features to parameter estimates.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Parameter estimates of shape (batch_size, num_params)
        """
        return self.mlp(x)

    @property
    def num_params(self) -> int:
        return self._num_params


class LinearEstimator(BasePhiNetwork):
    """
    Simple linear estimator (single layer, no hidden layers).

    Useful as a baseline or when the aggregated features are already
    highly informative.

    Args:
        input_dim: Input dimension (hidden_dim from aggregation)
        num_params: Number of parameters to estimate. Default: 4
    """

    def __init__(self, input_dim: int, num_params: int = 4):
        super().__init__()
        self._num_params = num_params
        self.linear = nn.Linear(input_dim, num_params)

    def forward(self, x: Tensor) -> Tensor:
        """
        Map aggregated features to parameter estimates.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Parameter estimates of shape (batch_size, num_params)
        """
        return self.linear(x)

    @property
    def num_params(self) -> int:
        return self._num_params


class BoundedMLPEstimator(BasePhiNetwork):
    """
    MLP estimator with bounded outputs using sigmoid activation.

    Constrains each parameter estimate to lie within specified bounds.
    Useful when parameters have known valid ranges.

    Args:
        input_dim: Input dimension (hidden_dim from aggregation)
        num_params: Number of parameters to estimate. Default: 4
        bounds: List of (low, high) tuples for each parameter.
                If None, outputs are in [0, 1].
        hidden_layers: List of hidden layer sizes. Default: [256, 128, 64]
        dropout: Dropout probability. Default: 0.1
    """

    def __init__(
        self,
        input_dim: int,
        num_params: int = 4,
        bounds: Optional[List[tuple]] = None,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._num_params = num_params

        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        # Default bounds: [0, 1] for all parameters
        if bounds is None:
            bounds = [(0.0, 1.0)] * num_params
        assert len(bounds) == num_params, "Must provide bounds for each parameter"

        # Register bounds as buffers (not parameters)
        lows = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        highs = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)

        layers = []
        in_features = input_dim

        # Build hidden layers
        for out_features in hidden_layers:
            layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_features = out_features

        # Final layer to num_params
        layers.append(nn.Linear(in_features, num_params))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Map aggregated features to bounded parameter estimates.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Parameter estimates of shape (batch_size, num_params),
            constrained to the specified bounds.
        """
        x = self.mlp(x)
        # Apply sigmoid and scale to bounds
        x = torch.sigmoid(x)
        x = self.lows + x * (self.highs - self.lows)
        return x

    @property
    def num_params(self) -> int:
        return self._num_params
