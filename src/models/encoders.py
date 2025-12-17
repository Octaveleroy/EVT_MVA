"""
Concrete encoder (psi network) implementations for trajectory encoding.

The encoder processes individual trajectories into fixed-dimensional
representations. Multiple architectures are provided:
- CNNEncoder: 1D convolutional network (recommended for time series)
- MLPEncoder: Multi-layer perceptron
"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import BasePsiNetwork


class CNNEncoder(BasePsiNetwork):
    """
    1D Convolutional Neural Network encoder for trajectories.

    Architecture:
        Conv1d → BatchNorm → ReLU (repeated)
        AdaptiveAvgPool1d → Flatten

    This architecture is recommended for time series data as it can
    capture local patterns in the trajectory at multiple scales.

    Args:
        input_length: Length of input sequence (N+1 time steps)
        hidden_dim: Output dimension of the encoder
        channels: List of channel sizes for conv layers. Default: [32, 64, 128]
    """

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 128,
        channels: list = None,
    ):
        super().__init__()
        self._output_dim = hidden_dim

        if channels is None:
            channels = [32, 64, 128]

        layers = []
        in_channels = 1

        # Build convolutional layers
        for i, out_channels in enumerate(channels):
            # Use larger kernel for first layer, smaller for subsequent
            kernel_size = 7 if i == 0 else 5
            padding = kernel_size // 2

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        # Final conv to hidden_dim
        layers.extend(
            [
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        )

        # Global average pooling to get fixed-size output
        layers.append(nn.AdaptiveAvgPool1d(1))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode trajectories.

        Args:
            x: Input tensor of shape (batch_size, 1, seq_len)

        Returns:
            Encoded tensor of shape (batch_size, hidden_dim)
        """
        x = self.conv_layers(x)  # (batch, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim)
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim


class MLPEncoder(BasePsiNetwork):
    """
    Multi-Layer Perceptron encoder for trajectories.

    Architecture:
        Flatten → Linear → ReLU → Linear → ReLU → ... → Linear

    A simpler alternative to CNN that treats the trajectory as a flat vector.
    May work well when combined with summary statistics or for shorter sequences.

    Args:
        input_length: Length of input sequence (N+1 time steps)
        hidden_dim: Output dimension of the encoder
        hidden_layers: List of hidden layer sizes. Default: [256, 128]
    """

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 128,
        hidden_layers: list = None,
    ):
        super().__init__()
        self._output_dim = hidden_dim
        self.input_length = input_length

        if hidden_layers is None:
            hidden_layers = [256, 128]

        layers = []
        in_features = input_length

        # Build hidden layers
        for out_features in hidden_layers:
            layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                ]
            )
            in_features = out_features

        # Final layer to hidden_dim
        layers.append(nn.Linear(in_features, hidden_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode trajectories.

        Args:
            x: Input tensor of shape (batch_size, 1, seq_len)

        Returns:
            Encoded tensor of shape (batch_size, hidden_dim)
        """
        x = x.squeeze(1)  # (batch, seq_len)
        x = self.mlp(x)  # (batch, hidden_dim)
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim


class ResidualBlock(nn.Module):
    """Residual block for ResNetEncoder."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ResNetEncoder(BasePsiNetwork):
    """
    ResNet-style 1D encoder with residual connections.

    Deeper architecture with skip connections for learning complex patterns.

    Args:
        input_length: Length of input sequence (N+1 time steps)
        hidden_dim: Output dimension of the encoder
        base_channels: Number of channels after initial conv. Default: 64
        num_blocks: Number of residual blocks. Default: 3
    """

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 128,
        base_channels: int = 64,
        num_blocks: int = 3,
    ):
        super().__init__()
        self._output_dim = hidden_dim

        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*blocks)

        # Output projection
        self.conv_out = nn.Sequential(
            nn.Conv1d(base_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode trajectories.

        Args:
            x: Input tensor of shape (batch_size, 1, seq_len)

        Returns:
            Encoded tensor of shape (batch_size, hidden_dim)
        """
        x = self.conv_in(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        x = x.squeeze(-1)
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim
    

class RNNEncoder(BasePsiNetwork):
    """
    Recurrent Neural Network (LSTM/GRU) encoder for trajectories.

    Suitable for time-series data where temporal dependencies and sequential
    order are critical. It processes the sequence step-by-step to produce
    a final summary embedding.

    Args:
        input_length: Length of input sequence (N+1 time steps).
        hidden_dim: Output dimension of the encoder.
        rnn_type: Type of RNN cell ('LSTM' or 'GRU'). Default: 'LSTM'.
        num_layers: Number of stacked RNN layers. Default: 1.
        bidirectional: If True, processes sequence in both directions.
                       Note: internal hidden size will be hidden_dim // 2.
    """

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 128,
        rnn_type: str = "LSTM",
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self._output_dim = hidden_dim
        self.input_length = input_length
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.upper()

        # Input to RNN is 1 because we have univariate time series
        input_size = 1
        
        # If bidirectional, we split the hidden dim between the two directions
        # to maintain the requested output size
        if bidirectional:
            assert hidden_dim % 2 == 0, "hidden_dim must be even for bidirectional RNN"
            self.rnn_hidden_dim = hidden_dim // 2
        else:
            self.rnn_hidden_dim = hidden_dim

        # Select RNN class
        if self.rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif self.rnn_type == "GRU":
            rnn_cls = nn.GRU
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}. Use 'LSTM' or 'GRU'.")

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=self.rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode trajectories using RNN.

        Args:
            x: Input tensor of shape (batch_size, 1, seq_len)
               Note: PyTorch RNN expects (batch, seq_len, features),
               so we permute inside.

        Returns:
            Encoded tensor of shape (batch_size, hidden_dim)
        """
        # Permute from (batch, 1, seq_len) -> (batch, seq_len, 1)
        # to match PyTorch RNN input requirements
        x = x.permute(0, 2, 1)

        # Run RNN
        # out shape: (batch, seq_len, num_directions * hidden_size)
        # hidden shape (LSTM): ((num_layers*dir, batch, hidden), (c_n...))
        # hidden shape (GRU): (num_layers*dir, batch, hidden)
        if self.rnn_type == "LSTM":
            _, (hidden, _) = self.rnn(x)
        else:
            _, hidden = self.rnn(x)

        # Extract the last layer's hidden state
        # hidden is (num_layers * num_directions, batch, hidden_size)
        
        if self.bidirectional:
            # Separate forward and backward states of the last layer
            # The last two entries in dimension 0 correspond to the last layer
            # (one forward, one backward)
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            # Concatenate to get shape (batch, hidden_dim)
            out = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            # Take the last layer's state
            # Shape: (batch, hidden_dim)
            out = hidden[-1, :, :]

        return out

    @property
    def output_dim(self) -> int:
        return self._output_dim
