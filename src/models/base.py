"""
Abstract base classes for Neural Bayes Estimator components.

These base classes define the interface for modular architecture swapping,
allowing easy experimentation with different transforms, encoders,
aggregations, and estimators.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor


class BaseTransform(nn.Module, ABC):
    """
    Abstract base class for pre-transformations applied to trajectory data.

    Transforms are applied before encoding and can be used for:
    - Variance stabilization (e.g., log, cube root)
    - Normalization (e.g., standardization)
    - Scale transformations
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply transformation to trajectory data.

        Args:
            x: Input tensor of shape (batch_size, M, seq_len)
               where M is the number of replicates and seq_len is N+1 time steps.

        Returns:
            Transformed tensor of same shape (batch_size, M, seq_len)
        """
        pass


class BasePsiNetwork(nn.Module, ABC):
    """
    Abstract base class for the psi (ψ) network - trajectory encoder.

    The psi network encodes individual trajectories into a fixed-dimensional
    representation. It is applied independently to each of the M replicates.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode a batch of individual trajectories.

        Args:
            x: Input tensor of shape (batch_size, 1, seq_len)
               where seq_len is N+1 time steps.

        Returns:
            Encoded tensor of shape (batch_size, hidden_dim)
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension (hidden_dim) of the encoder."""
        pass


class BaseAggregation(nn.Module, ABC):
    """
    Abstract base class for permutation-invariant aggregation functions.

    The aggregation function combines the encoded representations of M
    replicates into a single fixed-dimensional summary. It must be
    permutation-invariant (order of replicates should not matter).
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Aggregate encoded representations over replicates.

        Args:
            x: Input tensor of shape (batch_size, M, hidden_dim)
               where M is the number of replicates.

        Returns:
            Aggregated tensor of shape (batch_size, hidden_dim)
        """
        pass


class BasePhiNetwork(nn.Module, ABC):
    """
    Abstract base class for the phi (φ) network - parameter estimator.

    The phi network maps the aggregated representation to the final
    parameter estimates. It outputs one value per parameter being estimated.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Map aggregated features to parameter estimates.

        Args:
            x: Input tensor of shape (batch_size, hidden_dim)

        Returns:
            Parameter estimates of shape (batch_size, num_params)
        """
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Return the number of parameters being estimated."""
        pass
