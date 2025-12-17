"""
Concrete aggregation implementations for combining replicate representations.

Aggregation functions must be permutation-invariant, meaning the output
should not depend on the order of the M replicates. This is a key property
of the DeepSets framework (Zaheer et al., 2017).
"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseAggregation


class MeanAggregation(BaseAggregation):
    """
    Elementwise mean aggregation over replicates.

    This is the default and most commonly used aggregation in the paper.
    It computes the average representation across all M replicates.

    Output: mean over dim=1 (the replicate dimension)
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Aggregate by computing elementwise mean.

        Args:
            x: Input tensor of shape (batch_size, M, hidden_dim)

        Returns:
            Aggregated tensor of shape (batch_size, hidden_dim)
        """
        return x.mean(dim=1)


class SumAggregation(BaseAggregation):
    """
    Elementwise sum aggregation over replicates.

    Sums the representations across all M replicates.
    Note: The magnitude scales with M, which may affect training.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Aggregate by computing elementwise sum.

        Args:
            x: Input tensor of shape (batch_size, M, hidden_dim)

        Returns:
            Aggregated tensor of shape (batch_size, hidden_dim)
        """
        return x.sum(dim=1)


class MaxAggregation(BaseAggregation):
    """
    Elementwise max aggregation over replicates.

    Takes the maximum value across all M replicates for each feature.
    Useful when the most extreme/prominent features are most informative.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Aggregate by computing elementwise max.

        Args:
            x: Input tensor of shape (batch_size, M, hidden_dim)

        Returns:
            Aggregated tensor of shape (batch_size, hidden_dim)
        """
        return x.max(dim=1)[0]


class MinAggregation(BaseAggregation):
    """
    Elementwise min aggregation over replicates.

    Takes the minimum value across all M replicates for each feature.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Aggregate by computing elementwise min.

        Args:
            x: Input tensor of shape (batch_size, M, hidden_dim)

        Returns:
            Aggregated tensor of shape (batch_size, hidden_dim)
        """
        return x.min(dim=1)[0]


class MeanMaxAggregation(BaseAggregation):
    """
    Combined mean and max aggregation.

    Concatenates the mean and max aggregations, doubling the output dimension.
    This can capture both average behavior and extreme values.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Aggregate by concatenating mean and max.

        Args:
            x: Input tensor of shape (batch_size, M, hidden_dim)

        Returns:
            Aggregated tensor of shape (batch_size, 2 * hidden_dim)
        """
        mean_agg = x.mean(dim=1)
        max_agg = x.max(dim=1)[0]
        return torch.cat([mean_agg, max_agg], dim=1)


class MeanStdAggregation(BaseAggregation):
    """
    Combined mean and standard deviation aggregation.

    Concatenates the mean and std aggregations, doubling the output dimension.
    Captures both central tendency and variability across replicates.

    Args:
        eps: Small constant for numerical stability in std. Default: 1e-8
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Aggregate by concatenating mean and std.

        Args:
            x: Input tensor of shape (batch_size, M, hidden_dim)

        Returns:
            Aggregated tensor of shape (batch_size, 2 * hidden_dim)
        """
        mean_agg = x.mean(dim=1)
        std_agg = x.std(dim=1) + self.eps
        return torch.cat([mean_agg, std_agg], dim=1)
