"""
Concrete transform implementations for trajectory pre-processing.

These transforms are applied to trajectory data before encoding,
following recommendations from Sainsbury-Dale et al. (2024):
- Log transform for data with varying magnitudes
- Cube root for variance stabilization
- Standardization for normalization
"""

from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseTransform


class IdentityTransform(BaseTransform):
    """
    Identity transformation - returns input unchanged.

    This is the default transform when no pre-processing is needed.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class LogTransform(BaseTransform):
    """
    Logarithmic transformation: y = log(x + eps).

    Useful for data with highly varying magnitudes, such as
    max-stable processes on Fréchet margins (Section 3.3 of the paper).

    Args:
        eps: Small constant to avoid log(0). Default: 1e-8
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x + self.eps)


class CubeRootTransform(BaseTransform):
    """
    Cube root transformation: y = sign(x) * |x|^(1/3).

    Variance-stabilizing transformation, as used in Section 3.4 of
    Sainsbury-Dale et al. (2024) for the spatial conditional extremes model.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)


class PowerTransform(BaseTransform):
    """
    Generalized power transformation: y = sign(x) * |x|^p.

    Generalizes cube root (p=1/3) and square root (p=1/2).

    Args:
        power: The exponent to apply. Default: 0.5 (square root)
    """

    def __init__(self, power: float = 0.5):
        super().__init__()
        self.power = power

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.abs(x).pow(self.power)


class StandardizeTransform(BaseTransform):
    """
    Z-score standardization per trajectory.

    Each trajectory is independently standardized to have zero mean
    and unit variance: y = (x - mean(x)) / std(x).

    Args:
        eps: Small constant for numerical stability. Default: 1e-8
        dim: Dimension along which to compute statistics. Default: -1 (seq_len)
    """

    def __init__(self, eps: float = 1e-8, dim: int = -1):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=self.dim, keepdim=True)
        std = x.std(dim=self.dim, keepdim=True)
        return (x - mean) / (std + self.eps)


class MinMaxTransform(BaseTransform):
    """
    Min-max normalization per trajectory to [0, 1] range.

    y = (x - min(x)) / (max(x) - min(x))

    Args:
        eps: Small constant for numerical stability. Default: 1e-8
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Compute min and max along the sequence dimension
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + self.eps)


class ClipTransform(BaseTransform):
    """
    Clip values to a specified range.

    Args:
        min_val: Minimum value (or None for no lower bound)
        max_val: Maximum value (or None for no upper bound)
    """

    def __init__(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=self.min_val, max=self.max_val)


class DifferenceTransform(BaseTransform):
    """
    Compute finite differences: y[t] = x[t+1] - x[t].

    Useful for analyzing trajectory increments in SDE modeling.
    For order > 1, applies differencing recursively.

    Note: Output sequence length is reduced by `order` elements.
          Output shape: (batch_size, M, seq_len - order)

    Args:
        order: Order of differencing (1 = first differences, 2 = second, etc.)
               Default: 1

    Example:
        >>> transform = DifferenceTransform(order=1)
        >>> x = torch.tensor([[[1.0, 2.0, 4.0, 7.0]]])  # shape (1, 1, 4)
        >>> transform(x)  # Returns [[[1.0, 2.0, 3.0]]] shape (1, 1, 3)
    """

    def __init__(self, order: int = 1):
        super().__init__()
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.order = order

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.order):
            x = x[..., 1:] - x[..., :-1]
        return x


class LogReturnTransform(BaseTransform):
    """
    Compute log returns: y[t] = log(x[t+1] / x[t]).

    Standard transformation in stochastic process modeling, converting
    multiplicative dynamics to additive form.

    Note: Output sequence length is reduced by 1.
          Output shape: (batch_size, M, seq_len - 1)

    Args:
        eps: Small constant for numerical stability when x[t] is near zero.
             Default: 1e-8

    Example:
        >>> transform = LogReturnTransform()
        >>> x = torch.tensor([[[1.0, 2.0, 4.0]]])  # shape (1, 1, 3)
        >>> transform(x)  # Returns [[[log(2), log(2)]]] shape (1, 1, 2)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        ratios = x[..., 1:] / (x[..., :-1] + self.eps)
        return torch.log(ratios + self.eps)


class RobustStandardizeTransform(BaseTransform):
    """
    Robust standardization using median and IQR per trajectory.

    Each trajectory is standardized using robust statistics:
        y = (x - median(x)) / IQR(x)

    where IQR = Q3 - Q1 (interquartile range). More robust to outliers
    than mean/std standardization, particularly useful for extreme value data.

    Args:
        eps: Small constant for numerical stability. Default: 1e-8
        dim: Dimension along which to compute statistics. Default: -1 (seq_len)

    Example:
        >>> transform = RobustStandardizeTransform()
        >>> x = torch.randn(32, 100, 501)
        >>> y = transform(x)  # Shape preserved: (32, 100, 501)
    """

    def __init__(self, eps: float = 1e-8, dim: int = -1):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        median = x.median(dim=self.dim, keepdim=True)[0]
        q1 = torch.quantile(x, 0.25, dim=self.dim, keepdim=True)
        q3 = torch.quantile(x, 0.75, dim=self.dim, keepdim=True)
        iqr = q3 - q1
        return (x - median) / (iqr + self.eps)


class ComposedTransform(BaseTransform):
    """
    Compose multiple transforms sequentially.

    Transforms are applied in order: first transform first.

    Args:
        transforms: List of transforms to apply in sequence.

    Example:
        >>> transform = ComposedTransform([
        ...     ClipTransform(min_val=0.01),
        ...     LogTransform(),
        ...     StandardizeTransform()
        ... ])
    """

    def __init__(self, transforms: List[BaseTransform]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: Tensor) -> Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x
