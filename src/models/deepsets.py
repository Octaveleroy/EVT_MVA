"""
DeepSets Neural Bayes Estimator for SDE parameter inference.

This module implements the DeepSets architecture from Zaheer et al. (2017)
for Neural Bayes Estimation as described in Sainsbury-Dale et al. (2024).

The architecture processes M independent trajectory replicates through:
1. (Optional) Transform: Pre-process data (e.g., log, standardize)
2. Psi (ψ): Encode each trajectory independently
3. Aggregation (a): Combine encodings (permutation-invariant)
4. Phi (φ): Map to parameter estimates
"""

from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseTransform, BasePsiNetwork, BaseAggregation, BasePhiNetwork
from .transforms import IdentityTransform
from .aggregations import MeanAggregation
from .encoders import CNNEncoder
from .estimators import MLPEstimator


class DeepSetsNBE(nn.Module):
    """
    DeepSets Neural Bayes Estimator.

    Implements the DeepSets framework for parameter estimation from
    replicated trajectory data. The architecture is:

        X → Transform → ψ(each trajectory) → Aggregate → φ → θ̂

    All components (transform, psi, aggregation, phi) are modular and
    can be swapped for different implementations.

    Args:
        psi: Encoder network for individual trajectories.
        aggregation: Permutation-invariant aggregation function.
        phi: Parameter estimation network.
        transform: Optional pre-transformation for trajectory data.
                   Defaults to IdentityTransform (no transformation).

    Example:
        >>> from src.models import DeepSetsNBE, CNNEncoder, MeanAggregation, MLPEstimator
        >>> from src.models import StandardizeTransform
        >>>
        >>> # Create components
        >>> psi = CNNEncoder(input_length=501, hidden_dim=128)
        >>> agg = MeanAggregation()
        >>> phi = MLPEstimator(input_dim=128, num_params=4)
        >>> transform = StandardizeTransform()
        >>>
        >>> # Build model
        >>> model = DeepSetsNBE(psi, agg, phi, transform)
        >>>
        >>> # Forward pass
        >>> trajectories = torch.randn(32, 100, 501)  # (batch, M, N+1)
        >>> params = model(trajectories)  # (batch, 4)
    """

    def __init__(
        self,
        psi: BasePsiNetwork,
        aggregation: BaseAggregation,
        phi: BasePhiNetwork,
        transform: Optional[BaseTransform] = None,
    ):
        super().__init__()

        self.transform = transform if transform is not None else IdentityTransform()
        self.psi = psi
        self.aggregation = aggregation
        self.phi = phi

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the DeepSets architecture.

        Args:
            x: Input tensor of shape (batch_size, M, seq_len)
               where M is the number of replicates and seq_len is N+1 time steps.

        Returns:
            Parameter estimates of shape (batch_size, num_params)
        """
        batch_size, M, _ = x.shape

        # 1. Apply transform (may change seq_len for difference/logreturns)
        x = self.transform(x)
        seq_len = x.shape[-1]  # Get seq_len AFTER transform

        # 2. Reshape for psi: (batch_size * M, 1, seq_len)
        x = x.view(batch_size * M, 1, seq_len)

        # 3. Encode each trajectory independently
        x = self.psi(x)  # (batch_size * M, hidden_dim)

        # 4. Reshape back: (batch_size, M, hidden_dim)
        hidden_dim = x.shape[-1]
        x = x.view(batch_size, M, hidden_dim)

        # 5. Aggregate over M (permutation-invariant)
        x = self.aggregation(x)  # (batch_size, agg_output_dim)

        # 6. Map to parameters
        x = self.phi(x)  # (batch_size, num_params)

        return x

    @property
    def num_params(self) -> int:
        """Return the number of parameters being estimated."""
        return self.phi.num_params


# def build_default_nbe(
#     input_length: int,
#     num_params: int = 4,
#     hidden_dim: int = 128,
#     encoder_type: str = "cnn",
#     transform_type: str = "identity",
#     aggregation_type: str = "mean",
#     encoder_channels: Optional[List[int]] = None,
#     phi_hidden_layers: Optional[List[int]] = None,
# ) -> DeepSetsNBE:
#     """
#     Factory function to build a DeepSetsNBE with default configurations.

#     This is a convenience function for quickly creating a model with
#     common configurations.

#     Args:
#         input_length: Length of input trajectories (N+1 time steps)
#         num_params: Number of parameters to estimate. Default: 4
#         hidden_dim: Hidden dimension for encoder. Default: 128
#         encoder_type: Type of encoder ('cnn', 'mlp', 'resnet'). Default: 'cnn'
#         transform_type: Type of transform ('identity', 'log', 'standardize',
#                         'cuberoot', 'difference', 'logreturns',
#                         'robust_standardize'). Default: 'identity'
#         aggregation_type: Type of aggregation ('mean', 'sum', 'max',
#                           'meanmax', 'meanstd'). Default: 'mean'
#         encoder_channels: Channel sizes for CNN encoder. Default: [32, 64, 128]
#         phi_hidden_layers: Hidden layer sizes for MLP estimator. Default: [256, 128, 64]

#     Returns:
#         Configured DeepSetsNBE model

#     Example:
#         >>> model = build_default_nbe(input_length=501, hidden_dim=128)
#         >>> # Custom smaller model (~50k params)
#         >>> model = build_default_nbe(
#         ...     input_length=501, hidden_dim=64,
#         ...     encoder_channels=[16, 32, 64], phi_hidden_layers=[192, 64]
#         ... )
#     """
#     from .transforms import (
#         IdentityTransform,
#         LogTransform,
#         StandardizeTransform,
#         CubeRootTransform,
#         DifferenceTransform,
#         LogReturnTransform,
#         RobustStandardizeTransform,
#     )
#     from .encoders import CNNEncoder, MLPEncoder, ResNetEncoder
#     from .aggregations import (
#         MeanAggregation,
#         SumAggregation,
#         MaxAggregation,
#         MeanMaxAggregation,
#         MeanStdAggregation,
#     )
#     from .estimators import MLPEstimator

#     # Build transform
#     transform_map = {
#         "identity": IdentityTransform,
#         "log": LogTransform,
#         "standardize": StandardizeTransform,
#         "cuberoot": CubeRootTransform,
#         "difference": DifferenceTransform,
#         "logreturns": LogReturnTransform,
#         "robust_standardize": RobustStandardizeTransform,
#     }
#     if transform_type not in transform_map:
#         raise ValueError(f"Unknown transform type: {transform_type}")
#     transform = transform_map[transform_type]()

#     # Compute effective input length for length-changing transforms
#     effective_input_length = input_length
#     if transform_type in ("difference", "logreturns"):
#         effective_input_length = input_length - 1

#     # Build encoder
#     encoder_map = {
#         "cnn": CNNEncoder,
#         "mlp": MLPEncoder,
#         "resnet": ResNetEncoder,
#     }
#     if encoder_type not in encoder_map:
#         raise ValueError(f"Unknown encoder type: {encoder_type}")

#     encoder_kwargs = {"input_length": effective_input_length, "hidden_dim": hidden_dim}
#     if encoder_type == "cnn" and encoder_channels is not None:
#         encoder_kwargs["channels"] = encoder_channels
#     psi = encoder_map[encoder_type](**encoder_kwargs)

#     # Build aggregation
#     aggregation_map = {
#         "mean": MeanAggregation,
#         "sum": SumAggregation,
#         "max": MaxAggregation,
#         "meanmax": MeanMaxAggregation,
#         "meanstd": MeanStdAggregation,
#     }
#     if aggregation_type not in aggregation_map:
#         raise ValueError(f"Unknown aggregation type: {aggregation_type}")
#     aggregation = aggregation_map[aggregation_type]()

#     # Adjust phi input dimension for combined aggregations
#     phi_input_dim = hidden_dim
#     if aggregation_type in ["meanmax", "meanstd"]:
#         phi_input_dim = hidden_dim * 2

#     # Build estimator
#     estimator_kwargs = {"input_dim": phi_input_dim, "num_params": num_params}
#     if phi_hidden_layers is not None:
#         estimator_kwargs["hidden_layers"] = phi_hidden_layers
#     phi = MLPEstimator(**estimator_kwargs)

#     return DeepSetsNBE(psi, aggregation, phi, transform)
def build_default_nbe(
    input_length: int,
    num_params: int = 4,
    hidden_dim: int = 128,
    encoder_type: str = "cnn",
    transform_type: str = "identity",
    aggregation_type: str = "mean",
    encoder_channels: Optional[List[int]] = None,
    phi_hidden_layers: Optional[List[int]] = None,
    **kwargs,  # Ajout de kwargs pour capturer les params RNN (et futurs autres)
) -> DeepSetsNBE:
    """
    Factory function to build a DeepSetsNBE with default configurations.

    This is a convenience function for quickly creating a model with
    common configurations.

    Args:
        input_length: Length of input trajectories (N+1 time steps)
        num_params: Number of parameters to estimate. Default: 4
        hidden_dim: Hidden dimension for encoder. Default: 128
        encoder_type: Type of encoder ('cnn', 'mlp', 'resnet', 'rnn'). Default: 'cnn'
        transform_type: Type of transform ('identity', 'log', 'standardize',
                        'cuberoot', 'difference', 'logreturns',
                        'robust_standardize'). Default: 'identity'
        aggregation_type: Type of aggregation ('mean', 'sum', 'max',
                          'meanmax', 'meanstd'). Default: 'mean'
        encoder_channels: Channel sizes for CNN encoder. Default: [32, 64, 128]
        phi_hidden_layers: Hidden layer sizes for MLP estimator. Default: [256, 128, 64]
        **kwargs: Additional arguments for specific encoders (e.g., rnn_type, bidirectional)

    Returns:
        Configured DeepSetsNBE model
    """
    from .transforms import (
        IdentityTransform,
        LogTransform,
        StandardizeTransform,
        CubeRootTransform,
        DifferenceTransform,
        LogReturnTransform,
        RobustStandardizeTransform,
    )
    # AJOUT: Import du RNNEncoder
    from .encoders import CNNEncoder, MLPEncoder, ResNetEncoder, RNNEncoder
    from .aggregations import (
        MeanAggregation,
        SumAggregation,
        MaxAggregation,
        MeanMaxAggregation,
        MeanStdAggregation,
    )
    from .estimators import MLPEstimator

    # Build transform
    transform_map = {
        "identity": IdentityTransform,
        "log": LogTransform,
        "standardize": StandardizeTransform,
        "cuberoot": CubeRootTransform,
        "difference": DifferenceTransform,
        "logreturns": LogReturnTransform,
        "robust_standardize": RobustStandardizeTransform,
    }
    if transform_type not in transform_map:
        raise ValueError(f"Unknown transform type: {transform_type}")
    transform = transform_map[transform_type]()

    # Compute effective input length for length-changing transforms
    effective_input_length = input_length
    if transform_type in ("difference", "logreturns"):
        effective_input_length = input_length - 1

    # Build encoder
    encoder_map = {
        "cnn": CNNEncoder,
        "mlp": MLPEncoder,
        "resnet": ResNetEncoder,
        "rnn": RNNEncoder,  # AJOUT
    }
    if encoder_type not in encoder_map:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Base kwargs for all encoders
    encoder_kwargs = {"input_length": effective_input_length, "hidden_dim": hidden_dim}

    # Specific logic per encoder type
    if encoder_type == "cnn" and encoder_channels is not None:
        encoder_kwargs["channels"] = encoder_channels
    
    # AJOUT: Logique spécifique pour le RNN
    elif encoder_type == "rnn":
        encoder_kwargs["rnn_type"] = kwargs.get("rnn_type", "LSTM")
        encoder_kwargs["num_layers"] = kwargs.get("num_layers", 1)
        encoder_kwargs["bidirectional"] = kwargs.get("bidirectional", False)

    # Initialize the encoder
    psi = encoder_map[encoder_type](**encoder_kwargs)

    # Build aggregation
    aggregation_map = {
        "mean": MeanAggregation,
        "sum": SumAggregation,
        "max": MaxAggregation,
        "meanmax": MeanMaxAggregation,
        "meanstd": MeanStdAggregation,
    }
    if aggregation_type not in aggregation_map:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")
    aggregation = aggregation_map[aggregation_type]()

    # Adjust phi input dimension for combined aggregations
    phi_input_dim = hidden_dim
    if aggregation_type in ["meanmax", "meanstd"]:
        phi_input_dim = hidden_dim * 2

    # Build estimator
    estimator_kwargs = {"input_dim": phi_input_dim, "num_params": num_params}
    if phi_hidden_layers is not None:
        estimator_kwargs["hidden_layers"] = phi_hidden_layers
    phi = MLPEstimator(**estimator_kwargs)

    return DeepSetsNBE(psi, aggregation, phi, transform)