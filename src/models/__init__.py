"""
Neural Bayes Estimator models module.

This module provides modular components for building Neural Bayes Estimators
following the DeepSets architecture from Sainsbury-Dale et al. (2024).

Components:
- Base classes: Abstract interfaces for transforms, encoders, aggregations, estimators
- Transforms: Pre-processing transformations (log, standardize, etc.)
- Encoders: Trajectory encoders (CNN, MLP, ResNet)
- Aggregations: Permutation-invariant aggregations (mean, max, sum)
- Estimators: Parameter estimation networks (MLP)
- DeepSetsNBE: Main model combining all components
"""

# Base classes
from .base import (
    BaseTransform,
    BasePsiNetwork,
    BaseAggregation,
    BasePhiNetwork,
)

# Transforms
from .transforms import (
    IdentityTransform,
    LogTransform,
    CubeRootTransform,
    PowerTransform,
    StandardizeTransform,
    MinMaxTransform,
    ClipTransform,
    ComposedTransform,
    DifferenceTransform,
    LogReturnTransform,
    RobustStandardizeTransform,
)

# Encoders
from .encoders import (
    CNNEncoder,
    MLPEncoder,
    ResNetEncoder,
)

# Aggregations
from .aggregations import (
    MeanAggregation,
    SumAggregation,
    MaxAggregation,
    MinAggregation,
    MeanMaxAggregation,
    MeanStdAggregation,
)

# Estimators
from .estimators import (
    MLPEstimator,
    LinearEstimator,
    BoundedMLPEstimator,
)

# Main model
from .deepsets import (
    DeepSetsNBE,
    build_default_nbe,
)

__all__ = [
    # Base classes
    "BaseTransform",
    "BasePsiNetwork",
    "BaseAggregation",
    "BasePhiNetwork",
    # Transforms
    "IdentityTransform",
    "LogTransform",
    "CubeRootTransform",
    "PowerTransform",
    "StandardizeTransform",
    "MinMaxTransform",
    "ClipTransform",
    "ComposedTransform",
    "DifferenceTransform",
    "LogReturnTransform",
    "RobustStandardizeTransform",
    # Encoders
    "CNNEncoder",
    "MLPEncoder",
    "ResNetEncoder",
    # Aggregations
    "MeanAggregation",
    "SumAggregation",
    "MaxAggregation",
    "MinAggregation",
    "MeanMaxAggregation",
    "MeanStdAggregation",
    # Estimators
    "MLPEstimator",
    "LinearEstimator",
    "BoundedMLPEstimator",
    # Main model
    "DeepSetsNBE",
    "build_default_nbe",
]
