"""
Training module for Neural Bayes Estimators.

Provides training infrastructure including:
- NBETrainer: Main trainer class with TensorBoard logging
- EarlyStopping: Early stopping callback
- Metrics computation and checkpointing utilities
- Diagnostic plotting functions
"""

from .trainer import NBETrainer
from .utils import (
    compute_metrics,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    plot_predictions_vs_true,
    plot_training_history,
    PARAM_NAMES,
)

__all__ = [
    "NBETrainer",
    "compute_metrics",
    "EarlyStopping",
    "save_checkpoint",
    "load_checkpoint",
    "plot_predictions_vs_true",
    "plot_training_history",
    "PARAM_NAMES",
]
