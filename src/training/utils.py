"""
Training utilities for Neural Bayes Estimators.

Includes:
- Metrics computation
- Checkpointing (save/load)
- Early stopping
- Diagnostic plotting
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


# Parameter names for SDE model
PARAM_NAMES = ["A_0", "a", "B_0", "b"]


def compute_metrics(
    predictions: Tensor,
    targets: Tensor,
    param_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute per-parameter and aggregate metrics.

    Args:
        predictions: Predicted parameters, shape (batch_size, num_params)
        targets: True parameters, shape (batch_size, num_params)
        param_names: Names for each parameter. Defaults to PARAM_NAMES.

    Returns:
        Dictionary of metrics including:
        - {param}_mae: Mean absolute error for each parameter
        - {param}_mse: Mean squared error for each parameter
        - {param}_bias: Bias (mean error) for each parameter
        - {param}_rel_error: Relative error for each parameter
        - total_mae: Overall MAE
        - total_mse: Overall MSE
    """
    if param_names is None:
        param_names = PARAM_NAMES[: predictions.shape[1]]

    metrics = {}

    for i, name in enumerate(param_names):
        pred_i = predictions[:, i]
        target_i = targets[:, i]

        error = pred_i - target_i
        abs_error = error.abs()

        metrics[f"{name}_mae"] = abs_error.mean().item()
        metrics[f"{name}_mse"] = (error**2).mean().item()
        metrics[f"{name}_bias"] = error.mean().item()

        # Relative error (avoid division by zero)
        rel_error = abs_error / (target_i.abs() + 1e-8)
        metrics[f"{name}_rel_error"] = rel_error.mean().item()

    # Aggregate metrics
    total_error = predictions - targets
    metrics["total_mae"] = total_error.abs().mean().item()
    metrics["total_mse"] = (total_error**2).mean().item()

    return metrics


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.

    Monitors a metric (typically validation loss) and stops training
    if it doesn't improve for a specified number of epochs.

    Args:
        patience: Number of epochs to wait for improvement. Default: 10
        min_delta: Minimum change to qualify as improvement. Default: 1e-4
        mode: 'min' for metrics to minimize (loss), 'max' for metrics to
              maximize (accuracy). Default: 'min'

    Example:
        >>> early_stopping = EarlyStopping(patience=10)
        >>> for epoch in range(100):
        ...     val_loss = validate(model)
        ...     if early_stopping(val_loss):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value to monitor

        Returns:
            True if training should stop, False otherwise
        """
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.should_stop = False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    **kwargs,
):
    """
    Save a training checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save the checkpoint
        scheduler: Optional learning rate scheduler to save
        **kwargs: Additional items to save in the checkpoint
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint.update(kwargs)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
) -> Dict:
    """
    Load a training checkpoint.

    Args:
        path: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load the checkpoint to

    Returns:
        Dictionary containing checkpoint data (epoch, loss, etc.)
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def plot_predictions_vs_true(
    predictions: np.ndarray,
    targets: np.ndarray,
    param_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 10),
):
    """
    Create scatter plots of predicted vs true parameters.

    Args:
        predictions: Predicted parameters, shape (n_samples, num_params)
        targets: True parameters, shape (n_samples, num_params)
        param_names: Names for each parameter. Defaults to PARAM_NAMES.
        save_path: Optional path to save the figure
        figsize: Figure size. Default: (12, 10)

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    if param_names is None:
        param_names = PARAM_NAMES[: predictions.shape[1]]

    num_params = len(param_names)
    nrows = (num_params + 1) // 2
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes[:num_params], param_names)):
        pred = predictions[:, i]
        true = targets[:, i]

        ax.scatter(true, pred, alpha=0.3, s=10, label="Predictions")

        # Perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect")

        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name}: MAE={np.abs(pred - true).mean():.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[num_params:]:
        ax.set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 4),
):
    """
    Plot training history (loss curves).

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        save_path: Optional path to save the figure
        figsize: Figure size. Default: (12, 4)

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    ax = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate plot (if available)
    ax = axes[1]
    if "learning_rate" in history:
        ax.plot(epochs, history["learning_rate"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
