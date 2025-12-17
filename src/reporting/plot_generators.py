"""Plot generation utilities for model reports."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# Parameter names for SDE model
PARAM_NAMES = ["A_0", "a", "B_0", "b"]


def plot_predictions_vs_true_grid(
    predictions: np.ndarray,
    targets: np.ndarray,
    param_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Create scatter plots of predicted vs true for all parameters in a 2x2 grid.

    Each subplot shows:
    - Scatter plot of predictions vs true values
    - Perfect prediction line (red dashed)
    - Statistics box with MAE, MSE, R^2, Bias

    Args:
        predictions: Predicted parameters, shape (n_samples, num_params)
        targets: True parameters, shape (n_samples, num_params)
        param_names: Names for each parameter. Defaults to PARAM_NAMES.
        save_path: Optional path to save the figure
        figsize: Figure size. Default: (12, 10)

    Returns:
        matplotlib Figure object
    """
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

        # Scatter plot
        ax.scatter(true, pred, alpha=0.3, s=15, c="steelblue", edgecolors="none")

        # Perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        margin = (max_val - min_val) * 0.05
        ax.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            "r--",
            linewidth=2,
            label="Perfect",
        )

        # Compute metrics
        mae = np.abs(pred - true).mean()
        mse = ((pred - true) ** 2).mean()
        r2 = 1 - mse / np.var(true) if np.var(true) > 0 else 0
        bias = (pred - true).mean()

        # Add metrics to plot
        textstr = f"MAE: {mae:.4f}\nMSE: {mse:.6f}\nR²: {r2:.4f}\nBias: {bias:+.4f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )

        ax.set_xlabel(f"True {name}", fontsize=11)
        ax.set_ylabel(f"Predicted {name}", fontsize=11)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    # Hide unused subplots
    for ax in axes[num_params:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_error_distributions(
    predictions: np.ndarray,
    targets: np.ndarray,
    param_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Create error distribution histograms for all parameters.

    Each subplot shows:
    - Error histogram with density normalization
    - Gaussian fit overlay
    - Vertical line at zero (unbiased reference)
    - Mean error line (red dotted)

    Args:
        predictions: Predicted parameters, shape (n_samples, num_params)
        targets: True parameters, shape (n_samples, num_params)
        param_names: Names for each parameter. Defaults to PARAM_NAMES.
        save_path: Optional path to save the figure
        figsize: Figure size. Default: (12, 10)

    Returns:
        matplotlib Figure object
    """
    if param_names is None:
        param_names = PARAM_NAMES[: predictions.shape[1]]

    num_params = len(param_names)
    nrows = (num_params + 1) // 2
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes[:num_params], param_names)):
        errors = predictions[:, i] - targets[:, i]

        # Histogram
        n, bins, patches = ax.hist(
            errors,
            bins=50,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
        )

        # Gaussian fit overlay
        mu, sigma = errors.mean(), errors.std()
        x = np.linspace(errors.min(), errors.max(), 100)
        gaussian = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, gaussian, "r-", linewidth=2, label=f"N({mu:.4f}, {sigma:.4f}²)")

        # Vertical line at zero (ideal unbiased estimator)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

        # Mark mean error
        ax.axvline(
            x=mu, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"Mean"
        )

        ax.set_xlabel(f"{name} Error (Pred - True)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{name} Error Distribution", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[num_params:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_residuals_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    param_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Create residual analysis plots (residuals vs true values).

    Useful for detecting:
    - Heteroscedasticity (varying error variance)
    - Systematic bias patterns
    - Nonlinear estimation errors

    Each subplot shows:
    - Scatter of residuals vs true values
    - Zero line (red dashed)
    - Moving average trend line (orange)

    Args:
        predictions: Predicted parameters, shape (n_samples, num_params)
        targets: True parameters, shape (n_samples, num_params)
        param_names: Names for each parameter. Defaults to PARAM_NAMES.
        save_path: Optional path to save the figure
        figsize: Figure size. Default: (12, 10)

    Returns:
        matplotlib Figure object
    """
    if param_names is None:
        param_names = PARAM_NAMES[: predictions.shape[1]]

    num_params = len(param_names)
    nrows = (num_params + 1) // 2
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes[:num_params], param_names)):
        true = targets[:, i]
        residuals = predictions[:, i] - true

        # Scatter plot of residuals vs true values
        ax.scatter(true, residuals, alpha=0.3, s=10, c="steelblue", edgecolors="none")

        # Zero line
        ax.axhline(y=0, color="red", linestyle="--", linewidth=2)

        # Moving average trend line
        sorted_idx = np.argsort(true)
        true_sorted = true[sorted_idx]
        residuals_sorted = residuals[sorted_idx]

        # Simple moving average
        window_size = max(len(true) // 20, 10)
        if len(true) > window_size:
            # Cumulative sum approach for efficiency
            cumsum = np.cumsum(np.insert(residuals_sorted, 0, 0))
            smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
            x_smoothed = true_sorted[window_size // 2 : -window_size // 2 + 1]
            if len(x_smoothed) == len(smoothed):
                ax.plot(x_smoothed, smoothed, "orange", linewidth=2, label="Trend")

        ax.set_xlabel(f"True {name}", fontsize=10)
        ax.set_ylabel(f"Residual (Pred - True)", fontsize=10)
        ax.set_title(f"{name} Residuals vs True", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    # Hide unused subplots
    for ax in axes[num_params:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Create extended training curves visualization.

    Includes:
    - Loss curves (train/val)
    - Learning rate schedule
    - Per-parameter MAE over epochs (if available)

    Args:
        history: Dictionary with training history
            Expected keys: 'train_loss', 'val_loss', 'learning_rate'
            Optional: 'train_A_0_mae', 'val_A_0_mae', etc.
        save_path: Optional path to save the figure
        figsize: Figure size. Default: (14, 10)

    Returns:
        matplotlib Figure object
    """
    # Determine which data is available
    has_train_loss = "train_loss" in history
    has_val_loss = "val_loss" in history
    has_lr = "learning_rate" in history
    has_param_mae = any(k.endswith("_mae") for k in history.keys())

    if not has_train_loss and not has_val_loss:
        # No training history available
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No training history available",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    nrows = 2 if has_param_mae else 1
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # Determine epoch range
    n_epochs = 0
    if has_train_loss:
        n_epochs = len(history["train_loss"])
    elif has_val_loss:
        n_epochs = len(history["val_loss"])
    epochs = range(1, n_epochs + 1)

    # Loss curves
    ax = axes[0, 0]
    if has_train_loss:
        ax.plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    if has_val_loss:
        ax.plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[0, 1]
    if has_lr:
        lr_values = history["learning_rate"]
        lr_epochs = range(1, len(lr_values) + 1)
        ax.plot(lr_epochs, lr_values, "g-", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    else:
        ax.set_visible(False)

    # Per-parameter MAE (if available)
    if has_param_mae and nrows > 1:
        ax = axes[1, 0]
        for name in PARAM_NAMES:
            train_key = f"train_{name}_mae"
            val_key = f"val_{name}_mae"
            if train_key in history:
                mae_epochs = range(1, len(history[train_key]) + 1)
                ax.plot(
                    mae_epochs,
                    history[train_key],
                    "--",
                    label=f"{name} (train)",
                    alpha=0.7,
                )
            if val_key in history:
                mae_epochs = range(1, len(history[val_key]) + 1)
                ax.plot(
                    mae_epochs, history[val_key], "-", label=f"{name} (val)", linewidth=2
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.set_title("Per-Parameter MAE", fontweight="bold")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Per-parameter bias
        ax = axes[1, 1]
        has_bias = False
        for name in PARAM_NAMES:
            val_key = f"val_{name}_bias"
            if val_key in history:
                has_bias = True
                bias_epochs = range(1, len(history[val_key]) + 1)
                ax.plot(
                    bias_epochs, history[val_key], "-", label=f"{name}", linewidth=2
                )
        if has_bias:
            ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Bias")
            ax.set_title("Per-Parameter Bias", fontweight="bold")
            ax.legend(ncol=2, fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
