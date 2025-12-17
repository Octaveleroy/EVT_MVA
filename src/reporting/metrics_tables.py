"""Metrics and configuration table formatting utilities."""

from typing import Any, Dict, List, Optional

# Parameter names for SDE model
PARAM_NAMES = ["A_0", "a", "B_0", "b"]


def format_config_table(args: Dict[str, Any]) -> List[List[str]]:
    """
    Format checkpoint args into a config table.

    Args:
        args: Dictionary of training arguments from checkpoint

    Returns:
        List of [parameter, value] rows
    """
    config_keys = [
        ("train_size", "Training Size (steps/epoch)"),
        ("val_size", "Validation Size"),
        ("M", "Max M (replicates)"),
        ("min_M", "Min M (replicates)"),
        ("N", "Time Steps"),
        ("Delta_t", "Time Step Size"),
        ("X_0", "Initial Condition"),
        ("epochs", "Max Epochs"),
        ("batch_size", "Batch Size"),
        ("lr", "Learning Rate"),
        ("weight_decay", "Weight Decay"),
        ("loss", "Loss Function"),
        ("patience", "Early Stopping Patience"),
        ("seed", "Random Seed"),
    ]

    rows = []
    for key, display_name in config_keys:
        if key in args:
            value = args[key]
            if isinstance(value, float):
                if value < 0.001:
                    value = f"{value:.2e}"
                else:
                    value = f"{value:.6g}"
            rows.append([display_name, str(value)])

    return rows


def format_architecture_table(
    args: Dict[str, Any], total_params: int
) -> List[List[str]]:
    """
    Format model architecture into a table.

    Args:
        args: Dictionary of training arguments from checkpoint
        total_params: Total number of model parameters

    Returns:
        List of [component, configuration] rows
    """
    rows = [
        ["Encoder Type", args.get("encoder", "cnn").upper()],
        ["Transform", args.get("transform", "identity")],
        ["Aggregation", args.get("aggregation", "mean")],
        ["Hidden Dimension", str(args.get("hidden_dim", 128))],
    ]

    # Encoder channels
    channels = args.get("channels")
    if channels:
        rows.append(["Encoder Channels", channels])
    else:
        rows.append(["Encoder Channels", "32, 64, 128 (default)"])

    # Phi hidden layers
    phi_layers = args.get("phi_layers")
    if phi_layers:
        rows.append(["Phi Hidden Layers", phi_layers])
    else:
        rows.append(["Phi Hidden Layers", "256, 128, 64 (default)"])

    rows.append(["Total Parameters", f"{total_params:,}"])

    return rows


def format_metrics_table(
    metrics: Dict[str, float],
    param_names: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Format validation metrics into a table.

    Args:
        metrics: Dictionary of computed metrics
        param_names: Parameter names (defaults to PARAM_NAMES)

    Returns:
        List of rows: [header, param1_row, param2_row, ..., total_row]

    Structure:
        | Parameter | MAE | MSE | Bias | Rel. Error |
    """
    if param_names is None:
        param_names = PARAM_NAMES

    header = ["Parameter", "MAE", "MSE", "Bias", "Rel. Error"]
    rows = [header]

    for name in param_names:
        row = [
            name,
            f"{metrics.get(f'{name}_mae', 0):.6f}",
            f"{metrics.get(f'{name}_mse', 0):.8f}",
            f"{metrics.get(f'{name}_bias', 0):+.6f}",
            f"{metrics.get(f'{name}_rel_error', 0):.4f}",
        ]
        rows.append(row)

    # Add totals row
    rows.append(
        [
            "TOTAL",
            f"{metrics.get('total_mae', 0):.6f}",
            f"{metrics.get('total_mse', 0):.8f}",
            "-",
            "-",
        ]
    )

    return rows


def format_priors_table() -> List[List[str]]:
    """
    Format the prior distributions used for training.

    Returns:
        List of [parameter, prior distribution] rows
    """
    return [
        ["A_0", "Uniform(0.0, 1.0)"],
        ["a", "Uniform(0.0, 2.0)"],
        ["B_0", "Uniform(0.01, 0.5)"],
        ["b", "Uniform(0.0, 0.5)"],
    ]
