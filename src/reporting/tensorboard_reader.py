"""TensorBoard log extraction utilities."""

from pathlib import Path
from typing import Dict, List, Union


def read_tensorboard_from_subdirs(
    log_dir: Union[str, Path],
) -> Dict[str, List[float]]:
    """
    Read TensorBoard logs from subdirectory structure.

    The project uses a structure like:
        runs/nbe_cnn_M100_.../
            Loss_epoch_train/events.out.tfevents...
            Loss_epoch_val/events.out.tfevents...
            Params_A_0_mae_train/events.out.tfevents...

    Args:
        log_dir: Path to TensorBoard run directory

    Returns:
        Dictionary mapping metric names to lists of values per epoch
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("Warning: tensorboard not installed, cannot read training history")
        return {}

    log_dir = Path(log_dir)
    if not log_dir.exists():
        return {}

    history = {}

    # Iterate over subdirectories (each metric has its own subdir)
    for subdir in log_dir.iterdir():
        if subdir.is_dir():
            event_files = list(subdir.glob("events.out.tfevents.*"))
            if event_files:
                try:
                    ea = EventAccumulator(str(subdir))
                    ea.Reload()

                    tags = ea.Tags().get("scalars", [])
                    for tag in tags:
                        events = ea.Scalars(tag)
                        values = [e.value for e in events]

                        # Use subdirectory name as key
                        key = subdir.name
                        history[key] = values
                except Exception as e:
                    print(f"Warning: Could not read {subdir}: {e}")

    # Also check root directory for main event file (e.g., LearningRate)
    root_events = list(log_dir.glob("events.out.tfevents.*"))
    if root_events:
        try:
            ea = EventAccumulator(str(log_dir))
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            for tag in tags:
                events = ea.Scalars(tag)
                history[tag] = [e.value for e in events]
        except Exception as e:
            print(f"Warning: Could not read root events: {e}")

    return history


def normalize_history_keys(history: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Normalize TensorBoard history keys to standard format.

    Converts keys like:
        - 'Loss_epoch_train' -> 'train_loss'
        - 'Loss_epoch_val' -> 'val_loss'
        - 'Params_A_0_mae_train' -> 'train_A_0_mae'
        - 'LearningRate' -> 'learning_rate'

    Args:
        history: Raw history from read_tensorboard_from_subdirs

    Returns:
        History with normalized keys
    """
    normalized = {}

    key_mapping = {
        # Underscore format (legacy/manual)
        "Loss_epoch_train": "train_loss",
        "Loss_epoch_val": "val_loss",
        # Slash format (from add_scalars)
        "Loss/epoch_train": "train_loss",
        "Loss/epoch_val": "val_loss",
        # LearningRate (same in both formats)
        "LearningRate": "learning_rate",
    }

    for key, values in history.items():
        # Direct mapping
        if key in key_mapping:
            normalized[key_mapping[key]] = values
            continue

        # Handle Params_X_metric_split or Params/X_metric_split format
        if key.startswith("Params_") or key.startswith("Params/"):
            # e.g., Params_A_0_mae_train or Params/A_0_mae_train -> train_A_0_mae
            remainder = key[7:]  # Remove "Params_" or "Params/"
            parts = remainder.rsplit("_", 1)
            if len(parts) == 2:
                metric_part, split = parts
                normalized[f"{split}_{metric_part}"] = values
                continue

        # Keep original key if no mapping found
        normalized[key] = values

    return normalized
