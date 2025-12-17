"""
Training infrastructure for Neural Bayes Estimators.

Includes NBETrainer class with TensorBoard logging support.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import compute_metrics, EarlyStopping, save_checkpoint, PARAM_NAMES


class NBETrainer:
    """
    Trainer for Neural Bayes Estimators with TensorBoard logging.

    Handles the training loop, validation, logging, and checkpointing.

    Args:
        model: The DeepSetsNBE model to train
        optimizer: PyTorch optimizer
        loss_fn: Loss function (e.g., nn.L1Loss, nn.MSELoss)
        device: Device to train on ('cuda' or 'cpu')
        log_dir: Directory for TensorBoard logs. Default: 'runs/'
        scheduler: Optional learning rate scheduler
        param_names: Names of parameters being estimated. Default: PARAM_NAMES

    Example:
        >>> model = DeepSetsNBE(psi, agg, phi)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> trainer = NBETrainer(model, optimizer, nn.L1Loss(), device='cuda')
        >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: Union[str, torch.device] = "cuda",
        log_dir: Union[str, Path] = "runs/",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        param_names: Optional[List[str]] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.param_names = param_names if param_names is not None else PARAM_NAMES

        # Setup TensorBoard
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Track global step for logging
        self.global_step = 0

    def train_epoch(
        self, dataloader: DataLoader, steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            steps: Number of steps (batches) per epoch. If None, iterates through
                   entire dataloader. Required for IterableDataset (infinite data).

        Returns:
            Dictionary with training metrics (loss, per-param metrics)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        # Create progress bar with optional total
        pbar = tqdm(
            enumerate(dataloader),
            desc="Training",
            leave=False,
            total=steps,
        )

        for batch_idx, (trajectories, params) in pbar:
            # Stop after `steps` batches for IterableDataset
            if steps is not None and batch_idx >= steps:
                break

            # Move to device
            trajectories = trajectories.to(self.device)
            params = params.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(trajectories)

            # Compute loss
            loss = self.loss_fn(predictions, params)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(params.detach())
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Log batch loss to TensorBoard
            self.writer.add_scalar("Loss/train_batch", loss.item(), self.global_step)
            self.global_step += 1

        # Compute epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_predictions, all_targets, self.param_names)
        metrics["loss"] = avg_loss

        return metrics

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary with validation metrics (loss, per-param metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for trajectories, params in tqdm(dataloader, desc="Validating", leave=False):
                # Move to device
                trajectories = trajectories.to(self.device)
                params = params.to(self.device)

                # Forward pass
                predictions = self.model(trajectories)

                # Compute loss
                loss = self.loss_fn(predictions, params)

                # Track metrics
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(params)

        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_predictions, all_targets, self.param_names)
        metrics["loss"] = avg_loss

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        steps_per_epoch: Optional[int] = None,
        patience: int = 15,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping and TensorBoard logging.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs to train
            steps_per_epoch: Number of training steps (batches) per epoch.
                            Required for IterableDataset (infinite data).
                            If None, iterates through entire train_loader.
            patience: Early stopping patience. Default: 15
            checkpoint_dir: Directory to save checkpoints. Default: None
            save_best: Whether to save the best model. Default: True

        Returns:
            Dictionary with training history (train_loss, val_loss, etc.)
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Add per-param metrics to history
        for name in self.param_names:
            history[f"train_{name}_mae"] = []
            history[f"val_{name}_mae"] = []

        early_stopping = EarlyStopping(patience=patience)
        best_val_loss = float("inf")
        start_time = time.time()

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting training for {epochs} epochs...")
        if steps_per_epoch is not None:
            print(f"Steps per epoch: {steps_per_epoch}")
        else:
            print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"TensorBoard logs: {self.log_dir}")
        print("-" * 60)

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, steps=steps_per_epoch)
            history["train_loss"].append(train_metrics["loss"])

            # Validate
            val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_metrics["loss"])

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(current_lr)

            # Log per-param metrics
            for name in self.param_names:
                history[f"train_{name}_mae"].append(train_metrics.get(f"{name}_mae", 0))
                history[f"val_{name}_mae"].append(val_metrics.get(f"{name}_mae", 0))

            # Log to TensorBoard
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Print progress
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            if save_best and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                if checkpoint_dir is not None:
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_metrics["loss"],
                        checkpoint_dir / "best_model.pt",
                        scheduler=self.scheduler,
                    )
                    print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")

            # Early stopping
            if early_stopping(val_metrics["loss"]):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")

        return history

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
    ):
        """Log epoch metrics to TensorBoard."""
        # Loss
        self.writer.add_scalars(
            "Loss/epoch",
            {"train": train_metrics["loss"], "val": val_metrics["loss"]},
            epoch,
        )

        # Learning rate
        self.writer.add_scalar("LearningRate", learning_rate, epoch)

        # Per-parameter MAE
        for name in self.param_names:
            train_mae = train_metrics.get(f"{name}_mae", 0)
            val_mae = val_metrics.get(f"{name}_mae", 0)
            self.writer.add_scalars(
                f"Params/{name}_mae",
                {"train": train_mae, "val": val_mae},
                epoch,
            )

        # Per-parameter bias
        for name in self.param_names:
            train_bias = train_metrics.get(f"{name}_bias", 0)
            val_bias = val_metrics.get(f"{name}_bias", 0)
            self.writer.add_scalars(
                f"Params/{name}_bias",
                {"train": train_bias, "val": val_bias},
                epoch,
            )

    def predict(self, dataloader: DataLoader) -> tuple:
        """
        Generate predictions for a dataset.

        Args:
            dataloader: Data loader to predict on

        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for trajectories, params in dataloader:
                trajectories = trajectories.to(self.device)
                predictions = self.model(trajectories)
                all_predictions.append(predictions.cpu())
                all_targets.append(params)

        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        return predictions, targets

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()

    def __del__(self):
        """Ensure TensorBoard writer is closed."""
        if hasattr(self, "writer"):
            self.writer.close()
