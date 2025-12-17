#!/usr/bin/env python
"""
Train a Neural Bayes Estimator for SDE parameter inference.

This script implements on-the-fly training following Sainsbury-Dale et al. (2024):
- Section 2.3.3: Fresh data simulated continuously during training
- Section 2.2.2 (Eq 7): Variable M ~ U[min_M, max_M] per batch

Usage:
    # Variable M training (paper-aligned)
    python scripts/train_nbe.py --min_M 10 --M 100 --epochs 100

    # Fixed M training
    python scripts/train_nbe.py --M 100 --epochs 100

    # With custom settings
    python scripts/train_nbe.py --encoder cnn --transform standardize --device cuda

TensorBoard:
    tensorboard --logdir runs/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import (
    SDESimulator,
    NeuralBayesDataGenerator,
    UniformPrior,
    SDEDataset,
    OnTheFlyDataset,
)
from src.models import (
    build_default_nbe,
)
from src.training import (
    NBETrainer,
    save_checkpoint,
    plot_predictions_vs_true,
    plot_training_history,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Neural Bayes Estimator for SDE parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data generation
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Training samples per epoch (steps_per_epoch = train_size // batch_size)",
    )
    parser.add_argument(
        "--val_size", type=int, default=2000, help="Validation samples (fixed dataset)"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=100,
        help="Max M (number of replicates). Also used as fixed M for validation.",
    )
    parser.add_argument(
        "--min_M",
        type=int,
        default=10,
        help="Min M for variable M training (Eq 7 in paper). If None, uses --M (fixed M).",
    )
    parser.add_argument("--N", type=int, default=500, help="Number of time steps")
    parser.add_argument("--Delta_t", type=float, default=0.01, help="Time step size")
    parser.add_argument("--X_0", type=float, default=1.0, help="Initial condition")

    # Model architecture
    parser.add_argument(
        "--encoder",
        type=str,
        default="cnn",
        choices=["cnn", "mlp", "resnet","rnn"],
        help="Encoder architecture",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="identity",
        choices=[
            "identity",
            "log",
            "standardize",
            "cuberoot",
            "difference",
            "logreturns",
            "robust_standardize",
        ],
        help="Data pre-transformation",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "sum", "max", "meanmax", "meanstd"],
        help="Aggregation function",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension for encoder"
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="CNN encoder channels, comma-separated (e.g., '16,32,64'). Default: 32,64,128",
    )

    parser.add_argument(
        "--rnn_type",
        type=str,
        default="LSTM",
        choices=["LSTM", "GRU"],
        help="Type of RNN cell (only used if encoder is rnn)",
    )
    parser.add_argument(
        "--rnn_layers",
        type=int,
        default=1,
        help="Number of stacked RNN layers",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional RNN (processes sequence forwards and backwards)",
    )
    parser.add_argument(
        "--phi_layers",
        type=str,
        default=None,
        help="MLP estimator hidden layers, comma-separated (e.g., '192,64'). Default: 256,128,64",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--loss",
        type=str,
        default="mae",
        choices=["mae", "mse", "huber"],
        help="Loss function",
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="DataLoader workers (0 for main process)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="TensorBoard log directory (auto-generated if not specified)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs (plots, etc.)",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_priors():
    """Create prior distributions for SDE parameters."""
    return {
        "A_0": UniformPrior(0.0, 1.0),
        "a": UniformPrior(0.0, 2.0),
        "B_0": UniformPrior(0.01, 0.5),
        "b": UniformPrior(0.0, 0.5),
    }


def create_data(args):
    """
    Create training (online) and validation (fixed) datasets.

    Following Sainsbury-Dale et al. (2024):
    - Training: OnTheFlyDataset with variable M (Section 2.3.3, Eq 7)
    - Validation: Fixed dataset with fixed M (kept constant for consistent metrics)

    Returns:
        train_dataset: OnTheFlyDataset (infinite, variable M)
        val_dataset: SDEDataset (fixed, constant M)
        steps_per_epoch: Number of training steps per epoch
    """
    print("Creating simulator and datasets...")

    # Create simulator with max M
    simulator = SDESimulator(
        M=args.M,
        N=args.N,
        Delta_t=args.Delta_t,
        X_0=args.X_0,
        backend="torch",
        device="cpu",  # Simulate on CPU, transfer to GPU during training
        seed=args.seed,
    )

    priors = create_priors()

    # Determine min_M (default to M for fixed M training)
    min_M = args.min_M if args.min_M is not None else args.M

    print(f"  M range: [{min_M}, {args.M}]")
    if min_M == args.M:
        print("  (Fixed M mode)")
    else:
        print("  (Variable M mode - paper Eq 7)")

    # Training: OnTheFlyDataset (infinite, variable M)
    train_dataset = OnTheFlyDataset(
        simulator=simulator,
        priors=priors,
        batch_size=args.batch_size,
        min_M=min_M,
        max_M=args.M,
        seed=args.seed,
    )

    # Validation: Fixed dataset with fixed M
    # Paper says "fix the validation data" for consistent metrics
    print(f"  Generating {args.val_size} validation samples (fixed M={args.M})...")
    generator = NeuralBayesDataGenerator(simulator, priors)
    trajectories, params = generator.generate_dataset(n_params=args.val_size)
    val_dataset = SDEDataset(trajectories, params)

    # Calculate steps per epoch
    steps_per_epoch = args.train_size // args.batch_size

    print(f"  Training: on-the-fly generation, {steps_per_epoch} steps/epoch")
    print(f"  Validation: {len(val_dataset)} fixed samples")
    print(f"  Trajectory shape: (M, {args.N + 1}) where M ∈ [{min_M}, {args.M}]")

    return train_dataset, val_dataset, steps_per_epoch


def create_model(args):
    """Create the DeepSetsNBE model."""
    input_length = args.N + 1
    num_params = 4

    # Parse channels and phi_layers if provided
    encoder_channels = None
    if args.channels:
        encoder_channels = [int(x) for x in args.channels.split(",")]

    phi_hidden_layers = None
    if args.phi_layers:
        phi_hidden_layers = [int(x) for x in args.phi_layers.split(",")]

    print(f"\nBuilding model:")
    print(f"  Encoder: {args.encoder}")
    print(f"  Transform: {args.transform}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Hidden dim: {args.hidden_dim}")

    if args.encoder == "cnn" and encoder_channels:
        print(f"  Encoder channels: {encoder_channels}")
    elif args.encoder == "rnn":
        print(f"  RNN Type: {args.rnn_type}")
        print(f"  RNN Layers: {args.rnn_layers}")
        print(f"  Bidirectional: {args.bidirectional}")
    if phi_hidden_layers:
        print(f"  Phi hidden layers: {phi_hidden_layers}")

    # Build using factory function
    model = build_default_nbe(
        input_length=input_length,
        num_params=num_params,
        hidden_dim=args.hidden_dim,
        encoder_type=args.encoder,
        transform_type=args.transform,
        aggregation_type=args.aggregation,
        encoder_channels=encoder_channels,
        phi_hidden_layers=phi_hidden_layers,
        rnn_type=args.rnn_type,
        num_layers=args.rnn_layers, 
        bidirectional=args.bidirectional,
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def get_loss_function(loss_type: str) -> nn.Module:
    """Get loss function by name."""
    loss_map = {
        "mae": nn.L1Loss(),
        "mse": nn.MSELoss(),
        "huber": nn.SmoothL1Loss(),
    }
    return loss_map[loss_type]


def worker_init_fn(worker_id):
    """Initialize worker with unique seed for proper multiprocessing."""
    # Each worker gets a different seed based on worker_id
    # This is handled internally by OnTheFlyDataset, but we set numpy/torch seeds too
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(torch.initial_seed() + worker_id)


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Auto-generate log directory name
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        m_str = f"M{args.min_M}-{args.M}" if args.min_M else f"M{args.M}"
        args.log_dir = f"runs/nbe_{args.encoder}_{m_str}_{timestamp}"

    print("=" * 60)
    print("Neural Bayes Estimator Training (On-the-fly)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Log directory: {args.log_dir}")

    # Create datasets
    train_dataset, val_dataset, steps_per_epoch = create_data(args)

    # Create data loaders
    # OnTheFlyDataset yields pre-batched data, so batch_size=None
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # OnTheFlyDataset yields pre-batched data
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        pin_memory=(args.device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # Create model
    model = create_model(args)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Create loss function
    loss_fn = get_loss_function(args.loss)
    print(f"\nLoss function: {args.loss}")

    # Create trainer
    trainer = NBETrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        log_dir=args.log_dir,
        scheduler=scheduler,
    )

    # Train with steps_per_epoch
    print("\n" + "=" * 60)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        save_best=True,
    )

    # Generate predictions on validation set
    print("\nGenerating predictions on validation set...")
    predictions, targets = trainer.predict(val_loader)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save diagnostic plots
    print("Saving diagnostic plots...")
    plot_predictions_vs_true(
        predictions,
        targets,
        save_path=output_dir / "predictions_vs_true.png",
    )
    plot_training_history(
        history,
        save_path=output_dir / "training_history.png",
    )

    # Save final model
    final_model_path = Path(args.checkpoint_dir) / "final_model.pt"
    save_checkpoint(
        model,
        optimizer,
        epoch=len(history["train_loss"]) - 1,
        loss=history["val_loss"][-1],
        path=final_model_path,
        scheduler=scheduler,
        args=vars(args),
    )
    print(f"\nFinal model saved to: {final_model_path}")

    # Close trainer (TensorBoard writer)
    trainer.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}")
    print(f"Plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
