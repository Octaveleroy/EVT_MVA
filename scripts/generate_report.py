#!/usr/bin/env python
"""
Generate a PDF report for a trained Neural Bayes Estimator model.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.reporting import ReportGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate PDF report for a trained NBE model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )

    # Optional TensorBoard
    parser.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="Path to TensorBoard run directory for training curves",
    )

    # Validation settings
    parser.add_argument(
        "--K_val",
        type=int,
        default=2000,
        help="Number of validation parameter sets to generate",
    )

    parser.add_argument(
        "--M",
        type=int,
        default=100,
        help="Number of replicates per validation sample",
    )

    # Model architecture (for checkpoints without args)
    parser.add_argument(
        "--N",
        type=int,
        default=500,
        help="Number of time steps (required if checkpoint has no args)",
    )

    parser.add_argument(
        "--Delta_t",
        type=float,
        default=0.01,
        help="Time step size",
    )

    parser.add_argument(
        "--X_0",
        type=float,
        default=1.0,
        help="Initial condition",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="cnn",
        # --- MODIFICATION ICI : Ajout de rnn, gru, lstm ---
        choices=["cnn", "mlp", "resnet", "rnn", "gru", "lstm"], 
        help="Encoder architecture",
    )

    parser.add_argument(
        "--transform",
        type=str,
        default="identity",
        choices=["identity", "log", "standardize", "cuberoot", "difference", "logreturns", "robust_standardize"],
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
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for encoder",
    )

    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="CNN encoder channels, comma-separated (e.g., '16,32,64')",
    )

    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional RNN/GRU/LSTM",
    )

    parser.add_argument(
        "--phi_layers",
        type=str,
        default=None,
        help="MLP estimator hidden layers, comma-separated (e.g., '192,64')",
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path. Default: reports/{checkpoint_name}_report.pdf",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible validation data",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        reports_dir = project_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = reports_dir / f"{checkpoint_path.stem}_report_{timestamp}.pdf"

    # Build fallback args from CLI for checkpoints without args
    cli_args = {
        "N": args.N,
        "Delta_t": args.Delta_t,
        "X_0": args.X_0,
        "encoder": args.encoder,
        "transform": args.transform,
        "aggregation": args.aggregation,
        "hidden_dim": args.hidden_dim,
        "channels": args.channels,
        "phi_layers": args.phi_layers,
        "bidirectional": args.bidirectional,
        "train_size": "N/A",
        "val_size": "N/A",
        "M": args.M,
        "min_M": "N/A",
        "epochs": "N/A",
        "batch_size": "N/A",
        "lr": "N/A",
        "weight_decay": "N/A",
        "loss": "N/A",
        "patience": "N/A",
        "seed": args.seed,
    }

    print("=" * 60)
    print("NBE Model Report Generator")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"TensorBoard: {args.tensorboard or 'Not provided'}")
    print(f"Validation: K={args.K_val}, M={args.M}")
    print(f"Device: {args.device}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Create report generator
    generator = ReportGenerator(
        checkpoint_path=checkpoint_path,
        tensorboard_dir=args.tensorboard,
        K_val=args.K_val,
        M_val=args.M,
        device=args.device,
        seed=args.seed,
        fallback_args=cli_args,
    )

    # Generate report
    try:
        report_path = generator.generate_report(output_path)
        print("\n" + "=" * 60)
        print("Report generated successfully!")
        print(f"Output: {report_path}")
        print("=" * 60)
    except Exception as e:
        print(f"\nError generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()