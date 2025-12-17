"""Main report generator orchestrating the full pipeline."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics_tables import (
    format_architecture_table,
    format_config_table,
    format_metrics_table,
    format_priors_table,
)
from .pdf_builder import PDFReportBuilder
from .plot_generators import (
    plot_error_distributions,
    plot_predictions_vs_true_grid,
    plot_residuals_analysis,
    plot_training_curves,
)
from .tensorboard_reader import (
    normalize_history_keys,
    read_tensorboard_from_subdirs,
)


class ReportGenerator:
    """
    Orchestrates model report generation.

    Handles:
    - Checkpoint loading and model reconstruction
    - Fresh validation data generation
    - Metrics computation
    - Plot generation
    - PDF assembly
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        tensorboard_dir: Optional[Union[str, Path]] = None,
        K_val: int = 2000,
        M_val: int = 100,
        device: str = "cuda",
        seed: int = 42,
        fallback_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize report generator.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            tensorboard_dir: Optional path to TensorBoard run directory
            K_val: Number of validation parameter sets
            M_val: Number of replicates per validation sample
            device: Device for inference ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            fallback_args: Optional args to use if checkpoint doesn't contain 'args'
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir else None
        self.K_val = K_val
        self.M_val = M_val
        self.device = device
        self.seed = seed
        self.fallback_args = fallback_args or {}

        # Will be populated during generation
        self.checkpoint: Dict[str, Any] = {}
        self.args: Dict[str, Any] = {}
        self.model: Optional[nn.Module] = None
        self.predictions: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.metrics: Dict[str, float] = {}
        self.training_history: Dict[str, list] = {}

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint and extract training args.

        Uses fallback_args if checkpoint doesn't contain 'args'.

        Returns:
            Checkpoint dictionary

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint has no args and no fallback_args provided
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        if "args" in self.checkpoint:
            self.args = self.checkpoint["args"]
            print("    Using args from checkpoint")
        elif self.fallback_args:
            self.args = self.fallback_args
            print("    Using CLI-provided args (checkpoint has no args)")
        else:
            raise ValueError(
                f"Checkpoint does not contain 'args' and no fallback_args provided. "
                f"Either use 'final_model.pt' or provide model architecture via CLI."
            )

        return self.checkpoint

    def reconstruct_model(self) -> nn.Module:
        """
        Rebuild model from checkpoint args using build_default_nbe().

        Returns:
            Reconstructed model with loaded weights

        Raises:
            RuntimeError: If model cannot be reconstructed
        """
        # Import here to avoid circular imports
        from src.models import build_default_nbe

        args = self.args

        # Parse channels and phi_layers if they're strings
        encoder_channels = None
        if args.get("channels"):
            channels_str = args["channels"]
            if isinstance(channels_str, str):
                encoder_channels = [int(x) for x in channels_str.split(",")]
            elif isinstance(channels_str, list):
                encoder_channels = channels_str

        phi_hidden_layers = None
        if args.get("phi_layers"):
            layers_str = args["phi_layers"]
            if isinstance(layers_str, str):
                phi_hidden_layers = [int(x) for x in layers_str.split(",")]
            elif isinstance(layers_str, list):
                phi_hidden_layers = layers_str

        # Build model
        self.model = build_default_nbe(
            input_length=args["N"] + 1,
            num_params=4,
            hidden_dim=args.get("hidden_dim", 128),
            encoder_type=args.get("encoder", "cnn"),
            transform_type=args.get("transform", "identity"),
            aggregation_type=args.get("aggregation", "mean"),
            encoder_channels=encoder_channels,
            phi_hidden_layers=phi_hidden_layers,
            bidirectional=args.get("bidirectional", False),
        )

        # Load weights
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        return self.model

    def generate_validation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate fresh validation dataset using same priors as training.

        Returns:
            Tuple of (trajectories, params) tensors
        """
        # Import here to avoid circular imports
        from src import NeuralBayesDataGenerator, SDESimulator, UniformPrior

        # Set seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create simulator with same settings as training
        args = self.args
        simulator = SDESimulator(
            M=self.M_val,
            N=args["N"],
            Delta_t=args["Delta_t"],
            X_0=args["X_0"],
            backend="torch",
            device="cpu",  # Generate on CPU, transfer later
            seed=self.seed,
        )

        # Create priors (same as training)
        priors = {
            "A_0": UniformPrior(0.0, 1.0),
            "a": UniformPrior(0.0, 2.0),
            "B_0": UniformPrior(0.01, 0.5),
            "b": UniformPrior(0.0, 0.5),
        }

        # Generate validation data
        generator = NeuralBayesDataGenerator(simulator, priors)
        trajectories, params = generator.generate_dataset(n_params=self.K_val)

        return trajectories, params

    def compute_predictions(
        self, trajectories: torch.Tensor, params: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on validation data.

        Args:
            trajectories: Validation trajectories tensor
            params: Dictionary of parameter arrays

        Returns:
            Tuple of (predictions, targets) numpy arrays
        """
        from src import SDEDataset

        # Create dataset and dataloader
        dataset = SDEDataset(trajectories, params)
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=(self.device == "cuda"),
        )

        all_preds = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.numpy())

        self.predictions = np.concatenate(all_preds, axis=0)
        self.targets = np.concatenate(all_targets, axis=0)

        return self.predictions, self.targets

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute validation metrics.

        Returns:
            Dictionary of metrics per parameter and totals
        """
        from src.training.utils import compute_metrics

        preds_tensor = torch.from_numpy(self.predictions)
        targets_tensor = torch.from_numpy(self.targets)

        self.metrics = compute_metrics(preds_tensor, targets_tensor)
        return self.metrics

    def load_tensorboard_history(self) -> Dict[str, list]:
        """
        Extract training history from TensorBoard logs.

        Returns:
            Dictionary of training history (empty if not available)
        """
        if self.tensorboard_dir is None or not self.tensorboard_dir.exists():
            return {}

        raw_history = read_tensorboard_from_subdirs(self.tensorboard_dir)
        self.training_history = normalize_history_keys(raw_history)
        return self.training_history

    def generate_report(self, output_path: Union[str, Path]) -> Path:
        """
        Generate complete PDF report.

        Args:
            output_path: Path for output PDF file

        Returns:
            Path to generated PDF
        """
        output_path = Path(output_path)
        print(f"Generating report: {output_path}")

        # Step 1: Load checkpoint
        print("  Loading checkpoint...")
        self.load_checkpoint()

        # Step 2: Reconstruct model
        print("  Reconstructing model...")
        self.reconstruct_model()

        # Step 3: Generate validation data
        print(f"  Generating validation data (K={self.K_val}, M={self.M_val})...")
        trajectories, params = self.generate_validation_data()

        # Step 4: Compute predictions
        print("  Computing predictions...")
        self.compute_predictions(trajectories, params)

        # Step 5: Compute metrics
        print("  Computing metrics...")
        self.compute_metrics()

        # Step 6: Load TensorBoard history
        print("  Loading training history...")
        self.load_tensorboard_history()

        # Step 7: Generate plots and build PDF
        print("  Building PDF...")
        self._build_pdf(output_path)

        print(f"Report saved to: {output_path}")
        return output_path

    def _build_pdf(self, output_path: Path):
        """Build the PDF document with all sections."""
        pdf = PDFReportBuilder(output_path)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Model name from checkpoint path
        model_name = self.checkpoint_path.stem

        # Total parameters
        total_params = sum(p.numel() for p in self.model.parameters())

        # Add header
        pdf.add_header(
            model_name=model_name,
            checkpoint_path=str(self.checkpoint_path),
            generation_time=timestamp,
        )

        # Add configuration table
        config_rows = format_config_table(self.args)
        pdf.add_config_table(config_rows)

        # Add architecture table
        arch_rows = format_architecture_table(self.args, total_params)
        pdf.add_architecture_table(arch_rows)

        # Add detailed model structure
        pdf.add_model_structure(str(self.model))

        # Add priors table
        priors_rows = format_priors_table()
        pdf.add_priors_table(priors_rows)

        # Page break before metrics
        pdf.add_page_break()

        # Add validation summary
        pdf.add_validation_summary(
            k_val=self.K_val,
            m_val=self.M_val,
            total_mae=self.metrics.get("total_mae", 0),
            total_mse=self.metrics.get("total_mse", 0),
        )

        # Add metrics table
        metrics_rows = format_metrics_table(self.metrics)
        pdf.add_metrics_table(metrics_rows)

        # Generate plots to temp files and add to PDF
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Pred vs True scatter plots
            pdf.add_page_break()
            scatter_path = tmpdir / "scatter.png"
            plot_predictions_vs_true_grid(
                self.predictions, self.targets, save_path=scatter_path
            )
            plt.close()
            pdf.add_plot(scatter_path, "Predicted vs True Values")

            # Error distributions
            pdf.add_page_break()
            error_path = tmpdir / "error_dist.png"
            plot_error_distributions(
                self.predictions, self.targets, save_path=error_path
            )
            plt.close()
            pdf.add_plot(error_path, "Error Distributions")

            # Residual analysis
            pdf.add_page_break()
            residuals_path = tmpdir / "residuals.png"
            plot_residuals_analysis(
                self.predictions, self.targets, save_path=residuals_path
            )
            plt.close()
            pdf.add_plot(residuals_path, "Residual Analysis")

            # Training curves (if available)
            if self.training_history:
                pdf.add_page_break()
                curves_path = tmpdir / "training_curves.png"
                plot_training_curves(self.training_history, save_path=curves_path)
                plt.close()
                pdf.add_plot(curves_path, "Training History")

            # Build final PDF
            pdf.build()
