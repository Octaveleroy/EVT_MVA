"""
Report generation module for Neural Bayes Estimators.

Provides PDF report generation with:
- Fresh validation metrics
- Diagnostic plots (scatter, error distributions, residuals, training curves)
- Training configuration summary
- Model architecture details
"""

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
from .report_generator import ReportGenerator
from .tensorboard_reader import (
    normalize_history_keys,
    read_tensorboard_from_subdirs,
)

__all__ = [
    "ReportGenerator",
    "PDFReportBuilder",
    "plot_predictions_vs_true_grid",
    "plot_error_distributions",
    "plot_residuals_analysis",
    "plot_training_curves",
    "read_tensorboard_from_subdirs",
    "normalize_history_keys",
    "format_config_table",
    "format_architecture_table",
    "format_metrics_table",
    "format_priors_table",
]
