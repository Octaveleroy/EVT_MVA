"""PDF report builder using reportlab."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PDFReportBuilder:
    """
    Builds PDF reports using reportlab.

    Layout:
    - Page 1: Header, Config Table, Architecture Summary
    - Page 2: Validation Metrics Table
    - Page 3: Pred vs True Scatter Plots
    - Page 4: Error Distribution Histograms
    - Page 5: Residual Analysis
    - Page 6: Training Curves (if TensorBoard available)
    """

    def __init__(self, output_path: Union[str, Path]):
        """
        Initialize PDF builder.

        Args:
            output_path: Path for output PDF file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.elements: List = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define custom paragraph and table styles."""
        # Title style
        self.styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                spaceAfter=20,
                alignment=1,  # Center
            )
        )

        # Section header style
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=14,
                spaceBefore=15,
                spaceAfter=10,
                textColor=colors.darkblue,
            )
        )

        # Subsection header style
        self.styles.add(
            ParagraphStyle(
                name="SubsectionHeader",
                parent=self.styles["Heading3"],
                fontSize=12,
                spaceBefore=10,
                spaceAfter=5,
            )
        )

        # Info text style
        self.styles.add(
            ParagraphStyle(
                name="InfoText",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.grey,
            )
        )

    def add_header(
        self,
        model_name: str,
        checkpoint_path: str,
        generation_time: str,
    ):
        """
        Add report header section.

        Args:
            model_name: Name/identifier for the model
            checkpoint_path: Path to the checkpoint file
            generation_time: Timestamp when report was generated
        """
        # Title
        self.elements.append(
            Paragraph("NBE Model Report", self.styles["ReportTitle"])
        )

        # Model info
        info_data = [
            ["Model:", model_name],
            ["Checkpoint:", checkpoint_path],
            ["Generated:", generation_time],
        ]

        info_table = Table(info_data, colWidths=[2 * inch, 4.5 * inch])
        info_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.darkblue),
                    ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                    ("ALIGN", (1, 0), (1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        self.elements.append(info_table)
        self.elements.append(Spacer(1, 0.3 * inch))

    def add_section_header(self, title: str):
        """Add a section header."""
        self.elements.append(
            Paragraph(title, self.styles["SectionHeader"])
        )

    def add_config_table(self, config_rows: List[List[str]]):
        """
        Add training configuration table.

        Args:
            config_rows: List of [parameter, value] rows from format_config_table()
        """
        self.add_section_header("Training Configuration")

        # Add header row
        data = [["Parameter", "Value"]] + config_rows

        table = Table(data, colWidths=[2.5 * inch, 3 * inch])
        table.setStyle(
            TableStyle(
                [
                    # Header row
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    # Data rows
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("ALIGN", (0, 1), (0, -1), "LEFT"),
                    ("ALIGN", (1, 1), (1, -1), "LEFT"),
                    # Alternating row colors
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    # Grid
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        self.elements.append(table)
        self.elements.append(Spacer(1, 0.2 * inch))

    def add_architecture_table(self, arch_rows: List[List[str]]):
        """
        Add model architecture summary table.

        Args:
            arch_rows: List of [component, configuration] rows from format_architecture_table()
        """
        self.add_section_header("Model Architecture")

        # Add header row
        data = [["Component", "Configuration"]] + arch_rows

        table = Table(data, colWidths=[2.5 * inch, 3 * inch])
        table.setStyle(
            TableStyle(
                [
                    # Header row
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    # Data rows
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("ALIGN", (0, 1), (0, -1), "LEFT"),
                    ("ALIGN", (1, 1), (1, -1), "LEFT"),
                    # Alternating row colors
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    # Grid
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        self.elements.append(table)
        self.elements.append(Spacer(1, 0.2 * inch))

    def add_model_structure(self, model_str: str):
        """
        Add detailed model structure as a code block.

        Displays the PyTorch model's string representation showing
        the complete layer-by-layer architecture.

        Args:
            model_str: String representation of the model from str(model)
        """
        self.add_section_header("Detailed Model Structure")

        # Create a monospace style for code-like appearance
        code_style = ParagraphStyle(
            name='ModelCode',
            parent=self.styles['Normal'],
            fontName='Courier',
            fontSize=7,
            leading=9,
            leftIndent=10,
        )

        # Escape special characters for XML/HTML and preserve whitespace
        model_str = model_str.replace('&', '&amp;')
        model_str = model_str.replace('<', '&lt;')
        model_str = model_str.replace('>', '&gt;')
        model_str = model_str.replace(' ', '&nbsp;')
        model_str = model_str.replace('\n', '<br/>')

        self.elements.append(Paragraph(model_str, code_style))
        self.elements.append(Spacer(1, 0.2 * inch))

    def add_metrics_table(self, metrics_rows: List[List[str]]):
        """
        Add validation metrics table.

        Args:
            metrics_rows: Rows from format_metrics_table() including header
        """
        self.add_section_header("Validation Metrics")

        # metrics_rows already includes header
        table = Table(metrics_rows, colWidths=[1.2 * inch, 1.2 * inch, 1.4 * inch, 1.2 * inch, 1.2 * inch])
        table.setStyle(
            TableStyle(
                [
                    # Header row
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    # Data rows
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("ALIGN", (0, 1), (0, -1), "LEFT"),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    # Total row highlight
                    ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                    # Alternating row colors (except total)
                    ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.lightgrey]),
                    # Grid
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        self.elements.append(table)
        self.elements.append(Spacer(1, 0.2 * inch))

    def add_priors_table(self, priors_rows: List[List[str]]):
        """
        Add prior distributions table.

        Args:
            priors_rows: Rows from format_priors_table()
        """
        self.add_section_header("Prior Distributions")

        # Add header row
        data = [["Parameter", "Prior Distribution"]] + priors_rows

        table = Table(data, colWidths=[2 * inch, 3.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    # Header row
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    # Data rows
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("ALIGN", (0, 1), (-1, -1), "LEFT"),
                    # Alternating row colors
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    # Grid
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        self.elements.append(table)
        self.elements.append(Spacer(1, 0.2 * inch))

    def add_plot(
        self,
        image_path: Union[str, Path],
        caption: str,
        width: float = 6 * inch,
    ):
        """
        Add a plot image to the report.

        Args:
            image_path: Path to the image file
            caption: Caption text for the plot
            width: Width of the image in the PDF
        """
        image_path = Path(image_path)
        if not image_path.exists():
            self.elements.append(
                Paragraph(f"[Image not found: {image_path}]", self.styles["Normal"])
            )
            return

        self.add_section_header(caption)

        # Calculate height maintaining aspect ratio
        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            img_width, img_height = img.size
            aspect_ratio = img_height / img_width
            height = width * aspect_ratio

        img = Image(str(image_path), width=width, height=height)
        self.elements.append(img)
        self.elements.append(Spacer(1, 0.2 * inch))

    def add_page_break(self):
        """Insert a page break."""
        self.elements.append(PageBreak())

    def add_text(self, text: str, style: str = "Normal"):
        """
        Add a paragraph of text.

        Args:
            text: The text content
            style: Style name from styles dict
        """
        self.elements.append(Paragraph(text, self.styles[style]))
        self.elements.append(Spacer(1, 0.1 * inch))

    def add_validation_summary(
        self,
        k_val: int,
        m_val: int,
        total_mae: float,
        total_mse: float,
    ):
        """
        Add a summary box for validation results.

        Args:
            k_val: Number of validation samples
            m_val: Number of replicates per sample
            total_mae: Total mean absolute error
            total_mse: Total mean squared error
        """
        self.add_section_header("Validation Summary")

        summary_text = f"""
        <b>Validation Set:</b> {k_val:,} parameter sets, {m_val} replicates each<br/>
        <b>Total MAE:</b> {total_mae:.6f}<br/>
        <b>Total MSE:</b> {total_mse:.8f}
        """

        self.elements.append(
            Paragraph(summary_text, self.styles["Normal"])
        )
        self.elements.append(Spacer(1, 0.2 * inch))

    def build(self) -> Path:
        """
        Finalize and save the PDF.

        Returns:
            Path to the generated PDF file
        """
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        doc.build(self.elements)
        return self.output_path
