"""
Enhanced PDF Generator Module for FlashDNA

This module provides enterprise-grade PDF report generation with charts, formatting,
and comprehensive error handling for the FlashDNA analytics platform.

Features:
- Professional multi-page reports with cover page and company branding
- Data visualization including CAMP radar charts and financial projections
- Section-based customization of report content
- Robust error handling and fallback mechanisms
- Memory-efficient processing of large datasets
"""

import os
import logging
import tempfile
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from fpdf import FPDF
import base64
from io import BytesIO

# Configure module logger
logger = logging.getLogger(__name__)

# Define color scheme constants for consistent branding
COLOR_PRIMARY = (31, 119, 180)     # Blue
COLOR_SECONDARY = (255, 127, 14)   # Orange
COLOR_SUCCESS = (44, 160, 44)      # Green
COLOR_WARNING = (214, 39, 40)      # Red
COLOR_BACKGROUND = (248, 248, 248) # Light gray

class EnhancedReportPDF(FPDF):
    """
    Enhanced PDF report generator with professional formatting and visual elements.
    
    Extends FPDF with custom headers, footers, and specialized content blocks
    for financial and startup analytics reports.
    """
    
    def __init__(self, title: str = "Investor Report", doc: Dict[str, Any] = None, *args, **kwargs):
        """
        Initialize the enhanced PDF generator.
        
        Args:
            title: The report title
            doc: The document data dictionary
            *args: Additional positional arguments for FPDF
            **kwargs: Additional keyword arguments for FPDF
        """
        super().__init__(*args, **kwargs)
        self.title = title
        self.doc = doc or {}
        self.company_name = doc.get('name', 'Startup') if doc else 'Startup'
        
        # Configure page setup
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(True, margin=15)
        
        # Track if we've added a cover page
        self.has_cover_page = False
        
        # Register fonts (with fallbacks for missing fonts)
        self._setup_fonts()
    
    def _setup_fonts(self) -> None:
        """Setup fonts with fallbacks for different platforms."""
        # Standard fonts are always available in FPDF
        self.default_font = 'Arial'
    
    def header(self) -> None:
        """
        Add a custom header to each page except the cover page.
        Includes company name, report title, and date.
        """
        # Skip header on the cover page
        if self.page_no() == 1 and self.has_cover_page:
            return
        
        # Header with company name
        self.set_font(self.default_font, 'B', 10)
        self.cell(30, 10, self.company_name, 0, 0, 'L')
        
        # Report title in the middle
        self.cell(self.w - 60, 10, self.title, 0, 0, 'C')
        
        # Date on the right
        self.set_font(self.default_font, 'I', 8)
        self.cell(30, 10, datetime.now().strftime('%Y-%m-%d'), 0, 0, 'R')
        
        # Line break and separator
        self.ln(12)
        self.set_draw_color(*COLOR_PRIMARY)
        self.line(15, 20, self.w - 15, 20)
        self.ln(5)
    
    def footer(self) -> None:
        """
        Add a custom footer to each page except the cover page.
        Includes page number and separator line.
        """
        # Skip footer on the cover page
        if self.page_no() == 1 and self.has_cover_page:
            return
        
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        
        # Draw separator line
        self.set_draw_color(*COLOR_PRIMARY)
        self.line(15, self.h - 15, self.w - 15, self.h - 15)
        
        # Add page number
        self.set_font(self.default_font, 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def create_cover_page(self) -> None:
        """
        Create a professional cover page with logo, title, and company info.
        """
        self.add_page()
        self.has_cover_page = True
        
        # Try to get the logo from common locations
        logo_found = False
        for logo_path in ["static/logo.png", "logo.png", "static/img/logo.png"]:
            if os.path.exists(logo_path):
                try:
                    # Calculate logo dimensions for proper positioning
                    logo_height = 40
                    
                    # Get image dimensions if PIL is available
                    try:
                        import PIL.Image
                        img = PIL.Image.open(logo_path)
                        width, height = img.size
                        ratio = height / logo_height
                        logo_width = width / ratio
                        
                        # Center the logo
                        x_pos = (self.w - logo_width) / 2
                        self.image(logo_path, x=x_pos, y=30, h=logo_height)
                        logo_found = True
                        break
                    except ImportError:
                        # If PIL is not available, use a fixed width
                        self.image(logo_path, x=(self.w - 40)/2, y=30, h=logo_height)
                        logo_found = True
                        break
                except Exception as e:
                    logger.warning(f"Error loading logo from {logo_path}: {e}")
        
        # Extra space if logo was added
        if logo_found:
            self.ln(80)  # Space after logo
        else:
            self.ln(40)  # Less space if no logo
            
        # Add title with styling
        self.set_font(self.default_font, 'B', 24)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 20, "Investor Report", 0, 1, 'C')
        
        # Company name (larger font)
        self.set_font(self.default_font, 'B', 28)
        self.set_text_color(0, 0, 0)
        self.cell(0, 20, self.company_name, 0, 1, 'C')
        
        # Company information block
        self.ln(10)
        self.set_font(self.default_font, '', 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, f"Sector: {self.doc.get('sector', 'Technology')}", 0, 1, 'C')
        self.cell(0, 10, f"Stage: {self.doc.get('stage', 'Growth')}", 0, 1, 'C')
        
        # CAMP Score
        self.ln(15)
        self.set_font(self.default_font, 'B', 16)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 10, f"CAMP Score: {self.doc.get('camp_score', 0):.1f}/100", 0, 1, 'C')
        
        # Date at bottom
        self.set_y(-50)
        self.set_font(self.default_font, 'I', 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%B %d, %Y')}", 0, 1, 'C')
        
        # Confidentiality notice
        self.set_y(-30)
        self.set_font(self.default_font, 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, "CONFIDENTIAL", 0, 1, 'C')
        self.multi_cell(0, 4, "This report contains confidential information about the company and is intended only for the named recipient. If you are not the intended recipient, please notify the sender immediately.")
    
    def add_section_title(self, title: str) -> None:
        """
        Add a styled section title.
        
        Args:
            title: The section title text
        """
        self.set_font(self.default_font, 'B', 14)
        self.set_text_color(*COLOR_PRIMARY)
        self.set_fill_color(*COLOR_BACKGROUND)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)
    
    def add_subsection_title(self, title: str) -> None:
        """
        Add a styled subsection title.
        
        Args:
            title: The subsection title text
        """
        self.set_font(self.default_font, 'B', 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
    
    def add_paragraph(self, text: str) -> None:
        """
        Add a styled paragraph of text.
        
        Args:
            text: The paragraph text
        """
        self.set_font(self.default_font, '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)
        self.ln(3)
    
    def add_metric(self, label: str, value: str, width: int = 90) -> None:
        """
        Add a labeled metric with value.
        
        Args:
            label: The metric label
            value: The metric value
            width: Width of the label column
        """
        self.set_font(self.default_font, 'B', 10)
        self.set_text_color(80, 80, 80)
        self.cell(width, 8, label, 0, 0, 'L')
        
        self.set_font(self.default_font, '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, str(value), 0, 1, 'L')
    
    def add_metric_row(self, metrics: List[Tuple[str, str]]) -> None:
        """
        Add a row of equally-spaced metrics.
        
        Args:
            metrics: List of (label, value) tuples
        """
        col_width = (self.w - self.l_margin - self.r_margin) / len(metrics)
        
        # First row: labels
        for label, _ in metrics:
            self.set_font(self.default_font, 'B', 10)
            self.set_text_color(80, 80, 80)
            self.cell(col_width, 8, label, 0, 0, 'L')
        
        self.ln()
        
        # Second row: values
        for _, value in metrics:
            self.set_font(self.default_font, '', 10)
            self.set_text_color(0, 0, 0)
            self.cell(col_width, 8, str(value), 0, 0, 'L')
            
        self.ln(12)
    
    def add_table(self, headers: List[str], data: List[List[str]]) -> None:
        """
        Add a styled table with headers and data.
        
        Args:
            headers: List of column headers
            data: List of data rows, each a list of cell values
        """
        # Calculate column widths
        col_width = (self.w - self.l_margin - self.r_margin) / len(headers)
        row_height = 7
        
        # Add headers with styling
        self.set_font(self.default_font, 'B', 10)
        self.set_fill_color(*COLOR_PRIMARY)
        self.set_text_color(255, 255, 255)
        
        for header in headers:
            self.cell(col_width, row_height, header, 1, 0, 'C', 1)
        self.ln()
        
        # Add data rows with alternating background
        self.set_text_color(0, 0, 0)
        for i, row in enumerate(data):
            # Set background color for alternating rows
            if i % 2 == 0:
                self.set_fill_color(255, 255, 255)
            else:
                self.set_fill_color(*COLOR_BACKGROUND)
            
            for cell in row:
                self.set_font(self.default_font, '', 9)
                self.cell(col_width, row_height, str(cell), 1, 0, 'L', 1)
            self.ln()
        
        self.ln(5)
    
    def add_chart(self, chart_data: bytes) -> None:
        """
        Add a chart image to the PDF.
        
        Args:
            chart_data: The chart image data as bytes
        """
        if not chart_data:
            self.add_paragraph("Chart data not available")
            return
            
        # Save the chart data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            try:
                tmp.write(chart_data)
                tmp.flush()
                
                # Add image with center alignment
                img_width = 170
                x_pos = (self.w - img_width) / 2
                self.image(tmp.name, x=x_pos, y=self.get_y(), w=img_width)
                
                # Add extra space after chart
                self.ln(100)
                
            except Exception as e:
                logger.error(f"Error adding chart to PDF: {e}")
                self.add_paragraph(f"Error displaying chart")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp.name)
                except Exception as e:
                    logger.error(f"Error removing temporary chart file: {e}")


class ChartGenerator:
    """
    Generates charts and visualizations for PDF reports.
    
    Uses matplotlib for data visualization with fallbacks when not available.
    """
    
    def __init__(self):
        """Initialize the chart generator with matplotlib if available."""
        self.matplotlib_available = False
        self.np_available = False
        
        # Try to import matplotlib and numpy
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            
            self.plt = plt
            self.np = np
            self.matplotlib_available = True
            self.np_available = True
            logger.info("Matplotlib and NumPy are available for chart generation")
        except ImportError:
            logger.warning("Matplotlib or NumPy not available - chart generation will be limited")
            self.plt = None
            self.np = None
    
    def create_camp_radar_chart(self, doc: Dict[str, Any]) -> Optional[bytes]:
        """
        Create a CAMP framework radar chart.
        
        Args:
            doc: The document data dictionary
            
        Returns:
            bytes: The chart image data or None if chart creation failed
        """
        if not self.matplotlib_available or not self.np_available:
            return self._create_text_based_chart("CAMP Framework", doc)
        
        try:
            # Extract CAMP scores
            capital_score = doc.get("capital_score", 0)
            advantage_score = doc.get("advantage_score", 0)
            market_score = doc.get("market_score", 0)
            people_score = doc.get("people_score", 0)
            
            # Create radar chart
            categories = ['Capital Efficiency', 'Market Dynamics', 'Advantage Moat', 'People & Performance']
            values = [capital_score, market_score, advantage_score, people_score]
            
            # Close the loop for the radar chart
            categories = categories + [categories[0]]
            values = values + [values[0]]
            
            # Calculate angle for each category
            N = len(categories) - 1
            angles = [n / float(N) * 2 * self.np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
            
            # Draw one axis per variable and add labels
            self.plt.xticks(angles[:-1], categories[:-1], color='grey', size=10)
            
            # Draw y-labels
            ax.set_rlabel_position(0)
            self.plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=8)
            self.plt.ylim(0, 100)
            
            # Plot data
            ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
            
            # Fill area
            ax.fill(angles, values, 'blue', alpha=0.1)
            
            # Add title
            self.plt.title("CAMP Framework Analysis", size=14, color='blue', y=1.1)
            
            # Save to bytes
            return self._fig_to_bytes(fig)
            
        except Exception as e:
            logger.error(f"Error creating CAMP radar chart: {e}")
            return self._create_text_based_chart("CAMP Framework", doc)
    
    def create_growth_chart(self, doc: Dict[str, Any]) -> Optional[bytes]:
        """
        Create a user growth projection chart.
        
        Args:
            doc: The document data dictionary
            
        Returns:
            bytes: The chart image data or None if chart creation failed
        """
        if not self.matplotlib_available or not self.np_available:
            return None
            
        try:
            sys_dynamics = doc.get("system_dynamics", {})
            if isinstance(sys_dynamics, dict) and "users" in sys_dynamics:
                users = sys_dynamics.get("users", [])
                months = list(range(len(users)))
                
                # Create figure
                fig, ax = self.plt.subplots(figsize=(8, 4))
                ax.plot(months, users, marker='o', linestyle='-', color='blue')
                
                # Add labels and title
                ax.set_xlabel('Month')
                ax.set_ylabel('Users')
                ax.set_title('User Growth Projection')
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add trend line
                if len(users) > 1:
                    z = self.np.polyfit(months, users, 1)
                    p = self.np.poly1d(z)
                    ax.plot(months, p(months), "r--", alpha=0.8)
                
                # Save to bytes
                return self._fig_to_bytes(fig)
                
            return None
        except Exception as e:
            logger.error(f"Error creating growth chart: {e}")
            return None
    
    def create_competitor_chart(self, doc: Dict[str, Any]) -> Optional[bytes]:
        """
        Create a competitive positioning chart.
        
        Args:
            doc: The document data dictionary
            
        Returns:
            bytes: The chart image data or None if chart creation failed
        """
        if not self.matplotlib_available or not self.np_available:
            return None
            
        try:
            positioning = doc.get("competitive_positioning", {})
            
            if positioning and "dimensions" in positioning and "company_position" in positioning and "competitor_positions" in positioning:
                dimensions = positioning.get("dimensions", [])
                company_position = positioning.get("company_position", {})
                competitor_positions = positioning.get("competitor_positions", {})
                
                if len(dimensions) >= 2:
                    x_dim = dimensions[0]
                    y_dim = dimensions[1]
                    
                    # Create figure
                    fig, ax = self.plt.subplots(figsize=(8, 6))
                    
                    # Plot company position
                    company_x = company_position.get(x_dim, 50)
                    company_y = company_position.get(y_dim, 50)
                    ax.scatter(company_x, company_y, color='blue', s=100, marker='o', label='Your Company')
                    
                    # Plot competitor positions
                    for comp_name, comp_pos in competitor_positions.items():
                        comp_x = comp_pos.get(x_dim, 50)
                        comp_y = comp_pos.get(y_dim, 50)
                        ax.scatter(comp_x, comp_y, color='red', s=80, alpha=0.7)
                        ax.annotate(comp_name, (comp_x, comp_y), xytext=(5, 5), textcoords='offset points')
                    
                    # Add quadrant lines
                    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
                    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
                    
                    # Add labels and title
                    ax.set_xlabel(x_dim)
                    ax.set_ylabel(y_dim)
                    ax.set_title(f'Competitive Positioning: {x_dim} vs {y_dim}')
                    
                    # Set axis limits
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 100)
                    
                    # Add legend
                    ax.legend()
                    
                    # Add grid
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    # Save to bytes
                    return self._fig_to_bytes(fig)
                    
            return None
        except Exception as e:
            logger.error(f"Error creating competitor chart: {e}")
            return None
    
    def create_financial_chart(self, doc: Dict[str, Any]) -> Optional[bytes]:
        """
        Create a financial projection chart.
        
        Args:
            doc: The document data dictionary
            
        Returns:
            bytes: The chart image data or None if chart creation failed
        """
        if not self.matplotlib_available or not self.np_available:
            return None
            
        try:
            forecast = doc.get("financial_forecast", {})
            
            if forecast and "months" in forecast and "revenue" in forecast:
                months = forecast.get("months", [])
                revenue = forecast.get("revenue", [])
                profit = forecast.get("profit", []) if "profit" in forecast else None
                
                # Create figure
                fig, ax = self.plt.subplots(figsize=(8, 4))
                
                # Plot revenue
                ax.plot(months, revenue, marker='o', linestyle='-', color='green', label='Revenue')
                
                # Plot profit if available
                if profit and len(profit) == len(months):
                    ax.plot(months, profit, marker='s', linestyle='-', color='blue', label='Profit')
                
                # Add horizontal line at y=0
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                
                # Add labels and title
                ax.set_xlabel('Month')
                ax.set_ylabel('Amount ($)')
                ax.set_title('Financial Projections')
                
                # Add grid and legend
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Save to bytes
                return self._fig_to_bytes(fig)
                
            return None
        except Exception as e:
            logger.error(f"Error creating financial chart: {e}")
            return None
    
    def _fig_to_bytes(self, fig) -> Optional[bytes]:
        """
        Convert a matplotlib figure to bytes.
        
        Args:
            fig: The matplotlib figure
            
        Returns:
            bytes: The image data
        """
        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            self.plt.close(fig)  # Close the figure to free memory
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Error converting figure to bytes: {e}")
            return None
    
    def _create_text_based_chart(self, chart_type: str, doc: Dict[str, Any]) -> Optional[bytes]:
        """
        Create a text-based representation of a chart when matplotlib is not available.
        
        Args:
            chart_type: The type of chart
            doc: The document data
            
        Returns:
            bytes: The image data for a text chart
        """
        try:
            # Create a very simple representation using PIL
            try:
                from PIL import Image, ImageDraw, ImageFont
                
                # Create a blank image with white background
                img = Image.new('RGB', (600, 400), color=(255, 255, 255))
                draw = ImageDraw.Draw(img)
                
                # Try to load a font, or use default
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                    title_font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                    title_font = ImageFont.load_default()
                
                # Add title
                draw.text((20, 20), f"{chart_type} Analysis", fill=(0, 0, 0), font=title_font)
                
                # Add chart-specific data
                y_pos = 60
                if chart_type == "CAMP Framework":
                    metrics = [
                        ("Capital Efficiency", doc.get("capital_score", 0)),
                        ("Market Dynamics", doc.get("market_score", 0)),
                        ("Advantage Moat", doc.get("advantage_score", 0)),
                        ("People & Performance", doc.get("people_score", 0))
                    ]
                    
                    for name, value in metrics:
                        draw.text((40, y_pos), f"{name}: {value:.1f}/100", fill=(0, 0, 0), font=font)
                        y_pos += 30
                
                # Add note about matplotlib
                draw.text((20, 350), "Note: Install matplotlib for enhanced visualizations", 
                          fill=(150, 150, 150), font=font)
                
                # Save to bytes
                buf = BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                return buf.getvalue()
                
            except ImportError:
                logger.warning("PIL not available for text-based charts")
                return None
                
        except Exception as e:
            logger.error(f"Error creating text-based chart: {e}")
            return None


class PDFReportGenerator:
    """
    Main report generator for creating comprehensive PDF reports.
    
    Coordinates the process of creating different report sections,
    generating charts, and assembling the final PDF.
    """
    
    def __init__(self):
        """Initialize the PDF report generator."""
        self.chart_generator = ChartGenerator()
    
    def generate_report(self, doc: Dict[str, Any], report_type: str = "full", 
                        sections: Optional[Dict[str, bool]] = None) -> Optional[bytes]:
        """
        Generate a comprehensive PDF report.
        
        Args:
            doc: The document data dictionary
            report_type: The type of report ("full", "executive", "custom")
            sections: Dictionary of sections to include if report_type is "custom"
            
        Returns:
            bytes: The PDF data or None if generation failed
        """
        try:
            # Start creating the PDF
            pdf = EnhancedReportPDF(doc=doc)
            
            # Add cover page
            pdf.create_cover_page()
            
            # Get the sections to include
            if report_type == "custom" and sections is not None:
                active_sections = sections
            else:
                # Default sections for full report
                active_sections = {
                    "Executive Summary": True,
                    "Business Model": True,
                    "Market Analysis": True,
                    "Financial Projections": True,
                    "Team Assessment": True,
                    "Competitive Analysis": True,
                    "Growth Metrics": True,
                    "Risk Assessment": True,
                    "Exit Strategy": True,
                    "Technical Assessment": True
                }
                
                # For executive report, limit sections
                if report_type == "executive":
                    for section in ["Growth Metrics", "Risk Assessment", "Exit Strategy", "Technical Assessment"]:
                        active_sections[section] = False
            
            # Add sections based on active_sections
            if active_sections.get("Executive Summary", True):
                self._add_executive_summary(pdf, doc)
            
            if active_sections.get("Business Model", True):
                self._add_business_model(pdf, doc)
            
            if active_sections.get("Market Analysis", True):
                self._add_market_analysis(pdf, doc)
            
            if active_sections.get("Financial Projections", True):
                self._add_financial_projections(pdf, doc)
            
            if active_sections.get("Team Assessment", True):
                self._add_team_assessment(pdf, doc)
            
            if active_sections.get("Competitive Analysis", True):
                self._add_competitive_analysis(pdf, doc)
            
            if active_sections.get("Growth Metrics", True):
                self._add_growth_metrics(pdf, doc)
            
            if active_sections.get("Risk Assessment", True):
                self._add_risk_assessment(pdf, doc)
            
            if active_sections.get("Exit Strategy", True):
                self._add_exit_strategy(pdf, doc)
            
            if active_sections.get("Technical Assessment", True):
                self._add_technical_assessment(pdf, doc)
            
            # Return the PDF as bytes
            return pdf.output(dest='S').encode('latin1')
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return self._create_error_pdf(doc, str(e))
    
    def _add_executive_summary(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Executive Summary section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Executive Summary")
        
        # Key metrics summary
        metrics = [
            ("CAMP Score", f"{doc.get('camp_score', 0):.1f}/100"),
            ("Success Probability", f"{doc.get('success_prob', 0):.1f}%"),
            ("Runway", f"{doc.get('runway_months', 0):.1f} months")
        ]
        pdf.add_metric_row(metrics)
        
        # CAMP Framework chart
        camp_chart = self.chart_generator.create_camp_radar_chart(doc)
        if camp_chart:
            pdf.add_chart(camp_chart)
        
        # CAMP scores table
        pdf.add_subsection_title("CAMP Framework Scores")
        headers = ["Dimension", "Score"]
        data = [
            ["Capital Efficiency", f"{doc.get('capital_score', 0):.1f}/100"],
            ["Market Dynamics", f"{doc.get('market_score', 0):.1f}/100"],
            ["Advantage Moat", f"{doc.get('advantage_score', 0):.1f}/100"],
            ["People & Performance", f"{doc.get('people_score', 0):.1f}/100"]
        ]
        pdf.add_table(headers, data)
        
        # Key strengths and weaknesses
        pdf.add_subsection_title("Key Strengths & Weaknesses")
        
        # Get patterns
        patterns = doc.get("patterns_matched", [])
        if patterns:
            # Strengths
            pdf.set_font(pdf.default_font, 'B', 10)
            pdf.set_text_color(*COLOR_SUCCESS)
            pdf.cell(0, 8, "Strengths:", 0, 1, 'L')
            
            # Display positive patterns
            positive_patterns = [p for p in patterns if isinstance(p, dict) and p.get("is_positive", True)]
            for i, pattern in enumerate(positive_patterns[:3]):  # Show top 3
                pdf.set_font(pdf.default_font, '', 10)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 6, f"• {pattern.get('name', '')}", 0, 1, 'L')
            
            pdf.ln(3)
            
            # Weaknesses
            pdf.set_font(pdf.default_font, 'B', 10)
            pdf.set_text_color(*COLOR_WARNING)
            pdf.cell(0, 8, "Weaknesses:", 0, 1, 'L')
            
            # Display negative patterns
            negative_patterns = [p for p in patterns if isinstance(p, dict) and not p.get("is_positive", True)]
            for i, pattern in enumerate(negative_patterns[:3]):  # Show top 3
                pdf.set_font(pdf.default_font, '', 10)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 6, f"• {pattern.get('name', '')}", 0, 1, 'L')
    
    def _add_business_model(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Business Model section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Business Model")
        
        # Business model description
        business_model = doc.get("business_model", "")
        if business_model:
            pdf.add_paragraph(business_model)
        
        # Unit economics
        unit_econ = doc.get("unit_economics", {})
        if unit_econ and isinstance(unit_econ, dict):
            pdf.add_subsection_title("Unit Economics")
            
            # Unit economics metrics
            metrics = [
                ("LTV", f"${unit_econ.get('ltv', 0):,.2f}"),
                ("CAC", f"${unit_econ.get('cac', 0):,.2f}"),
                ("LTV:CAC Ratio", f"{unit_econ.get('ltv_cac_ratio', 0):.2f}")
            ]
            pdf.add_metric_row(metrics)
            
            metrics2 = [
                ("Gross Margin", f"{unit_econ.get('gross_margin', 0)*100:.1f}%"),
                ("CAC Payback", f"{unit_econ.get('cac_payback_months', 0):.1f} months"),
                ("Contribution Margin", f"{unit_econ.get('contribution_margin', 0)*100:.1f}%")
            ]
            pdf.add_metric_row(metrics2)
    
    def _add_market_analysis(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Market Analysis section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Market Analysis")
        
        # Market metrics
        metrics = [
            ("Market Size", f"${doc.get('market_size', 0)/1e6:.1f}M"),
            ("Market Growth Rate", f"{doc.get('market_growth_rate', 0):.1f}%/yr"),
            ("Market Share", f"{doc.get('market_share', 0):.2f}%")
        ]
        pdf.add_metric_row(metrics)
        
        # User growth metrics
        metrics2 = [
            ("User Growth Rate", f"{doc.get('user_growth_rate', 0):.1f}%/mo"),
            ("Churn Rate", f"{doc.get('churn_rate', 0):.1f}%/mo"),
            ("Viral Coefficient", f"{doc.get('viral_coefficient', 0):.2f}")
        ]
        pdf.add_metric_row(metrics2)
        
        # User growth chart
        growth_chart = self.chart_generator.create_growth_chart(doc)
        if growth_chart:
            pdf.add_subsection_title("User Growth Projection")
            pdf.add_chart(growth_chart)
        
        # PMF analysis
        pmf = doc.get("pmf_analysis", {})
        if pmf and isinstance(pmf, dict):
            pdf.add_subsection_title("Product-Market Fit")
            
            metrics3 = [
                ("PMF Score", f"{pmf.get('pmf_score', 0):.1f}/100"),
                ("PMF Stage", f"{pmf.get('stage', '')}"),
                ("Engagement Score", f"{pmf.get('engagement_score', 0):.1f}/100")
            ]
            pdf.add_metric_row(metrics3)
    
    def _add_financial_projections(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Financial Projections section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Financial Projections")
        
        # Financial metrics
        metrics = [
            ("Monthly Revenue", f"${doc.get('monthly_revenue', 0):,.2f}"),
            ("Burn Rate", f"${doc.get('burn_rate', 0):,.2f}"),
            ("Runway", f"{doc.get('runway_months', 0):.1f} months")
        ]
        pdf.add_metric_row(metrics)
        
        # Financial chart
        financial_chart = self.chart_generator.create_financial_chart(doc)
        if financial_chart:
            pdf.add_chart(financial_chart)
            
        # Cash flow data
        if doc.get("cash_flow"):
            pdf.add_subsection_title("Cash Flow Projection")
            
            # Calculate some periods of cash flow for display
            cash_flow = doc.get("cash_flow", [])
            periods = min(len(cash_flow), 6)  # Show first 6 months
            
            headers = ["Month"] + [str(i) for i in range(periods)]
            data = [["Cash"] + [f"${cash_flow[i]:,.0f}" for i in range(periods)]]
            
            pdf.add_table(headers, data)
            
        # Valuation metrics
        valuation = doc.get("valuation_metrics", {})
        if valuation and isinstance(valuation, dict):
            pdf.add_subsection_title("Valuation Estimates")
            
            try:
                val_metrics = [
                    ("Revenue Multiple", f"{valuation.get('revenue_multiple', 0):.1f}x"),
                    ("DCF Valuation", f"${valuation.get('dcf_valuation', 0)/1e6:.1f}M"),
                    ("Comparables", f"${valuation.get('comparable_valuation', 0)/1e6:.1f}M")
                ]
                pdf.add_metric_row(val_metrics)
            except Exception as e:
                logger.error(f"Error adding valuation metrics: {e}")
                pdf.add_paragraph("Valuation metrics not available")
    
    def _add_team_assessment(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Team Assessment section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Team Assessment")
        
        # Team metrics
        metrics = [
            ("Team Score", f"{doc.get('team_score', 0):.1f}/100"),
            ("Founder Experience", f"{doc.get('founder_domain_exp_yrs', 0)} years"),
            ("Previous Exits", f"{doc.get('founder_exits', 0)}")
        ]
        pdf.add_metric_row(metrics)
        
        metrics2 = [
            ("Team Size", f"{doc.get('employee_count', 0)} employees"),
            ("Tech Talent Ratio", f"{doc.get('tech_talent_ratio', 0)*100:.1f}%"),
            ("Team Diversity", f"{doc.get('founder_diversity_score', 0):.1f}/100")
        ]
        pdf.add_metric_row(metrics2)
        
        # Leadership presence
        pdf.add_subsection_title("Leadership Team")
        
        leadership = {
            "CEO": True,  # Assumed always present
            "CTO": doc.get("has_cto", False),
            "CMO": doc.get("has_cmo", False),
            "CFO": doc.get("has_cfo", False)
        }
        
        headers = list(leadership.keys())
        data = [["Yes" if v else "No" for v in leadership.values()]]
        
        pdf.add_table(headers, data)
        
        # Execution risk
        execution_risk = doc.get("execution_risk", {})
        if execution_risk and isinstance(execution_risk, dict) and "risk_factors" in execution_risk:
            pdf.add_subsection_title("Execution Risk Factors")
            
            risk_factors = execution_risk.get("risk_factors", {})
            
            if risk_factors:
                headers = ["Risk Factor", "Score"]
                data = [[factor, f"{score:.1f}/100"] for factor, score in risk_factors.items()]
                
                pdf.add_table(headers, data)
    
    def _add_competitive_analysis(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Competitive Analysis section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Competitive Analysis")
        
        # Competitors
        competitors = doc.get("competitors", [])
        if competitors and all(isinstance(comp, dict) for comp in competitors):
            pdf.add_subsection_title("Key Competitors")
            
            headers = ["Competitor", "Market Share", "Growth Rate"]
            data = []
            
            for comp in competitors:
                data.append([
                    comp.get("name", ""),
                    f"{comp.get('market_share', 0)*100:.1f}%",
                    f"{comp.get('growth_rate', 0)*100:.1f}%"
                ])
            
            pdf.add_table(headers, data)
        
        # Competitive positioning chart
        comp_chart = self.chart_generator.create_competitor_chart(doc)
        if comp_chart:
            pdf.add_chart(comp_chart)
            
        # Competitive positioning
        positioning = doc.get("competitive_positioning", {})
        if positioning:
            pdf.add_subsection_title("Competitive Position")
            
            position = positioning.get("position", "")
            if position:
                pdf.add_paragraph(f"Market Position: {position}")
            
            # Advantages and disadvantages
            advantages = positioning.get("advantages", [])
            disadvantages = positioning.get("disadvantages", [])
            
            if advantages:
                pdf.set_font(pdf.default_font, 'B', 10)
                pdf.set_text_color(*COLOR_SUCCESS)
                pdf.cell(0, 8, "Competitive Advantages:", 0, 1, 'L')
                
                for adv in advantages:
                    if isinstance(adv, dict):
                        pdf.set_font(pdf.default_font, '', 10)
                        pdf.set_text_color(0, 0, 0)
                        pdf.cell(0, 6, f"• {adv.get('name', '')}", 0, 1, 'L')
            
            if disadvantages:
                pdf.set_font(pdf.default_font, 'B', 10)
                pdf.set_text_color(*COLOR_WARNING)
                pdf.cell(0, 8, "Competitive Disadvantages:", 0, 1, 'L')
                
                for disadv in disadvantages:
                    if isinstance(disadv, dict):
                        pdf.set_font(pdf.default_font, '', 10)
                        pdf.set_text_color(0, 0, 0)
                        pdf.cell(0, 6, f"• {disadv.get('name', '')}", 0, 1, 'L')
    
    def _add_growth_metrics(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Growth Metrics section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Growth Metrics")
        
        # Growth metrics
        growth_metrics = doc.get("growth_metrics", {})
        if growth_metrics and isinstance(growth_metrics, dict):
            # Try to get viral, organic, and paid growth metrics
            viral_growth = growth_metrics.get("viral_growth", 0)
            organic_growth = growth_metrics.get("organic_growth", 0)
            paid_growth = growth_metrics.get("paid_growth", 0)
            
            metrics = [
                ("Viral Growth", f"{viral_growth:.1f}%"),
                ("Organic Growth", f"{organic_growth:.1f}%"),
                ("Paid Growth", f"{paid_growth:.1f}%")
            ]
            pdf.add_metric_row(metrics)
            
            # Acquisition channels
            channels = growth_metrics.get("acquisition_channels", {})
            if channels:
                pdf.add_subsection_title("Acquisition Channels")
                
                headers = ["Channel", "Contribution"]
                data = [[channel, f"{contribution:.1f}%"] for channel, contribution in channels.items()]
                
                pdf.add_table(headers, data)
    
    def _add_risk_assessment(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Risk Assessment section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Risk Assessment")
        
        # Risk factors
        risk_factors = doc.get("risk_factors", {})
        if risk_factors:
            headers = ["Risk Factor", "Level", "Impact"]
            data = []
            
            for factor, details in risk_factors.items():
                if isinstance(details, dict):
                    data.append([
                        factor,
                        f"High" if details.get('probability', 0) > 0.6 else 
                        ("Medium" if details.get('probability', 0) > 0.3 else "Low"),
                        f"High" if details.get('impact', 0) > 0.6 else 
                        ("Medium" if details.get('impact', 0) > 0.3 else "Low")
                    ])
                else:
                    # If details is just a value
                    data.append([
                        factor,
                        f"High" if float(details) > 60 else 
                        ("Medium" if float(details) > 30 else "Low"),
                        "Medium"
                    ])
            
            if data:
                pdf.add_table(headers, data)
    
    def _add_exit_strategy(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Exit Strategy section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Exit Strategy")
        
        # Exit analysis
        exit_analysis = doc.get("exit_path_analysis", {})
        exit_recs = doc.get("exit_recommendations", {})
        
        if exit_analysis or exit_recs:
            # Exit readiness
            readiness = exit_analysis.get("exit_readiness_score", 0)
            pdf.add_metric("Exit Readiness Score", f"{readiness:.1f}/100")
            
            # Optimal exit path
            if exit_recs:
                optimal_path = exit_recs.get("optimal_path", "")
                path_details = exit_recs.get("path_details", {})
                
                if optimal_path:
                    pdf.add_subsection_title("Optimal Exit Path")
                    if isinstance(path_details, dict) and "description" in path_details:
                        pdf.add_paragraph(path_details.get("description", optimal_path))
                    else:
                        pdf.add_paragraph(optimal_path)
                
                # Exit timeline
                timeline = exit_recs.get("timeline", {})
                if timeline and isinstance(timeline, dict):
                    pdf.add_subsection_title("Exit Timeline")
                    
                    metrics = [
                        ("Years to Exit", f"{timeline.get('years_to_exit', 0):.1f}"),
                        ("Exit Year", f"{timeline.get('exit_year', 0)}"),
                        ("Exit Valuation", f"${timeline.get('exit_valuation', 0)/1e6:.1f}M")
                    ]
                    pdf.add_metric_row(metrics)
            
            # Exit scenarios
            scenarios = exit_analysis.get("scenarios", [])
            if scenarios and all(isinstance(s, dict) for s in scenarios):
                pdf.add_subsection_title("Exit Path Scenarios")
                
                headers = ["Exit Path", "Valuation ($M)", "Probability", "Time (Years)"]
                data = []
                
                for scenario in scenarios:
                    data.append([
                        scenario.get("path_name", ""),
                        f"{scenario.get('exit_valuation', 0)/1e6:.1f}",
                        f"{scenario.get('probability', 0)*100:.1f}%",
                        f"{scenario.get('time_to_exit', 0):.1f}"
                    ])
                
                pdf.add_table(headers, data)
    
    def _add_technical_assessment(self, pdf: EnhancedReportPDF, doc: Dict[str, Any]) -> None:
        """
        Add the Technical Assessment section to the PDF.
        
        Args:
            pdf: The PDF document
            doc: The document data dictionary
        """
        pdf.add_page()
        pdf.add_section_title("Technical Assessment")
        
        # Tech assessment
        tech_assessment = doc.get("tech_assessment", {})
        
        if tech_assessment and isinstance(tech_assessment, dict):
            # Overall tech score
            tech_score = tech_assessment.get("overall_score", 0)
            pdf.add_metric("Technical Assessment Score", f"{tech_score:.1f}/100")
            
            # Component scores
            scores = tech_assessment.get("scores", {})
            if scores and isinstance(scores, dict):
                pdf.add_subsection_title("Component Scores")
                
                headers = ["Component", "Score"]
                data = [[component, f"{score:.1f}/100"] for component, score in scores.items()]
                
                pdf.add_table(headers, data)
            
            # Tech stack
            tech_stack = tech_assessment.get("tech_stack", {})
            if tech_stack:
                pdf.add_subsection_title("Technology Stack")
                
                if isinstance(tech_stack, dict):
                    for category, technologies in tech_stack.items():
                        pdf.set_font(pdf.default_font, 'B', 10)
                        pdf.cell(0, 8, category, 0, 1, 'L')
                        
                        if isinstance(technologies, list):
                            tech_str = ", ".join(technologies)
                        else:
                            tech_str = str(technologies)
                        
                        pdf.set_font(pdf.default_font, '', 10)
                        pdf.cell(0, 6, tech_str, 0, 1, 'L')
                        pdf.ln(2)
                else:
                    pdf.add_paragraph(str(tech_stack))
            
            # Technical recommendations
            recommendations = tech_assessment.get("recommendations", [])
            if recommendations:
                pdf.add_subsection_title("Technical Recommendations")
                
                for i, rec in enumerate(recommendations):
                    pdf.add_paragraph(f"{i+1}. {rec}")
    
    def _create_error_pdf(self, doc: Dict[str, Any], error_message: str) -> bytes:
        """
        Create a simple PDF with error information when the main report fails.
        
        Args:
            doc: The document data dictionary
            error_message: The error message
            
        Returns:
            bytes: The PDF data
        """
        try:
            # Create a basic PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f"{doc.get('name', 'Startup')} - Investor Report", 0, 1, 'C')
            
            # Add basic info
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f"CAMP Score: {doc.get('camp_score', 0):.1f}/100", 0, 1)
            pdf.cell(0, 10, f"Success Probability: {doc.get('success_prob', 0):.1f}%", 0, 1)
            pdf.cell(0, 10, f"Runway: {doc.get('runway_months', 0):.1f} months", 0, 1)
            
            # CAMP breakdown
            pdf.cell(0, 10, "", 0, 1)  # Empty line
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "CAMP Framework Scores", 0, 1)
            
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f"Capital Efficiency: {doc.get('capital_score', 0):.1f}/100", 0, 1)
            pdf.cell(0, 10, f"Market Dynamics: {doc.get('market_score', 0):.1f}/100", 0, 1)
            pdf.cell(0, 10, f"Advantage Moat: {doc.get('advantage_score', 0):.1f}/100", 0, 1)
            pdf.cell(0, 10, f"People & Performance: {doc.get('people_score', 0):.1f}/100", 0, 1)
            
            # Error message
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 10, "", 0, 1)
            pdf.multi_cell(0, 5, f"Note: Enhanced report generation encountered an error: {error_message}. This is a simplified fallback report.")
            
            return pdf.output(dest='S').encode('latin1')
            
        except Exception as fallback_e:
            logger.error(f"Error creating fallback error PDF: {fallback_e}")
            # If even the fallback fails, return a minimal bytes object to avoid breaking the UI
            return b'%PDF-1.3\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R/Resources<<>>/Contents 4 0 R>>\nendobj\n4 0 obj\n<</Length 22>>stream\nBT\n/F1 12 Tf\n100 700 Td\n(Error generating report) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000056 00000 n \n0000000111 00000 n \n0000000212 00000 n \ntrailer\n<</Size 5/Root 1 0 R>>\nstartxref\n285\n%%EOF\n'

# Simplified public function for use in streamlit
def generate_enhanced_pdf(doc: Dict[str, Any], report_type: str = "full", 
                         sections: Optional[Dict[str, bool]] = None) -> Optional[bytes]:
    """
    Generate an enhanced PDF report with charts, graphs, and proper formatting.
    
    Args:
        doc: The document data dictionary
        report_type: The type of report ("full", "executive", "custom")
        sections: Dictionary of sections to include if report_type is "custom"
        
    Returns:
        bytes: The PDF data or None if generation failed
    """
    generator = PDFReportGenerator()
    return generator.generate_report(doc, report_type, sections) 