"""
Enhanced PDF Generator Module
"""
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import traceback

logger = logging.getLogger("pdf_generator")

# Define colors for consistent styling
COLOR_PRIMARY = (31, 119, 180)     # Blue
COLOR_SECONDARY = (255, 127, 14)   # Orange
COLOR_SUCCESS = (44, 160, 44)      # Green
COLOR_WARNING = (214, 39, 40)      # Red
COLOR_BACKGROUND = (248, 248, 248) # Light gray


class EnhancedReportPDF(FPDF):
    def __init__(self, title="Investor Report", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(True, margin=15)
        
    def header(self):
        if self.page_no() == 1:
            return  # Skip header on cover page
        
        # Header with logo if available
        self.set_font('Arial', 'B', 10)
        self.cell(30, 10, f"{self.doc_name}", 0, 0, 'L')
        
        # Title in the middle
        self.cell(self.w - 60, 10, self.title, 0, 0, 'C')
        
        # Date on the right
        self.set_font('Arial', 'I', 8)
        self.cell(30, 10, datetime.now().strftime('%Y-%m-%d'), 0, 0, 'R')
        
        # Line break and separator
        self.ln(12)
        self.set_draw_color(*COLOR_PRIMARY)
        self.line(15, 20, self.w - 15, 20)
        self.ln(5)
        
    def footer(self):
        if self.page_no() == 1:
            return  # Skip footer on cover page
            
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        
        # Draw line
        self.set_draw_color(*COLOR_PRIMARY)
        self.line(15, self.h - 15, self.w - 15, self.h - 15)
        
        # Page number
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def create_cover_page(self, doc):
        """Create a stylish cover page with logo."""
        self.doc_name = doc.get('name', 'Startup')
        self.add_page()
        
        # Try to get the logo from static directory
        logo_path = "static/logo.png"
        logo_height = 40
        
        # Add logo if exists
        if os.path.exists(logo_path):
            # Center the logo
            logo_width = 0
            try:
                # Get image dimensions
                import PIL.Image
                img = PIL.Image.open(logo_path)
                width, height = img.size
                ratio = height / logo_height
                logo_width = width / ratio
                
                # Center the logo
                x_pos = (self.w - logo_width) / 2
                self.image(logo_path, x=x_pos, y=30, h=logo_height)
            except Exception as e:
                logging.error(f"Error loading logo: {e}")
        
        # Add title with sufficient spacing
        self.set_font('Arial', 'B', 24)
        self.set_text_color(*COLOR_PRIMARY)
        self.ln(60)  # Space after logo
        self.cell(0, 20, "Investor Report", 0, 1, 'C')
        
        # Company name
        self.set_font('Arial', 'B', 28)
        self.set_text_color(0, 0, 0)
        self.cell(0, 20, doc.get('name', 'Startup'), 0, 1, 'C')
        
        # Information block
        self.ln(10)
        self.set_font('Arial', '', 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, f"Sector: {doc.get('sector', 'Technology')}", 0, 1, 'C')
        self.cell(0, 10, f"Stage: {doc.get('stage', 'Growth')}", 0, 1, 'C')
        
        # CAMP Score
        self.ln(15)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 10, f"CAMP Score: {doc.get('camp_score', 0):.1f}/100", 0, 1, 'C')
        
        # Date at bottom
        self.set_y(-50)
        self.set_font('Arial', 'I', 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%B %d, %Y')}", 0, 1, 'C')
        
        # Confidentiality notice
        self.set_y(-30)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, "CONFIDENTIAL", 0, 1, 'C')
        self.multi_cell(0, 4, "This report contains confidential information about the company and is intended only for the named recipient. If you are not the intended recipient, please notify the sender immediately.")
    
    def add_section_title(self, title):
        """Add a styled section title."""
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*COLOR_PRIMARY)
        self.set_fill_color(*COLOR_BACKGROUND)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)
        
    def add_subsection_title(self, title):
        """Add a styled subsection title."""
        self.set_font('Arial', 'B', 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
        
    def add_paragraph(self, text):
        """Add a paragraph with proper styling."""
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)
        self.ln(3)
        
    def add_metric(self, label, value, width=90):
        """Add a styled metric label and value."""
        self.set_font('Arial', 'B', 10)
        self.set_text_color(80, 80, 80)
        self.cell(width, 8, label, 0, 0, 'L')
        
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, str(value), 0, 1, 'L')
        
    def add_metric_row(self, metrics):
        """Add a row of metrics with equal spacing."""
        col_width = (self.w - self.l_margin - self.r_margin) / len(metrics)
        
        for label, value in metrics:
            self.set_font('Arial', 'B', 10)
            self.set_text_color(80, 80, 80)
            self.cell(col_width, 8, label, 0, 0, 'L')
        
        self.ln()
        
        for label, value in metrics:
            self.set_font('Arial', '', 10)
            self.set_text_color(0, 0, 0)
            self.cell(col_width, 8, str(value), 0, 0, 'L')
            
        self.ln(12)
        
    def add_table(self, headers, data):
        """Add a styled table with headers and data."""
        # Calculate column widths
        col_width = (self.w - self.l_margin - self.r_margin) / len(headers)
        row_height = 7
        
        # Add headers
        self.set_font('Arial', 'B', 10)
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
                self.set_font('Arial', '', 9)
                self.cell(col_width, row_height, str(cell), 1, 0, 'L', 1)
            self.ln()
        
        self.ln(5)
        
    def add_chart(self, plt_figure):
        """Add a matplotlib figure as chart."""
        if plt_figure is None:
            self.add_paragraph("Chart generation not available - matplotlib required")
            return
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            try:
                plt_figure.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                plt.close(plt_figure)  # Close the figure to free memory
                
                # Add image with center alignment
                img_width = 170
                x_pos = (self.w - img_width) / 2
                self.image(tmp.name, x=x_pos, y=self.get_y(), w=img_width)
                
                # Add space after chart
                self.ln(100)  # Extra space after chart
                
            except Exception as e:
                logging.error(f"Error saving or displaying chart: {e}")
                self.add_paragraph(f"Error generating chart visualization: {str(e)}")
            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp.name)
                except Exception as e:
                    logging.error(f"Error removing temp chart file: {e}")


# Helper function to create CAMP radar chart    
def create_camp_radar_chart(doc):
    try:
        # Calculate CAMP scores
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
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories[:-1], color='grey', size=10)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=8)
        plt.ylim(0, 100)
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
        
        # Fill area
        ax.fill(angles, values, 'blue', alpha=0.1)
        
        # Add title
        plt.title("CAMP Framework Analysis", size=14, color='blue', y=1.1)
        
        return fig
    except Exception as e:
        logging.error(f"Error creating CAMP radar chart: {e}")
        return None

# Helper function to create growth chart
def create_growth_chart(doc):
    try:
        sys_dynamics = doc.get("system_dynamics", {})
        if isinstance(sys_dynamics, dict) and "users" in sys_dynamics:
            users = sys_dynamics.get("users", [])
            months = list(range(len(users)))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(months, users, marker='o', linestyle='-', color='blue')
            
            # Add labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Users')
            ax.set_title('User Growth Projection')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add trend line
            if len(users) > 1:
                z = np.polyfit(months, users, 1)
                p = np.poly1d(z)
                ax.plot(months, p(months), "r--", alpha=0.8)
            
            return fig
        return None
    except Exception as e:
        logging.error(f"Error creating growth chart: {e}")
        return None
        
# Helper function to create competitor chart
def create_competitor_chart(doc):
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
                fig, ax = plt.subplots(figsize=(8, 6))
                
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
                
                return fig
        return None
    except Exception as e:
        logging.error(f"Error creating competitor chart: {e}")
        return None
        
# Helper function to create financial chart
def create_financial_chart(doc):
    try:
        forecast = doc.get("financial_forecast", {})
        
        if forecast and "months" in forecast and "revenue" in forecast:
            months = forecast.get("months", [])
            revenue = forecast.get("revenue", [])
            profit = forecast.get("profit", []) if "profit" in forecast else None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 4))
            
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
            
            return fig
        return None
    except Exception as e:
        logging.error(f"Error creating financial chart: {e}")
        return None


def generate_enhanced_pdf(doc, report_type="full", sections=None):
    """Generate an enhanced PDF report with charts, graphs, and better formatting."""
    try:
        # Start creating the PDF
        pdf = EnhancedReportPDF()
        
        # Create a deep copy of the doc to avoid modifying the original
        doc_copy = copy.deepcopy(doc)
        
        # Add cover page
        pdf.create_cover_page(doc_copy)
        
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
        
        # Executive Summary
        if active_sections.get("Executive Summary", True):
            pdf.add_page()
            pdf.add_section_title("Executive Summary")
            
            # Key metrics summary
            metrics = [
                ("CAMP Score", f"{doc_copy.get('camp_score', 0):.1f}/100"),
                ("Success Probability", f"{doc_copy.get('success_prob', 0):.1f}%"),
                ("Runway", f"{doc_copy.get('runway_months', 0):.1f} months")
            ]
            pdf.add_metric_row(metrics)
            
            # CAMP Framework chart
            camp_fig = create_camp_radar_chart(doc_copy)
            if camp_fig:
                pdf.add_chart(camp_fig)
            
            # CAMP scores table
            pdf.add_subsection_title("CAMP Framework Scores")
            headers = ["Dimension", "Score"]
            data = [
                ["Capital Efficiency", f"{doc_copy.get('capital_score', 0):.1f}/100"],
                ["Market Dynamics", f"{doc_copy.get('market_score', 0):.1f}/100"],
                ["Advantage Moat", f"{doc_copy.get('advantage_score', 0):.1f}/100"],
                ["People & Performance", f"{doc_copy.get('people_score', 0):.1f}/100"]
            ]
            pdf.add_table(headers, data)
            
            # Key strengths and weaknesses
            pdf.add_subsection_title("Key Strengths & Weaknesses")
            
            # Get patterns
            patterns = doc_copy.get("patterns_matched", [])
            if patterns:
                # Strengths
                pdf.set_font('Arial', 'B', 10)
                pdf.set_text_color(*COLOR_SUCCESS)
                pdf.cell(0, 8, "Strengths:", 0, 1, 'L')
                
                # Display positive patterns
                positive_patterns = [p for p in patterns if isinstance(p, dict) and p.get("is_positive", True)]
                for i, pattern in enumerate(positive_patterns[:3]):  # Show top 3
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 6, f"• {pattern.get('name', '')}", 0, 1, 'L')
                
                pdf.ln(3)
                
                # Weaknesses
                pdf.set_font('Arial', 'B', 10)
                pdf.set_text_color(*COLOR_WARNING)
                pdf.cell(0, 8, "Weaknesses:", 0, 1, 'L')
                
                # Display negative patterns
                negative_patterns = [p for p in patterns if isinstance(p, dict) and not p.get("is_positive", True)]
                for i, pattern in enumerate(negative_patterns[:3]):  # Show top 3
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 6, f"• {pattern.get('name', '')}", 0, 1, 'L')
        
        # Only add essential sections to keep the file size manageable
        # Business Model 
        if active_sections.get("Business Model", True):
            pdf.add_page()
            pdf.add_section_title("Business Model")
            
            # Business model description
            business_model = doc_copy.get("business_model", "")
            if business_model:
                pdf.add_paragraph(business_model)
            
            # Unit economics
            unit_econ = doc_copy.get("unit_economics", {})
            if unit_econ:
                pdf.add_subsection_title("Unit Economics")
                
                # Unit economics metrics
                metrics = [
                    ("LTV", f"${unit_econ.get('ltv', 0):,.2f}"),
                    ("CAC", f"${unit_econ.get('cac', 0):,.2f}"),
                    ("LTV:CAC Ratio", f"{unit_econ.get('ltv_cac_ratio', 0):.2f}")
                ]
                pdf.add_metric_row(metrics)
        
        # Market Analysis
        if active_sections.get("Market Analysis", True):
            pdf.add_page()
            pdf.add_section_title("Market Analysis")
            
            # Market metrics
            metrics = [
                ("Market Size", f"${doc_copy.get('market_size', 0)/1e6:.1f}M"),
                ("Market Growth Rate", f"{doc_copy.get('market_growth_rate', 0):.1f}%/yr"),
                ("Market Share", f"{doc_copy.get('market_share', 0):.2f}%")
            ]
            pdf.add_metric_row(metrics)
            
            # User growth chart
            growth_fig = create_growth_chart(doc_copy)
            if growth_fig:
                pdf.add_subsection_title("User Growth Projection")
                pdf.add_chart(growth_fig)
        
        # Financial Projections
        if active_sections.get("Financial Projections", True):
            pdf.add_page()
            pdf.add_section_title("Financial Projections")
            
            # Financial metrics
            metrics = [
                ("Monthly Revenue", f"${doc_copy.get('monthly_revenue', 0):,.2f}"),
                ("Burn Rate", f"${doc_copy.get('burn_rate', 0):,.2f}"),
                ("Runway", f"{doc_copy.get('runway_months', 0):.1f} months")
            ]
            pdf.add_metric_row(metrics)
            
            # Financial chart
            financial_fig = create_financial_chart(doc_copy)
            if financial_fig:
                pdf.add_chart(financial_fig)
        
        # Competitive Analysis
        if active_sections.get("Competitive Analysis", True):
            pdf.add_page()
            pdf.add_section_title("Competitive Analysis")
            
            # Competitive positioning chart
            comp_fig = create_competitor_chart(doc_copy)
            if comp_fig:
                pdf.add_chart(comp_fig)
        
        # Return the PDF as bytes
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        logging.error(f"Error generating enhanced PDF: {traceback.format_exc()}")
        
        # Create a simple fallback PDF with just basic information
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
            
            # Error message
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 10, "", 0, 1)
            pdf.multi_cell(0, 5, f"Note: Enhanced report generation encountered an error: {str(e)}. This is a simplified fallback report.")
            
            return pdf.output(dest='S').encode('latin1')
        except Exception as fallback_e:
            logging.error(f"Fallback PDF generation also failed: {fallback_e}")
            return None 