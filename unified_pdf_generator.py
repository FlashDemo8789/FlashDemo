"""
Unified PDF Generator for FlashDNA

This module consolidates all PDF generation functionality into a single, 
well-organized file with proper fallbacks and error handling.
"""

import logging
import tempfile
import os
import io
import traceback
import copy
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

# Configure logging
logger = logging.getLogger("unified_pdf")
logger.setLevel(logging.INFO)

# Color constants for consistent styling
COLOR_PRIMARY = (31, 119, 180)  # Blue
COLOR_SECONDARY = (255, 127, 14)  # Orange
COLOR_SUCCESS = (44, 160, 44)  # Green
COLOR_WARNING = (214, 39, 40)  # Red
COLOR_BACKGROUND = (248, 248, 248)  # Light gray

class ReportPDF(FPDF):
    """Enhanced PDF class with support for charts and rich formatting."""
    
    def __init__(self, title="Investor Report", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        self.company_name = "Startup"
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(True, margin=15)
        self.has_cover = False
        self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
        
    def header(self):
        # Skip header on cover page
        if self.page_no() == 1 and self.has_cover:
            return
        
        # Header with logo if available
        self.set_font('Arial', 'B', 10)
        self.cell(30, 10, self.company_name, 0, 0, 'L')
        
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
        # Skip footer on cover page
        if self.page_no() == 1 and self.has_cover:
            return
            
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        
        # Draw line
        self.set_draw_color(*COLOR_PRIMARY)
        self.line(15, self.h - 15, self.w - 15, self.h - 15)
        
        # Page number
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def create_cover_page(self, doc):
        """Create a stylish cover page with logo and key metrics."""
        self.company_name = doc.get('name', 'Startup')
        self.add_page()
        self.has_cover = True
        
        # Try to get the logo from static directory
        logo_path = "static/logo.png"
        logo_height = 40
        
        # Add logo if exists
        if os.path.exists(logo_path):
            # Center the logo
            try:
                x_pos = (self.w - 40)/2
                self.image(logo_path, x=x_pos, y=30, h=logo_height)
            except Exception as e:
                logger.error(f"Error loading logo: {e}")
        
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
        
        # Add key metrics in a nice grid
        self.ln(5)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(80, 80, 80)
        
        # Draw metric boxes
        metrics = [
            ("Success Probability", f"{doc.get('success_prob', 0):.1f}%"),
            ("Runway", f"{doc.get('runway_months', 0):.1f} months"),
            ("Monthly Revenue", f"${doc.get('monthly_revenue', 0):,.0f}")
        ]
        
        self.ln(5)
        # Create a centered metrics display
        x_start = (self.w - 150) / 2
        self.set_xy(x_start, self.y)
        
        for i, (label, value) in enumerate(metrics):
            # Draw box
            self.set_fill_color(*COLOR_BACKGROUND)
            self.set_draw_color(*COLOR_PRIMARY)
            self.rect(x_start + (i * 50), self.y, 45, 25, 'DF')
            
            # Add label
            self.set_xy(x_start + (i * 50), self.y + 5)
            self.set_font('Arial', 'B', 8)
            self.set_text_color(80, 80, 80)
            self.cell(45, 5, label, 0, 1, 'C')
            
            # Add value
            self.set_xy(x_start + (i * 50), self.y + 5)
            self.set_font('Arial', 'B', 12)
            self.set_text_color(*COLOR_PRIMARY)
            self.cell(45, 10, value, 0, 1, 'C')
        
        # Reset position
        self.ln(30)
        
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
        self.multi_cell(0, 4, "This report contains confidential information about the company and is intended only for the named recipient.")
    
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
        """Add a row of metrics with equal spacing and styled boxes."""
        col_width = (self.w - self.l_margin - self.r_margin) / len(metrics)
        
        # Calculate positions
        base_y = self.get_y()
        box_height = 25
        
        for i, (label, value) in enumerate(metrics):
            x_pos = self.l_margin + (i * col_width)
            
            # Draw box
            self.set_fill_color(*COLOR_BACKGROUND)
            self.set_draw_color(*COLOR_PRIMARY)
            self.rect(x_pos, base_y, col_width - 2, box_height, 'DF')
            
            # Add label
            self.set_xy(x_pos, base_y + 5)
            self.set_font('Arial', 'B', 9)
            self.set_text_color(80, 80, 80)
            self.cell(col_width - 2, 5, label, 0, 1, 'C')
            
            # Add value
            self.set_xy(x_pos, base_y + 12)
            self.set_font('Arial', 'B', 11)
            self.set_text_color(*COLOR_PRIMARY)
            self.cell(col_width - 2, 10, value, 0, 1, 'C')
            
        # Move to next line
        self.set_y(base_y + box_height + 5)
        
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
    
    def add_radar_chart(self, categories, values, title="CAMP Framework Scores"):
        """Add a radar chart to the PDF."""
        try:
            # Create radar chart with matplotlib
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, polar=True)
            
            # Ensure the plot forms a complete circle by appending the first value at the end
            values = np.array(values + [values[0]])
            categories = categories + [categories[0]]
            
            # Compute angle for each category (in radians)
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            angles = np.append(angles, angles[0])
            
            # Draw the chart
            ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
            ax.fill(angles, values, alpha=0.25, color='#1f77b4')
            
            # Set category labels
            ax.set_thetagrids(angles * 180/np.pi, categories)
            
            # Set radial limits
            ax.set_ylim(0, 100)
            
            # Add gridlines
            ax.grid(True)
            
            # Set title
            plt.title(title, size=14, y=1.1)
            
            # Save to BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            
            # Get the current Y position
            current_y = self.get_y()
            
            # Add title
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'C')
            
            # Add image
            self.image(buf, x=(self.w - 100)/2, y=current_y + 10, w=100)
            
            # Move down after the chart
            self.set_y(current_y + 120)
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {str(e)}")
            self.add_paragraph(f"Error creating radar chart: {str(e)}")
    
    def add_bar_chart(self, categories, values, title="", ylabel="Value", colors=None):
        """Add a bar chart to the PDF."""
        try:
            # Create bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Generate colors if not provided
            if colors is None:
                colors = ['#1f77b4'] * len(categories)
                
            # Create bars
            bars = ax.bar(categories, values, color=colors)
            
            # Customize the chart
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Save to BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            
            # Get the current Y position
            current_y = self.get_y()
            
            # Add title if not already in the chart
            if not title:
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, title, 0, 1, 'C')
            
            # Add image
            self.image(buf, x=(self.w - 140)/2, y=current_y, w=140)
            
            # Move down after the chart
            self.set_y(current_y + 90)
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            self.add_paragraph(f"Error creating bar chart: {str(e)}")
    
    def add_line_chart(self, x_data, y_data, title="", xlabel="", ylabel="", color='#1f77b4'):
        """Add a line chart to the PDF."""
        try:
            # Create line chart with matplotlib
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Create line
            ax.plot(x_data, y_data, marker='o', color=color, linewidth=2)
            
            # Customize the chart
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save to BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            
            # Get the current Y position
            current_y = self.get_y()
            
            # Add title if not already in the chart
            if not title:
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, title, 0, 1, 'C')
            
            # Add image
            self.image(buf, x=(self.w - 140)/2, y=current_y, w=140)
            
            # Move down after the chart
            self.set_y(current_y + 90)
            
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            self.add_paragraph(f"Error creating line chart: {str(e)}")
    
    def add_styled_unit_economics(self, unit_econ):
        """Add a visually appealing unit economics section with a diagram."""
        self.add_subsection_title("Unit Economics")
        
        # Extract values with defaults
        ltv = unit_econ.get('ltv', 0)
        cac = unit_econ.get('cac', 0)
        ratio = unit_econ.get('ltv_cac_ratio', 0)
        payback = unit_econ.get('cac_payback_months', 0)
        
        # Add metrics
        metrics = [
            ("LTV", f"${ltv:,.2f}"),
            ("CAC", f"${cac:,.2f}"),
            ("LTV:CAC Ratio", f"{ratio:.2f}"),
            ("CAC Payback", f"{payback:.1f} months")
        ]
        
        self.add_metric_row(metrics)
        
        # Add visual LTV to CAC comparison
        self.ln(10)
        self.add_subsection_title("LTV vs CAC Visualization")
        
        # Create bar chart
        if ltv > 0 or cac > 0:
            self.add_bar_chart(
                ['LTV', 'CAC'], 
                [ltv, cac], 
                "Customer Value vs Acquisition Cost", 
                "Value ($)",
                ['#4CAF50', '#FF5722']
            )
        
        # Add interpretation
        self.ln(5)
        
        if ratio >= 3:
            assessment = "Strong unit economics with LTV significantly higher than CAC."
        elif ratio >= 1:
            assessment = "Positive unit economics, but room for improvement in the LTV:CAC ratio."
        else:
            assessment = "Concerning unit economics with CAC higher than LTV. Focus on improving this ratio."
        
        self.add_paragraph(f"Assessment: {assessment}")

def generate_enhanced_pdf(doc, report_type="full", sections=None):
    """
    Generate an enhanced PDF report with charts, graphs, and better formatting.
    
    Args:
        doc: The document data dictionary
        report_type: The type of report ("full", "executive", "custom")
        sections: Dictionary of sections to include if report_type is "custom"
        
    Returns:
        bytes: The PDF data or None if generation failed
    """
    logger.info(f"Starting PDF generation with report type: {report_type}")
    
    try:
        # Create a deep copy of the doc to avoid modifying the original
        doc_copy = copy.deepcopy(doc)
        
        # Initialize PDF
        pdf = ReportPDF(title=f"{doc_copy.get('name', 'Startup')} Investment Analysis")
        
        # Add cover page
        pdf.create_cover_page(doc_copy)
        
        # Determine which sections to include
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
        
        # Executive Summary section
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
            
            # Add CAMP radar chart
            camp_categories = ['Capital', 'Market', 'Advantage', 'People']
            camp_values = [
                doc_copy.get('capital_score', 0),
                doc_copy.get('market_score', 0),
                doc_copy.get('advantage_score', 0),
                doc_copy.get('people_score', 0)
            ]
            
            pdf.add_radar_chart(camp_categories, camp_values, "CAMP Framework Scores")
            
            # Key strengths and weaknesses
            pdf.add_subsection_title("Key Strengths")
            
            # Extract strengths from patterns
            strengths = []
            for pattern in doc_copy.get("patterns_matched", []):
                if isinstance(pattern, dict) and pattern.get("is_positive", False):
                    strengths.append(pattern.get("name", ""))
            
            # Add default strengths if none found
            if not strengths:
                if doc_copy.get('capital_score', 0) > 70:
                    strengths.append("Strong capital efficiency")
                if doc_copy.get('market_score', 0) > 70:
                    strengths.append("Strong market positioning")
                if doc_copy.get('advantage_score', 0) > 70:
                    strengths.append("Strong competitive advantage")
                if doc_copy.get('people_score', 0) > 70:
                    strengths.append("Strong team execution")
                
                # Add at least one strength
                if not strengths:
                    max_score = max([
                        doc_copy.get('capital_score', 0),
                        doc_copy.get('market_score', 0),
                        doc_copy.get('advantage_score', 0),
                        doc_copy.get('people_score', 0)
                    ])
                    
                    if max_score == doc_copy.get('capital_score', 0):
                        strengths.append("Relative strength in capital efficiency")
                    elif max_score == doc_copy.get('market_score', 0):
                        strengths.append("Relative strength in market positioning")
                    elif max_score == doc_copy.get('advantage_score', 0):
                        strengths.append("Relative strength in competitive advantage")
                    elif max_score == doc_copy.get('people_score', 0):
                        strengths.append("Relative strength in team execution")
            
            # Add strengths to PDF
            for strength in strengths[:3]:
                pdf.add_paragraph(f"✓ {strength}")
            
            # Add weaknesses section
            pdf.add_subsection_title("Areas for Improvement")
            
            # Extract weaknesses from patterns
            weaknesses = []
            for pattern in doc_copy.get("patterns_matched", []):
                if isinstance(pattern, dict) and not pattern.get("is_positive", True):
                    weaknesses.append(pattern.get("name", ""))
            
            # Add default weaknesses if none found
            if not weaknesses:
                if doc_copy.get('capital_score', 0) < 50:
                    weaknesses.append("Improve capital efficiency")
                if doc_copy.get('market_score', 0) < 50:
                    weaknesses.append("Strengthen market positioning")
                if doc_copy.get('advantage_score', 0) < 50:
                    weaknesses.append("Build stronger competitive moat")
                if doc_copy.get('people_score', 0) < 50:
                    weaknesses.append("Enhance team capabilities")
                
                # Add at least one weakness
                if not weaknesses:
                    min_score = min([
                        doc_copy.get('capital_score', 0),
                        doc_copy.get('market_score', 0),
                        doc_copy.get('advantage_score', 0),
                        doc_copy.get('people_score', 0)
                    ])
                    
                    if min_score == doc_copy.get('capital_score', 0):
                        weaknesses.append("Consider improving capital efficiency")
                    elif min_score == doc_copy.get('market_score', 0):
                        weaknesses.append("Consider improving market positioning")
                    elif min_score == doc_copy.get('advantage_score', 0):
                        weaknesses.append("Consider strengthening competitive advantage")
                    elif min_score == doc_copy.get('people_score', 0):
                        weaknesses.append("Consider enhancing team capabilities")
            
            # Add weaknesses to PDF
            for weakness in weaknesses[:3]:
                pdf.add_paragraph(f"! {weakness}")
        
        # Business Model section
        if active_sections.get("Business Model", True):
            pdf.add_page()
            pdf.add_section_title("Business Model")
            
            # Business model description
            business_model = doc_copy.get("business_model", "")
            if business_model:
                pdf.add_paragraph(business_model)
            else:
                pdf.add_paragraph("No business model description available.")
            
            # Unit economics
            unit_econ = doc_copy.get("unit_economics", {})
            if unit_econ:
                pdf.add_styled_unit_economics(unit_econ)
            
            # Growth metrics
            pdf.add_subsection_title("Key Growth Metrics")
            
            growth_metrics = [
                ("Monthly Revenue", f"${doc_copy.get('monthly_revenue', 0):,.2f}"),
                ("Revenue Growth", f"{doc_copy.get('revenue_growth_rate', 0):.1f}%/mo"),
                ("User Growth", f"{doc_copy.get('user_growth_rate', 0):.1f}%/mo")
            ]
            pdf.add_metric_row(growth_metrics)
            
            # Projected growth chart
            if isinstance(doc_copy.get("system_dynamics", {}), dict) and "users" in doc_copy.get("system_dynamics", {}):
                users = doc_copy.get("system_dynamics", {}).get("users", [])
                months = list(range(len(users)))
                
                if users and months:
                    pdf.add_line_chart(
                        months, 
                        users, 
                        "Projected User Growth", 
                        "Months", 
                        "Users"
                    )
        
        # Market Analysis section
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
            
            # Competitive position
            pdf.add_subsection_title("Competitive Position")
            
            # Extract competitive data
            competitive_pos = doc_copy.get("competitive_positioning", {})
            position = competitive_pos.get("position", "challenger")
            
            pdf.add_paragraph(f"Current Position: {position.capitalize()}")
            
            # Add competitive advantages
            advantages = competitive_pos.get("advantages", [])
            if advantages:
                pdf.add_subsection_title("Competitive Advantages")
                
                advantages_names = []
                advantages_scores = []
                
                for adv in advantages:
                    if isinstance(adv, dict):
                        advantages_names.append(adv.get("name", ""))
                        advantages_scores.append(adv.get("score", 0))
                
                if advantages_names and advantages_scores:
                    pdf.add_bar_chart(
                        advantages_names,
                        advantages_scores,
                        "Competitive Advantages",
                        "Score"
                    )
            
            # Market penetration chart
            market_penetration = doc_copy.get("market_penetration", {})
            if isinstance(market_penetration, dict) and "timeline" in market_penetration and "penetration" in market_penetration:
                timeline = market_penetration.get("timeline", [])
                penetration = [p * 100 for p in market_penetration.get("penetration", [])]  # Convert to percentage
                
                if timeline and penetration and len(timeline) == len(penetration):
                    pdf.add_line_chart(
                        timeline,
                        penetration,
                        "Market Penetration Projection",
                        "Month",
                        "Penetration (%)"
                    )
        
        # Team Assessment section
        if active_sections.get("Team Assessment", True):
            pdf.add_page()
            pdf.add_section_title("Team Assessment")
            
            # Team metrics
            metrics = [
                ("Team Score", f"{doc_copy.get('team_score', 0):.1f}/100"),
                ("Founder Experience", f"{doc_copy.get('founder_domain_exp_yrs', 0)} years"),
                ("Previous Exits", f"{doc_copy.get('founder_exits', 0)}")
            ]
            pdf.add_metric_row(metrics)
            
            metrics2 = [
                ("Team Size", f"{doc_copy.get('employee_count', 0)} employees"),
                ("Tech Talent Ratio", f"{doc_copy.get('tech_talent_ratio', 0)*100:.1f}%"),
                ("Team Diversity", f"{doc_copy.get('founder_diversity_score', 0):.1f}/100")
            ]
            pdf.add_metric_row(metrics2)
            
            # Execution risk
            execution_risk = doc_copy.get("execution_risk", {})
            if isinstance(execution_risk, dict) and "risk_factors" in execution_risk:
                risk_factors = execution_risk.get("risk_factors", {})
                
                if risk_factors:
                    pdf.add_subsection_title("Execution Risk Factors")
                    
                    factors = list(risk_factors.keys())
                    scores = list(risk_factors.values())
                    
                    if factors and scores:
                        pdf.add_bar_chart(
                            factors,
                            scores,
                            "Risk Factors",
                            "Risk Level"
                        )
            
            # Team leadership
            pdf.add_subsection_title("Leadership Team")
            
            leadership = {
                "CEO": True,  # Assumed always present
                "CTO": doc_copy.get("has_cto", False),
                "CMO": doc_copy.get("has_cmo", False),
                "CFO": doc_copy.get("has_cfo", False)
            }
            
            # Create a visual representation of the leadership team
            leaders = list(leadership.keys())
            status = [1 if v else 0 for v in leadership.values()]
            
            pdf.add_bar_chart(
                leaders,
                status,
                "Leadership Positions",
                "Present (1) / Absent (0)",
                ['#4CAF50' if s else '#FF5722' for s in status]
            )
            
        # Return the PDF as bytes
        logger.info("PDF generation completed successfully")
        return pdf.output(dest='S').encode('latin1')
        
    except Exception as e:
        logger.error(f"Error generating enhanced PDF: {traceback.format_exc()}")
        return generate_emergency_pdf(doc)

def generate_emergency_pdf(doc):
    """
    Generate a minimal emergency PDF when the enhanced version fails.
    
    Args:
        doc: Document data dictionary
        
    Returns:
        bytes: PDF report as bytes
    """
    logger.info("Generating emergency PDF")
    
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
        
        # Emergency message
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "", 0, 1)
        pdf.multi_cell(0, 5, "Note: This is a simplified emergency report. For a full report, please try again later.")
        
        return pdf.output(dest='S').encode('latin1')
        
    except Exception as e:
        logger.error(f"Emergency PDF generation also failed: {e}")
        # Return an empty PDF if all else fails
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Error Generating Report", 0, 1, 'C')
        return pdf.output(dest='S').encode('latin1')

# Alias for backward compatibility
generate_investor_report = generate_enhanced_pdf 