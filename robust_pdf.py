"""
Robust PDF Generator for FlashDNA

This module provides a unified, robust PDF generation system for FlashDNA reports.
It consolidates the functionality from multiple existing modules and adds enhanced
error handling with graceful degradation.

Key features:
1. Unified interface for all PDF generation needs
2. Multi-tier fallback system for maximum reliability
3. Enhanced visuals with better chart formatting
4. Thread-safe implementation for concurrent report generation
"""

import os
import io
import logging
import tempfile
import traceback
import copy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("robust_pdf")

# Define types to ensure proper type safety
DocumentData = Dict[str, Any]
PdfBytes = bytes
ChartData = Dict[str, Any]

# Try to import libraries with proper error handling
try:
    # Primary PDF generation library
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError as e:
    logger.error(f"ReportLab import error: {str(e)}")
    REPORTLAB_AVAILABLE = False

try:
    # Visualization libraries
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    logger.error(f"Matplotlib import error: {str(e)}")
    MATPLOTLIB_AVAILABLE = False

try:
    # Optional advanced charting library
    import plotly.graph_objects as go
    from plotly.io import to_image
    PLOTLY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Plotly import error: {str(e)}")
    PLOTLY_AVAILABLE = False

# Constants for styling
PRIMARY_COLOR = (31, 119, 180)  # Blue
SECONDARY_COLOR = (255, 127, 14)  # Orange
SUCCESS_COLOR = (44, 160, 44)  # Green
WARNING_COLOR = (214, 39, 40)  # Red
BACKGROUND_COLOR = (248, 248, 248)  # Light gray

class RobustPDFGenerator:
    """
    Main PDF generation class with robust error handling and fallbacks.
    """
    
    def __init__(self, doc_data: DocumentData, report_type: str = "full", sections: Optional[Dict[str, bool]] = None):
        """
        Initialize the PDF generator with document data and report options.
        
        Args:
            doc_data: Dictionary containing all startup analysis data
            report_type: Type of report to generate ("full", "executive", "custom")
            sections: Dictionary mapping section names to boolean inclusion flags
        """
        self.doc_data = copy.deepcopy(doc_data)  # Deep copy to avoid modifying original
        self.report_type = report_type
        self.sections = sections or self._default_sections(report_type)
        self.company_name = self.doc_data.get('name', self.doc_data.get('company_name', 'Startup'))
        self.generation_date = datetime.now()
        
        # Check available libraries and set capabilities
        self.capabilities = {
            "reportlab": REPORTLAB_AVAILABLE,
            "matplotlib": MATPLOTLIB_AVAILABLE,
            "plotly": PLOTLY_AVAILABLE
        }
        
        # Determine the best available method for PDF generation
        self.best_method = self._determine_best_method()
        logger.info(f"Using PDF generation method: {self.best_method}")
    
    def _default_sections(self, report_type: str) -> Dict[str, bool]:
        """Return default sections based on report type."""
        if report_type == "executive":
            return {
                "Executive Summary": True,
                "Business Model": True,
                "Market Analysis": True,
                "Team Assessment": True,
                "Financial Projections": False,
                "Competitive Analysis": False,
                "Growth Metrics": False,
                "Risk Assessment": False,
                "Exit Strategy": False,
                "Technical Assessment": False
            }
        elif report_type == "full":
            return {
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
        else:
            # Default to executive summary for unknown types
            return {"Executive Summary": True}
    
    def _determine_best_method(self) -> str:
        """Determine the best available method for PDF generation."""
        if self.capabilities["reportlab"] and self.capabilities["matplotlib"]:
            return "full"
        elif self.capabilities["reportlab"]:
            return "basic"
        else:
            return "emergency"
    
    def generate_pdf(self) -> PdfBytes:
        """
        Generate the PDF report using the best available method.
        
        Returns:
            bytes: The generated PDF as bytes
        """
        try:
            if self.best_method == "full":
                return self._generate_full_pdf()
            elif self.best_method == "basic":
                return self._generate_basic_pdf()
            else:
                return self._generate_emergency_pdf()
        except Exception as e:
            logger.error(f"Error generating PDF with {self.best_method} method: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try next best method
            try:
                if self.best_method == "full":
                    logger.info("Falling back to basic PDF generation")
                    return self._generate_basic_pdf()
                elif self.best_method == "basic":
                    logger.info("Falling back to emergency PDF generation")
                    return self._generate_emergency_pdf()
            except Exception as e2:
                logger.error(f"Error in fallback PDF generation: {str(e2)}")
                logger.error(traceback.format_exc())
            
            # Ultimate fallback - return an absolute minimal PDF
            logger.critical("All PDF generation methods failed, returning minimal PDF")
            return self._generate_minimal_pdf()
    
    def _generate_full_pdf(self) -> PdfBytes:
        """Generate a fully featured PDF with charts and complete analysis."""
        buffer = io.BytesIO()
        
        # Create the document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=0.5*inch,
            rightMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=20,
            alignment=TA_CENTER,
            textColor=colors.Color(*[c/255 for c in PRIMARY_COLOR])
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=16,
            alignment=TA_LEFT,
            textColor=colors.Color(*[c/255 for c in PRIMARY_COLOR])
        )
        
        subheading_style = ParagraphStyle(
            'Subheading',
            parent=styles['Heading3'],
            fontSize=14,
            alignment=TA_LEFT,
            textColor=colors.Color(*[c/255 for c in PRIMARY_COLOR])
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_LEFT
        )
        
        # Create the story (content elements)
        story = []
        
        # Add title
        story.append(Paragraph(f"{self.company_name} - Investor Report", title_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Add generation date
        story.append(Paragraph(
            f"Generated: {self.generation_date.strftime('%Y-%m-%d %H:%M')}",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.5*inch))
        
        # Add sections based on configuration
        if self.sections.get("Executive Summary", True):
            self._add_executive_summary(story, styles, heading_style, normal_style, subheading_style)
        
        if self.sections.get("Business Model", False):
            self._add_business_model(story, styles, heading_style, normal_style)
        
        if self.sections.get("Market Analysis", False):
            self._add_market_analysis(story, styles, heading_style, normal_style)
        
        if self.sections.get("Financial Projections", False):
            self._add_financial_projections(story, styles, heading_style, normal_style)
        
        if self.sections.get("Team Assessment", False):
            self._add_team_assessment(story, styles, heading_style, normal_style)
        
        if self.sections.get("Competitive Analysis", False):
            self._add_competitive_analysis(story, styles, heading_style, normal_style)
        
        if self.sections.get("Risk Assessment", False):
            self._add_risk_assessment(story, styles, heading_style, normal_style)
            
        if self.sections.get("Exit Strategy", False):
            self._add_exit_strategy(story, styles, heading_style, normal_style)
            
        if self.sections.get("Technical Assessment", False):
            self._add_technical_assessment(story, styles, heading_style, normal_style)
        
        # Build the document
        doc.build(story)
        
        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    def _add_executive_summary(self, story, styles, heading_style, normal_style, subheading_style):
        """Add executive summary section to the report."""
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Get key metrics
        camp_score = self.doc_data.get('camp_score', 0)
        success_prob = self.doc_data.get('success_prob', 0)
        runway = self.doc_data.get('runway_months', 0)
        
        # Create summary text
        summary = f"""
        {self.company_name} has been analyzed using the CAMP Framework, resulting in an overall score of {camp_score:.1f}/100. 
        The analysis indicates a {success_prob:.1f}% probability of success with a current runway of {runway:.1f} months.
        """
        
        story.append(Paragraph(summary, normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Add CAMP breakdown table
        camp_data = [
            ["CAMP Framework Component", "Score"],
            ["Capital Efficiency", f"{self.doc_data.get('capital_score', 0):.1f}/100"],
            ["Market Dynamics", f"{self.doc_data.get('market_score', 0):.1f}/100"],
            ["Advantage Moat", f"{self.doc_data.get('advantage_score', 0):.1f}/100"],
            ["People & Performance", f"{self.doc_data.get('people_score', 0):.1f}/100"]
        ]
        
        # Create the table
        camp_table = Table(camp_data, colWidths=[4*inch, 1.5*inch])
        camp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
        ]))
        
        story.append(camp_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Add CAMP radar chart if possible
        if self.capabilities["matplotlib"]:
            try:
                radar_chart = self._create_camp_radar_chart()
                if radar_chart:
                    story.append(Image(radar_chart, width=5*inch, height=4*inch))
                    story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.error(f"Error creating radar chart: {str(e)}")
        
        # Key strengths and weaknesses
        story.append(Paragraph("Key Strengths:", subheading_style))
        
        # Get patterns for strengths
        strengths = []
        for pattern in self.doc_data.get("patterns_matched", []):
            if isinstance(pattern, dict) and pattern.get("is_positive", False):
                strengths.append(pattern.get("name", ""))
        
        # If no strengths found, use CAMP scores to determine
        if not strengths:
            camp_scores = {
                "Capital Efficiency": self.doc_data.get('capital_score', 0),
                "Market Dynamics": self.doc_data.get('market_score', 0),
                "Advantage Moat": self.doc_data.get('advantage_score', 0),
                "People & Performance": self.doc_data.get('people_score', 0)
            }
            
            # Find top 2 highest scores
            top_strengths = sorted(camp_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            for area, score in top_strengths:
                strengths.append(f"Strong {area} ({score:.1f}/100)")
        
        # Add strengths
        for strength in strengths[:3]:  # Top 3 strengths
            story.append(Paragraph(f"• {strength}", normal_style))
        
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Areas for Improvement:", subheading_style))
        
        # Get patterns for weaknesses
        weaknesses = []
        for pattern in self.doc_data.get("patterns_matched", []):
            if isinstance(pattern, dict) and not pattern.get("is_positive", True):
                weaknesses.append(pattern.get("name", ""))
        
        # If no weaknesses found, use CAMP scores to determine
        if not weaknesses:
            camp_scores = {
                "Capital Efficiency": self.doc_data.get('capital_score', 0),
                "Market Dynamics": self.doc_data.get('market_score', 0),
                "Advantage Moat": self.doc_data.get('advantage_score', 0),
                "People & Performance": self.doc_data.get('people_score', 0)
            }
            
            # Find top 2 lowest scores
            top_weaknesses = sorted(camp_scores.items(), key=lambda x: x[1])[:2]
            for area, score in top_weaknesses:
                weaknesses.append(f"Improve {area} ({score:.1f}/100)")
        
        # Add weaknesses
        for weakness in weaknesses[:3]:  # Top 3 weaknesses
            story.append(Paragraph(f"• {weakness}", normal_style))
        
        story.append(Spacer(1, 0.5*inch))
    
    def _add_business_model(self, story, styles, heading_style, normal_style):
        """Add business model section to the report."""
        story.append(Paragraph("Business Model", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Business model description
        business_model = self.doc_data.get("business_model", "")
        if business_model:
            story.append(Paragraph(business_model, normal_style))
        else:
            story.append(Paragraph(
                f"{self.company_name} operates in the {self.doc_data.get('sector', 'technology')} "
                f"sector at the {self.doc_data.get('stage', 'early')} stage.",
                normal_style
            ))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Unit economics
        story.append(Paragraph("Unit Economics", styles['Heading3']))
        
        unit_econ = self.doc_data.get("unit_economics", {})
        if unit_econ and isinstance(unit_econ, dict):
            # Create unit economics table
            unit_data = [
                ["Metric", "Value"],
                ["Customer LTV", self._format_currency(unit_econ.get("ltv", 0))],
                ["Customer CAC", self._format_currency(unit_econ.get("cac", 0))],
                ["LTV:CAC Ratio", f"{unit_econ.get('ltv_cac_ratio', 0):.2f}"],
                ["Gross Margin", f"{unit_econ.get('gross_margin', 0)*100:.1f}%"],
                ["CAC Payback", f"{unit_econ.get('cac_payback_months', 0):.1f} months"]
            ]
            
            unit_table = Table(unit_data, colWidths=[2.5*inch, 2.5*inch])
            unit_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
            ]))
            
            story.append(unit_table)
        else:
            story.append(Paragraph("Unit economics data not available.", normal_style))
        
        story.append(Spacer(1, 0.5*inch))
    
    def _add_market_analysis(self, story, styles, heading_style, normal_style):
        """Add market analysis section to the report."""
        story.append(Paragraph("Market Analysis", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Create market data table
        market_data = [
            ["Metric", "Value"],
            ["Market Size", self._format_currency(self.doc_data.get("market_size", 0))],
            ["Market Growth Rate", f"{self.doc_data.get('market_growth_rate', 0):.1f}%/year"],
            ["Market Share", f"{self.doc_data.get('market_share', 0):.2f}%"],
            ["User Growth Rate", f"{self.doc_data.get('user_growth_rate', 0):.1f}%/month"],
            ["Current Users", f"{self.doc_data.get('current_users', 0):,}"]
        ]
        
        market_table = Table(market_data, colWidths=[2.5*inch, 2.5*inch])
        market_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
        ]))
        
        story.append(market_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Add user growth chart if available
        sys_dynamics = self.doc_data.get("system_dynamics", {})
        if isinstance(sys_dynamics, dict) and "users" in sys_dynamics and self.capabilities["matplotlib"]:
            try:
                user_chart = self._create_user_growth_chart(sys_dynamics)
                if user_chart:
                    story.append(Paragraph("User Growth Projection", styles['Heading3']))
                    story.append(Image(user_chart, width=5*inch, height=3*inch))
                    story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.error(f"Error creating user growth chart: {str(e)}")
        
        # PMF analysis
        pmf = self.doc_data.get("pmf_analysis", {})
        if pmf and isinstance(pmf, dict):
            story.append(Paragraph("Product-Market Fit", styles['Heading3']))
            
            # Create PMF table
            pmf_data = [
                ["Metric", "Value"],
                ["PMF Score", f"{pmf.get('pmf_score', 0):.1f}/100"],
                ["PMF Stage", f"{pmf.get('stage', 'Unknown')}"],
                ["Engagement Score", f"{pmf.get('engagement_score', 0):.1f}/100"],
                ["Retention Score", f"{pmf.get('retention_score', 0):.1f}/100"]
            ]
            
            pmf_table = Table(pmf_data, colWidths=[2.5*inch, 2.5*inch])
            pmf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
            ]))
            
            story.append(pmf_table)
        
        story.append(Spacer(1, 0.5*inch))
    
    def _add_financial_projections(self, story, styles, heading_style, normal_style):
        """Add financial projections section to the report."""
        story.append(Paragraph("Financial Projections", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Key financial metrics
        story.append(Paragraph("Key Financial Metrics", styles['Heading3']))
        
        fin_data = [
            ["Metric", "Value"],
            ["Monthly Revenue", self._format_currency(self.doc_data.get("monthly_revenue", 0))],
            ["Monthly Burn", self._format_currency(self.doc_data.get("burn_rate", 0))],
            ["Runway", f"{self.doc_data.get('runway_months', 0):.1f} months"],
            ["Revenue Growth Rate", f"{self.doc_data.get('revenue_growth_rate', 0):.1f}%/month"]
        ]
        
        fin_table = Table(fin_data, colWidths=[2.5*inch, 2.5*inch])
        fin_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
        ]))
        
        story.append(fin_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Add revenue projections chart if available
        forecast = self.doc_data.get("financial_forecast", {})
        if isinstance(forecast, dict) and "months" in forecast and "revenue" in forecast and self.capabilities["matplotlib"]:
            try:
                revenue_chart = self._create_revenue_forecast_chart(forecast)
                if revenue_chart:
                    story.append(Paragraph("Revenue Forecast", styles['Heading3']))
                    story.append(Image(revenue_chart, width=5*inch, height=3*inch))
                    story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.error(f"Error creating revenue forecast chart: {str(e)}")
        
        # Add cash flow chart if available
        if "cash_flow" in self.doc_data and self.capabilities["matplotlib"]:
            try:
                cash_chart = self._create_cash_flow_chart()
                if cash_chart:
                    story.append(Paragraph("Cash Flow Projection", styles['Heading3']))
                    story.append(Image(cash_chart, width=5*inch, height=3*inch))
                    story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.error(f"Error creating cash flow chart: {str(e)}")
        
        story.append(Spacer(1, 0.5*inch))
    
    def _add_team_assessment(self, story, styles, heading_style, normal_style):
        """Add team assessment section to the report."""
        story.append(Paragraph("Team Assessment", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Team metrics
        story.append(Paragraph("Team Metrics", styles['Heading3']))
        
        team_data = [
            ["Metric", "Value"],
            ["Team Score", f"{self.doc_data.get('team_score', 0):.1f}/100"],
            ["Founder Domain Experience", f"{self.doc_data.get('founder_domain_exp_yrs', 0)} years"],
            ["Previous Exits", f"{self.doc_data.get('founder_exits', 0)}"],
            ["Team Size", f"{self.doc_data.get('employee_count', 0)} employees"],
            ["Tech Talent Ratio", f"{self.doc_data.get('tech_talent_ratio', 0)*100:.1f}%"]
        ]
        
        team_table = Table(team_data, colWidths=[2.5*inch, 2.5*inch])
        team_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
        ]))
        
        story.append(team_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Leadership team
        story.append(Paragraph("Leadership Team", styles['Heading3']))
        
        leadership = {
            "CEO": True,  # Assumed always present
            "CTO": self.doc_data.get("has_cto", False),
            "CMO": self.doc_data.get("has_cmo", False),
            "CFO": self.doc_data.get("has_cfo", False)
        }
        
        leadership_data = [["Position", "Present"]]
        for role, present in leadership.items():
            leadership_data.append([role, "Yes" if present else "No"])
        
        leadership_table = Table(leadership_data, colWidths=[2.5*inch, 2.5*inch])
        leadership_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
            # Color yes/no cells
            ('TEXTCOLOR', (1, 1), (1, -1), colors.green if leadership["CEO"] else colors.red),
            ('TEXTCOLOR', (1, 2), (1, 2), colors.green if leadership["CTO"] else colors.red),
            ('TEXTCOLOR', (1, 3), (1, 3), colors.green if leadership["CMO"] else colors.red),
            ('TEXTCOLOR', (1, 4), (1, 4), colors.green if leadership["CFO"] else colors.red),
        ]))
        
        story.append(leadership_table)
        story.append(Spacer(1, 0.5*inch))
    
    def _add_competitive_analysis(self, story, styles, heading_style, normal_style):
        """Add competitive analysis section to the report."""
        story.append(Paragraph("Competitive Analysis", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Get competitive positioning
        comp_pos = self.doc_data.get("competitive_positioning", {})
        if comp_pos and isinstance(comp_pos, dict):
            position = comp_pos.get("position", "challenger")
            story.append(Paragraph(f"Competitive Position: {position.capitalize()}", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Advantages and disadvantages
            advantages = comp_pos.get("advantages", [])
            disadvantages = comp_pos.get("disadvantages", [])
            
            if advantages:
                story.append(Paragraph("Key Advantages:", styles['Heading3']))
                for adv in advantages[:3]:  # Top 3 advantages
                    if isinstance(adv, dict):
                        name = adv.get("name", "")
                        score = adv.get("score", 0)
                        story.append(Paragraph(f"• {name} ({score:.1f}/100)", normal_style))
            
            if disadvantages:
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph("Key Challenges:", styles['Heading3']))
                for dis in disadvantages[:3]:  # Top 3 disadvantages
                    if isinstance(dis, dict):
                        name = dis.get("name", "")
                        score = dis.get("score", 0)
                        story.append(Paragraph(f"• {name} ({score:.1f}/100)", normal_style))
        
        # Competitors
        competitors = self.doc_data.get("competitors", [])
        if competitors and all(isinstance(comp, dict) for comp in competitors):
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("Key Competitors", styles['Heading3']))
            
            # Create competitors table
            comp_data = [["Competitor", "Threat Level", "Key Strength"]]
            
            for comp in competitors[:5]:  # Top 5 competitors
                name = comp.get("name", "Unknown")
                threat = comp.get("threat_level", "Medium")
                
                strengths = comp.get("strengths", [])
                key_strength = strengths[0] if strengths else "Unknown"
                
                comp_data.append([name, threat, key_strength])
            
            comp_table = Table(comp_data, colWidths=[2*inch, 1.5*inch, 2*inch])
            comp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(comp_table)
        
        # Market trends
        trends = self.doc_data.get("market_trends", {})
        if trends and isinstance(trends, dict):
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("Market Trends", styles['Heading3']))
            
            trend_items = trends.get("trends", [])
            if trend_items:
                for trend in trend_items[:3]:  # Top 3 trends
                    if isinstance(trend, str):
                        story.append(Paragraph(f"• {trend}", normal_style))
                    elif isinstance(trend, dict):
                        name = trend.get("name", "")
                        desc = trend.get("description", "")
                        story.append(Paragraph(f"• {name}: {desc}", normal_style))
        
        story.append(Spacer(1, 0.5*inch))
    
    def _add_risk_assessment(self, story, styles, heading_style, normal_style):
        """Add risk assessment section to the report."""
        story.append(Paragraph("Risk Assessment", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Execution risk
        exec_risk = self.doc_data.get("execution_risk", {})
        if exec_risk and isinstance(exec_risk, dict):
            story.append(Paragraph("Execution Risk Factors", styles['Heading3']))
            
            risk_factors = exec_risk.get("risk_factors", {})
            if risk_factors and isinstance(risk_factors, dict):
                # Create risk factors table
                risk_data = [["Risk Factor", "Risk Level"]]
                
                for factor, level in risk_factors.items():
                    risk_data.append([factor.replace("_", " ").title(), f"{level:.1f}/100"])
                
                risk_table = Table(risk_data, colWidths=[3*inch, 2*inch])
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                    ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
                    ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
                ]))
                
                story.append(risk_table)
        
        # Monte Carlo simulation
        monte_carlo = self.doc_data.get("monte_carlo", {})
        if monte_carlo and isinstance(monte_carlo, dict):
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("Monte Carlo Simulation Results", styles['Heading3']))
            
            # Create Monte Carlo results table
            mc_data = [
                ["Metric", "Value"],
                ["Success Probability", f"{monte_carlo.get('success_probability', 0):.1f}%"],
                ["Median Runway", f"{monte_carlo.get('median_runway_months', 0):.1f} months"],
                ["Failure Probability", f"{monte_carlo.get('failure_probability', 0):.1f}%"],
                ["Cash-out Risk", f"{monte_carlo.get('cash_out_risk', 0):.1f}%"]
            ]
            
            mc_table = Table(mc_data, colWidths=[2.5*inch, 2.5*inch])
            mc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
            ]))
            
            story.append(mc_table)
            
            # Add Monte Carlo visualization if available
            if "projections" in monte_carlo and self.capabilities["matplotlib"]:
                try:
                    mc_chart = self._create_monte_carlo_chart(monte_carlo)
                    if mc_chart:
                        story.append(Spacer(1, 0.2*inch))
                        story.append(Image(mc_chart, width=5*inch, height=3*inch))
                except Exception as e:
                    logger.error(f"Error creating Monte Carlo chart: {str(e)}")
        
        story.append(Spacer(1, 0.5*inch))
    
    def _add_exit_strategy(self, story, styles, heading_style, normal_style):
        """Add exit strategy section to the report."""
        story.append(Paragraph("Exit Strategy Analysis", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Exit path analysis
        exit_analysis = self.doc_data.get("exit_path_analysis", {})
        if exit_analysis and isinstance(exit_analysis, dict):
            # Exit readiness
            readiness = exit_analysis.get("exit_readiness_score", 0)
            story.append(Paragraph(f"Exit Readiness Score: {readiness:.1f}/100", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Success factors by exit path
            success_factors = exit_analysis.get("success_factors", {})
            if success_factors and isinstance(success_factors, dict):
                story.append(Paragraph("Success Factors by Exit Path", styles['Heading3']))
                
                # Create success factors table
                factor_data = [["Exit Path", "Success Factor"]]
                
                for path, factor in success_factors.items():
                    factor_data.append([path.replace("_", " ").title(), f"{factor:.1f}/100"])
                
                factor_table = Table(factor_data, colWidths=[2.5*inch, 2.5*inch])
                factor_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                    ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
                    ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
                ]))
                
                story.append(factor_table)
                story.append(Spacer(1, 0.2*inch))
            
            # Exit recommendations
            exit_recs = self.doc_data.get("exit_recommendations", {})
            if exit_recs and isinstance(exit_recs, dict):
                optimal_path = exit_recs.get("optimal_path", "")
                if optimal_path:
                    story.append(Paragraph("Recommended Exit Strategy", styles['Heading3']))
                    
                    path_details = exit_recs.get("path_details", {})
                    path_desc = path_details.get("description", optimal_path)
                    
                    story.append(Paragraph(path_desc, normal_style))
                    story.append(Spacer(1, 0.1*inch))
                
                # Timeline
                timeline = exit_recs.get("timeline", {})
                if timeline and isinstance(timeline, dict):
                    story.append(Paragraph("Exit Timeline", styles['Heading3']))
                    
                    timeline_data = [
                        ["Metric", "Value"],
                        ["Years to Exit", f"{timeline.get('years_to_exit', 0):.1f} years"],
                        ["Exit Year", f"{timeline.get('exit_year', 0)}"],
                        ["Exit Valuation", self._format_currency(timeline.get('exit_valuation', 0))]
                    ]
                    
                    timeline_table = Table(timeline_data, colWidths=[2.5*inch, 2.5*inch])
                    timeline_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
                        ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                        ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
                        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
                    ]))
                    
                    story.append(timeline_table)
                    story.append(Spacer(1, 0.2*inch))
                
                # Recommendations
                recommendations = exit_recs.get("recommendations", [])
                if recommendations:
                    story.append(Paragraph("Key Recommendations", styles['Heading3']))
                    
                    for i, rec in enumerate(recommendations[:3]):  # Top 3 recommendations
                        story.append(Paragraph(f"{i+1}. {rec}", normal_style))
        else:
            story.append(Paragraph("Exit strategy analysis not available.", normal_style))
        
        story.append(Spacer(1, 0.5*inch))
    
    def _add_technical_assessment(self, story, styles, heading_style, normal_style):
        """Add technical assessment section to the report."""
        story.append(Paragraph("Technical Assessment", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Technical assessment
        tech_assessment = self.doc_data.get("tech_assessment", {})
        if tech_assessment and isinstance(tech_assessment, dict):
            # Overall score
            tech_score = tech_assessment.get("overall_score", 0)
            story.append(Paragraph(f"Technical Assessment Score: {tech_score:.1f}/100", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Component scores
            scores = tech_assessment.get("scores", {})
            if scores and isinstance(scores, dict):
                story.append(Paragraph("Component Scores", styles['Heading3']))
                
                # Create scores table
                score_data = [["Component", "Score"]]
                
                for component, score in scores.items():
                    score_data.append([component.replace("_", " ").title(), f"{score:.1f}/100"])
                
                score_table = Table(score_data, colWidths=[2.5*inch, 2.5*inch])
                score_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                    ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
                    ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
                ]))
                
                story.append(score_table)
                story.append(Spacer(1, 0.2*inch))
            
            # Tech stack
            tech_stack = tech_assessment.get("tech_stack", {})
            if tech_stack and isinstance(tech_stack, dict):
                story.append(Paragraph("Technology Stack", styles['Heading3']))
                
                # Prepare tech stack items
                stack_items = []
                for category, technologies in tech_stack.items():
                    if isinstance(technologies, list):
                        tech_str = ", ".join(technologies)
                    elif isinstance(technologies, dict):
                        tech_str = ", ".join([f"{k} ({v})" for k, v in technologies.items()])
                    else:
                        tech_str = str(technologies)
                    
                    stack_items.append(f"• {category}: {tech_str}")
                
                for item in stack_items:
                    story.append(Paragraph(item, normal_style))
                
                story.append(Spacer(1, 0.2*inch))
            
            # Technical debt
            tech_debt = tech_assessment.get("technical_debt", {})
            if tech_debt and isinstance(tech_debt, dict):
                story.append(Paragraph("Technical Debt Assessment", styles['Heading3']))
                
                debt_score = tech_debt.get("score", 0)
                story.append(Paragraph(f"Technical Debt Score: {debt_score:.1f}/100", normal_style))
                
                # Add technical debt areas
                debt_areas = tech_debt.get("areas", {})
                if debt_areas and isinstance(debt_areas, dict):
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph("Technical Debt Areas:", normal_style))
                    
                    for area, severity in debt_areas.items():
                        story.append(Paragraph(f"• {area.replace('_', ' ').title()}: {severity:.1f}/100", normal_style))
            
            # Recommendations
            recommendations = tech_assessment.get("recommendations", [])
            if recommendations:
                story.append(Spacer(1, 0.2*inch))
                story.append(Paragraph("Technical Recommendations", styles['Heading3']))
                
                for i, rec in enumerate(recommendations[:3]):  # Top 3 recommendations
                    story.append(Paragraph(f"{i+1}. {rec}", normal_style))
        else:
            story.append(Paragraph("Technical assessment not available.", normal_style))
        
        story.append(Spacer(1, 0.5*inch))
    
    def _create_camp_radar_chart(self) -> bytes:
        """Create a radar chart for CAMP framework scores."""
        if not self.capabilities["matplotlib"]:
            return None
        
        try:
            # Get CAMP scores
            capital_score = self.doc_data.get('capital_score', 0)
            market_score = self.doc_data.get('market_score', 0)
            advantage_score = self.doc_data.get('advantage_score', 0)
            people_score = self.doc_data.get('people_score', 0)
            
            # Create categories and values for the radar chart
            categories = ['Capital\nEfficiency', 'Market\nDynamics', 'Advantage\nMoat', 'People &\nPerformance']
            values = [capital_score, market_score, advantage_score, people_score]
            
            # Close the loop for the radar chart
            values = values + [values[0]]
            categories = categories + [categories[0]]
            
            # Calculate angles for each category
            N = len(categories) - 1  # -1 because we added one to close the loop
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create figure
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)
            
            # Draw the chart
            ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
            ax.fill(angles, values, alpha=0.25, color='blue')
            
            # Set category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories[:-1])
            
            # Set y-axis limits
            ax.set_ylim(0, 100)
            plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=8)
            
            # Add title
            plt.title("CAMP Framework Scores", size=14, color='blue', y=1.1)
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Return the image data
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error creating CAMP radar chart: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _create_user_growth_chart(self, sys_dynamics: Dict[str, Any]) -> bytes:
        """Create a user growth projection chart."""
        if not self.capabilities["matplotlib"]:
            return None
        
        try:
            # Extract users data
            users = sys_dynamics.get("users", [])
            months = list(range(len(users)))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Plot data
            ax.plot(months, users, marker='o', linestyle='-', color='blue')
            
            # Add labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Users')
            ax.set_title('User Growth Projection')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Return the image data
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error creating user growth chart: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _create_revenue_forecast_chart(self, forecast: Dict[str, Any]) -> bytes:
        """Create a revenue forecast chart."""
        if not self.capabilities["matplotlib"]:
            return None
        
        try:
            # Extract data
            months = forecast.get("months", [])
            revenue = forecast.get("revenue", [])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Plot data
            ax.plot(months, revenue, marker='o', linestyle='-', color='green')
            
            # Add labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Revenue ($)')
            ax.set_title('Revenue Forecast')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format y-axis with dollar sign
            import matplotlib.ticker as mtick
            formatter = mtick.StrMethodFormatter('${x:,.0f}')
            ax.yaxis.set_major_formatter(formatter)
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Return the image data
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error creating revenue forecast chart: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _create_cash_flow_chart(self) -> bytes:
        """Create a cash flow projection chart."""
        if not self.capabilities["matplotlib"]:
            return None
        
        try:
            # Extract data
            cash_flow = self.doc_data.get("cash_flow", [])
            months = list(range(len(cash_flow)))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Plot data
            ax.plot(months, cash_flow, marker='o', linestyle='-', color='blue')
            
            # Add runway marker
            runway = self.doc_data.get("runway_months", 0)
            if runway > 0:
                ax.axvline(x=runway, linestyle='--', color='red')
                ax.text(
                    runway, min(cash_flow) + (max(cash_flow) - min(cash_flow)) * 0.1,
                    f'Runway: {runway:.1f} months',
                    color='red',
                    ha='right'
                )
            
            # Add labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Cash ($)')
            ax.set_title('Cash Flow Projection')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, linestyle='-', color='black', alpha=0.3)
            
            # Format y-axis with dollar sign
            import matplotlib.ticker as mtick
            formatter = mtick.StrMethodFormatter('${x:,.0f}')
            ax.yaxis.set_major_formatter(formatter)
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Return the image data
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error creating cash flow chart: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _create_monte_carlo_chart(self, monte_carlo: Dict[str, Any]) -> bytes:
        """Create a Monte Carlo simulation visualization."""
        if not self.capabilities["matplotlib"]:
            return None
        
        try:
            # Check if we have projections data
            if "projections" not in monte_carlo:
                return None
            
            projections = monte_carlo.get("projections", {})
            
            # Choose which projection to visualize (prefer revenue if available)
            if "revenue" in projections:
                proj_key = "revenue"
                y_label = "Revenue ($)"
                title = "Revenue Monte Carlo Simulation"
            elif "users" in projections:
                proj_key = "users"
                y_label = "Users"
                title = "User Growth Monte Carlo Simulation"
            elif "cash" in projections:
                proj_key = "cash"
                y_label = "Cash ($)"
                title = "Cash Monte Carlo Simulation"
            else:
                return None
            
            proj_data = projections.get(proj_key, {})
            
            # Check if we have percentiles and months
            if "percentiles" not in proj_data or "months" not in proj_data:
                return None
            
            percentiles = proj_data.get("percentiles", {})
            months = proj_data.get("months", [])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Plot each percentile
            for percentile, values in percentiles.items():
                if len(values) != len(months):
                    continue
                
                if percentile == "p50":
                    # Median line
                    ax.plot(months, values, linewidth=2, color='blue', label='Median')
                elif percentile == "p90":
                    # 90th percentile
                    ax.plot(months, values, linewidth=1, color='green', label='90th Percentile')
                elif percentile == "p10":
                    # 10th percentile
                    ax.plot(months, values, linewidth=1, color='red', label='10th Percentile')
            
            # Add labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Add appropriate formatting for the y-axis
            if proj_key in ["revenue", "cash"]:
                import matplotlib.ticker as mtick
                formatter = mtick.StrMethodFormatter('${x:,.0f}')
                ax.yaxis.set_major_formatter(formatter)
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Return the image data
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error creating Monte Carlo chart: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _generate_basic_pdf(self) -> PdfBytes:
        """Generate a basic PDF with tables but no charts."""
        buffer = io.BytesIO()
        
        # Create the document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=0.5*inch,
            rightMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=20,
            alignment=TA_CENTER,
            textColor=colors.Color(*[c/255 for c in PRIMARY_COLOR])
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=16,
            alignment=TA_LEFT,
            textColor=colors.Color(*[c/255 for c in PRIMARY_COLOR])
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_LEFT
        )
        
        # Create the story (content elements)
        story = []
        
        # Add title
        story.append(Paragraph(f"{self.company_name} - Investor Report", title_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Add generation date
        story.append(Paragraph(
            f"Generated: {self.generation_date.strftime('%Y-%m-%d %H:%M')}",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.5*inch))
        
        # Add summary section
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Add key metrics table
        camp_score = self.doc_data.get('camp_score', 0)
        success_prob = self.doc_data.get('success_prob', 0)
        runway = self.doc_data.get('runway_months', 0)
        
        metrics_data = [
            ["Metric", "Value"],
            ["CAMP Score", f"{camp_score:.1f}/100"],
            ["Success Probability", f"{success_prob:.1f}%"],
            ["Runway", f"{runway:.1f} months"],
            ["Monthly Revenue", self._format_currency(self.doc_data.get('monthly_revenue', 0))],
            ["Burn Rate", self._format_currency(self.doc_data.get('burn_rate', 0))]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Add CAMP breakdown table
        camp_data = [
            ["CAMP Framework Component", "Score"],
            ["Capital Efficiency", f"{self.doc_data.get('capital_score', 0):.1f}/100"],
            ["Market Dynamics", f"{self.doc_data.get('market_score', 0):.1f}/100"],
            ["Advantage Moat", f"{self.doc_data.get('advantage_score', 0):.1f}/100"],
            ["People & Performance", f"{self.doc_data.get('people_score', 0):.1f}/100"]
        ]
        
        camp_table = Table(camp_data, colWidths=[3*inch, 2*inch])
        camp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
        ]))
        
        story.append(camp_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Add market section
        story.append(Paragraph("Market Analysis", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        market_data = [
            ["Metric", "Value"],
            ["Market Size", self._format_currency(self.doc_data.get("market_size", 0))],
            ["Market Growth Rate", f"{self.doc_data.get('market_growth_rate', 0):.1f}%/year"],
            ["Market Share", f"{self.doc_data.get('market_share', 0):.2f}%"],
            ["User Growth Rate", f"{self.doc_data.get('user_growth_rate', 0):.1f}%/month"]
        ]
        
        market_table = Table(market_data, colWidths=[2.5*inch, 2.5*inch])
        market_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.Color(*[c/255 for c in BACKGROUND_COLOR])),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
        ]))
        
        story.append(market_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Build the document
        doc.build(story)
        
        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    def _generate_emergency_pdf(self) -> PdfBytes:
        """Generate a minimal but reliable PDF report with essential information."""
        # Check if ReportLab is available
        if not self.capabilities["reportlab"]:
            logger.error("ReportLab not available - returning minimal PDF")
            return self._generate_minimal_pdf()
        
        try:
            # Create a buffer for the PDF
            buffer = io.BytesIO()
            
            # Create the document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                leftMargin=0.5*inch,
                rightMargin=0.5*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Create custom style for title
            title_style = ParagraphStyle(
                'EmergencyTitle',
                parent=styles['Title'],
                fontSize=20,
                alignment=TA_CENTER,
                textColor=colors.Color(*[c/255 for c in PRIMARY_COLOR])
            )
            
            # Create the story (content elements)
            story = []
            
            # Add title
            story.append(Paragraph(f"{self.company_name} - Investor Report", title_style))
            story.append(Spacer(1, 0.25*inch))
            
            # Add generation date
            story.append(Paragraph(
                f"Generated: {self.generation_date.strftime('%Y-%m-%d %H:%M')}",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.5*inch))
            
            # Add key metrics table
            story.append(Paragraph("Key Metrics", styles['Heading2']))
            story.append(Spacer(1, 0.15*inch))
            
            metrics_data = [
                ["Metric", "Value"],
                ["CAMP Score", f"{self.doc_data.get('camp_score', 0):.1f}/100"],
                ["Success Probability", f"{self.doc_data.get('success_prob', 0):.1f}%"],
                ["Runway", f"{self.doc_data.get('runway_months', 0):.1f} months"],
                ["Monthly Revenue", self._format_currency(self.doc_data.get('monthly_revenue', 0))],
                ["Burn Rate", self._format_currency(self.doc_data.get('burn_rate', 0))]
            ]
            
            # Create table
            metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
                ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Add CAMP breakdown
            story.append(Paragraph("CAMP Framework Scores", styles['Heading2']))
            story.append(Spacer(1, 0.15*inch))
            
            camp_data = [
                ["Dimension", "Score"],
                ["Capital Efficiency", f"{self.doc_data.get('capital_score', 0):.1f}/100"],
                ["Market Dynamics", f"{self.doc_data.get('market_score', 0):.1f}/100"],
                ["Advantage Moat", f"{self.doc_data.get('advantage_score', 0):.1f}/100"],
                ["People & Performance", f"{self.doc_data.get('people_score', 0):.1f}/100"]
            ]
            
            camp_table = Table(camp_data, colWidths=[2.5*inch, 2.5*inch])
            camp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
                ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
            ]))
            
            story.append(camp_table)
            
            # Build the document
            doc.build(story)
            
            # Get the PDF data
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return pdf_data
            
        except Exception as e:
            logger.critical(f"Emergency PDF generation failed: {str(e)}")
            logger.critical(traceback.format_exc())
            
            # Ultimate fallback
            return self._generate_minimal_pdf()
    
    def _generate_minimal_pdf(self) -> PdfBytes:
        """
        Generate an absolute minimal PDF as the ultimate fallback.
        This uses manual PDF construction without any libraries.
        """
        # Create a minimal PDF file with company name and current date
        pdf_content = f"""
%PDF-1.4
1 0 obj
<</Type /Catalog /Pages 2 0 R>>
endobj
2 0 obj
<</Type /Pages /Kids [3 0 R] /Count 1>>
endobj
3 0 obj
<</Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R>>
endobj
4 0 obj
<</Font <</F1 6 0 R>>>>
endobj
5 0 obj
<</Length 160>>
stream
BT
/F1 24 Tf
72 720 Td
({self.company_name} - Investor Report) Tj
/F1 12 Tf
0 -40 Td
(Generated: {self.generation_date.strftime('%Y-%m-%d %H:%M')}) Tj
0 -20 Td
(CAMP Score: {self.doc_data.get('camp_score', 0):.1f}/100) Tj
ET
endstream
endobj
6 0 obj
<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>
endobj
xref
0 7
0000000000 65535 f
0000000010 00000 n
0000000056 00000 n
0000000111 00000 n
0000000212 00000 n
0000000253 00000 n
0000000463 00000 n
trailer
<</Size 7 /Root 1 0 R>>
startxref
531
%%EOF
"""
        return pdf_content.strip().encode('latin1')
    
    def _format_currency(self, value: Union[int, float]) -> str:
        """Format a value as currency with appropriate scale."""
        try:
            value = float(value)
            if value >= 1_000_000_000:
                return f"${value/1_000_000_000:.2f}B"
            elif value >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            elif value >= 1_000:
                return f"${value/1_000:.1f}K"
            else:
                return f"${value:.2f}"
        except (ValueError, TypeError):
            return "$0.00"

def generate_pdf(doc_data: DocumentData, 
                output_path: Optional[str] = None, 
                report_type: str = "full", 
                sections: Optional[Dict[str, bool]] = None) -> Union[PdfBytes, bool]:
    """
    Generate a PDF report from the provided document data.
    
    Args:
        doc_data: Dictionary containing all startup analysis data
        output_path: Path where the PDF should be saved (if None, returns bytes)
        report_type: Type of report to generate ("full", "executive", "custom")
        sections: Dictionary mapping section names to boolean inclusion flags
        
    Returns:
        bytes or bool: PDF data as bytes if output_path is None, otherwise True/False for success
    """
    try:
        # Create PDF generator
        generator = RobustPDFGenerator(doc_data, report_type, sections)
        
        # Generate PDF
        pdf_data = generator.generate_pdf()
        
        # Save to file if path provided
        if output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Write PDF to file
            with open(output_path, 'wb') as f:
                f.write(pdf_data)
            
            logger.info(f"PDF report saved to {output_path}")
            return True
        
        # Otherwise return PDF data
        return pdf_data
    
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        logger.error(traceback.format_exc())
        
        if output_path:
            return False
        else:
            # Return a minimal PDF as fallback
            return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Resources<<>>/Contents 4 0 R/Parent 2 0 R>>endobj 4 0 obj<</Length 21>>stream\nBT /F1 12 Tf 100 700 Td (Error generating report) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000199 00000 n\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n269\n%%EOF"

# Compatibility aliases for the existing functions
generate_enhanced_pdf = generate_pdf
generate_investor_report = generate_pdf
