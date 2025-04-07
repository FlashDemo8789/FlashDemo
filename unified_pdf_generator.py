"""
Investor-Grade Report Generator for FlashDNA

Built with ReportLab to produce investment banking quality reports for fundraising.
"""

import os
import io
import logging
import tempfile
import copy
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from io import BytesIO
import traceback

# Install ReportLab if not present
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.platypus import PageBreak, KeepTogether, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.charts.textlabels import Label
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.platypus import PageBreak, KeepTogether, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.charts.textlabels import Label

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("report_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("investor_report")

# Brand colors for consistent styling
BRAND_PRIMARY = colors.HexColor('#1F77B4')  # Blue
BRAND_SECONDARY = colors.HexColor('#FF7F0E')  # Orange
BRAND_SUCCESS = colors.HexColor('#2CA02C')  # Green
BRAND_WARNING = colors.HexColor('#D62728')  # Red
BRAND_NEUTRAL = colors.HexColor('#7F7F7F')  # Gray
BRAND_ACCENT = colors.HexColor('#17BECF')  # Light blue

class InvestorReport:
    """
    Class for generating professional investor-grade reports with ReportLab.
    """
    
    def __init__(self, doc_data, report_type="full", sections=None):
        """Initialize the report generator with configuration and styles."""
        self.doc_data = copy.deepcopy(doc_data) if doc_data else {}
        self.report_type = report_type
        self.sections = sections
        self.company_name = self.doc_data.get('name', 'Startup')
        
        # Determine which sections to include
        self._determine_active_sections()
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Story will contain all flowable elements
        self.story = []
        
        logger.info(f"Initialized InvestorReport for {self.company_name}")
    
    def _determine_active_sections(self):
        """Determine which sections to include in the report."""
        if self.report_type == "custom" and self.sections is not None:
            self.active_sections = self.sections
        else:
            # Default sections for full report
            self.active_sections = {
                "Executive Summary": True,
                "Business Model": True,
                "Market Analysis": True,
                "Financial Projections": True,
                "Team Assessment": True,
                "Competitive Analysis": True,
                "Growth Metrics": True,
                "Risk Assessment": True,
                "Exit Strategy": True,
                "Technical Assessment": True,
                "CAMP Details": True,
                "PMF Analysis": True
            }
            
            # For executive report, limit sections
            if self.report_type == "executive":
                for section in ["Growth Metrics", "Risk Assessment", "Technical Assessment", "PMF Analysis"]:
                    self.active_sections[section] = False
    
    def _setup_custom_styles(self):
        """Set up custom paragraph and table styles."""
        # Title styles
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Title'],
            fontSize=24,
            leading=28,
            textColor=BRAND_PRIMARY
        ))
        
        # Heading styles
        self.styles.add(ParagraphStyle(
            name='Heading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=BRAND_PRIMARY,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='Heading2',
            parent=self.styles['Heading2'],
            fontSize=16,
            leading=20,
            textColor=BRAND_PRIMARY,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='Heading3',
            parent=self.styles['Heading3'],
            fontSize=14,
            leading=18,
            textColor=BRAND_PRIMARY,
            spaceAfter=8
        ))
        
        # Body styles
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=6
        ))
        
        # Special styles
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=14,
            textColor=BRAND_PRIMARY,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.darkgrey,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Italic'],
            fontSize=9,
            leading=11,
            alignment=TA_CENTER
        ))
        
        # Financial styles
        self.styles.add(ParagraphStyle(
            name='Financial',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=12,
            fontName='Helvetica-Bold'
        ))
        
        # Table styles
        self.table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BRAND_PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('BOX', (0, 0), (-1, -1), 0.25, colors.grey),
        ])
        
        # Alternating row style
        self.alternating_row_style = TableStyle([
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('BACKGROUND', (0, 2), (-1, 2), colors.white),
            ('BACKGROUND', (0, 4), (-1, 4), colors.white),
            ('BACKGROUND', (0, 6), (-1, 6), colors.white),
            ('BACKGROUND', (0, 8), (-1, 8), colors.white),
        ])
    
    def format_currency(self, value):
        """Format value as currency with proper notation for large numbers."""
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
        except:
            return str(value)
    
    def create_cover_page(self):
        """Create an attractive cover page."""
        try:
            # Add startup logo if available
            logo_paths = [
                "static/logo.png",
                "static/img/logo.png",
                os.path.join(os.path.dirname(__file__), "logo.png"),
                "/app/static/logo.png"
            ]
            
            logo_added = False
            for logo_path in logo_paths:
                if os.path.exists(logo_path):
                    try:
                        img = Image(logo_path, width=2*inch, height=2*inch)
                        img.hAlign = 'CENTER'
                        self.story.append(img)
                        self.story.append(Spacer(1, 0.5*inch))
                        logo_added = True
                        logger.info(f"Added logo from {logo_path}")
                        break
                    except Exception as logo_err:
                        logger.warning(f"Error adding logo from {logo_path}: {str(logo_err)}")
            
            # Add report title
            self.story.append(Paragraph("Investor Report", self.styles['Title']))
            self.story.append(Spacer(1, 0.5*inch))
            
            # Add company name in large font
            company_style = ParagraphStyle(
                'CompanyName',
                parent=self.styles['Title'],
                fontSize=36,
                leading=40,
                alignment=TA_CENTER,
                spaceAfter=30
            )
            self.story.append(Paragraph(self.company_name, company_style))
            self.story.append(Spacer(1, 0.25*inch))
            
            # Add company info
            info_style = ParagraphStyle(
                'CompanyInfo',
                parent=self.styles['Normal'],
                fontSize=14,
                leading=18,
                alignment=TA_CENTER
            )
            
            self.story.append(Paragraph(f"Sector: {self.doc_data.get('sector', 'Technology')}", info_style))
            self.story.append(Paragraph(f"Stage: {self.doc_data.get('stage', 'Growth')}", info_style))
            self.story.append(Spacer(1, 0.5*inch))
            
            # Add CAMP score
            camp_style = ParagraphStyle(
                'CampScore',
                parent=self.styles['Heading1'],
                fontSize=20,
                leading=24,
                alignment=TA_CENTER,
                textColor=BRAND_PRIMARY
            )
            
            camp_score = self.doc_data.get('camp_score', 0)
            self.story.append(Paragraph(f"CAMP Score: {camp_score:.1f}/100", camp_style))
            self.story.append(Spacer(1, 0.5*inch))
            
            # Add key metrics in a table
            data = [
                ["Success Probability", "Runway", "Monthly Revenue"],
                [
                    f"{self.doc_data.get('success_prob', 0):.1f}%", 
                    f"{self.doc_data.get('runway_months', 0):.1f} months",
                    self.format_currency(self.doc_data.get('monthly_revenue', 0))
                ]
            ]
            
            metrics_table = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
                ('TEXTCOLOR', (0, 0), (-1, 0), BRAND_PRIMARY),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, 1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, 1), BRAND_PRIMARY),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, 1), 14),
                ('TOPPADDING', (0, 1), (-1, 1), 12),
                ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
                ('BOX', (0, 0), (-1, 1), 1, colors.lightgrey),
                ('GRID', (0, 0), (-1, 1), 0.5, colors.lightgrey),
            ]))
            
            self.story.append(metrics_table)
            self.story.append(Spacer(1, inch))
            
            # Add date and confidentiality notice
            date_style = ParagraphStyle(
                'Date',
                parent=self.styles['Normal'],
                fontSize=12,
                leading=14,
                alignment=TA_CENTER,
                textColor=colors.darkgrey
            )
            
            self.story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", date_style))
            self.story.append(Spacer(1, 0.5*inch))
            
            confidential_style = ParagraphStyle(
                'Confidential',
                parent=self.styles['Italic'],
                fontSize=9,
                leading=11,
                alignment=TA_CENTER,
                textColor=colors.darkgrey
            )
            
            self.story.append(Paragraph("CONFIDENTIAL", confidential_style))
            self.story.append(Paragraph("This report contains confidential information about the company and is intended only for the named recipient.", confidential_style))
            
            # Add page break
            self.story.append(PageBreak())
            logger.info("Added cover page")
            
        except Exception as e:
            logger.error(f"Error creating cover page: {str(e)}\n{traceback.format_exc()}")
            # Add minimal cover page if error occurs
            self.story.append(Paragraph("Investor Report", self.styles['Title']))
            self.story.append(Spacer(1, 0.25*inch))
            self.story.append(Paragraph(self.company_name, self.styles['Heading1']))
            self.story.append(PageBreak())

    def create_table_of_contents(self):
        """Create a table of contents for the report."""
        try:
            self.story.append(Paragraph("Table of Contents", self.styles['Heading1']))
            self.story.append(Spacer(1, 0.2*inch))
            
            toc_sections = []
            
            if self.active_sections.get("Executive Summary", True):
                toc_sections.append("Executive Summary")
                
            if self.active_sections.get("CAMP Details", True):
                toc_sections.append("CAMP Framework Details")
                
            for section, include in self.active_sections.items():
                if include and section not in ["Executive Summary", "CAMP Details"]:
                    toc_sections.append(section)
            
            toc_style = ParagraphStyle(
                'TOC',
                parent=self.styles['Normal'],
                fontSize=12,
                leading=24  # Increased spacing between lines
            )
            
            for section in toc_sections:
                self.story.append(Paragraph(section, toc_style))
            
            self.story.append(PageBreak())
            logger.info("Added table of contents")
            
        except Exception as e:
            logger.error(f"Error creating table of contents: {str(e)}")
            # Skip TOC if error occurs
    
    def create_executive_summary(self):
        """Create the executive summary section."""
        try:
            if not self.active_sections.get("Executive Summary", True):
                return
                
            self.story.append(Paragraph("Executive Summary", self.styles['Heading1']))
            self.story.append(Spacer(1, 0.1*inch))
            
            # Add key metrics
            self.add_metric_row([
                ("CAMP Score", f"{self.doc_data.get('camp_score', 0):.1f}/100"),
                ("Success Probability", f"{self.doc_data.get('success_prob', 0):.1f}%"),
                ("Runway", f"{self.doc_data.get('runway_months', 0):.1f} months")
            ])
            
            self.story.append(Spacer(1, 0.2*inch))
            
            # Add CAMP radar chart
            self.add_camp_radar_chart()
            
            # Add CAMP scores table
            self.story.append(Paragraph("CAMP Framework Scores", self.styles['Heading3']))
            
            camp_data = [
                ["Dimension", "Score"],
                ["Capital Efficiency", f"{self.doc_data.get('capital_score', 0):.1f}/100"],
                ["Market Dynamics", f"{self.doc_data.get('market_score', 0):.1f}/100"],
                ["Advantage Moat", f"{self.doc_data.get('advantage_score', 0):.1f}/100"],
                ["People & Performance", f"{self.doc_data.get('people_score', 0):.1f}/100"]
            ]
            
            camp_table = Table(camp_data, colWidths=[3*inch, 1.5*inch])
            camp_table.setStyle(self.table_style)
            
            # Add alternating row style
            camp_table.setStyle(self.alternating_row_style)
            
            self.story.append(camp_table)
            self.story.append(Spacer(1, 0.2*inch))
            
            # Key strengths and weaknesses
            self.story.append(Paragraph("Key Strengths", self.styles['Heading3']))
            
            # Extract strengths from patterns
            strengths = []
            for pattern in self.doc_data.get("patterns_matched", []):
                if isinstance(pattern, dict) and pattern.get("is_positive", False):
                    strengths.append(pattern.get("name", ""))
            
            # Add default strengths if none found
            if not strengths:
                if self.doc_data.get('capital_score', 0) > 70:
                    strengths.append("Strong capital efficiency")
                if self.doc_data.get('market_score', 0) > 70:
                    strengths.append("Strong market positioning")
                if self.doc_data.get('advantage_score', 0) > 70:
                    strengths.append("Strong competitive advantage")
                if self.doc_data.get('people_score', 0) > 70:
                    strengths.append("Strong team execution")
                
                # Add at least one strength
                if not strengths:
                    max_score = max([
                        self.doc_data.get('capital_score', 0),
                        self.doc_data.get('market_score', 0),
                        self.doc_data.get('advantage_score', 0),
                        self.doc_data.get('people_score', 0)
                    ])
                    
                    if max_score == self.doc_data.get('capital_score', 0):
                        strengths.append("Relative strength in capital efficiency")
                    elif max_score == self.doc_data.get('market_score', 0):
                        strengths.append("Relative strength in market positioning")
                    elif max_score == self.doc_data.get('advantage_score', 0):
                        strengths.append("Relative strength in competitive advantage")
                    elif max_score == self.doc_data.get('people_score', 0):
                        strengths.append("Relative strength in team execution")
            
            # Add strengths as bullet points
            strength_items = []
            for strength in strengths[:3]:
                strength_items.append(ListItem(Paragraph(strength, self.styles['BodyText'])))
            
            if strength_items:
                self.story.append(ListFlowable(strength_items, bulletType='bullet', start=''))
            
            self.story.append(Spacer(1, 0.2*inch))
            
            # Add weaknesses section
            self.story.append(Paragraph("Areas for Improvement", self.styles['Heading3']))
            
            # Extract weaknesses from patterns
            weaknesses = []
            for pattern in self.doc_data.get("patterns_matched", []):
                if isinstance(pattern, dict) and not pattern.get("is_positive", True):
                    weaknesses.append(pattern.get("name", ""))
            
            # Add default weaknesses if none found
            if not weaknesses:
                if self.doc_data.get('capital_score', 0) < 50:
                    weaknesses.append("Improve capital efficiency")
                if self.doc_data.get('market_score', 0) < 50:
                    weaknesses.append("Strengthen market positioning")
                if self.doc_data.get('advantage_score', 0) < 50:
                    weaknesses.append("Build stronger competitive moat")
                if self.doc_data.get('people_score', 0) < 50:
                    weaknesses.append("Enhance team capabilities")
                
                # Add at least one weakness
                if not weaknesses:
                    min_score = min([
                        self.doc_data.get('capital_score', 0),
                        self.doc_data.get('market_score', 0),
                        self.doc_data.get('advantage_score', 0),
                        self.doc_data.get('people_score', 0)
                    ])
                    
                    if min_score == self.doc_data.get('capital_score', 0):
                        weaknesses.append("Consider improving capital efficiency")
                    elif min_score == self.doc_data.get('market_score', 0):
                        weaknesses.append("Consider improving market positioning")
                    elif min_score == self.doc_data.get('advantage_score', 0):
                        weaknesses.append("Consider strengthening competitive advantage")
                    elif min_score == self.doc_data.get('people_score', 0):
                        weaknesses.append("Consider enhancing team capabilities")
            
            # Add weaknesses as bullet points
            weakness_items = []
            for weakness in weaknesses[:3]:
                weakness_items.append(ListItem(Paragraph(weakness, self.styles['BodyText'])))
            
            if weakness_items:
                self.story.append(ListFlowable(weakness_items, bulletType='bullet', start=''))
            
            self.story.append(PageBreak())
            logger.info("Added executive summary")
            
        except Exception as e:
            logger.error(f"Error creating executive summary: {str(e)}\n{traceback.format_exc()}")
            # Add basic executive summary if error occurs
            self.story.append(Paragraph("Executive Summary", self.styles['Heading1']))
            self.story.append(Paragraph("Error generating full executive summary.", self.styles['BodyText']))
            self.story.append(PageBreak())
    
    def create_camp_details(self):
        """Create the CAMP framework details section."""
        try:
            if not self.active_sections.get("CAMP Details", True):
                return
                
            self.story.append(Paragraph("CAMP Framework Details", self.styles['Heading1']))
            
            # Capital efficiency
            self.story.append(Paragraph("Capital Efficiency", self.styles['Heading2']))
            self.story.append(Paragraph(f"Score: {self.doc_data.get('capital_score', 0):.1f}/100", self.styles['Heading3']))
            
            capital_metrics = [
                ("Monthly Revenue", self.format_currency(self.doc_data.get('monthly_revenue', 0))),
                ("Burn Rate", self.format_currency(self.doc_data.get('burn_rate', 0))),
                ("Runway", f"{self.doc_data.get('runway_months', 0):.1f} months")
            ]
            
            self.add_metric_row(capital_metrics)
            
            capital_metrics2 = [
                ("Gross Margin", f"{self.doc_data.get('gross_margin_percent', 0):.1f}%"),
                ("LTV:CAC Ratio", f"{self.doc_data.get('ltv_cac_ratio', 0):.2f}"),
                ("CAC", self.format_currency(self.doc_data.get('customer_acquisition_cost', 0)))
            ]
            
            self.add_metric_row(capital_metrics2)
            
            # Add cash flow chart if available
            cash_flow = self.doc_data.get("cash_flow", [])
            if cash_flow:
                self.add_line_chart(
                    list(range(len(cash_flow))),
                    cash_flow,
                    "Cash Flow Projection",
                    "Month",
                    "Cash ($)"
                )
            
            # Market dynamics
            self.story.append(Paragraph("Market Dynamics", self.styles['Heading2']))
            self.story.append(Paragraph(f"Score: {self.doc_data.get('market_score', 0):.1f}/100", self.styles['Heading3']))
            
            market_metrics = [
                ("Market Size", self.format_currency(self.doc_data.get('market_size', 0))),
                ("Market Growth", f"{self.doc_data.get('market_growth_rate', 0):.1f}%/yr"),
                ("Market Share", f"{self.doc_data.get('market_share', 0):.2f}%")
            ]
            
            self.add_metric_row(market_metrics)
            
            market_metrics2 = [
                ("User Growth", f"{self.doc_data.get('user_growth_rate', 0):.1f}%/mo"),
                ("Revenue Growth", f"{self.doc_data.get('revenue_growth_rate', 0):.1f}%/mo"),
                ("Churn Rate", f"{self.doc_data.get('churn_rate', 0):.1f}%/mo")
            ]
            
            self.add_metric_row(market_metrics2)
            
            # Advantage moat
            self.story.append(Paragraph("Advantage Moat", self.styles['Heading2']))
            self.story.append(Paragraph(f"Score: {self.doc_data.get('advantage_score', 0):.1f}/100", self.styles['Heading3']))
            
            advantage_metrics = [
                ("Technical Innovation", f"{self.doc_data.get('technical_innovation_score', 0):.1f}/100"),
                ("Product Maturity", f"{self.doc_data.get('product_maturity_score', 0):.1f}/100"),
                ("Moat Score", f"{self.doc_data.get('moat_score', 0):.1f}/100")
            ]
            
            self.add_metric_row(advantage_metrics)
            
            # People & performance
            self.story.append(Paragraph("People & Performance", self.styles['Heading2']))
            self.story.append(Paragraph(f"Score: {self.doc_data.get('people_score', 0):.1f}/100", self.styles['Heading3']))
            
            people_metrics = [
                ("Team Score", f"{self.doc_data.get('team_score', 0):.1f}/100"),
                ("Founder Experience", f"{self.doc_data.get('founder_domain_exp_yrs', 0)} years"),
                ("Previous Exits", f"{self.doc_data.get('founder_exits', 0)}")
            ]
            
            self.add_metric_row(people_metrics)
            
            self.story.append(PageBreak())
            logger.info("Added CAMP details")
            
        except Exception as e:
            logger.error(f"Error creating CAMP details: {str(e)}\n{traceback.format_exc()}")
            # Add basic CAMP details if error occurs
            self.story.append(Paragraph("CAMP Framework Details", self.styles['Heading1']))
            self.story.append(Paragraph("Error generating full CAMP details.", self.styles['BodyText']))
            self.story.append(PageBreak())
    
    def create_business_model_section(self):
        """Create the business model section."""
        try:
            if not self.active_sections.get("Business Model", True):
                return
                
            self.story.append(Paragraph("Business Model", self.styles['Heading1']))
            
            # Business model description
            business_model = self.doc_data.get("business_model", "")
            if business_model:
                self.story.append(Paragraph(business_model, self.styles['BodyText']))
            else:
                self.story.append(Paragraph("No business model description available.", self.styles['BodyText']))
            
            self.story.append(Spacer(1, 0.2*inch))
            
            # Unit economics
            unit_econ = self.doc_data.get("unit_economics", {})
            if unit_econ:
                self.story.append(Paragraph("Unit Economics", self.styles['Heading2']))
                
                # Extract values with defaults
                ltv = unit_econ.get('ltv', 0)
                cac = unit_econ.get('cac', 0)
                ratio = unit_econ.get('ltv_cac_ratio', 0)
                payback = unit_econ.get('cac_payback_months', 0)
                
                # Add metrics
                unit_metrics = [
                    ("LTV", self.format_currency(ltv)),
                    ("CAC", self.format_currency(cac)),
                    ("LTV:CAC Ratio", f"{ratio:.2f}"),
                    ("CAC Payback", f"{payback:.1f} months")
                ]
                
                self.add_metric_row(unit_metrics)
                
                # Add visual LTV to CAC comparison
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("LTV vs CAC Comparison", self.styles['Heading3']))
                
                # Create bar chart
                if ltv > 0 or cac > 0:
                    self.add_bar_chart(
                        ['LTV', 'CAC'], 
                        [ltv, cac], 
                        "Customer Value vs Acquisition Cost", 
                        "Value ($)",
                        [BRAND_SUCCESS, BRAND_WARNING]
                    )
                
                # Add interpretation
                if ratio >= 3:
                    assessment = "Strong unit economics with LTV significantly higher than CAC."
                elif ratio >= 1:
                    assessment = "Positive unit economics, but room for improvement in the LTV:CAC ratio."
                else:
                    assessment = "Concerning unit economics with CAC higher than LTV. Focus on improving this ratio."
                
                self.story.append(Paragraph(f"Assessment: {assessment}", self.styles['BodyText']))
            
            # Financial projections
            financial_forecast = self.doc_data.get("financial_forecast", {})
            if financial_forecast:
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("Financial Projections", self.styles['Heading2']))
                
                # Extract revenue and profit data
                months = financial_forecast.get("months", [])
                revenue = financial_forecast.get("revenue", [])
                profit = financial_forecast.get("profit", [])
                
                if months and revenue and len(months) == len(revenue):
                    # Display summary metrics
                    total_revenue = sum(revenue)
                    avg_monthly_revenue = total_revenue / len(revenue) if revenue else 0
                    growth_rate = ((revenue[-1] / revenue[0]) - 1) * 100 if len(revenue) > 1 and revenue[0] > 0 else 0
                    
                    fin_metrics = [
                        ("Total Revenue", self.format_currency(total_revenue)),
                        ("Avg. Monthly Revenue", self.format_currency(avg_monthly_revenue)),
                        ("Growth Rate", f"{growth_rate:.1f}%")
                    ]
                    
                    self.add_metric_row(fin_metrics)
                    
                    # Add revenue chart
                    self.add_line_chart(
                        months,
                        revenue,
                        "Revenue Projection",
                        "Month",
                        "Revenue ($)"
                    )
                    
                    # Add profit chart if available
                    if profit and len(profit) == len(months):
                        self.add_line_chart(
                            months,
                            profit,
                            "Profit Projection",
                            "Month",
                            "Profit ($)"
                        )
            
            self.story.append(PageBreak())
            logger.info("Added business model section")
            
        except Exception as e:
            logger.error(f"Error creating business model section: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Business Model", self.styles['Heading1']))
            self.story.append(Paragraph("Error generating full business model section.", self.styles['BodyText']))
            self.story.append(PageBreak())

    def create_market_analysis_section(self):
        """Create the market analysis section."""
        try:
            if not self.active_sections.get("Market Analysis", True):
                return
                
            self.story.append(Paragraph("Market Analysis", self.styles['Heading1']))
            
            # Market metrics
            market_metrics = [
                ("Market Size", self.format_currency(self.doc_data.get('market_size', 0))),
                ("Market Growth Rate", f"{self.doc_data.get('market_growth_rate', 0):.1f}%/yr"),
                ("Market Share", f"{self.doc_data.get('market_share', 0):.2f}%")
            ]
            
            self.add_metric_row(market_metrics)
            
            # Market breakdown
            market_trends = self.doc_data.get("market_trends", {})
            if isinstance(market_trends, dict) and "trends" in market_trends:
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("Market Trends", self.styles['Heading2']))
                
                trends = market_trends.get("trends", [])
                trend_items = []
                
                for trend in trends:
                    if isinstance(trend, dict):
                        trend_text = f"{trend.get('name', '')}: {trend.get('description', '')}"
                        trend_items.append(ListItem(Paragraph(trend_text, self.styles['BodyText'])))
                    elif isinstance(trend, str):
                        trend_items.append(ListItem(Paragraph(trend, self.styles['BodyText'])))
                
                if trend_items:
                    self.story.append(ListFlowable(trend_items, bulletType='bullet', start=''))
            
            # Competitive position
            competitive_pos = self.doc_data.get("competitive_positioning", {})
            if competitive_pos:
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("Competitive Position", self.styles['Heading2']))
                
                position = competitive_pos.get("position", "challenger")
                self.story.append(Paragraph(f"Current Position: {position.capitalize()}", self.styles['BodyText']))
                
                # Add competitive advantages
                advantages = competitive_pos.get("advantages", [])
                if advantages:
                    self.story.append(Spacer(1, 0.1*inch))
                    self.story.append(Paragraph("Competitive Advantages", self.styles['Heading3']))
                    
                    advantages_names = []
                    advantages_scores = []
                    
                    for adv in advantages:
                        if isinstance(adv, dict):
                            advantages_names.append(adv.get("name", ""))
                            advantages_scores.append(adv.get("score", 0))
                    
                    if advantages_names and advantages_scores:
                        self.add_bar_chart(
                            advantages_names,
                            advantages_scores,
                            "Competitive Advantages",
                            "Score"
                        )
            
            # PMF Analysis if available
            pmf = self.doc_data.get("pmf_analysis", {})
            if pmf:
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("Product-Market Fit", self.styles['Heading2']))
                
                pmf_score = pmf.get("pmf_score", 0)
                pmf_stage = pmf.get("stage", "")
                
                pmf_metrics = [
                    ("PMF Score", f"{pmf_score:.1f}/100"),
                    ("Stage", f"{pmf_stage}"),
                    ("Retention Rate", f"{pmf.get('retention_rate', 0):.1f}%")
                ]
                
                self.add_metric_row(pmf_metrics)
            
            # Market penetration chart
            market_penetration = self.doc_data.get("market_penetration", {})
            if isinstance(market_penetration, dict) and "timeline" in market_penetration and "penetration" in market_penetration:
                timeline = market_penetration.get("timeline", [])
                penetration = [p * 100 for p in market_penetration.get("penetration", [])]  # Convert to percentage
                
                if timeline and penetration and len(timeline) == len(penetration):
                    self.story.append(Spacer(1, 0.2*inch))
                    self.add_line_chart(
                        timeline,
                        penetration,
                        "Market Penetration Projection",
                        "Month",
                        "Penetration (%)"
                    )
            
            self.story.append(PageBreak())
            logger.info("Added market analysis section")
            
        except Exception as e:
            logger.error(f"Error creating market analysis section: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Market Analysis", self.styles['Heading1']))
            self.story.append(Paragraph("Error generating full market analysis section.", self.styles['BodyText']))
            self.story.append(PageBreak())
    
    def create_team_assessment_section(self):
        """Create the team assessment section."""
        try:
            if not self.active_sections.get("Team Assessment", True):
                return
                
            self.story.append(Paragraph("Team Assessment", self.styles['Heading1']))
            
            # Team metrics
            team_metrics = [
                ("Team Score", f"{self.doc_data.get('team_score', 0):.1f}/100"),
                ("Founder Experience", f"{self.doc_data.get('founder_domain_exp_yrs', 0)} years"),
                ("Previous Exits", f"{self.doc_data.get('founder_exits', 0)}")
            ]
            
            self.add_metric_row(team_metrics)
            
            team_metrics2 = [
                ("Team Size", f"{self.doc_data.get('employee_count', 0)} employees"),
                ("Tech Talent Ratio", f"{self.doc_data.get('tech_talent_ratio', 0)*100:.1f}%"),
                ("Team Diversity", f"{self.doc_data.get('founder_diversity_score', 0):.1f}/100")
            ]
            
            self.add_metric_row(team_metrics2)
            
            # Leadership team
            self.story.append(Spacer(1, 0.2*inch))
            self.story.append(Paragraph("Leadership Team", self.styles['Heading2']))
            
            leadership = {
                "CEO": True,  # Assumed always present
                "CTO": self.doc_data.get("has_cto", False),
                "CMO": self.doc_data.get("has_cmo", False),
                "CFO": self.doc_data.get("has_cfo", False)
            }
            
            # Create a visual representation of the leadership team
            leaders = list(leadership.keys())
            status = [1 if v else 0 for v in leadership.values()]
            
            self.add_bar_chart(
                leaders,
                status,
                "Leadership Positions",
                "Present (1) / Absent (0)",
                [BRAND_SUCCESS if s else BRAND_WARNING for s in status]
            )
            
            # Execution risk
            execution_risk = self.doc_data.get("execution_risk", {})
            if isinstance(execution_risk, dict) and "risk_factors" in execution_risk:
                risk_factors = execution_risk.get("risk_factors", {})
                
                if risk_factors:
                    self.story.append(Spacer(1, 0.2*inch))
                    self.story.append(Paragraph("Execution Risk Factors", self.styles['Heading2']))
                    
                    factors = list(risk_factors.keys())
                    scores = list(risk_factors.values())
                    
                    if factors and scores:
                        self.add_bar_chart(
                            factors,
                            scores,
                            "Risk Factors",
                            "Risk Level"
                        )
            
            self.story.append(PageBreak())
            logger.info("Added team assessment section")
            
        except Exception as e:
            logger.error(f"Error creating team assessment section: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Team Assessment", self.styles['Heading1']))
            self.story.append(Paragraph("Error generating full team assessment section.", self.styles['BodyText']))
            self.story.append(PageBreak())
    
    def create_competitive_analysis(self):
        """Create the competitive analysis section."""
        try:
            if not self.active_sections.get("Competitive Analysis", True):
                return
                
            self.story.append(Paragraph("Competitive Analysis", self.styles['Heading1']))
            
            # Competitors list
            competitors = self.doc_data.get("competitors", [])
            if competitors and all(isinstance(comp, dict) for comp in competitors):
                self.story.append(Paragraph("Key Competitors", self.styles['Heading2']))
                
                # Prepare header and data for the table
                headers = ["Competitor", "Funding", "Founded", "Threat Level"]
                data = [headers]
                
                for comp in competitors:
                    row = [
                        comp.get("name", ""),
                        self.format_currency(comp.get("funding", 0)),
                        comp.get("founded", ""),
                        comp.get("threat_level", "Medium")
                    ]
                    data.append(row)
                
                # Create the table
                competitors_table = Table(data, colWidths=[2*inch, inch, inch, 1.5*inch])
                competitors_table.setStyle(self.table_style)
                
                # Add alternating row style
                competitors_table.setStyle(self.alternating_row_style)
                
                self.story.append(competitors_table)
                self.story.append(Spacer(1, 0.2*inch))
            
            # Competitive positioning
            positioning = self.doc_data.get("competitive_positioning", {})
            if positioning:
                self.story.append(Paragraph("Competitive Positioning", self.styles['Heading2']))
                
                # Extract positioning data
                dimensions = positioning.get("dimensions", [])
                company_position = positioning.get("company_position", {})
                competitor_positions = positioning.get("competitor_positions", {})
                
                if dimensions and company_position and competitor_positions and len(dimensions) >= 2:
                    # Add description of positioning
                    position_text = f"Current competitive position: {positioning.get('position', 'Challenger').capitalize()}"
                    self.story.append(Paragraph(position_text, self.styles['BodyText']))
                    self.story.append(Spacer(1, 0.1*inch))
                    
                    # Create a competitor comparison table
                    if len(dimensions) >= 2:
                        x_dim = dimensions[0]
                        y_dim = dimensions[1]
                        
                        headers = ["Company", x_dim, y_dim]
                        data = [headers]
                        
                        # Add company data
                        company_x = company_position.get(x_dim, 50)
                        company_y = company_position.get(y_dim, 50)
                        data.append([self.company_name, f"{company_x:.1f}", f"{company_y:.1f}"])
                        
                        # Add competitor data
                        for comp_name, comp_pos in competitor_positions.items():
                            comp_x = comp_pos.get(x_dim, 50)
                            comp_y = comp_pos.get(y_dim, 50)
                            data.append([comp_name, f"{comp_x:.1f}", f"{comp_y:.1f}"])
                        
                        # Create the table
                        position_table = Table(data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                        position_table.setStyle(self.table_style)
                        
                        # Add alternating row style
                        position_table.setStyle(self.alternating_row_style)
                        
                        self.story.append(position_table)
            
            # Network effects
            network = self.doc_data.get("network_analysis", {})
            if network:
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("Network Effects Analysis", self.styles['Heading2']))
                
                # Network effect score
                ne_score = network.get("network_effect_score", 0)
                self.story.append(Paragraph(f"Network Effect Score: {ne_score:.1f}/100", self.styles['BodyText']))
                
                # Network effect types
                ne_types = network.get("network_effect_types", {})
                
                if ne_types:
                    types = list(ne_types.keys())
                    scores = list(ne_types.values())
                    
                    if types and scores:
                        self.add_bar_chart(
                            types,
                            scores,
                            "Network Effect Strength by Type",
                            "Strength"
                        )
            
            self.story.append(PageBreak())
            logger.info("Added competitive analysis section")
            
        except Exception as e:
            logger.error(f"Error creating competitive analysis section: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Competitive Analysis", self.styles['Heading1']))
            self.story.append(Paragraph("Error generating full competitive analysis section.", self.styles['BodyText']))
            self.story.append(PageBreak())

    def add_remaining_sections(self):
        """Add any remaining sections based on active_sections configuration."""
        try:
            # Forecast section
            if self.active_sections.get("Growth Metrics", True):
                self.story.append(Paragraph("Growth Metrics & Forecasts", self.styles['Heading1']))
                
                # User growth projection
                sys_dynamics = self.doc_data.get("system_dynamics", {})
                if isinstance(sys_dynamics, dict) and "users" in sys_dynamics:
                    users = sys_dynamics.get("users", [])
                    months = list(range(len(users)))
                    
                    if users and months:
                        self.story.append(Paragraph("User Growth Projection", self.styles['Heading2']))
                        self.add_line_chart(
                            months, 
                            users, 
                            "Projected User Growth", 
                            "Months", 
                            "Users"
                        )
                
                # Monte Carlo simulation
                monte_carlo = self.doc_data.get("monte_carlo", {})
                if monte_carlo and "user_projections" in monte_carlo:
                    self.story.append(Paragraph("Monte Carlo Simulation", self.styles['Heading2']))
                    
                    # Extract key metrics
                    success_prob = monte_carlo.get("success_probability", 0)
                    median_runway = monte_carlo.get("median_runway_months", 0)
                    
                    # Add metrics
                    mc_metrics = [
                        ("Success Probability", f"{success_prob:.1f}%"),
                        ("Median Runway", f"{median_runway:.1f} months"),
                        ("Simulations", f"{monte_carlo.get('simulation_count', 0)}")
                    ]
                    
                    self.add_metric_row(mc_metrics)
                
                self.story.append(PageBreak())
                logger.info("Added growth metrics section")
            
            # Risk assessment
            if self.active_sections.get("Risk Assessment", True):
                self.story.append(Paragraph("Risk Assessment", self.styles['Heading1']))
                
                # Extract risk factors
                risk_factors = self.doc_data.get("risk_factors", {})
                if risk_factors:
                    self.story.append(Paragraph("Key Risk Factors", self.styles['Heading2']))
                    
                    # Create risk table
                    headers = ["Risk Factor", "Severity", "Mitigation"]
                    data = [headers]
                    
                    for factor, details in risk_factors.items():
                        if isinstance(details, dict):
                            row = [
                                factor,
                                f"{details.get('severity', 0):.1f}/10",
                                details.get("mitigation", "")
                            ]
                            data.append(row)
                        else:
                            row = [factor, f"{details:.1f}/10", ""]
                            data.append(row)
                    
                    # Create the table
                    risk_table = Table(data, colWidths=[2*inch, inch, 3*inch])
                    risk_table.setStyle(self.table_style)
                    
                    # Add alternating row style
                    risk_table.setStyle(self.alternating_row_style)
                    
                    self.story.append(risk_table)
                
                self.story.append(PageBreak())
                logger.info("Added risk assessment section")
            
            # Exit strategy
            if self.active_sections.get("Exit Strategy", True):
                self.story.append(Paragraph("Exit Strategy", self.styles['Heading1']))
                
                exit_analysis = self.doc_data.get("exit_path_analysis", {})
                exit_recs = self.doc_data.get("exit_recommendations", {})
                
                if exit_analysis or exit_recs:
                    # Top metrics and optimal path
                    optimal_path = exit_recs.get("optimal_path", "")
                    readiness = exit_recs.get("readiness", 0)
                    
                    self.story.append(Paragraph(f"Exit Readiness: {readiness:.1f}/100", self.styles['BodyText']))
                    
                    if optimal_path:
                        self.story.append(Paragraph(f"Optimal Exit Path: {exit_recs.get('path_details', {}).get('description', optimal_path)}", self.styles['BodyText']))
                    
                    # Exit timeline
                    timeline = exit_recs.get("timeline", {})
                    if timeline:
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Exit Timeline", self.styles['Heading2']))
                        
                        timeline_metrics = [
                            ("Years to Exit", f"{timeline.get('years_to_exit', 0):.1f}"),
                            ("Exit Year", f"{timeline.get('exit_year', 0)}"),
                            ("Exit Valuation", self.format_currency(timeline.get('exit_valuation', 0)))
                        ]
                        
                        self.add_metric_row(timeline_metrics)
                    
                    # Exit scenarios
                    scenarios = exit_analysis.get("scenarios", [])
                    if scenarios and all(isinstance(s, dict) for s in scenarios):
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Exit Path Scenarios", self.styles['Heading2']))
                        
                        # Create scenarios table
                        headers = ["Exit Path", "Valuation", "Probability", "Time to Exit"]
                        data = [headers]
                        
                        for scenario in scenarios:
                            row = [
                                scenario.get("path_name", ""),
                                self.format_currency(scenario.get("exit_valuation", 0)),
                                f"{scenario.get('probability', 0)*100:.1f}%",
                                f"{scenario.get('time_to_exit', 0):.1f} years"
                            ]
                            data.append(row)
                        
                        # Create the table
                        scenario_table = Table(data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                        scenario_table.setStyle(self.table_style)
                        
                        # Add alternating row style
                        scenario_table.setStyle(self.alternating_row_style)
                        
                        self.story.append(scenario_table)
                
                self.story.append(PageBreak())
                logger.info("Added exit strategy section")
            
            # Technical assessment
            if self.active_sections.get("Technical Assessment", True):
                self.story.append(Paragraph("Technical Assessment", self.styles['Heading1']))
                
                tech_assessment = self.doc_data.get("tech_assessment", {})
                if tech_assessment:
                    # Overall tech score
                    tech_score = tech_assessment.get("overall_score", 0)
                    self.story.append(Paragraph(f"Technical Assessment Score: {tech_score:.1f}/100", self.styles['BodyText']))
                    
                    # Component scores
                    scores = tech_assessment.get("scores", {})
                    if scores:
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Component Scores", self.styles['Heading2']))
                        
                        categories = list(scores.keys())
                        values = list(scores.values())
                        
                        if categories and values:
                            self.add_bar_chart(
                                categories,
                                values,
                                "Technical Component Scores",
                                "Score"
                            )
                    
                    # Tech stack
                    tech_stack = tech_assessment.get("tech_stack", {})
                    if tech_stack:
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Technology Stack", self.styles['Heading2']))
                        
                        stack_categories = list(tech_stack.keys())
                        
                        for category in stack_categories:
                            self.story.append(Paragraph(category, self.styles['Heading3']))
                            
                            technologies = tech_stack[category]
                            
                            if isinstance(technologies, list):
                                tech_text = ", ".join(technologies)
                                self.story.append(Paragraph(tech_text, self.styles['BodyText']))
                            elif isinstance(technologies, dict):
                                tech_items = []
                                for tech, details in technologies.items():
                                    tech_items.append(ListItem(Paragraph(f"{tech}: {details}", self.styles['BodyText'])))
                                
                                if tech_items:
                                    self.story.append(ListFlowable(tech_items, bulletType='bullet', start=''))
                    
                    # Recommendations
                    recommendations = tech_assessment.get("recommendations", [])
                    if recommendations:
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Technical Recommendations", self.styles['Heading2']))
                        
                        rec_items = []
                        for rec in recommendations:
                            rec_items.append(ListItem(Paragraph(rec, self.styles['BodyText'])))
                        
                        if rec_items:
                            self.story.append(ListFlowable(rec_items, bulletType='bullet', start=''))
                
                self.story.append(PageBreak())
                logger.info("Added technical assessment section")
            
        except Exception as e:
            logger.error(f"Error adding remaining sections: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Additional Sections", self.styles['Heading1']))
            self.story.append(Paragraph("Error generating additional sections.", self.styles['BodyText']))
            self.story.append(PageBreak())
    
    def add_metric_row(self, metrics):
        """Add a row of metrics with equal styling."""
        try:
            # Calculate column widths
            col_width = 6.5 * inch / len(metrics)
            
            # Create data for the table
            labels = [m[0] for m in metrics]
            values = [m[1] for m in metrics]
            
            data = [labels, values]
            
            # Create the table
            table = Table(data, colWidths=[col_width] * len(metrics))
            
            # Style the table
            table.setStyle(TableStyle([
                # Headers (labels)
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                
                # Values
                ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
                ('TEXTCOLOR', (0, 1), (-1, 1), BRAND_PRIMARY),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, 1), 14),
                ('TOPPADDING', (0, 1), (-1, 1), 6),
                
                # Table styling
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            self.story.append(table)
            self.story.append(Spacer(1, 0.3*inch))
            
        except Exception as e:
            logger.error(f"Error adding metric row: {str(e)}")
            # Fallback to simple format
            for label, value in metrics:
                self.story.append(Paragraph(f"{label}: {value}", self.styles['BodyText']))
            self.story.append(Spacer(1, 0.2*inch))
    
    def add_camp_radar_chart(self):
        """Add a CAMP framework radar chart using matplotlib."""
        try:
            # Get CAMP scores
            camp_scores = [
                self.doc_data.get('capital_score', 0),
                self.doc_data.get('market_score', 0),
                self.doc_data.get('advantage_score', 0),
                self.doc_data.get('people_score', 0)
            ]
            
            categories = ['Capital', 'Market', 'Advantage', 'People']
            
            # Create radar chart with matplotlib
            plt.clf()
            plt.close('all')
            
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)
            
            # Ensure the plot forms a complete circle by appending the first value at the end
            values = np.array(camp_scores + [camp_scores[0]])
            cat_labels = categories + [categories[0]]
            
            # Compute angle for each category (in radians)
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            angles = np.append(angles, angles[0])
            
            # Draw the chart
            ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
            ax.fill(angles, values, alpha=0.25, color='#1f77b4')
            
            # Set category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # Set radial limits
            ax.set_ylim(0, 100)
            
            # Add gridlines
            ax.grid(True)
            
            # Set title
            plt.title("CAMP Framework Scores", size=14, color='#1f77b4', y=1.1)
            
            # Save to BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            
            # Create image and add to story
            img = Image(buf)
            img.drawHeight = 3.5*inch
            img.drawWidth = 4*inch
            img.hAlign = 'CENTER'
            
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
            
            logger.info("Added CAMP radar chart")
            
        except Exception as e:
            logger.error(f"Error creating CAMP radar chart: {str(e)}\n{traceback.format_exc()}")
            # Skip chart if error occurs
            self.story.append(Paragraph("CAMP Framework Visualization", self.styles['Heading3']))
            self.story.append(Paragraph("Error generating CAMP radar chart.", self.styles['BodyText']))
            self.story.append(Spacer(1, 0.2*inch))
    
    def add_bar_chart(self, categories, values, title="", ylabel="Value", colors=None):
        """Add a bar chart using matplotlib."""
        try:
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            
            # Create bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Generate colors if not provided
            if colors is None:
                colors = [BRAND_PRIMARY.hexval()] * len(categories)
                
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
            
            # Create image and add to story
            img = Image(buf)
            img.drawHeight = 3*inch
            img.drawWidth = 6*inch
            img.hAlign = 'CENTER'
            
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
            
            logger.info(f"Added bar chart: {title}")
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}\n{traceback.format_exc()}")
            # Skip chart if error occurs
            self.story.append(Paragraph(title, self.styles['Heading3']))
            self.story.append(Paragraph("Error generating bar chart.", self.styles['BodyText']))
            self.story.append(Spacer(1, 0.2*inch))
    
    def add_line_chart(self, x_data, y_data, title="", xlabel="", ylabel="", color=None):
        """Add a line chart using matplotlib."""
        try:
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            
            # Create line chart with matplotlib
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Set color
            if color is None:
                color = '#1f77b4'
            
            # Create line
            ax.plot(x_data, y_data, marker='o', color=color, linewidth=2)
            
            # Customize the chart
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Format y-axis with comma for thousands
            if "Revenue" in ylabel or "Cash" in ylabel or "$" in ylabel:
                formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
                ax.yaxis.set_major_formatter(formatter)
            
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save to BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            
            # Create image and add to story
            img = Image(buf)
            img.drawHeight = 3*inch
            img.drawWidth = 6*inch
            img.hAlign = 'CENTER'
            
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
            
            logger.info(f"Added line chart: {title}")
            
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}\n{traceback.format_exc()}")
            # Skip chart if error occurs
            self.story.append(Paragraph(title, self.styles['Heading3']))
            self.story.append(Paragraph("Error generating line chart.", self.styles['BodyText']))
            self.story.append(Spacer(1, 0.2*inch))
    
    def generate_report(self):
        """Generate the complete report."""
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
            
            # Add cover page
            self.create_cover_page()
            
            # Add table of contents
            self.create_table_of_contents()
            
            # Add executive summary
            self.create_executive_summary()
            
            # Add CAMP details
            self.create_camp_details()
            
            # Add business model section
            self.create_business_model_section()
            
            # Add market analysis section
            self.create_market_analysis_section()
            
            # Add team assessment section
            self.create_team_assessment_section()
            
            # Add competitive analysis
            self.create_competitive_analysis()
            
            # Add remaining sections
            self.add_remaining_sections()
            
            # Build the document
            doc.build(self.story)
            
            # Get the PDF data
            pdf_data = buffer.getvalue()
            buffer.close()
            
            logger.info("PDF report generation completed successfully")
            return pdf_data
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}\n{traceback.format_exc()}")
            return generate_emergency_pdf(self.doc_data)

def generate_enhanced_pdf(doc, report_type="full", sections=None):
    """
    Generate an investment banking quality PDF report with ReportLab.
    
    Args:
        doc: The document data dictionary
        report_type: The type of report ("full", "executive", "custom")
        sections: Dictionary of sections to include if report_type is "custom"
        
    Returns:
        bytes: The PDF data
    """
    logger.info(f"Starting enhanced PDF generation with report type: {report_type}")
    
    try:
        # Create the report generator
        report = InvestorReport(doc, report_type, sections)
        
        # Generate the report
        pdf_data = report.generate_report()
        
        return pdf_data
        
    except Exception as e:
        logger.error(f"Error in generate_enhanced_pdf: {str(e)}\n{traceback.format_exc()}")
        return generate_emergency_pdf(doc)

def generate_emergency_pdf(doc):
    """Generate a minimal emergency PDF when the enhanced version fails."""
    logger.info("Generating emergency PDF")
    
    try:
        # Create a buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the document
        doc_template = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            leftMargin=0.5*inch,
            rightMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create the story
        story = []
        
        # Add title
        story.append(Paragraph(f"{doc.get('name', 'Startup')} - Investor Report", styles['Title']))
        story.append(Spacer(1, 0.25*inch))
        
        # Add basic info
        story.append(Paragraph(f"CAMP Score: {doc.get('camp_score', 0):.1f}/100", styles['Normal']))
        story.append(Paragraph(f"Success Probability: {doc.get('success_prob', 0):.1f}%", styles['Normal']))
        story.append(Paragraph(f"Runway: {doc.get('runway_months', 0):.1f} months", styles['Normal']))
        
        # CAMP breakdown
        story.append(Spacer(1, 0.25*inch))
        story.append(Paragraph("CAMP Framework Scores", styles['Heading2']))
        
        story.append(Paragraph(f"Capital Efficiency: {doc.get('capital_score', 0):.1f}/100", styles['Normal']))
        story.append(Paragraph(f"Market Dynamics: {doc.get('market_score', 0):.1f}/100", styles['Normal']))
        story.append(Paragraph(f"Advantage Moat: {doc.get('advantage_score', 0):.1f}/100", styles['Normal']))
        story.append(Paragraph(f"People & Performance: {doc.get('people_score', 0):.1f}/100", styles['Normal']))
        
        # Add core metrics
        story.append(Spacer(1, 0.25*inch))
        story.append(Paragraph("Key Metrics", styles['Heading2']))
        
        def format_currency(value):
            value = float(value)
            if value >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            elif value >= 1_000:
                return f"${value/1_000:.1f}K"
            else:
                return f"${value:.2f}"
        
        story.append(Paragraph(f"Monthly Revenue: {format_currency(doc.get('monthly_revenue', 0))}", styles['Normal']))
        story.append(Paragraph(f"Burn Rate: {format_currency(doc.get('burn_rate', 0))}", styles['Normal']))
        story.append(Paragraph(f"LTV:CAC Ratio: {doc.get('ltv_cac_ratio', 0):.2f}", styles['Normal']))
        
        # Emergency message
        story.append(Spacer(1, 0.25*inch))
        note_style = ParagraphStyle('Note', parent=styles['Italic'], textColor=colors.red)
        story.append(Paragraph("Note: This is a simplified emergency report. A fully detailed report with visualizations will be available soon.", note_style))
        
        # Build the document
        doc_template.build(story)
        
        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
        
    except Exception as e:
        logger.error(f"Emergency PDF generation also failed: {e}")
        # Return an empty PDF if all else fails
        buffer = io.BytesIO()
        doc_template = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph("Error Generating Report", styles['Title'])]
        doc_template.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data

# Alias for backward compatibility
generate_investor_report = generate_enhanced_pdf 