"""
Emergency PDF Generator

This module provides ultra-reliable PDF generation for when all other approaches fail.
It uses only the most basic ReportLab features to maximize reliability.
"""

import logging
import io
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emergency_pdf.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("emergency_pdf")

# Try to import ReportLab
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError as e:
    logger.error(f"ReportLab import error: {str(e)}")
    REPORTLAB_AVAILABLE = False
    # Try to install ReportLab
    try:
        import subprocess
        import sys
        logger.info("Attempting to install ReportLab...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        REPORTLAB_AVAILABLE = True
        logger.info("Successfully installed and imported ReportLab")
    except Exception as install_error:
        logger.error(f"Failed to install ReportLab: {str(install_error)}")
        logger.error(traceback.format_exc())

def emergency_generate_pdf(doc, report_type="full", sections=None):
    """
    Generate a minimal but reliable PDF report with essential information.
    
    Args:
        doc: Document data dictionary
        report_type: Report type (ignored in emergency mode)
        sections: Sections to include (ignored in emergency mode)
    
    Returns:
        bytes: PDF data
    """
    logger.info("Generating emergency PDF report")
    
    # Check if ReportLab is available
    if not REPORTLAB_AVAILABLE:
        logger.error("ReportLab not available - returning minimal PDF")
        # Create an absolute minimal PDF (manually constructed)
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Resources<<>>/Contents 4 0 R/Parent 2 0 R>>endobj 4 0 obj<</Length 21>>stream\nBT /F1 12 Tf 100 700 Td (Error) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000199 00000 n\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n269\n%%EOF"
    
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
        
        # Create custom styles
        title_style = ParagraphStyle(
            'EmergencyTitle',
            parent=styles['Title'],
            fontSize=20,
            alignment=TA_CENTER,
            textColor=colors.navy
        )
        
        heading_style = ParagraphStyle(
            'EmergencyHeading',
            parent=styles['Heading2'],
            fontSize=16,
            alignment=TA_LEFT,
            textColor=colors.navy
        )
        
        metric_label = ParagraphStyle(
            'MetricLabel',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_LEFT,
            textColor=colors.darkgrey
        )
        
        metric_value = ParagraphStyle(
            'MetricValue',
            parent=styles['Normal'],
            fontSize=12,
            alignment=TA_LEFT,
            textColor=colors.black
        )
        
        warning_style = ParagraphStyle(
            'Warning',
            parent=styles['Italic'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.red
        )
        
        # Create the story
        story = []
        
        # Add title
        company_name = doc.get('name', 'Startup')
        story.append(Paragraph(f"{company_name} - Investment Analysis", title_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Add timestamp
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Add key metrics table
        story.append(Paragraph("Key Investment Metrics", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Format currency helper
        def format_currency(value):
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
                return f"${0:.2f}"
        
        # Create data for metrics table
        metrics_data = [
            ["Metric", "Value"],
            ["CAMP Score", f"{doc.get('camp_score', 0):.1f}/100"],
            ["Success Probability", f"{doc.get('success_prob', 0):.1f}%"],
            ["Monthly Revenue", format_currency(doc.get('monthly_revenue', 0))],
            ["Burn Rate", format_currency(doc.get('burn_rate', 0))],
            ["Runway", f"{doc.get('runway_months', 0):.1f} months"],
            ["LTV:CAC Ratio", f"{doc.get('ltv_cac_ratio', 0):.2f}"]
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
            ('BOX', (0, 0), (1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, 1), (1, -1), colors.white),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Add CAMP breakdown
        story.append(Paragraph("CAMP Framework Scores", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        camp_data = [
            ["Dimension", "Score"],
            ["Capital Efficiency", f"{doc.get('capital_score', 0):.1f}/100"],
            ["Market Dynamics", f"{doc.get('market_score', 0):.1f}/100"],
            ["Advantage Moat", f"{doc.get('advantage_score', 0):.1f}/100"],
            ["People & Performance", f"{doc.get('people_score', 0):.1f}/100"]
        ]
        
        camp_table = Table(camp_data, colWidths=[2.5*inch, 2.5*inch])
        camp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (1, -1), 0.25, colors.grey),
            ('BOX', (0, 0), (1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, 1), (1, -1), colors.white),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
        ]))
        
        story.append(camp_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Add emergency message
        story.append(Paragraph("EMERGENCY REPORT MODE", warning_style))
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph("This is a simplified emergency report generated when the full report system encountered a problem.", warning_style))
        story.append(Paragraph("A complete report with charts and detailed analysis will be available soon.", warning_style))
        
        # Build the document
        doc_template.build(story)
        
        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        logger.info("Successfully generated emergency PDF")
        return pdf_data
        
    except Exception as e:
        logger.critical(f"Emergency PDF generation failed: {str(e)}")
        logger.critical(traceback.format_exc())
        
        # Ultimate fallback - return a hand-crafted minimal PDF
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Resources<<>>/Contents 4 0 R/Parent 2 0 R>>endobj 4 0 obj<</Length 21>>stream\nBT /F1 12 Tf 100 700 Td (Emergency Error Report) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000199 00000 n\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n269\n%%EOF" 