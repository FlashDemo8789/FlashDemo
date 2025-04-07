"""
Global PDF Functions

This module provides global access to PDF generation functions from various sources.
It attempts to import from the most reliable source first, then falls back to alternatives.
"""

import logging
import sys
import traceback
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("global_pdf_functions")

# Global variables to store functions once imported
_generate_enhanced_pdf = None
_generate_emergency_pdf = None
_generate_investor_report = None

def _import_pdf_functions():
    """Import PDF functions from various sources, with fallbacks."""
    global _generate_enhanced_pdf, _generate_emergency_pdf, _generate_investor_report
    
    # Try importing from unified_pdf_generator (primary source)
    try:
        logger.info("Attempting to import from unified_pdf_generator")
        from unified_pdf_generator import generate_enhanced_pdf, generate_emergency_pdf, generate_investor_report
        _generate_enhanced_pdf = generate_enhanced_pdf
        _generate_emergency_pdf = generate_emergency_pdf
        _generate_investor_report = generate_investor_report
        logger.info("Successfully imported from unified_pdf_generator")
        return True
    except ImportError as e:
        logger.warning(f"Failed to import from unified_pdf_generator: {str(e)}")
    
    # Try importing from pdf_generator (compatibility layer)
    try:
        logger.info("Attempting to import from pdf_generator")
        from pdf_generator import generate_enhanced_pdf, generate_emergency_pdf, generate_investor_report
        _generate_enhanced_pdf = generate_enhanced_pdf
        _generate_emergency_pdf = generate_emergency_pdf
        _generate_investor_report = generate_investor_report
        logger.info("Successfully imported from pdf_generator")
        return True
    except ImportError as e:
        logger.warning(f"Failed to import from pdf_generator: {str(e)}")
    
    # Define minimal emergency functions
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        logger.info("Creating minimal emergency PDF functions using ReportLab")
        
        def minimal_emergency_pdf(doc):
            """Ultra-minimal emergency PDF generator."""
            buffer = BytesIO()
            doc_template = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Create error style
            error_style = ParagraphStyle(
                'Error',
                parent=styles['Normal'],
                textColor=colors.red
            )
            
            story = [
                Paragraph(f"{doc.get('name', 'Startup')} - Error Report", styles['Title']),
                Spacer(1, 0.25*inch),
                Paragraph("ERROR: PDF generation modules could not be loaded.", error_style),
                Spacer(1, 0.1*inch),
                Paragraph("Please check your installation and dependencies.", styles['Normal'])
            ]
            
            doc_template.build(story)
            return buffer.getvalue()
        
        # Set emergency functions
        _generate_emergency_pdf = minimal_emergency_pdf
        _generate_enhanced_pdf = minimal_emergency_pdf
        _generate_investor_report = minimal_emergency_pdf
        
        logger.info("Created minimal emergency PDF functions")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create minimal emergency functions: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Attempt to import functions at module load time
_import_successful = _import_pdf_functions()

def generate_enhanced_pdf(doc, report_type="full", sections=None):
    """
    Generate an enhanced PDF report with charts and graphs.
    
    This is a proxy function that calls the actual implementation.
    """
    global _generate_enhanced_pdf
    
    if _generate_enhanced_pdf is None:
        if not _import_pdf_functions():
            raise ImportError("Could not import PDF generation functions")
    
    return _generate_enhanced_pdf(doc, report_type, sections)

def generate_emergency_pdf(doc):
    """
    Generate a minimal emergency PDF report when the enhanced version fails.
    
    This is a proxy function that calls the actual implementation.
    """
    global _generate_emergency_pdf
    
    if _generate_emergency_pdf is None:
        if not _import_pdf_functions():
            raise ImportError("Could not import PDF generation functions")
    
    return _generate_emergency_pdf(doc)

def generate_investor_report(doc, report_type="full", sections=None):
    """
    Legacy function name for generate_enhanced_pdf.
    
    This is a proxy function that calls the actual implementation.
    """
    global _generate_investor_report
    
    if _generate_investor_report is None:
        if not _import_pdf_functions():
            raise ImportError("Could not import PDF generation functions")
    
    return _generate_investor_report(doc, report_type, sections)

logger.info("Global PDF functions module initialized") 