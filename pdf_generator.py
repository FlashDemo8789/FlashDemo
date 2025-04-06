"""
PDF Generator Compatibility Module

This module imports functions from unified_pdf_generator.py for backward compatibility.
"""

import logging
logger = logging.getLogger("pdf_generator")

try:
    # Import from unified_pdf_generator
    from unified_pdf_generator import generate_enhanced_pdf, generate_investor_report
    
    logger.info("Successfully imported PDF generation functions from unified_pdf_generator")
    
except ImportError as e:
    logger.error(f"Failed to import PDF generation functions: {e}")
    
    # Define fallback functions
    def generate_enhanced_pdf(doc, report_type="full", sections=None):
        """Fallback function when the real function is not available."""
        logger.error("PDF generation failed: unified_pdf_generator not found")
        raise ImportError("PDF generation failed: unified_pdf_generator not found")
    
    def generate_investor_report(doc, report_type="full", sections=None):
        """Fallback function when the real function is not available."""
        logger.error("PDF generation failed: unified_pdf_generator not found")
        raise ImportError("PDF generation failed: unified_pdf_generator not found") 