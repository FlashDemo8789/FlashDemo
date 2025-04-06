"""
Global PDF Functions

This module ensures that PDF generation functions are available globally.
Any module that imports this file will have access to the PDF generation functions.
"""

import logging
import sys
import builtins

logger = logging.getLogger("global_pdf_functions")
logger.setLevel(logging.INFO)

# Add the functions to builtins if not already there
if not hasattr(builtins, 'generate_enhanced_pdf'):
    logger.info("Adding PDF functions to builtins namespace")
    
    try:
        # Try to import from unified_pdf_generator first
        from unified_pdf_generator import generate_enhanced_pdf, generate_investor_report
        
        builtins.generate_enhanced_pdf = generate_enhanced_pdf
        builtins.generate_investor_report = generate_investor_report
        
        logger.info("Successfully added generate_enhanced_pdf and generate_investor_report to builtins")
        
    except ImportError:
        logger.warning("Could not import from unified_pdf_generator, trying pdf_generator")
        
        try:
            # Try to import from pdf_generator as fallback
            from pdf_generator import generate_enhanced_pdf, generate_investor_report
            
            builtins.generate_enhanced_pdf = generate_enhanced_pdf
            builtins.generate_investor_report = generate_investor_report
            
            logger.info("Successfully added PDF functions from pdf_generator to builtins")
            
        except ImportError:
            logger.error("Could not find PDF generation functions in any module")
            
            # Define fallback functions
            def generate_enhanced_pdf(doc, report_type="full", sections=None):
                """Fallback function when the real function is not available."""
                logger.error("PDF generation failed: No PDF generator module found")
                raise ImportError("PDF generation failed: No PDF generator module found")
            
            def generate_investor_report(doc, report_type="full", sections=None):
                """Fallback function when the real function is not available."""
                logger.error("PDF generation failed: No PDF generator module found")
                raise ImportError("PDF generation failed: No PDF generator module found")
            
            builtins.generate_enhanced_pdf = generate_enhanced_pdf
            builtins.generate_investor_report = generate_investor_report
            
            logger.warning("Added fallback PDF functions to builtins")

# Export the functions at the module level for import statements
try:
    generate_enhanced_pdf = builtins.generate_enhanced_pdf
    generate_investor_report = builtins.generate_investor_report
except AttributeError:
    logger.error("Failed to export PDF functions at module level")

logger.info("Global PDF functions module initialized") 