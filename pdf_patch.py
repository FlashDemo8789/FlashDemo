"""
PDF Generation Compatibility Patch

This module ensures all PDF generation functions are available globally
and in all the expected modules.
"""

import sys
import os
import importlib
import logging

logger = logging.getLogger("pdf_patch")
logger.setLevel(logging.INFO)

def apply_patch():
    """Apply the PDF generation compatibility patch."""
    logger.info("Applying PDF generation compatibility patch")
    
    # First import our unified generator
    try:
        from unified_pdf_generator import generate_enhanced_pdf, generate_investor_report
        logger.info("Successfully imported unified PDF generator")
    except ImportError:
        logger.error("Failed to import unified_pdf_generator.py - please ensure it exists")
        return False
    
    # Make the functions available in the global namespace
    import builtins
    builtins.generate_enhanced_pdf = generate_enhanced_pdf
    builtins.generate_investor_report = generate_investor_report
    logger.info("Added PDF generation functions to builtins")
    
    # Patch existing modules if they exist
    modules_to_patch = [
        "report_generator", 
        "pdf_generator", 
        "enhanced_pdf_generator",
        "emergency_pdf_generator"
    ]
    
    for module_name in modules_to_patch:
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            
            # Add our functions to the module
            module.generate_enhanced_pdf = generate_enhanced_pdf
            module.generate_investor_report = generate_investor_report
            
            logger.info(f"Successfully patched {module_name}")
        except ImportError:
            logger.warning(f"Module {module_name} not found, skipping")
        except Exception as e:
            logger.error(f"Error patching {module_name}: {e}")
    
    logger.info("PDF generation compatibility patch applied successfully")
    return True

# Apply the patch when this module is imported
apply_patch() 