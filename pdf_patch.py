"""
AI-Enhanced PDF Generation Compatibility Patch

This module ensures all PDF generation functions are available globally
and replaced with our AI-enhanced versions.
"""

import sys
import os
import importlib
import logging

logger = logging.getLogger("ai_pdf_patch")
logger.setLevel(logging.INFO)

def apply_patch():
    """Apply the AI-enhanced PDF generation compatibility patch."""
    logger.info("Applying AI-enhanced PDF generation compatibility patch")
    
    # First import our AI-enhanced generator
    try:
        from ai_enhanced_pdf_generator import generate_ai_enhanced_pdf, generate_enhanced_pdf, generate_investor_report
        logger.info("Successfully imported AI-enhanced PDF generator")
    except ImportError:
        # Try to find our file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        try:
            from ai_enhanced_pdf_generator import generate_ai_enhanced_pdf, generate_enhanced_pdf, generate_investor_report
            logger.info("Successfully imported AI-enhanced PDF generator (via path adjustment)")
        except ImportError:
            logger.error("Failed to import ai_enhanced_pdf_generator.py - please ensure it exists")
            return False
    
    # Make the functions available in the global namespace
    import builtins
    builtins.generate_enhanced_pdf = generate_enhanced_pdf
    builtins.generate_investor_report = generate_investor_report
    builtins.generate_ai_enhanced_pdf = generate_ai_enhanced_pdf
    logger.info("Added AI-enhanced PDF generation functions to builtins")
    
    # Patch existing modules if they exist
    modules_to_patch = [
        "report_generator", 
        "pdf_generator", 
        "enhanced_pdf_generator",
        "emergency_pdf_generator",
        "unified_pdf_generator"
    ]
    
    for module_name in modules_to_patch:
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            
            # Add our functions to the module
            module.generate_enhanced_pdf = generate_enhanced_pdf
            module.generate_investor_report = generate_investor_report
            if hasattr(module, 'generate_ai_enhanced_pdf'):
                module.generate_ai_enhanced_pdf = generate_ai_enhanced_pdf
            else:
                setattr(module, 'generate_ai_enhanced_pdf', generate_ai_enhanced_pdf)
            
            logger.info(f"Successfully patched {module_name} with AI-enhanced functions")
        except ImportError:
            logger.warning(f"Module {module_name} not found, skipping")
        except Exception as e:
            logger.error(f"Error patching {module_name}: {e}")
    
    # Also patch global_pdf_functions
    try:
        import global_pdf_functions
        global_pdf_functions.generate_enhanced_pdf = generate_enhanced_pdf
        global_pdf_functions.generate_investor_report = generate_investor_report
        if hasattr(global_pdf_functions, 'generate_ai_enhanced_pdf'):
            global_pdf_functions.generate_ai_enhanced_pdf = generate_ai_enhanced_pdf
        else:
            setattr(global_pdf_functions, 'generate_ai_enhanced_pdf', generate_ai_enhanced_pdf)
        logger.info("Successfully patched global_pdf_functions")
    except ImportError:
        logger.warning("Module global_pdf_functions not found, skipping")
    except Exception as e:
        logger.error(f"Error patching global_pdf_functions: {e}")
    
    logger.info("AI-enhanced PDF generation compatibility patch applied successfully")
    return True

# Apply the patch when this module is imported
apply_patch() 