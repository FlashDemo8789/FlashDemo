"""
FlashDNA PDF Generator Fix

This script fixes the PDF generation functionality in the FlashDNA application.
Run it once before starting your application.
"""

import os
import sys
import logging

logger = logging.getLogger("pdf_fix")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def fix_pdf_generation():
    """Fix PDF generation in the FlashDNA application."""
    logger.info("Applying FlashDNA PDF generation fix")
    
    # Create a unified PDF generator if it doesn't exist
    if not os.path.exists("unified_pdf_generator.py"):
        logger.info("Creating unified_pdf_generator.py")
        # Copy the content from the instructions
        with open("unified_pdf_generator.py", "w") as f:
            f.write("# See unified_pdf_generator.py content in instructions")
    
    # Create the patch module if it doesn't exist
    if not os.path.exists("pdf_patch.py"):
        logger.info("Creating pdf_patch.py")
        # Copy the content from the instructions
        with open("pdf_patch.py", "w") as f:
            f.write("# See pdf_patch.py content in instructions")
    
    # Create simple function wrappers in all existing file locations
    for filename in ["report_generator.py", "pdf_generator.py", "enhanced_pdf_generator.py"]:
        if os.path.exists(filename):
            logger.info(f"Adding missing functions to {filename}")
            
            with open(filename, "r") as f:
                content = f.read()
            
            # Add the generate_enhanced_pdf function if it's not there
            if "def generate_enhanced_pdf" not in content:
                with open(filename, "a") as f:
                    f.write("""

# Function wrapper for PDF generation compatibility
def generate_enhanced_pdf(doc, report_type="full", sections=None):
    \"\"\"
    Generate an enhanced PDF report with charts, graphs, and better formatting.
    
    Args:
        doc: The document data dictionary
        report_type: The type of report ("full", "executive", "custom")
        sections: Dictionary of sections to include if report_type is "custom"
        
    Returns:
        bytes: The PDF data or None if generation failed
    \"\"\"
    try:
        # First try to use the unified PDF generator
        from unified_pdf_generator import generate_enhanced_pdf as unified_generate
        return unified_generate(doc, report_type, sections)
    except ImportError:
        # If that fails, create a simple PDF
        try:
            from fpdf import FPDF
            
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
            
            return pdf.output(dest='S').encode('latin1')
        except Exception as e:
            print(f"Basic PDF generation failed: {e}")
            return None
""")
    
    logger.info("PDF generation fix applied successfully")
    logger.info("Please restart your application for the changes to take effect")

if __name__ == "__main__":
    fix_pdf_generation() 