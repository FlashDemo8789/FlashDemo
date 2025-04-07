"""
Compatibility layer for PDF generation functions.
Imports and re-exports the main PDF generation functions.
"""

import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pdf_generator")

# Try to import the enhanced PDF generator
try:
    from unified_pdf_generator import generate_enhanced_pdf, generate_emergency_pdf, generate_investor_report
    logger.info("Successfully imported PDF generation functions from unified_pdf_generator")
except ImportError as e:
    logger.error(f"Error importing from unified_pdf_generator: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Define fallback functions
    def generate_emergency_pdf(doc):
        """Emergency fallback function when imports fail."""
        logger.error("Using last-resort emergency PDF generator")
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
            import io
            
            # Create a minimal PDF
            buffer = io.BytesIO()
            doc_template = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [Paragraph(f"{doc.get('name', 'Startup')} - Error Report", styles['Title']),
                     Paragraph("Error: Failed to load PDF generation module", styles['Normal'])]
            doc_template.build(story)
            return buffer.getvalue()
        except Exception as inner_e:
            logger.critical(f"Even emergency PDF generation failed: {str(inner_e)}")
            return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Resources<<>>/Contents 4 0 R/Parent 2 0 R>>endobj 4 0 obj<</Length 21>>stream\nBT /F1 12 Tf 100 700 Td (Error) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000199 00000 n\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n269\n%%EOF"
    
    def generate_enhanced_pdf(doc, report_type="full", sections=None):
        """Fallback for the main PDF generation function."""
        logger.error("Using fallback generate_enhanced_pdf due to import failure")
        return generate_emergency_pdf(doc)
    
    # Alias for backward compatibility
    generate_investor_report = generate_enhanced_pdf 