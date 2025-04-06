"""
Emergency PDF Generator Module

This module provides a last-resort PDF generation capability when all other
PDF generators fail. It uses minimal dependencies and simplifies the report
generation process to maximize the chance of success.
"""

import logging
from fpdf import FPDF
from datetime import datetime
import copy
import traceback

logger = logging.getLogger("emergency_pdf_generator")

class EmergencyPDF(FPDF):
    """Simplified PDF report generator for emergency fallback."""
    
    def __init__(self, title="Investor Report", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        # Set up margins
        self.set_margins(15, 15, 15)
        # Set auto page break
        self.set_auto_page_break(True, margin=25)
        
    def header(self):
        # Report title
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, self.title, 0, 1, 'C')
        
        # Line break
        self.ln(10)
    
    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def emergency_generate_pdf(doc, report_type="full", sections=None):
    """Generate a minimal emergency PDF report when other generators fail."""
    try:
        logger.info("Generating emergency PDF report")
        
        # Create a deep copy of the doc to avoid modifying the original
        doc_copy = copy.deepcopy(doc)
        
        # Initialize PDF report object
        pdf = EmergencyPDF(title=f"{doc_copy.get('name', 'Startup')} Investment Analysis")
        
        # Add cover page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 20, "Investor Report", 0, 1, 'C')
        
        # Startup name
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 30, doc_copy.get('name', 'Startup'), 0, 1, 'C')
        
        # Stage and sector
        pdf.set_font('Arial', '', 14)
        if doc_copy.get('stage'):
            pdf.cell(0, 15, f"Stage: {doc_copy.get('stage')}", 0, 1, 'C')
        if doc_copy.get('sector'):
            pdf.cell(0, 15, f"Sector: {doc_copy.get('sector')}", 0, 1, 'C')
        
        # Date
        pdf.set_font('Arial', 'I', 12)
        pdf.cell(0, 15, f"Generated: {datetime.now().strftime('%B %d, %Y')}", 0, 1, 'C')
        
        # Add CAMP score section if available
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "CAMP Framework Analysis", 0, 1, 'L')
        pdf.ln(5)
        
        camp_score = doc_copy.get('camp_score', 0)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f"Overall CAMP Score: {camp_score:.1f}/100", 0, 1, 'L')
        pdf.ln(5)
        
        # Add individual CAMP components
        for component, label in [
            ('capital_score', 'Capital Efficiency'), 
            ('advantage_score', 'Advantage'), 
            ('market_score', 'Market'),
            ('people_score', 'People')
        ]:
            score = doc_copy.get(component, 0)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f"{label}: {score:.1f}/100", 0, 1, 'L')
        
        # Add summary section
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Executive Summary", 0, 1, 'L')
        pdf.ln(5)
        
        summary = doc_copy.get('summary', '')
        if summary:
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 7, summary)
        else:
            pdf.set_font('Arial', 'I', 11)
            pdf.multi_cell(0, 7, "No summary available")
        
        # Add a note about emergency generation
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "About This Report", 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 7, "This is an emergency version of the investor report. " +
                        "The standard report could not be generated due to technical issues. " +
                        "For a full report, please contact support.")
        
        # Return the PDF as bytes
        try:
            pdf_bytes = pdf.output(dest='S')
            if isinstance(pdf_bytes, str):
                pdf_bytes = pdf_bytes.encode('latin1')
            return pdf_bytes
        except Exception as e:
            logger.error(f"Error in PDF output: {str(e)}")
            # Last resort - try with minimal content
            basic_pdf = EmergencyPDF()
            basic_pdf.add_page()
            basic_pdf.set_font('Arial', 'B', 16)
            basic_pdf.cell(0, 10, "Error Report", 0, 1, 'C')
            basic_pdf.set_font('Arial', '', 12)
            basic_pdf.multi_cell(0, 7, "Unable to generate investor report due to technical issues.")
            basic_bytes = basic_pdf.output(dest='S')
            if isinstance(basic_bytes, str):
                basic_bytes = basic_bytes.encode('latin1')
            return basic_bytes
    
    except Exception as e:
        logger.error(f"Emergency PDF generation failed with error: {str(e)}\n{traceback.format_exc()}")
        # Create an absolute minimal PDF
        try:
            error_pdf = FPDF()
            error_pdf.add_page()
            error_pdf.set_font('Arial', 'B', 16)
            error_pdf.cell(0, 10, "Error Generating Report", 0, 1, 'C')
            error_pdf.set_font('Arial', '', 12)
            error_pdf.multi_cell(0, 7, f"Report generation failed: {str(e)}")
            error_bytes = error_pdf.output(dest='S')
            if isinstance(error_bytes, str):
                error_bytes = error_bytes.encode('latin1')
            return error_bytes
        except:
            # If even this fails, return an empty PDF
            return b"%PDF-1.3\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/Resources <<\n/Font <<\n/F1 4 0 R\n>>\n>>\n/MediaBox [0 0 612 792]\n/Contents 5 0 R\n>>\nendobj\n4 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\nendobj\n5 0 obj\n<< /Length 68 >>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Error: Unable to generate PDF report) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000233 00000 n\n0000000300 00000 n\ntrailer\n<<\n/Size 6\n/Root 1 0 R\n>>\nstartxref\n419\n%%EOF" 