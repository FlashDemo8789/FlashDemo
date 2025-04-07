#!/usr/bin/env python
"""
Test script for PDF generation.
"""

import unified_pdf_generator
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pdf_tester")

# Create a test document
test_doc = {
    'name': 'Test Startup',
    'camp_score': 85,
    'capital_score': 80,
    'market_score': 90,
    'advantage_score': 85,
    'people_score': 85,
    'success_prob': 72,
    'runway_months': 18,
    'monthly_revenue': 85000,
    'burn_rate': 35000,
    'ltv_cac_ratio': 3.2,
    'user_growth_rate': 12.5,
    'revenue_growth_rate': 15.2,
    'market_size': 5000000000,
    'market_growth_rate': 8.5,
    'market_share': 0.5
}

def run_test():
    """Generate a test PDF report."""
    logger.info("Generating PDF report...")
    
    # Try the enhanced PDF first
    try:
        logger.info("Attempting enhanced PDF generation...")
        pdf_data = unified_pdf_generator.generate_enhanced_pdf(test_doc)
        logger.info(f"Generated enhanced PDF with {len(pdf_data)} bytes")
        
        # Save the PDF
        with open('test_report.pdf', 'wb') as f:
            f.write(pdf_data)
        logger.info("Saved to test_report.pdf")
        
        return True
    except Exception as e:
        logger.error(f"Error with enhanced PDF, falling back to emergency PDF: {e}")
        
        try:
            # Fall back to emergency PDF
            logger.info("Attempting emergency PDF generation...")
            pdf_data = unified_pdf_generator.generate_emergency_pdf(test_doc)
            logger.info(f"Generated emergency PDF with {len(pdf_data)} bytes")
            
            # Save the PDF
            with open('test_report_emergency.pdf', 'wb') as f:
                f.write(pdf_data)
            logger.info("Saved to test_report_emergency.pdf")
            
            return True
        except Exception as e2:
            logger.error(f"Emergency PDF generation also failed: {e2}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    success = run_test()
    if success:
        print("PDF generation successful! Check logs for details on which version was generated.")
    else:
        print("All PDF generation methods failed! Check logs for details.") 