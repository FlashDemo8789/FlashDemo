#!/usr/bin/env python
"""
Test script for the robust PDF generator.
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("robust_pdf_test")

# Create the reports directory if it doesn't exist
if not os.path.exists("reports"):
    os.makedirs("reports")

# Create a test document
test_doc = {
    'name': 'Test Startup',
    'company_name': 'Test Startup',
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
    'market_share': 0.5,
    'patterns_matched': [
        {'name': 'Strong Team', 'is_positive': True},
        {'name': 'Growing Market', 'is_positive': True},
        {'name': 'High Burn Rate', 'is_positive': False}
    ],
    'unit_economics': {
        'ltv': 1500,
        'cac': 500,
        'ltv_cac_ratio': 3.0,
        'gross_margin': 0.65,
        'cac_payback_months': 8.5
    },
    'pmf_analysis': {
        'pmf_score': 72,
        'stage': 'Promising',
        'engagement_score': 75,
        'retention_score': 68
    },
    'system_dynamics': {
        'users': [1000, 1200, 1450, 1750, 2100, 2550, 3100, 3800]
    },
    'financial_forecast': {
        'months': list(range(12)),
        'revenue': [85000, 97750, 112413, 129275, 148666, 170966, 196611, 226102, 260018, 299020, 343874, 395455]
    },
    'team_score': 85,
    'founder_domain_exp_yrs': 7,
    'founder_exits': 1,
    'employee_count': 15,
    'tech_talent_ratio': 0.6,
    'has_cto': True,
    'has_cmo': True,
    'has_cfo': False,
    'cash_flow': [500000, 465000, 430000, 395000, 360000, 325000, 290000, 340000, 390000, 440000, 490000, 540000]
}

def test_robust_pdf():
    """Test the robust PDF generator."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test direct import from robust_pdf
    try:
        logger.info("Testing direct import from robust_pdf")
        from robust_pdf import generate_pdf
        
        # Generate full report
        output_path = f"reports/test_robust_full_{timestamp}.pdf"
        result = generate_pdf(test_doc, output_path=output_path)
        if result:
            logger.info(f"Successfully generated full report at {output_path}")
        else:
            logger.error("Failed to generate full report")
            
        # Generate executive report
        output_path = f"reports/test_robust_exec_{timestamp}.pdf"
        result = generate_pdf(test_doc, output_path=output_path, report_type="executive")
        if result:
            logger.info(f"Successfully generated executive report at {output_path}")
        else:
            logger.error("Failed to generate executive report")
            
        # Generate custom report
        output_path = f"reports/test_robust_custom_{timestamp}.pdf"
        sections = {
            "Executive Summary": True,
            "Market Analysis": True,
            "Financial Projections": True,
            "Competitive Analysis": False,
            "Team Assessment": True,
            "Risk Assessment": False
        }
        result = generate_pdf(test_doc, output_path=output_path, report_type="custom", sections=sections)
        if result:
            logger.info(f"Successfully generated custom report at {output_path}")
        else:
            logger.error("Failed to generate custom report")
            
        # Generate PDF bytes directly
        logger.info("Testing direct bytes generation")
        pdf_bytes = generate_pdf(test_doc)
        if isinstance(pdf_bytes, bytes):
            # Save the bytes to a file
            output_path = f"reports/test_robust_bytes_{timestamp}.pdf"
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
            logger.info(f"Successfully generated PDF bytes and saved to {output_path}")
        else:
            logger.error(f"Failed to generate PDF bytes, got {type(pdf_bytes)} instead")
        
        return True
    except Exception as e:
        logger.error(f"Error testing robust_pdf direct import: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test import through pdf_patch
    try:
        logger.info("Testing import through pdf_patch")
        import pdf_patch
        pdf_patch.apply_patch()
        
        # After patching, these functions should be available globally
        output_path = f"reports/test_patch_full_{timestamp}.pdf"
        result = generate_enhanced_pdf(test_doc, output_path)
        if result:
            logger.info(f"Successfully generated report through pdf_patch at {output_path}")
        else:
            logger.error("Failed to generate report through pdf_patch")
        
        return True
    except Exception as e:
        logger.error(f"Error testing pdf_patch: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test import through global_pdf_functions
    try:
        logger.info("Testing import through global_pdf_functions")
        import global_pdf_functions
        
        output_path = f"reports/test_global_full_{timestamp}.pdf"
        result = global_pdf_functions.generate_enhanced_pdf(test_doc, output_path)
        if result:
            logger.info(f"Successfully generated report through global_pdf_functions at {output_path}")
        else:
            logger.error("Failed to generate report through global_pdf_functions")
        
        return True
    except Exception as e:
        logger.error(f"Error testing global_pdf_functions: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return False

if __name__ == "__main__":
    success = test_robust_pdf()
    if success:
        print("Tests completed. Check logs for details.")
    else:
        print("All tests failed. Check logs for details.")
        sys.exit(1) 