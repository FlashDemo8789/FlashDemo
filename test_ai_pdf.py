"""
Test script for AI-enhanced PDF generation system.
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import our AI-enhanced PDF generator
from ai_enhanced_pdf_generator import generate_ai_enhanced_pdf

def create_test_document():
    """Create a test document with sample data."""
    return {
        "company_name": "Test AI Startup",
        "industry": "Artificial Intelligence",
        "founding_date": "2020-01-01",
        "revenue_model": "SaaS",
        "monthly_revenue": 50000,
        "growth_rate": 0.15,
        "ltv_cac_ratio": 3.5,
        "market_share": 0.05,
        "runway_months": 18,
        "team_size": 15,
        "funding_round": "Series A",
        "total_funding": 5000000,
        "valuation": 25000000,
        "target_market": "Enterprise AI Solutions",
        "competitors": [
            {"name": "Competitor A", "strength": "Market Leader", "weakness": "High Prices"},
            {"name": "Competitor B", "strength": "Innovative Tech", "weakness": "Limited Scale"}
        ],
        "growth_metrics": {
            "monthly_active_users": 10000,
            "customer_acquisition_cost": 1000,
            "customer_lifetime_value": 3500,
            "churn_rate": 0.02
        },
        "risks": [
            {"category": "Market", "description": "Increasing competition", "mitigation": "Focus on niche markets"},
            {"category": "Technical", "description": "Scalability challenges", "mitigation": "Cloud infrastructure"}
        ],
        "exit_strategy": "Acquisition by major tech company",
        "technical_stack": ["Python", "TensorFlow", "AWS", "React"],
        "intellectual_property": ["2 Patents", "3 Trademarks"],
        "partnerships": ["Tech Partner A", "Distribution Partner B"]
    }

def main():
    """Main test function."""
    try:
        logger.info("Starting AI-enhanced PDF generation test")
        
        # Create test document
        doc_data = create_test_document()
        logger.info("Created test document data")
        
        # Generate AI-enhanced PDF
        output_path = f"test_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        logger.info(f"Generating AI-enhanced PDF to {output_path}")
        
        # Generate the report with all sections
        generate_ai_enhanced_pdf(
            doc_data=doc_data,
            output_path=output_path,
            report_type="investor",
            sections=[
                "executive_summary",
                "business_model",
                "market_analysis",
                "team",
                "competitive_analysis",
                "growth_metrics",
                "risk_assessment",
                "exit_strategy",
                "technical_assessment"
            ]
        )
        
        # Check if file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Successfully generated AI-enhanced PDF: {output_path} ({file_size} bytes)")
        else:
            logger.error("Failed to generate PDF file")
            
    except Exception as e:
        logger.error(f"Error during test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 