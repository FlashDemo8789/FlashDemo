"""
Enhanced Visualization Integration with PDF Reports

This file contains example implementations showing how to integrate the enhanced 
visualization system with the FlashDNA PDF report generator.
"""

import logging
import traceback
from io import BytesIO
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Spacer
from enhanced_visualization import get_enhanced_visualization, ChartType, Backend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pdf_integration")

# Example implementation for unified_pdf_generator.py - modified methods to use enhanced visualization

def example_add_camp_radar_chart(self):
    """Add a CAMP framework radar chart using enhanced visualization."""
    try:
        # Initialize visualization system if not already done
        if not hasattr(self, 'viz'):
            self.viz = get_enhanced_visualization(dark_mode=False)
        
        # Generate the chart using enhanced visualization
        result = self.viz.generate_camp_radar_chart(self.doc)
        chart = result['chart']
        backend = result['backend']
        
        # Convert chart to an image and add to the report
        chart_image = self.viz.chart_to_image(
            chart=chart,
            backend=backend,
            format='png',
            width=600,
            height=400,
            scale=2.0  # Higher resolution for PDF
        )
        
        # Create a ReportLab Image object
        img = Image(BytesIO(chart_image), width=450, height=300)
        
        # Add to the story
        self.story.append(Paragraph("CAMP Framework Analysis", self.styles['Heading2']))
        self.story.append(img)
        self.story.append(Spacer(1, 0.2 * inch))
        
        logger.info("Added enhanced CAMP radar chart")
        
    except Exception as e:
        logger.error(f"Error creating CAMP radar chart: {str(e)}\n{traceback.format_exc()}")
        # Skip chart if error occurs
        self.story.append(Paragraph("Error generating CAMP radar chart.", self.styles['CustomBodyText']))

def example_add_financial_chart(self):
    """Add a financial projection chart using enhanced visualization."""
    try:
        # Initialize visualization system if not already done
        if not hasattr(self, 'viz'):
            self.viz = get_enhanced_visualization(dark_mode=False)
        
        # Generate the chart using enhanced visualization
        result = self.viz.generate_financial_chart(self.doc)
        chart = result['chart']
        backend = result['backend']
        
        # Convert chart to an image and add to the report
        chart_image = self.viz.chart_to_image(
            chart=chart,
            backend=backend,
            format='png',
            width=800,
            height=400,
            scale=2.0  # Higher resolution for PDF
        )
        
        # Create a ReportLab Image object
        img = Image(BytesIO(chart_image), width=500, height=250)
        
        # Add to the story
        self.story.append(Paragraph("Financial Projections", self.styles['Heading2']))
        self.story.append(img)
        self.story.append(Spacer(1, 0.2 * inch))
        
        logger.info("Added enhanced financial chart")
        
    except Exception as e:
        logger.error(f"Error creating financial chart: {str(e)}\n{traceback.format_exc()}")
        # Skip chart if error occurs
        self.story.append(Paragraph("Error generating financial chart.", self.styles['CustomBodyText']))

def example_add_user_growth_chart(self):
    """Add a user growth projection chart using enhanced visualization."""
    try:
        # Initialize visualization system if not already done
        if not hasattr(self, 'viz'):
            self.viz = get_enhanced_visualization(dark_mode=False)
        
        # Generate the chart using enhanced visualization
        result = self.viz.generate_user_growth_chart(self.doc)
        chart = result['chart']
        backend = result['backend']
        
        # Convert chart to an image and add to the report
        chart_image = self.viz.chart_to_image(
            chart=chart,
            backend=backend,
            format='png',
            width=800,
            height=400,
            scale=2.0  # Higher resolution for PDF
        )
        
        # Create a ReportLab Image object
        img = Image(BytesIO(chart_image), width=500, height=250)
        
        # Add to the story
        self.story.append(Paragraph("User Growth Projection", self.styles['Heading2']))
        self.story.append(img)
        self.story.append(Spacer(1, 0.2 * inch))
        
        logger.info("Added enhanced user growth chart")
        
    except Exception as e:
        logger.error(f"Error creating user growth chart: {str(e)}\n{traceback.format_exc()}")
        # Skip chart if error occurs
        self.story.append(Paragraph("Error generating user growth chart.", self.styles['CustomBodyText']))

def example_add_competitive_chart(self):
    """Add a competitive positioning chart using enhanced visualization."""
    try:
        # Initialize visualization system if not already done
        if not hasattr(self, 'viz'):
            self.viz = get_enhanced_visualization(dark_mode=False)
        
        # Generate the chart using enhanced visualization
        result = self.viz.generate_competitive_chart(self.doc)
        chart = result['chart']
        backend = result['backend']
        
        # Convert chart to an image and add to the report
        chart_image = self.viz.chart_to_image(
            chart=chart,
            backend=backend,
            format='png',
            width=700,
            height=500,
            scale=2.0  # Higher resolution for PDF
        )
        
        # Create a ReportLab Image object
        img = Image(BytesIO(chart_image), width=450, height=320)
        
        # Add to the story
        self.story.append(Paragraph("Competitive Positioning", self.styles['Heading2']))
        self.story.append(img)
        self.story.append(Spacer(1, 0.2 * inch))
        
        logger.info("Added enhanced competitive positioning chart")
        
    except Exception as e:
        logger.error(f"Error creating competitive chart: {str(e)}\n{traceback.format_exc()}")
        # Skip chart if error occurs
        self.story.append(Paragraph("Error generating competitive positioning chart.", self.styles['CustomBodyText']))

def example_add_custom_bar_chart(self, categories, values, title, ylabel="Value"):
    """Add a custom bar chart using enhanced visualization."""
    try:
        # Initialize visualization system if not already done
        if not hasattr(self, 'viz'):
            self.viz = get_enhanced_visualization(dark_mode=False)
        
        # Create DataFrame for chart
        import pandas as pd
        df = pd.DataFrame({"Category": categories, "Value": values})
        
        # Create the chart
        chart, backend = self.viz.create_chart(
            chart_type=ChartType.BAR,
            data=df,
            x="Category",
            y="Value",
            title=title,
            ylabel=ylabel
        )
        
        # Convert chart to an image and add to the report
        chart_image = self.viz.chart_to_image(
            chart=chart,
            backend=backend,
            format='png',
            width=800,
            height=400,
            scale=2.0  # Higher resolution for PDF
        )
        
        # Create a ReportLab Image object
        img = Image(BytesIO(chart_image), width=500, height=250)
        
        # Add to the story
        self.story.append(Paragraph(title, self.styles['Heading2']))
        self.story.append(img)
        self.story.append(Spacer(1, 0.2 * inch))
        
        logger.info(f"Added enhanced bar chart: {title}")
        
    except Exception as e:
        logger.error(f"Error creating bar chart: {str(e)}\n{traceback.format_exc()}")
        # Skip chart if error occurs
        self.story.append(Paragraph(f"Error generating chart: {title}", self.styles['CustomBodyText']))

def example_add_custom_line_chart(self, x_data, y_data, title, xlabel="", ylabel=""):
    """Add a custom line chart using enhanced visualization."""
    try:
        # Initialize visualization system if not already done
        if not hasattr(self, 'viz'):
            self.viz = get_enhanced_visualization(dark_mode=False)
        
        # Create DataFrame for chart
        import pandas as pd
        df = pd.DataFrame({xlabel if xlabel else "X": x_data, 
                          ylabel if ylabel else "Y": y_data})
        
        # Create the chart
        chart, backend = self.viz.create_chart(
            chart_type=ChartType.LINE,
            data=df,
            x=xlabel if xlabel else "X",
            y=ylabel if ylabel else "Y",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel
        )
        
        # Convert chart to an image and add to the report
        chart_image = self.viz.chart_to_image(
            chart=chart,
            backend=backend,
            format='png',
            width=800,
            height=400,
            scale=2.0  # Higher resolution for PDF
        )
        
        # Create a ReportLab Image object
        img = Image(BytesIO(chart_image), width=500, height=250)
        
        # Add to the story
        self.story.append(Paragraph(title, self.styles['Heading2']))
        self.story.append(img)
        self.story.append(Spacer(1, 0.2 * inch))
        
        logger.info(f"Added enhanced line chart: {title}")
        
    except Exception as e:
        logger.error(f"Error creating line chart: {str(e)}\n{traceback.format_exc()}")
        # Skip chart if error occurs
        self.story.append(Paragraph(f"Error generating chart: {title}", self.styles['CustomBodyText']))

# Full integration example - how to modify InvestorReport class

"""
# Add to the imports at the top of unified_pdf_generator.py:
from enhanced_visualization import get_enhanced_visualization, ChartType, Backend

# Add to the __init__ method of InvestorReport class:
self.viz = get_enhanced_visualization(dark_mode=False)

# Replace the existing chart methods with the enhanced versions
InvestorReport.add_camp_radar_chart = example_add_camp_radar_chart
InvestorReport.add_bar_chart = example_add_custom_bar_chart
InvestorReport.add_line_chart = example_add_custom_line_chart

# Add the new chart methods
InvestorReport.add_financial_chart = example_add_financial_chart
InvestorReport.add_user_growth_chart = example_add_user_growth_chart
InvestorReport.add_competitive_chart = example_add_competitive_chart
"""

# Example usage (for documentation purposes)
def example_usage():
    """Example of how to use the enhanced visualization in PDF reports."""
    explanation = """
    # Example usage in generate_investor_report function:
    
    def generate_investor_report(doc, output_path):
        report = InvestorReport(doc, output_path)
        
        # Build the report structure
        report.add_title_page()
        report.add_toc()
        
        # Executive Summary
        report.add_executive_summary()
        
        # CAMP Analysis with enhanced charts
        report.add_section("CAMP Framework Analysis")
        report.add_camp_radar_chart()  # Uses enhanced visualization
        
        # Financial Projections with enhanced charts
        report.add_section("Financial Projections")
        report.add_financial_chart()   # Uses enhanced visualization
        
        # User Growth with enhanced charts
        report.add_section("User Growth")
        report.add_user_growth_chart() # Uses enhanced visualization
        
        # Competitive Analysis with enhanced charts
        report.add_section("Competitive Analysis")
        report.add_competitive_chart() # Uses enhanced visualization
        
        # Generate the PDF
        report.build()
        
        return output_path
    """
    
    return explanation 