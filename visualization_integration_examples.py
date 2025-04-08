"""
Enhanced Visualization Integration Examples

This file contains examples of how to integrate the enhanced visualization system 
with the existing FlashDNA application (dashboard and PDF reports).
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Make sure current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the visualization module
from enhanced_visualization import get_enhanced_visualization, ChartType, Backend

# Main dashboard integration example
def main():
    """Show examples of integrating with Streamlit dashboard."""
    st.title("Enhanced Visualization System - Demo Dashboard")
    
    # Initialize the visualization system (can be light or dark mode)
    viz = get_enhanced_visualization(dark_mode=False)
    
    # Basic chart types section
    st.header("Basic Chart Types")
    
    chart_type = st.selectbox(
        "Choose a chart type",
        ["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart"]
    )
    
    if chart_type == "Line Chart":
        # Create sample data
        months = list(range(12))
        revenue = [100000 * (1 + 0.1) ** x for x in months]
        expenses = [80000 * (1 + 0.05) ** x for x in months]
        
        df = pd.DataFrame({
            "Month": months,
            "Revenue": revenue,
            "Expenses": expenses
        })
        
        # Create a line chart with enhanced visualization
        chart, backend = viz.create_chart(
            chart_type=ChartType.LINE,
            data=df,
            x="Month",
            y=["Revenue", "Expenses"],
            title="Revenue vs Expenses",
            xlabel="Month",
            ylabel="Amount ($)"
        )
        
        # Display the chart in Streamlit
        if backend == Backend.PLOTLY:
            st.plotly_chart(chart, use_container_width=True)
        elif backend == Backend.MATPLOTLIB:
            st.pyplot(chart)
        elif backend == Backend.ALTAIR:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.json(chart)  # Display D3 spec as JSON
    
    elif chart_type == "Bar Chart":
        # Create sample data
        categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
        values = [25, 40, 30, 50, 45]
        
        df = pd.DataFrame({"Category": categories, "Value": values})
        
        # Create a bar chart with enhanced visualization
        chart, backend = viz.create_chart(
            chart_type=ChartType.BAR,
            data=df,
            x="Category",
            y="Value",
            title="Category Values",
            xlabel="Category",
            ylabel="Value"
        )
        
        # Display the chart in Streamlit
        if backend == Backend.PLOTLY:
            st.plotly_chart(chart, use_container_width=True)
        elif backend == Backend.MATPLOTLIB:
            st.pyplot(chart)
        elif backend == Backend.ALTAIR:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.json(chart)  # Display D3 spec as JSON
    
    elif chart_type == "Scatter Plot":
        # Create sample data
        n = 100
        x = np.random.normal(0, 1, n)
        y = 0.5 * x + np.random.normal(0, 0.5, n)
        categories = np.random.choice(['A', 'B', 'C'], n)
        
        df = pd.DataFrame({
            "X Value": x,
            "Y Value": y,
            "Category": categories
        })
        
        # Create a scatter plot with enhanced visualization
        chart, backend = viz.create_chart(
            chart_type=ChartType.SCATTER,
            data=df,
            x="X Value",
            y="Y Value",
            color="Category",
            title="Sample Scatter Plot",
            xlabel="X Axis",
            ylabel="Y Axis"
        )
        
        # Display the chart in Streamlit
        if backend == Backend.PLOTLY:
            st.plotly_chart(chart, use_container_width=True)
        elif backend == Backend.MATPLOTLIB:
            st.pyplot(chart)
        elif backend == Backend.ALTAIR:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.json(chart)  # Display D3 spec as JSON
    
    elif chart_type == "Area Chart":
        # Create sample data for area chart
        dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        revenue_streams = {
            "Product A": [10000 * (1 + 0.05) ** x for x in range(12)],
            "Product B": [5000 * (1 + 0.08) ** x for x in range(12)],
            "Product C": [7000 * (1 + 0.06) ** x for x in range(12)]
        }
        
        df = pd.DataFrame({
            "Date": dates,
            **revenue_streams
        })
        
        # Create an area chart with enhanced visualization
        chart, backend = viz.create_chart(
            chart_type=ChartType.AREA,
            data=df,
            x="Date",
            y=list(revenue_streams.keys()),
            title="Revenue Streams Over Time",
            xlabel="Date",
            ylabel="Revenue ($)"
        )
        
        # Display the chart in Streamlit
        if backend == Backend.PLOTLY:
            st.plotly_chart(chart, use_container_width=True)
        elif backend == Backend.MATPLOTLIB:
            st.pyplot(chart)
        elif backend == Backend.ALTAIR:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.json(chart)  # Display D3 spec as JSON
    
    # Advanced chart types section
    st.header("Advanced Chart Types")
    
    advanced_type = st.selectbox(
        "Choose an advanced visualization",
        ["CAMP Radar Chart", "Competitive Positioning", "User Growth", "3D Scenario Exploration"]
    )
    
    if advanced_type == "CAMP Radar Chart":
        # Create sample data for CAMP framework
        doc = {
            "camp_framework": {
                "category": [0.75, 0.65, 0.85, 0.70],
                "approach": [0.65, 0.60, 0.80, 0.50],
                "market": [0.85, 0.70, 0.50, 0.60],
                "potential": [0.80, 0.60, 0.70, 0.90]
            }
        }
        
        # Generate CAMP radar chart
        result = viz.generate_camp_radar_chart(doc)
        
        # Display the chart in Streamlit
        if "chart" in result and result["chart"] is not None:
            backend = result.get("backend")
            if backend == Backend.PLOTLY:
                st.plotly_chart(result["chart"], use_container_width=True)
            elif backend == Backend.MATPLOTLIB:
                st.pyplot(result["chart"])
            else:
                st.json(result)
        else:
            st.warning("Failed to create CAMP radar chart")
    
    elif advanced_type == "Competitive Positioning":
        # Create sample data for competitive positioning
        doc = {
            "competitors": {
                "market_size": [0.75, 0.65, 0.50, 0.80],
                "technology": [0.60, 0.80, 0.70, 0.65],
                "names": ["Competitor A", "Competitor B", "Competitor C", "Our Company"]
            }
        }
        
        # Generate competitive positioning chart
        result = viz.generate_competitive_chart(doc)
        
        # Display the chart in Streamlit
        if "chart" in result and result["chart"] is not None:
            backend = result.get("backend")
            if backend == Backend.PLOTLY:
                st.plotly_chart(result["chart"], use_container_width=True)
            elif backend == Backend.MATPLOTLIB:
                st.pyplot(result["chart"])
            else:
                st.json(result)
        else:
            st.warning("Failed to create competitive positioning chart")
    
    elif advanced_type == "User Growth":
        # Create sample data for user growth
        doc = {
            "user_growth": {
                "month": list(range(1, 25)),
                "users": [100, 150, 200, 300, 450, 600, 800, 1000, 1200, 1500, 1800, 2200,
                          2600, 3100, 3700, 4400, 5200, 6100, 7200, 8500, 10000, 12000, 14000, 16000]
            }
        }
        
        # Generate user growth chart
        result = viz.generate_user_growth_chart(doc)
        
        # Display the chart in Streamlit
        if "chart" in result and result["chart"] is not None:
            backend = result.get("backend")
            if backend == Backend.PLOTLY:
                st.plotly_chart(result["chart"], use_container_width=True)
            elif backend == Backend.MATPLOTLIB:
                st.pyplot(result["chart"])
            else:
                st.json(result)
        else:
            st.warning("Failed to create user growth chart")
    
    elif advanced_type == "3D Scenario Exploration":
        # Generate sample scenario data
        import numpy as np
        scenarios = []
        
        for _ in range(100):
            churn_rate = np.random.uniform(0.01, 0.2)
            referral_rate = np.random.uniform(0.01, 0.15)
            growth_rate = np.random.uniform(0.05, 0.3)
            acquisition_cost = np.random.uniform(5, 50)
            
            # Calculate outcomes based on these parameters
            final_users = 1000 * (1 + (growth_rate + referral_rate - churn_rate) * 12)
            success_probability = min(1.0, max(0.0, 0.5 + 0.5 * (referral_rate / 0.1 - churn_rate / 0.1)))
            
            scenarios.append({
                "churn_rate": churn_rate,
                "referral_rate": referral_rate,
                "growth_rate": growth_rate,
                "acquisition_cost": acquisition_cost,
                "final_users": final_users,
                "success_probability": success_probability
            })
        
        # Generate 3D scenario visualization
        result = viz.generate_scenario_visualization(scenarios)
        
        # Display the chart in Streamlit if figure is available
        if 'figure' in result and result['figure'] is not None:
            st.plotly_chart(result['figure'], use_container_width=True)
        else:
            st.warning("3D visualization not available in current backend")

if __name__ == "__main__":
    main()

# Example 2: PDF Report Integration
def pdf_report_integration_example():
    """Examples of how to integrate with PDF reports."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from io import BytesIO
    import base64
    
    # Initialize the visualization system
    viz = get_enhanced_visualization(dark_mode=False)
    
    # Example function to add a chart to a PDF
    def add_chart_to_pdf(pdf_canvas, chart, backend, x, y, width, height):
        """Add a chart to a PDF at the specified position."""
        # Convert chart to an image
        img_data = viz.chart_to_image(
            chart=chart,
            backend=backend,
            format='png',
            width=width,
            height=height
        )
        
        # Create a temporary file-like object
        img_stream = BytesIO(img_data)
        
        # Add the image to the PDF
        pdf_canvas.drawImage(img_stream, x, y, width=width, height=height)
    
    # Create a sample PDF with charts
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Add title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(72, height - 72, "Enhanced Visualization in PDF Reports")
    
    # Create a sample CAMP radar chart
    doc = {
        "capital_score": 75,
        "advantage_score": 60,
        "market_score": 85,
        "people_score": 65
    }
    
    result = viz.generate_camp_radar_chart(doc)
    chart = result['chart']
    backend = result['backend']
    
    # Add the chart to the PDF
    add_chart_to_pdf(p, chart, backend, 72, height - 400, 300, 300)
    
    # Add a description
    p.setFont("Helvetica", 12)
    p.drawString(400, height - 200, "CAMP Framework Analysis")
    p.setFont("Helvetica", 10)
    p.drawString(400, height - 220, "Capital: 75/100")
    p.drawString(400, height - 235, "Advantage: 60/100")
    p.drawString(400, height - 250, "Market: 85/100")
    p.drawString(400, height - 265, "People: 65/100")
    
    # Save the PDF
    p.save()
    buffer.seek(0)
    
    # In a real application, you would save this to a file
    # For this example, we'll just return the buffer
    return buffer

# Integration with unified_pdf_generator.py
def unified_pdf_generator_integration():
    """Example of integrating with unified_pdf_generator.py."""
    code_example = """
    # In unified_pdf_generator.py
    
    from enhanced_visualization import get_enhanced_visualization, ChartType
    
    class InvestorReport:
        def __init__(self, doc, output_path):
            self.doc = doc
            self.output_path = output_path
            self.viz = get_enhanced_visualization(dark_mode=False)
            # ... rest of initialization ...
            
        def add_camp_radar_chart(self):
            """Add a CAMP framework radar chart."""
            try:
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
                    height=400
                )
                
                # Create a ReportLab Image object
                img = Image(BytesIO(chart_image), width=450, height=300)
                
                # Add to the story
                self.story.append(Paragraph("CAMP Framework Analysis", self.styles['Heading2']))
                self.story.append(img)
                self.story.append(Spacer(1, 0.2 * inch))
                
                logger.info("Added enhanced CAMP radar chart")
                
            except Exception as e:
                logger.error(f"Error creating CAMP radar chart: {str(e)}\\n{traceback.format_exc()}")
                # Skip chart if error occurs
                self.story.append(Paragraph("Error generating CAMP radar chart.", self.styles['CustomBodyText']))
    """
    
    return code_example 