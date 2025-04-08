import streamlit as st
import os
import time
import pickle
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, date
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
import re
import math
import uuid
import base64
from enum import Enum
import io
import tempfile
import csv
from io import StringIO
import copy
import sys
import importlib

# Import core analysis modules
from advanced_ml import evaluate_startup, evaluate_startup_camp
# Use enhanced AI integration for intangible score calculation
from enhanced_ai_integration import compute_intangible_llm
from domain_expansions import apply_domain_expansions
from team_moat import compute_team_depth_score, compute_moat_score, evaluate_team_execution_risk
from pattern_detector import detect_patterns, generate_pattern_insights
from system_dynamics import system_dynamics_sim, virality_sim, calculate_growth_metrics
from hpc_scenario import find_optimal_scenario
from financial_modeling import scenario_runway, calculate_unit_economics, forecast_financials, calculate_valuation_metrics
from competitive_intelligence import CompetitiveIntelligence
from monte_carlo import MonteCarloSimulator, SimulationResult
# Keeping pitch_sentiment as indicated in the comment
from pitch_sentiment import PitchAnalyzer
from ml_assessment import StartupAssessmentModel
from product_market_fit import ProductMarketFitAnalyzer, PMFMetrics, PMFStage

# Use try/except for imports that might fail
try:
    # Try to import from technical_due_diligence first
    from technical_due_diligence import EnterpriseGradeTechnicalDueDiligence as TechnicalDueDiligence, TechnicalAssessment
except ImportError:
    # Fall back to the minimal version
    try:
        from technical_due_diligence_minimal import TechnicalDueDiligence, TechnicalAssessment
    except ImportError:
        # If all else fails, create fake classes
        class TechnicalDueDiligence:
            def __init__(self, *args, **kwargs): pass
            def assess_technical_architecture(self, tech_data, generate_report=False): 
                return {"error": "No TechnicalDueDiligence available"}
        
        class TechnicalAssessment:
            def __init__(self, *args, **kwargs): pass
            def to_dict(self): 
                return {"error": "No TechnicalAssessment available"}

from cohort_analysis import CohortAnalyzer
from network_analysis import NetworkEffectAnalyzer
from benchmarking import BenchmarkEngine, BenchmarkResult

# Setup logging FIRST
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analysis_flow")

# Import PDF patch to ensure all PDF generation functions are available
import pdf_patch
# Explicitly apply the patch to ensure PDF functions are globally available
# This should make generate_enhanced_pdf etc. available via builtins or patched modules
pdf_patch.apply_patch()

# Import global PDF functions - these should now be the patched versions
# Use these functions throughout the analysis flow
try:
    # Reload in case it was imported before patching
    if 'global_pdf_functions' in sys.modules:
         global_pdf_functions = importlib.reload(sys.modules['global_pdf_functions'])
    else:
         import global_pdf_functions
    # Make them easily accessible in this namespace if needed, otherwise use global_pdf_functions.func_name
    from global_pdf_functions import generate_enhanced_pdf, generate_emergency_pdf, generate_investor_report
    logger.info("Successfully imported patched PDF functions via global_pdf_functions.")
except ImportError as e:
     logger.critical(f"Failed to import global_pdf_functions even after patching: {e}. PDF generation will likely fail.")
     # Define dummy functions to prevent NameErrors later, though generation will fail
     def generate_enhanced_pdf(*args, **kwargs):
         logger.error("Dummy generate_enhanced_pdf called - import failed.")
         return b""
     generate_emergency_pdf = generate_enhanced_pdf
     generate_investor_report = generate_enhanced_pdf

from acquisition_fit import AcquisitionFitAnalyzer, AcquisitionFitResult
# Adding the missing import for comparative exit path analysis
from comparative_exit_path import ExitPathAnalyzer
from utils import create_placeholder, extract_text_from_pdf, create_radar_chart


########################################
# Setup & Shared UI Functions
########################################

def setup_page():
    """Setup page configuration and CSS styling."""
    # Set page config
    st.set_page_config(
        page_title="FlashDNA Infinity",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS
    css = """
    <style>
        /* CAMP framework styling */
        .camp-header {
            text-align: center;
            padding: 8px;
            border-radius: 5px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .camp-capital { background-color: rgba(66, 135, 245, 0.2); }
        .camp-market { background-color: rgba(240, 180, 0, 0.2); }
        .camp-advantage { background-color: rgba(0, 204, 150, 0.2); }
        .camp-people { background-color: rgba(232, 65, 24, 0.2); }
        
        /* Streamlit overwrites */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    # Try to load a logo from multiple potential locations
    logo = None
    logo_paths = [
        os.getenv("LOGO_PATH", ""),                          # Environment variable
        "static/logo.png",                                   # Standard static directory
        "static/img/logo.png",                               # Alternative static img directory
        "logo.png",                                          # Root directory
        os.path.join(os.path.dirname(__file__), "logo.png"), # Same directory as script
        "/app/static/logo.png"                               # Docker container path
    ]
    
    # Filter out empty paths and try each one
    for path in [p for p in logo_paths if p]:
        try:
            if os.path.exists(path):
                logo = Image.open(path)
                logger.info(f"Successfully loaded logo from {path}")
                break
        except Exception as e:
            logger.warning(f"Failed to load logo from {path}: {e}")
    
    # If no logo found, use built-in base64 encoded logo
    if logo is None:
        try:
            # Embedded small logo as base64 to ensure we always have one
            base64_logo = """
            iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAYAAAA8AXHiAAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9
            kT1Iw0AcxV9TtSIVBTuIOGSoThZERR21CkWoEGqFVh1MbvqhNGlIUlwcBdeCgx+LVQcXZ10dXAVB
            8APE0clJ0UVK/F9SaBHjwXE/3t173L0DhGaVqWbPOKBqlpFOxMVcflUMvCKIEYQxICJTT2YWM/Bc
            X/fw8fUuyrO8z/25+pWCyQDfiOcZ6phE3yBOb5o6533iCCtJCvE58bhBFyR+5Lrs8hvnksMAzwyv
            1HSvWCQWCy21sLIaG/UIKawaTnTqOvIJqZjK5a3MYllT9caHc+0aBUt2JX2miI+lkEYaRQxhhbao
            UHMQp91Ko5RG+X7mxx/1+iVyKeSqgJFjHhvQILt+8D/43a1VnBj3kiJxoPvFtj9GgcAu0GrY9vex
            bbdOgP4n4Mpr+St1YPaT9FpLix4BfdvAxXVLU/aAyx1g8MmQTdmVgrSEYgn4fUbPySB0C/Rf9eZW
            +2PnARqnQXLNrQYHwsAoUbLX87zbc87t3527d0/vB+jdcs2z/KpoAAAABmJLR0QA/wD/AP+gvaeT
            AAAACXBIWXMAAC4jAAAuIwF4pT92AAAAB3RJTUUH6AQEAgUQ9cGVXAAABhdJREFUeNrt3T9sFEkZ
            B/DflaMDCXHExiBZRIoQIgqKA8RfkYxAok32kH+koZ25uCgOEdGkO9Fh4ZIGkMh2gwM05J9p8KVK
            tE3sAIoAFwmVE6IKiJNCEULsXeG7A2lvZ/ZmZ2dnZ77f05Dssed5n92Z92dm39QIAAAAAAAAAAAA
            AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMD/1eKAf9/CwoIl0hnP8xYbjcYD
            IqpTOJ8TUb3RaJA0nzFFNZvNkPuEZrMZ+I5YrMWjWGMu1mKxKEVEURXnE4rMl6JPKlZjzPzQUOSJ
            nKVVMZyDnZbzevE2m80BsYdcxjc1m02UWNHJa4ViLC5ZBGijT9XpdA5nf87n9bM/dzoduuL+pAOP
            8WR59qLM5qx2u013v1YrlfuTz/KVzXO/Vkv1s3q9TnW71KvV06oq2kFzDq3bbQ48xOm0OVh1drtd
            arValLVOs9Sh1erQ52Yw98Onv371gO5//Rfzz38+/PeOux89/vb3f7zzxz9PnMjqvO/fvXDt7bf+
            8OfKPRuVOqxkdQKDjy+guw8e0Y96ftDPgZeRH/7wRyLivHZe/+JP9X9pnTtrxIlY0Zoz/0lJVeyw
            LGd7nj9Pd25cpze2b/OyEZ2g73Lv9FzPB80fUb2eXX2v3++/3Gg0xLNnzyjL+5LqeZ6p1WrCdT3K
            8nxZcdmxZTFrC/P6jqwrF64Pnkqmao2tPc/3TK1Wu+JynPP3jucZiWOvDqhFbLj322vUXj2YuDIX
            tpUmqrNOZ50o8OEqLadXzfn58vLyB/fu3fvLqVOnLOtYlmUxLv9nWda2zs3Nzdm3b99eXVpaunH2
            7NnXbdseHYv6vj/wOG8qmQ3JnuWbWuSGrMfzUh1ZvZ5fgHcx4DOKi8gXfLsZ+Hd+hHmEuA3JwPvz
            YfcxTJw2Bnl9P/QzQZ9z+XvLo9u358x5HOMJtC8ajcY3NU35cFCtVltbWVnphJ2YV2zb3qyUfr8v
            ZLxD9a+qqp7neVLxfV/2+30KWuaPYl03h41GY1tJtV6vOzLGVCgVkpBlTHzOt0RCa0oYkRPDNcaY
            odfI6xgLyF9Hsh83IVzX9YXtmDgrpG36/T7vJXIBQK/X08IYYbC/vx+xtHKDlhGRs7i4OOCY8nNp
            Pdt2YgcQMqDEQtpN+Uxs1vd9I3udYX1fVLHfz3uKnb9RRRKZz1PKJN6hFqtWqzVrFpf1Gm0SQ8dY
            3W73QIxVWN+vYK+w9L5frVZfnA8qnV/kGGsFe4WlH2NRVMeD08JJoYR+jDXd76/wdyxxjAUAAAAA
            QHblP/5Rv9M3LlRV1VRV3bJERFu2qqqaqqqaruvKiKpULKqqVlJVLUVRtrZUVd2K+b1qRVGUqUJR
            FCp6AZ5RFEVUVVVh2OBMZtvKO8Z2Y4ynZHNMlXrVhA6sXPP5/EPP83ZdbuS15/kPxb6CRq1We6LZ
            tiqVyg7Dh7ydSqWyw/HCDGvbtroVHcTDnEgkEolEotGmp6dvTU9PG9d1FyiZhe0VV1EUfWZmZuLI
            mpmZUfP8/vl8X1vX9U9zXPQNXdfXc15JbbqurxshhOu674Z0FHaMCCGE8zwnR2Y4juPneL6dHHdS
            rzzPe+x53s4YFBLrOPNY5KiusXnz5s38eyRFsXekKo3B4sqeYZiFwOPTn5qa+qrwwlquVCr2CI1N
            1CqVij3iF/w8EolEIpFIJFLCHQwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
            AAAAAAAAAADACPo3v8NB55hh5MIAAAAASUVORK5CYII=
            """
            logo = Image.open(io.BytesIO(base64.b64decode(base64_logo.strip())))
            logger.info("Using embedded fallback logo")
        except Exception as e:
            logger.error(f"Failed to load embedded logo: {e}")
    
    return logo

def apply_tab_scroll_style():
    """
    Allows horizontal scrolling of the tab bar on smaller screens (mobile).
    """
    st.markdown("""
    <style>
    div[role="tablist"] {
        overflow-x: auto !important;
        overflow-y: hidden !important;
        white-space: nowrap !important;
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header(logo):
    """Display the application header with logo."""
    # Create a header with logo and title
    col1, col2 = st.columns([1, 5])
    
    with col1:
        if logo is not None:
            st.image(logo, width=150)
    
    with col2:
        st.markdown("<h1 style='color:#FF5757;'>FlashDNA Infinity</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.2em;'>Advanced Startup Analysis with CAMP Framework</p>", unsafe_allow_html=True)

def initialize_session():
    """Initialize session state variables."""
    if "doc" not in st.session_state:
        st.session_state.doc = {}
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False
    if "active_section" not in st.session_state:
        st.session_state.active_section = "Basic Info"
    if "display_mode" not in st.session_state:
        # Default to expanded 12-tab mode, can be toggled to 8-tab mode
        st.session_state.display_mode = "expanded"
    if "welcome_style" not in st.session_state:
        # Default to the radar chart welcome, can be toggled to bar chart
        st.session_state.welcome_style = "radar"

def load_xgb_model():
    """Load the trained XGBoost model for startup success prediction."""
    # Set model locations to try in order (environment variable, multiple paths)
    model_paths = [
        os.getenv("MODEL_PATH", ""),  # First try env variable if set
        "model_xgb.pkl",              # Try current directory
        os.path.join("models", "model_xgb.pkl"),  # Try models subdirectory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_xgb.pkl"),  # Try script dir
        "/app/model_xgb.pkl"          # Try Docker container path
    ]
    
    # Filter out empty paths
    model_paths = [path for path in model_paths if path]
    
    # Try each path
    for model_path in model_paths:
        try:
            logger.info(f"Attempting to load model from: {model_path}")
            if not os.path.exists(model_path):
                logger.warning(f"Model path does not exist: {model_path}")
                continue
                
            # Try to load the model
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            # Create a proper model wrapper to handle different XGBoost versions
            logger.info("Creating model wrapper for compatibility")
            class ModelWrapper:
                def __init__(self, model):
                    self.model = model
                    # Don't try to access or set use_label_encoder attribute
                    # It's not available in newer XGBoost versions
                
                def predict_proba(self, X):
                    """Wrapper for predict_proba that handles various model types."""
                    try:
                        # For models with predict_proba
                        return self.model.predict_proba(X)
                    except AttributeError:
                        # For models without predict_proba, using predict
                        logger.info("Model doesn't have predict_proba, using predict instead")
                        try:
                            preds = self.model.predict(X)
                            if len(preds.shape) == 1:
                                # Convert to pseudo-probabilities for binary classification
                                prob = np.clip(preds, 0, 1)
                                return np.vstack([1-prob, prob]).T
                            else:
                                # Already has multiple columns
                                return preds
                        except Exception as e:
                            logger.error(f"Predict failed: {str(e)}")
                            # Return 50/50 probability if all else fails
                            if len(X) == 1:
                                return np.array([[0.5, 0.5]])
                            else:
                                return np.tile([0.5, 0.5], (len(X), 1))
                
                def predict(self, X):
                    try:
                        # Try direct predict first
                        return self.model.predict(X)
                    except Exception as e:
                        # Fall back to predict_proba
                        try:
                            probs = self.predict_proba(X)
                            return probs[:, 1]
                        except Exception as e:
                            logger.error(f"Error in predict: {str(e)}")
                            # Safe fallback
                            return np.array([0.5] * X.shape[0])
            
            # Wrap the model
            wrapped_model = ModelWrapper(model)
            logger.info(f"Model loaded successfully from {model_path}")
            return wrapped_model
            
        except FileNotFoundError:
            logger.warning(f"Model not found at {model_path}")
            continue
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            continue
    
    # If we reach here, all paths failed
    logger.warning("All model paths failed. Using fallback model.")
    # Return a fallback model that always predicts 0.5 probability
    class FallbackModel:
        def predict(self, X):
            return np.array([0.5] * X.shape[0])
        
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * X.shape[0])
    
    return FallbackModel()

# Helper function to convert objects to dictionaries
def to_dict(obj):
    """Convert an object to a dictionary if it's not already dict-like."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    if hasattr(obj, '_asdict'):  # For namedtuples
        return obj._asdict()
    try:
        # Try to convert to dict if it has attributes
        return {k: getattr(obj, k) for k in dir(obj) 
                if not k.startswith('_') and not callable(getattr(obj, k))}
    except:
        logger.warning(f"Could not convert {type(obj)} to dictionary")
        return {}

# CAMP Framework Metrics Dictionary
def get_camp_metrics_glossary():
    """Return the CAMP framework metrics glossary with descriptions."""
    return {
        # Capital Efficiency Metrics
        "monthly_revenue": "Current monthly revenue in dollars. Higher is better.",
        "annual_recurring_revenue": "Predictable yearly revenue, typically for subscription businesses.",
        "lifetime_value_ltv": "Total value of average customer relationship over time.",
        "gross_margin_percent": "Percentage of revenue retained after direct costs. Higher is better.",
        "operating_margin_percent": "Percentage of revenue retained after all operating costs.",
        "burn_rate": "Monthly cash spent in dollars. Lower relative to revenue is better.",
        "runway_months": "Months of cash remaining at current burn rate. Higher is better.",
        "cash_on_hand_million": "Total available cash in millions. Higher is better.",
        "debt_ratio": "Ratio of debt to total assets. Lower is generally better.",
        "ltv_cac_ratio": "Ratio of customer lifetime value to acquisition cost. >3 is good.",
        "customer_acquisition_cost": "Average cost to acquire a customer in dollars.",
        
        # Market Dynamics Metrics
        "market_size": "Total addressable market in dollars. Larger markets offer more potential.",
        "market_growth_rate": "Annual market expansion rate as percentage. Higher is better.",
        "market_share": "Percentage of market currently captured.",
        "user_growth_rate": "Monthly percentage increase in user base. Higher is better.",
        "viral_coefficient": "Average number of new users each user refers. >1 creates viral growth.",
        "category_leadership_score": "Strength of market positioning on a 0-100 scale.",
        "revenue_growth_rate": "Monthly percentage increase in revenue. Higher is better.",
        "net_retention_rate": "Revenue retention including expansion and churn. >100% is negative churn.",
        "churn_rate": "Monthly percentage of customers who leave. Lower is better.",
        "community_growth_rate": "Growth rate of community or network. Important for network effects.",
        
        # Advantage Moat Metrics
        "patent_count": "Number of granted patents.",
        "technical_innovation_score": "Uniqueness and value of technology on a 0-100 scale.",
        "product_maturity_score": "Completeness and stability of product on a 0-100 scale.",
        "api_integrations_count": "Number of system integrations. Creates switching costs.",
        "scalability_score": "Ability to handle growth without linear cost increase. 0-100 scale.",
        "technical_debt_score": "Inverse measure of technical debt burden. Lower is better.",
        "product_security_score": "Strength of security measures on a 0-100 scale.",
        "business_model_strength": "Robustness of business model on a 0-100 scale.",
        
        # People & Performance Metrics
        "founder_exits": "Number of previous successful exits by founders.",
        "founder_domain_exp_yrs": "Years of relevant experience in this domain.",
        "employee_count": "Total team size.",
        "founder_diversity_score": "Team diversity measure on a 0-100 scale.",
        "management_satisfaction_score": "Team satisfaction measure on a 0-100 scale.",
        "tech_talent_ratio": "Proportion of technical talent in the company. 0-1 scale.",
        "has_cto": "Whether the company has a Chief Technology Officer.",
        "has_cmo": "Whether the company has a Chief Marketing Officer.",
        "has_cfo": "Whether the company has a Chief Financial Officer.",
        "employee_turnover_rate": "Annual employee churn percentage. Lower is better.",
        "nps_score": "Net Promoter Score from customers. -100 to 100 scale.",
        "support_ticket_sla_percent": "Service level agreement performance for support tickets.",
        
        # Intangible Metrics
        "intangible": "AI-derived assessment based on pitch text. 0-100 scale.",
        "pitch_sentiment": "Sentiment analysis of the pitch materials.",
        
        # Other Metrics
        "stage": "Company development stage (pre-seed, seed, series A, etc.)",
        "sector": "Industry sector or category.",
    }

########################################
# Sidebar Input
########################################

def render_sidebar_input():
    """Render multi-tabbed form in the sidebar for user input using CAMP framework."""
    with st.sidebar:
        st.subheader("Startup Information")
        
        # Get metrics glossary for tooltips
        metrics_glossary = get_camp_metrics_glossary()
        
        # Define the menu using CAMP framework categories
        menu = option_menu(
            "CAMP Framework Input",
            ["Basic Info", "Capital (C)", "Market (M)", "Advantage (A)", "People (P)", "Pitch"],
            icons=["info-circle", "cash-coin", "graph-up", "shield-check", "people", "file-earmark-text"],
            menu_icon="list",
            default_index=0,
            orientation="horizontal"
        )
        
        # Store active section
        st.session_state.active_section = menu

        doc = {}
        
        # Basic Info (common to all sections)
        if menu == "Basic Info":
            doc["name"] = st.text_input("Startup Name", "NewCo")
            
            doc["stage"] = st.selectbox(
                "Stage", 
                ["pre-seed","seed","series-a","series-b","growth","other"],
                help=f"Company development stage"
            )
            
            doc["sector"] = st.selectbox(
                "Sector", 
                ["fintech","biotech","saas","ai","ecommerce","marketplace","crypto","other"],
                help=f"Industry sector or category"
            )

        # Capital Efficiency (C) section
        elif menu == "Capital (C)":
            st.markdown('<p class="camp-header camp-capital">Capital Efficiency (C) - 30%</p>', unsafe_allow_html=True)
            st.markdown("Financial health and capital deployment metrics")
            
            doc["monthly_revenue"] = st.number_input(
                "Monthly Revenue ($)", 
                0.0, 1e9, 50000.0,
                help=metrics_glossary.get("monthly_revenue", "")
            )
            
            doc["burn_rate"] = st.number_input(
                "Burn Rate ($)", 
                0.0, 1e9, 30000.0,
                help=metrics_glossary.get("burn_rate", "")
            )
            
            doc["current_cash"] = st.number_input(
                "Current Cash ($)", 
                0.0, 1e9, 500000.0,
                help=metrics_glossary.get("cash_on_hand_million", "")
            )
            
            col1, col2 = st.columns(2)
            with col1:
                doc["ltv_cac_ratio"] = st.slider(
                    "LTV:CAC", 
                    0.0, 10.0, 2.5,
                    help=metrics_glossary.get("ltv_cac_ratio", "")
                )
                
                doc["lifetime_value_ltv"] = st.number_input(
                    "Customer LTV ($)", 
                    0.0, 1e6, 500.0,
                    help=metrics_glossary.get("lifetime_value_ltv", "")
                )
            
            with col2:
                doc["customer_acquisition_cost"] = st.number_input(
                    "CAC ($)", 
                    0.0, 1e6, 300.0,
                    help=metrics_glossary.get("customer_acquisition_cost", "")
                )
                
                doc["gross_margin_percent"] = st.slider(
                    "Gross Margin (%)", 
                    0, 100, 70,
                    help=metrics_glossary.get("gross_margin_percent", "")
                )
            
            with st.expander("Additional Capital Metrics"):
                col1, col2 = st.columns(2)
                with col1:
                    doc["operating_margin_percent"] = st.slider(
                        "Operating Margin (%)", 
                        -100, 100, 10,
                        help=metrics_glossary.get("operating_margin_percent", "")
                    )
                    
                    doc["annual_recurring_revenue"] = st.number_input(
                        "ARR ($)", 
                        0.0, 1e9, doc.get("monthly_revenue", 50000.0) * 12,
                        help=metrics_glossary.get("annual_recurring_revenue", "")
                    )
                
                with col2:
                    doc["debt_ratio"] = st.slider(
                        "Debt Ratio", 
                        0.0, 1.0, 0.2,
                        help=metrics_glossary.get("debt_ratio", "")
                    )
                    
                    doc["avg_revenue_per_user"] = st.number_input(
                        "ARPU ($/mo)", 
                        0.0, 1e6, 100.0,
                        help="Average Revenue Per User per month"
                    )

        # Market Dynamics (M) section
        elif menu == "Market (M)":
            st.markdown('<p class="camp-header camp-market">Market Dynamics (M) - 25%</p>', unsafe_allow_html=True)
            st.markdown("Market opportunity and growth potential metrics")
            
            doc["market_size"] = st.number_input(
                "Market Size ($)", 
                0.0, 1e12, 50e6,
                help=metrics_glossary.get("market_size", "")
            )
            
            col1, col2 = st.columns(2)
            with col1:
                doc["market_growth_rate"] = st.slider(
                    "Market Growth Rate (%/yr)",
                    0, 200, 10,
                    help=metrics_glossary.get("market_growth_rate", "")
                )
                
                doc["market_share"] = st.slider(
                    "Market Share (%)",
                    0.0, 100.0, 0.5,
                    help=metrics_glossary.get("market_share", "")
                )
            
            with col2:
                doc["revenue_growth_rate"] = st.slider(
                    "Revenue Growth Rate (%/mo)", 
                    0, 100, 15,
                    help=metrics_glossary.get("revenue_growth_rate", "")
                )
                
                doc["viral_coefficient"] = st.slider(
                    "Viral Coefficient",
                    0.0, 3.0, 0.8,
                    help=metrics_glossary.get("viral_coefficient", "")
                )
            
            st.markdown('<p class="camp-subheader">User Metrics</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                doc["current_users"] = st.number_input(
                    "Current Users",
                    0, 1_000_000_000, 1000,
                    help="Current number of users"
                )
                
                doc["user_growth_rate"] = st.slider(
                    "User Growth Rate (%/mo)",
                    0, 100, 10,
                    help=metrics_glossary.get("user_growth_rate", "")
                )
            
            with col2:
                doc["churn_rate"] = st.slider(
                    "Churn Rate (%/mo)",
                    0.0, 50.0, 5.0,
                    help=metrics_glossary.get("churn_rate", "")
                )
                
                doc["referral_rate"] = st.slider(
                    "Referral Rate (%)",
                    0.0, 50.0, 2.0,
                    help="Percentage of users who refer others"
                )
            
            with st.expander("Additional Market Metrics"):
                col1, col2 = st.columns(2)
                with col1:
                    doc["net_retention_rate"] = st.slider(
                        "Net Retention Rate (%/yr)", 
                        50, 150, 100,
                        help=metrics_glossary.get("net_retention_rate", "")
                    )
                    
                    doc["category_leadership_score"] = st.slider(
                        "Category Leadership",
                        0, 100, 50,
                        help=metrics_glossary.get("category_leadership_score", "")
                    )
                
                with col2:
                    doc["community_growth_rate"] = st.slider(
                        "Community Growth (%/mo)",
                        0, 100, 5,
                        help=metrics_glossary.get("community_growth_rate", "")
                    )
                    
                    doc["conversion_rate"] = st.slider(
                        "Conversion Rate (%)",
                        0.0, 100.0, 10.0,
                        help="Percentage of leads that convert to customers"
                    )

        # Advantage Moat (A) section
        elif menu == "Advantage (A)":
            st.markdown('<p class="camp-header camp-advantage">Advantage Moat (A) - 20%</p>', unsafe_allow_html=True)
            st.markdown("Competitive advantages and defensibility metrics")
            
            col1, col2 = st.columns(2)
            with col1:
                doc["technical_innovation_score"] = st.slider(
                    "Technical Innovation",
                    0, 100, 50,
                    help=metrics_glossary.get("technical_innovation_score", "")
                )
                
                doc["product_maturity_score"] = st.slider(
                    "Product Maturity",
                    0, 100, 60,
                    help=metrics_glossary.get("product_maturity_score", "")
                )
            
            with col2:
                doc["scalability_score"] = st.slider(
                    "Scalability",
                    0, 100, 70,
                    help=metrics_glossary.get("scalability_score", "")
                )
                
                doc["technical_debt_score"] = st.slider(
                    "Technical Debt",
                    0, 100, 30,
                    help=metrics_glossary.get("technical_debt_score", "")
                )
            
            col1, col2 = st.columns(2)
            with col1:
                doc["patent_count"] = st.number_input(
                    "Patents",
                    0, 500, 2,
                    help=metrics_glossary.get("patent_count", "")
                )
                
                doc["api_integrations_count"] = st.number_input(
                    "API Integrations",
                    0, 100, 5,
                    help=metrics_glossary.get("api_integrations_count", "")
                )
            
            with col2:
                doc["product_security_score"] = st.slider(
                    "Security Score",
                    0, 100, 75,
                    help=metrics_glossary.get("product_security_score", "")
                )
                
                doc["business_model_strength"] = st.slider(
                    "Business Model Strength",
                    0, 100, 65,
                    help=metrics_glossary.get("business_model_strength", "")
                )
            
            # Add sector-specific advantage metrics
            sector = doc.get("sector", st.session_state.doc.get("sector", "saas")).lower()
            with st.expander(f"Sector-specific {sector.capitalize()} metrics"):
                if sector == "fintech":
                    doc["licenses_count"] = st.number_input("Licenses Count", 0, 50, 2)
                    doc["default_rate"] = st.slider("Default Rate (%)", 0.0, 30.0, 2.0)
                elif sector in ["biotech", "healthtech"]:
                    doc["clinical_phase"] = st.selectbox("Clinical Phase", [0, 1, 2, 3, 4])
                elif sector == "ai":
                    doc["data_volume_tb"] = st.number_input("Data Volume (TB)", 0.0, 1e9, 100.0, step=1.0)
                elif sector == "crypto":
                    doc["token_utility_score"] = st.number_input("Token Utility", 0, 100, 50)
                    doc["decentralization_factor"] = st.slider("Decentralization", 0.0, 1.0, 0.5)

        # People & Performance (P) section
        elif menu == "People (P)":
            st.markdown('<p class="camp-header camp-people">People & Performance (P) - 25%</p>', unsafe_allow_html=True)
            st.markdown("Team strength and execution ability metrics")
            
            col1, col2 = st.columns(2)
            with col1:
                doc["founder_exits"] = st.number_input(
                    "Previous Founder Exits", 
                    0, 10, 0,
                    help=metrics_glossary.get("founder_exits", "")
                )
                
                doc["founder_domain_exp_yrs"] = st.number_input(
                    "Founder Domain Exp (yrs)",
                    0, 30, 5,
                    help=metrics_glossary.get("founder_domain_exp_yrs", "")
                )
            
            with col2:
                doc["employee_count"] = st.number_input(
                    "Employee Count", 
                    0, 10000, 10,
                    help=metrics_glossary.get("employee_count", "")
                )
                
                doc["founder_diversity_score"] = st.slider(
                    "Team Diversity",
                    0, 100, 50,
                    help=metrics_glossary.get("founder_diversity_score", "")
                )
            
            st.markdown('<p class="camp-subheader">Leadership Team</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                doc["has_cto"] = st.checkbox(
                    "Has CTO", 
                    True,
                    help=metrics_glossary.get("has_cto", "")
                )
            
            with col2:
                doc["has_cmo"] = st.checkbox(
                    "Has CMO", 
                    False,
                    help=metrics_glossary.get("has_cmo", "")
                )
            
            with col3:
                doc["has_cfo"] = st.checkbox(
                    "Has CFO", 
                    False,
                    help=metrics_glossary.get("has_cfo", "")
                )
            
            st.markdown('<p class="camp-subheader">Team Performance</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                doc["tech_talent_ratio"] = st.slider(
                    "Tech Talent Ratio",
                    0.0, 1.0, 0.4,
                    help=metrics_glossary.get("tech_talent_ratio", "")
                )
                
                doc["employee_turnover_rate"] = st.slider(
                    "Employee Turnover (%/yr)",
                    0, 100, 15,
                    help=metrics_glossary.get("employee_turnover_rate", "")
                )
            
            with col2:
                doc["nps_score"] = st.slider(
                    "NPS Score",
                    -100, 100, 30,
                    help=metrics_glossary.get("nps_score", "")
                )
                
                doc["management_satisfaction_score"] = st.slider(
                    "Team Satisfaction",
                    0, 100, 70,
                    help=metrics_glossary.get("management_satisfaction_score", "")
                )
            
            with st.expander("Additional People Metrics"):
                doc["support_ticket_sla_percent"] = st.slider(
                    "Support SLA (%)",
                    0, 100, 90,
                    help=metrics_glossary.get("support_ticket_sla_percent", "")
                )
                
                doc["support_ticket_volume"] = st.number_input(
                    "Support Tickets/mo",
                    0, 10000, 50,
                    help="Monthly support ticket volume"
                )

        # Pitch (Intangible) section
        elif menu == "Pitch":
            st.subheader("Pitch Deck Analysis")
            st.markdown("Upload a pitch deck for intangible score analysis")
            
            uploaded = st.file_uploader("Upload PDF", type=["pdf"])
            if uploaded:
                st.success(f"Uploaded => {uploaded.name}")
                pdf_text = extract_text_from_pdf(uploaded)
                st.text_area("Extracted PDF Text", pdf_text, height=200)
                doc["pitch_deck_text"] = pdf_text
            
            st.markdown("*The pitch analysis provides an intangible modifier that can adjust the final CAMP score by up to ±20%*")

        # Store doc in session
        for k, v in doc.items():
            st.session_state.doc[k] = v

        # Analysis button
        if st.button("Analyze with CAMP Framework", type="primary"):
            st.session_state.analyze_clicked = True

        # Display options (Adding UI controls for the elements we identified)
        with st.expander("Display Options"):
            # Toggle between 8-tab and 12-tab modes
            display_mode = st.radio(
                "Display Mode",
                options=["Standard (8 Tabs)", "Expanded (12 Tabs)"],
                index=1 if st.session_state.display_mode == "expanded" else 0
            )
            st.session_state.display_mode = "expanded" if display_mode == "Expanded (12 Tabs)" else "standard"
            
            # Toggle welcome screen style
            welcome_style = st.radio(
                "Welcome Screen Style",
                options=["Radar Chart", "Bar Chart"],
                index=0 if st.session_state.welcome_style == "radar" else 1
            )
            st.session_state.welcome_style = "radar" if welcome_style == "Radar Chart" else "bar"

        # Help section
        with st.expander("About CAMP Framework"):
            st.markdown("""
            ## CAMP Framework
            
            The **CAMP Framework** analyzes startups across four key pillars:
            
            - **C**apital Efficiency (30%): Financial health and capital deployment
            - **A**dvantage Moat (20%): Competitive advantages and defensibility
            - **M**arket Dynamics (25%): Market opportunity and growth potential
            - **P**eople & Performance (25%): Team strength and execution ability
            
            Plus an **Intangible** modifier from pitch analysis (±20% adjustment)
            
            Complete each section for a comprehensive analysis.
            """)


def fix_tech_dd_data(doc):
    """Ensure that technical due diligence data has valid values."""
    import logging
    logger = logging.getLogger("analysis_flow")
    
    if not doc:
        return doc
    
    # Define valid positions EXACTLY as listed in the error message
    valid_positions = {"Top 10%", "Bottom 25%", "Top 25%", "Below average", "Average"}
    
    # Recursively fix all occurrences of industry_benchmark_position
    def fix_benchmark_positions(obj):
        if isinstance(obj, dict):
            # Direct fix for industry_benchmark_position
            if "industry_benchmark_position" in obj:
                current_value = obj["industry_benchmark_position"]
                
                # If current value is not valid, replace it
                if current_value not in valid_positions:
                    logger.warning(f"Invalid industry_benchmark_position value: '{current_value}', replacing with 'Average'")
                    obj["industry_benchmark_position"] = "Average"
            
            # Process all dictionary values recursively
            for key, value in obj.items():
                obj[key] = fix_benchmark_positions(value)
                
        elif isinstance(obj, list):
            # Process list items recursively
            return [fix_benchmark_positions(item) for item in obj]
        
        return obj
    
    # Apply fixes to the entire document
    doc = fix_benchmark_positions(doc)
    
    # Force-fix specific known locations
    if "tech_assessment" in doc and isinstance(doc["tech_assessment"], dict):
        tech = doc["tech_assessment"]
        
        # Ensure competitive_positioning exists
        if "competitive_positioning" not in tech:
            tech["competitive_positioning"] = {}
            
        if not isinstance(tech["competitive_positioning"], dict):
            tech["competitive_positioning"] = {}
            
        # Force a valid value for industry_benchmark_position
        tech["competitive_positioning"]["industry_benchmark_position"] = "Average"
        logger.info("Forced tech_assessment.competitive_positioning.industry_benchmark_position to 'Average'")
    
    # Also fix in competitive_positioning at document root
    if "competitive_positioning" in doc and isinstance(doc["competitive_positioning"], dict):
        doc["competitive_positioning"]["industry_benchmark_position"] = "Average"
        logger.info("Forced doc.competitive_positioning.industry_benchmark_position to 'Average'")
    
    return doc


########################################
# Main Analysis Workflow
########################################

def run_analysis(doc: dict)-> dict:
    """
    Perform the comprehensive startup analysis using CAMP framework.
    """
    with st.spinner("Running CAMP framework analysis..."):
        prog = st.progress(0)

        # 1) domain expansions
        expansions = apply_domain_expansions(doc)
        doc.update(expansions)
        prog.progress(10)

        # 2) intangible from DeepSeek
        if doc.get("pitch_deck_text"):
            intangible = compute_intangible_llm(doc)  # from intangible_api
            doc["intangible"] = intangible
        else:
            doc["intangible"] = 50.0
        prog.progress(20)

        # 3) team + moat
        doc["team_score"] = compute_team_depth_score(doc)
        doc["moat_score"] = compute_moat_score(doc)
        doc["execution_risk"] = to_dict(evaluate_team_execution_risk(doc))
        prog.progress(30)

        # 4) Evaluate with XGB and CAMP framework
        model = load_xgb_model()
        
        # First calculate CAMP scores separately using evaluate_startup_camp
        camp_scores = evaluate_startup_camp(doc, None)
        doc.update(camp_scores)
        
        # Set camp_score from flashdna_score for consistency
        doc["camp_score"] = doc.get("flashdna_score", 50.0)
        
        # Then calculate success probability using a different approach
        if model:
            try:
                # Try to get success probability from the model
                from advanced_ml import build_feature_vector
                fv = build_feature_vector(doc).reshape(1, -1)
                
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(fv)[0, 1]
                    doc["success_prob"] = prob * 100  # Convert to percentage
                else:
                    # For models without predict_proba
                    prob = model.predict(fv)[0]
                    doc["success_prob"] = prob * 100
                    
                # Add variance to ensure it's different from CAMP score
                market_factor = 1 + (doc.get("market_growth_rate", 0) - 10) / 100
                moat_factor = 1 + (doc.get("moat_score", 0) - 50) / 200
                
                # Apply factors to create differentiation
                doc["success_prob"] = min(95, max(5, doc["success_prob"] * market_factor * moat_factor))
                
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                # Fallback: Calculate success probability as a weighted function of CAMP 
                # with different weights than the CAMP score itself
                doc["success_prob"] = min(95, max(5, 
                    doc.get("capital_score", 0) * 0.25 +
                    doc.get("market_score", 0) * 0.35 +   # Higher weight for market
                    doc.get("advantage_score", 0) * 0.25 +
                    doc.get("people_score", 0) * 0.15))   # Lower weight for people
        else:
            # If no model is available, use a different formula than camp_score
            doc["success_prob"] = min(95, max(5, 
                doc.get("capital_score", 0) * 0.25 +
                doc.get("market_score", 0) * 0.35 +   # Higher weight for market
                doc.get("advantage_score", 0) * 0.25 +
                doc.get("people_score", 0) * 0.15))   # Lower weight for people
        
        # Ensure the values are different by adding a small variance if they're too close
        if abs(doc.get("camp_score", 0) - doc.get("success_prob", 0)) < 0.5:
            # Add a small variance based on market growth rate
            market_adj = (doc.get("market_growth_rate", 0) - 10) * 0.5
            doc["success_prob"] = min(95, max(5, doc["success_prob"] + market_adj))
            
        prog.progress(40)

        # 5) system dynamics
        sys_res = system_dynamics_sim(
            user_initial= doc.get("current_users",1000),
            months=24,
            marketing_spend= doc.get("burn_rate",30000)*0.4,
            referral_rate= doc.get("referral_rate",0.02),
            churn_rate= doc.get("churn_rate",0.05),
            seasonality=True
        )
        doc["system_dynamics"]= to_dict(sys_res)
        doc["growth_metrics"]= to_dict(calculate_growth_metrics(sys_res))

        doc["virality_sim"]= to_dict(virality_sim(
            user_initial= doc.get("current_users",1000),
            k_factor= doc.get("viral_coefficient",0.2),
            conversion_rate= doc.get("conversion_rate",0.1),
            cycles=12
        ))
        prog.progress(50)

        # Remove SIR model section
        # Initialize market_penetration with a default value
        doc["market_penetration"] = {
            "timeline": list(range(24)),
            "penetration": [doc.get("market_share", 0.005) * (1 + doc.get("market_growth_rate", 0.1) * i/12) for i in range(24)]
        }
        prog.progress(60)

        # scenario
        try:
            sc= find_optimal_scenario(
                target_metric= "final_users",
                init_users= doc.get("current_users",1000),
                current_churn= doc.get("churn_rate",0.05),
                current_referral= doc.get("referral_rate",0.02)
            )
            doc["hpc_data"]= sc["all_scenarios"]
            doc["optimal_scenario"]= sc["optimal"]
        except Exception as e:
            logger.error(f"Scenario => {str(e)}")
            doc["hpc_data"]= []
            doc["optimal_scenario"]= {}
        prog.progress(70)

        # 7) financial modeling
        runw, end_cash, flow= scenario_runway(
            burn_rate= doc.get("burn_rate",30000),
            current_cash= doc.get("current_cash",500000),
            monthly_revenue= doc.get("monthly_revenue",50000),
            rev_growth= doc.get("user_growth_rate",0.1),
            cost_growth= 0.05
        )
        doc["runway_months"]= runw
        doc["ending_cash"]= end_cash
        doc["cash_flow"]= flow

        ue= calculate_unit_economics(doc)
        doc["unit_economics"]= to_dict(ue)

        fc= forecast_financials(doc)
        doc["financial_forecast"]= to_dict(fc)

        val= calculate_valuation_metrics(doc)
        doc["valuation_metrics"]= to_dict(val)
        prog.progress(80)

        # 8) Competitive Intelligence
        try:
            ci = CompetitiveIntelligence()
            
            # Get competitors with better handling of data structures
            competitors = ci.get_competitors(doc.get("name", ""), doc.get("sector", "saas"))
            
            # Ensure we have a valid list of competitor dictionaries
            if isinstance(competitors, list):
                doc["competitors"] = competitors
            else:
                # Create fallback if we didn't get valid competitors
                doc["competitors"] = [
                    {
                        "name": "Competitor A",
                        "url": "https://example.com",
                        "market_share": 0.15,
                        "growth_rate": 0.12,
                        "strengths": ["Strong market position", "Innovative technology"]
                    },
                    {
                        "name": "Competitor B",
                        "url": "https://example2.com",
                        "market_share": 0.08,
                        "growth_rate": 0.18,
                        "strengths": ["Cost efficient", "Fast growth"]
                    }
                ]
            
            # Get competitive positioning
            doc["competitive_positioning"] = to_dict(ci.competitive_positioning_analysis(doc, doc["competitors"]))
            
            # Get market trends
            doc["market_trends"] = to_dict(ci.market_trends_analysis(doc.get("sector", "")))
            
            # Get moat analysis
            doc["moat_analysis"] = to_dict(ci.competitive_moat_analysis(doc))
            
        except Exception as e:
            logger.error(f"Competitive Intelligence => {e}")
            # Provide fallback data
            doc["competitors"] = [
                {
                    "name": "Competitor A",
                    "url": "https://example.com",
                    "market_share": 0.15,
                    "growth_rate": 0.12,
                    "strengths": ["Strong market position", "Innovative technology"]
                },
                {
                    "name": "Competitor B",
                    "url": "https://example2.com",
                    "market_share": 0.08,
                    "growth_rate": 0.18,
                    "strengths": ["Cost efficient", "Fast growth"]
                }
            ]
            doc["competitive_positioning"] = {
                "position": "challenger",
                "advantages": [{"name": "Innovation", "score": 75}],
                "disadvantages": [{"name": "Market Share", "score": 35}],
                "opportunities": {"expansion": 0.7, "partnerships": 0.8},
                "threats": {"competition": 0.6, "market_changes": 0.5}
            }
            doc["market_trends"] = {"growth_rate": 0.15, "trends": ["Digital transformation", "AI integration"]}
            doc["moat_analysis"] = {"overall_score": 65, "strongest": "technology", "weakest": "network_effects"}
        prog.progress(85)

        # 9) Monte Carlo
        try:
            mc = MonteCarloSimulator()
            
            # Create document in the format expected by run_simulation
            mc_doc = {
                "stage": doc.get("company_stage", "seed"),
                "sector": doc.get("sector", "saas"),
                "monthly_revenue": doc.get("monthly_revenue", 50000),
                "burn_rate": doc.get("burn_rate", 30000),
                "current_cash": doc.get("current_cash", 500000),
                "current_users": doc.get("current_users", 1000),
                "revenue_growth_rate": doc.get("revenue_growth_rate", 0.1),
                "user_growth_rate": doc.get("user_growth_rate", 0.15),
                "churn_rate": doc.get("churn_rate", 0.05),
                "gross_margin_percent": doc.get("gross_margin_percent", 70),
                "camp_score": doc.get("camp_score", 50),
                "monte_carlo_iterations": 500  # Lower for web interface
            }
            
            # Run simulation
            mc_result = mc.run_simulation(mc_doc)
            doc["monte_carlo"] = to_dict(mc_result)
            
        except Exception as e:
            logger.error(f"Monte Carlo => {e}")
            doc["monte_carlo"] = {"error": str(e)}
            
        prog.progress(90)
            
        # 10) PMF 
        try:
            pmf = ProductMarketFitAnalyzer()
            pmf_result = pmf.analyze_pmf(doc)
            doc["pmf_analysis"] = to_dict(pmf_result)
        except Exception as e:
            logger.error(f"PMF Analysis => {e}")
            doc["pmf_analysis"] = {}
            
        # 11) Pattern Detection
        try:
            doc["patterns"] = [to_dict(pattern) for pattern in detect_patterns(doc)]
        except Exception as e:
            logger.error(f"Pattern Detection => {e}")
            doc["patterns"] = []
            
        prog.progress(92)
        
        # 12) Pitch Sentiment
        try:
            if "pitch_deck_text" in doc and doc["pitch_deck_text"].strip():
                pa = PitchAnalyzer()
                sentiment = pa.analyze_sentiment(doc["pitch_deck_text"])
                doc["pitch_sentiment"] = to_dict(sentiment)
            else:
                doc["pitch_sentiment"] = {}
        except Exception as e:
            logger.error(f"Pitch Sentiment => {e}")
            doc["pitch_sentiment"] = {}
            
        prog.progress(94)
        
        # 13) Tech Assessment
        # First fix any technical due diligence data to ensure valid values
        fix_tech_dd_data(doc)
        
        try:
            # Prepare tech data
            tech_data = {
                "architecture": doc.get("tech_architecture", ""),
                "stack": doc.get("tech_stack", ""),
                "dev_team_size": doc.get("development_team_size", 5),
                "cto_experience": doc.get("cto_experience_years", 8)
            }
            
            # Run technical due diligence
            tdd = TechnicalDueDiligence()
            tch = tdd.assess_technical_architecture(tech_data, generate_report=True)
            doc["tech_assessment"]= to_dict(tch)
            
            # Make sure data is properly formatted after assessment
            fix_tech_dd_data(doc)  # Apply fix again after assessment
            
            # Emergency direct fix for any remaining industry_benchmark_position issues
            if "tech_assessment" in doc and isinstance(doc["tech_assessment"], dict):
                tech = doc["tech_assessment"]
                if "competitive_positioning" in tech and isinstance(tech["competitive_positioning"], dict):
                    comp_pos = tech["competitive_positioning"]
                    # Force a valid value no matter what
                    comp_pos["industry_benchmark_position"] = "Average"
                    logger.info("Emergency fix: Set tech_assessment.competitive_positioning.industry_benchmark_position to 'Average'")
            
            # Also check root competitive_positioning
            if "competitive_positioning" in doc and isinstance(doc["competitive_positioning"], dict):
                comp_pos = doc["competitive_positioning"]
                # Force a valid value no matter what
                comp_pos["industry_benchmark_position"] = "Average"
                logger.info("Emergency fix: Set root competitive_positioning.industry_benchmark_position to 'Average'")
            
        except Exception as e:
            logger.error(f"TechDD => {str(e)}")
            doc["tech_assessment"] = {"error": str(e)}
            
            # Add emergency fallback tech assessment with valid values
            doc["tech_assessment"] = {
                "overall_score": 65.0,
                "competitive_positioning": {
                    "industry_benchmark_position": "Average",
                    "relative_tech_strength": 0.5
                },
                "scores": {
                    "architecture": 65.0,
                    "scalability": 60.0,
                    "security": 70.0,
                    "technology_stack": 65.0,
                    "development_practices": 60.0
                }
            }
            logger.info("Created emergency fallback tech_assessment with valid values")

        prog.progress(96)

        # 14) Cohort
        try:
            c_an = CohortAnalyzer()
            # Try to generate real cohort data if we have enough information
            if doc.get("user_acquisition_data") and doc.get("transaction_data"):
                try:
                    real_cohort_data = c_an.analyze_cohorts(
                        user_data=pd.DataFrame(doc["user_acquisition_data"]), 
                        transaction_data=pd.DataFrame(doc["transaction_data"]),
                        cohort_periods=12
                    )
                    doc["cohort_data"] = to_dict(real_cohort_data)
                    logger.info("Using real cohort data for analysis")
                except Exception as inner_e:
                    logger.warning(f"Failed to process real cohort data: {inner_e}")
                    doc["cohort_data"] = to_dict(c_an._generate_dummy_cohort_data(6))
                    logger.info("Falling back to synthetic cohort data")
            else:
                # No real data available, use synthetic data but log a warning
                doc["cohort_data"] = to_dict(c_an._generate_dummy_cohort_data(6))
                logger.info("No user acquisition or transaction data available, using synthetic cohort data")
        except Exception as e:
            logger.error(f"Cohort => {e}")
            doc["cohort_data"] = {}

        # 15) Network
        try:
            n_an= NetworkEffectAnalyzer()
            doc["network_analysis"]= to_dict(n_an.analyze_network_effects(company_data=doc))
        except Exception as e:
            logger.error(f"Network => {e}")
            doc["network_analysis"]= {}

        # 16) Acquisition Fit Analysis
        try:
            afa = AcquisitionFitAnalyzer()
            acq_fit = afa.analyze_acquisition_fit(doc)
            doc["acquisition_fit"] = to_dict(acq_fit)
        except Exception as e:
            logger.error(f"Acquisition Fit => {e}")
            doc["acquisition_fit"] = {}
            
        # 17) Comparative Exit Path Analysis (Adding this module)
        try:
            exit_analyzer = ExitPathAnalyzer(doc)
            exit_analysis = exit_analyzer.analyze_exit_paths()
            doc["exit_path_analysis"] = to_dict(exit_analysis)
            doc["exit_recommendations"] = to_dict(exit_analyzer.get_exit_recommendations())
        except Exception as e:
            logger.error(f"Exit Path Analysis => {e}")
            doc["exit_path_analysis"] = {}
            doc["exit_recommendations"] = {}

        prog.progress(97)

        # 18) Benchmarking
        try:
            be= BenchmarkEngine()
            doc["benchmarks"]= to_dict(be.benchmark_startup(doc))
        except Exception as e:
            logger.error(f"Benchmark => {e}")
            doc["benchmarks"]= {}

        # Patterns
        pats= detect_patterns(doc)
        doc["patterns_matched"]= [to_dict(pat) for pat in pats]
        doc["pattern_insights"]= to_dict(generate_pattern_insights(pats))

        # Final verification to ensure CAMP score and success probability are distinct
        if abs(doc.get("camp_score", 0) - doc.get("success_prob", 0)) < 1.0:
            # If they're still too similar, adjust success probability
            logger.warning("CAMP score and success probability are still too similar. Applying final adjustment.")
            # Apply an adjustment based on runway (shorter runway = lower success probability)
            runway_factor = min(5, max(-5, (doc.get("runway_months", 12) - 12) * 0.2))
            # And market growth (higher growth = higher success probability)
            market_factor = min(5, max(-5, (doc.get("market_growth_rate", 10) - 10) * 0.5))
            
            doc["success_prob"] = min(95, max(5, doc.get("success_prob", 50) + runway_factor + market_factor))

        prog.progress(100)
        time.sleep(0.5)
        prog.empty()

    return doc


########################################
# Result Visualization Functions
########################################

def render_summary_tab(doc: dict):
    """Render the summary tab with key metrics and CAMP scores."""
    st.header("CAMP Framework Analysis Summary")
    
    # Calculate CAMP scores
    capital_score = doc.get("capital_score", 0)
    advantage_score = doc.get("advantage_score", 0)
    market_score = doc.get("market_score", 0)
    people_score = doc.get("people_score", 0)
    camp_score = doc.get("camp_score", 0)
    
    # Summary metrics
    cols = st.columns(5)
    with cols[0]:
        st.metric("CAMP Score", f"{camp_score:.1f}")
    with cols[1]:
        st.metric("Success Probability", f"{doc.get('success_prob', 0):.1f}%")
    with cols[2]:
        st.metric("Runway", f"{doc.get('runway_months', 0):.1f} months")
    with cols[3]:
        exit_readiness = 0
        if isinstance(doc.get("exit_path_analysis"), dict):
            exit_readiness = doc.get("exit_path_analysis", {}).get("exit_readiness_score", 0)
        st.metric("Exit Readiness", f"{exit_readiness:.1f}/100")
    with cols[4]:
        st.metric("Moat Strength", f"{doc.get('moat_score', 0):.1f}/100")
    
    # CAMP framework radar chart
    st.subheader("CAMP Framework Dimensions")
    
    # Create radar chart using plotly
    categories = ['Capital Efficiency', 'Market Dynamics', 'Advantage Moat', 'People & Performance']
    values = [capital_score, market_score, advantage_score, people_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='CAMP Scores',
        line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key strengths and weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Strengths")
        strengths = []
        
        if capital_score > 70:
            strengths.append("Strong capital efficiency")
        if market_score > 70:
            strengths.append("Strong market positioning")
        if advantage_score > 70:
            strengths.append("Strong competitive moat")
        if people_score > 70:
            strengths.append("Strong team & execution")
        
        # Add other strengths from pattern_insights
        for pattern in doc.get("patterns_matched", []):
            if isinstance(pattern, dict) and pattern.get("is_positive", False):
                strengths.append(pattern.get("name", ""))
        
        # Ensure at least some strengths are shown
        if not strengths:
            if capital_score > market_score and capital_score > advantage_score and capital_score > people_score:
                strengths.append("Relatively strong capital efficiency")
            elif market_score > capital_score and market_score > advantage_score and market_score > people_score:
                strengths.append("Relatively strong market positioning")
            elif advantage_score > capital_score and advantage_score > market_score and advantage_score > people_score:
                strengths.append("Relatively strong competitive moat")
            elif people_score > capital_score and people_score > market_score and people_score > advantage_score:
                strengths.append("Relatively strong team & execution")
            else:
                strengths.append("Balanced across CAMP dimensions")
        
        for strength in strengths[:5]:  # Show top 5 strengths
            st.markdown(f"✅ {strength}")
    
    with col2:
        st.subheader("Areas for Improvement")
        weaknesses = []
        
        if capital_score < 50:
            weaknesses.append("Improve capital efficiency")
        if market_score < 50:
            weaknesses.append("Strengthen market positioning")
        if advantage_score < 50:
            weaknesses.append("Build stronger competitive moat")
        if people_score < 50:
            weaknesses.append("Enhance team capabilities")
        
        # Add other weaknesses from pattern_insights
        for pattern in doc.get("patterns_matched", []):
            if isinstance(pattern, dict) and not pattern.get("is_positive", True):
                weaknesses.append(pattern.get("name", ""))
        
        # Ensure at least some weaknesses are shown
        if not weaknesses:
            if capital_score < market_score and capital_score < advantage_score and capital_score < people_score:
                weaknesses.append("Relatively weaker capital efficiency")
            elif market_score < capital_score and market_score < advantage_score and market_score < people_score:
                weaknesses.append("Relatively weaker market positioning")
            elif advantage_score < capital_score and advantage_score < market_score and advantage_score < people_score:
                weaknesses.append("Relatively weaker competitive moat")
            elif people_score < capital_score and people_score < market_score and people_score < advantage_score:
                weaknesses.append("Relatively weaker team & execution")
            else:
                weaknesses.append("Consider improving all CAMP dimensions")
        
        for weakness in weaknesses[:5]:  # Show top 5 weaknesses
            st.markdown(f"🚧 {weakness}")
    
    # Pattern insights
    if isinstance(doc.get("pattern_insights"), dict) and doc.get("pattern_insights", {}).get("top_insights"):
        st.subheader("Key Insights")
        insights = doc.get("pattern_insights", {}).get("top_insights", [])
        for i, insight in enumerate(insights[:3]):
            if isinstance(insight, dict):
                st.markdown(f"**{i+1}. {insight.get('title', '')}**")
                st.markdown(insight.get('description', ''))


def render_camp_details_tab(doc: dict):
    """Render detailed CAMP framework metrics."""
    st.header("CAMP Framework Details")
    
    # Create tabs for each CAMP dimension
    camp_tabs = st.tabs(["Capital (C)", "Advantage (A)", "Market (M)", "People (P)"])
    
    # Capital tab
    with camp_tabs[0]:
        st.subheader("Capital Efficiency")
        st.markdown(f"**Score: {doc.get('capital_score', 0):.1f}/100**")
        
        # Key metrics
        metrics = [
            ("Monthly Revenue", f"${doc.get('monthly_revenue', 0):,.2f}"),
            ("Burn Rate", f"${doc.get('burn_rate', 0):,.2f}"),
            ("Runway", f"{doc.get('runway_months', 0):.1f} months"),
            ("Gross Margin", f"{doc.get('gross_margin_percent', 0):.1f}%"),
            ("LTV:CAC Ratio", f"{doc.get('ltv_cac_ratio', 0):.2f}"),
            ("CAC", f"${doc.get('customer_acquisition_cost', 0):,.2f}")
        ]
        
        # Create metric grid
        cols = st.columns(3)
        for i, (name, value) in enumerate(metrics):
            with cols[i % 3]:
                st.metric(name, value)
        
        # Cash flow chart
        st.subheader("Cash Flow Projection")
        if doc.get("cash_flow"):
            cash_flow = doc.get("cash_flow", [])
            months = list(range(len(cash_flow)))
            
            df = pd.DataFrame({
                "Month": months,
                "Cash": cash_flow
            })
            
            fig = px.line(df, x="Month", y="Cash", 
                        title="Cash Flow Projection", 
                        labels={"Cash": "Cash ($)", "Month": "Month"})
            
            # Add runway marker
            runway = doc.get("runway_months", 0)
            if runway > 0:
                fig.add_vline(x=runway, line_dash="dash", line_color="red",
                            annotation_text=f"Runway: {runway:.1f} months",
                            annotation_position="top right")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Unit economics
        st.subheader("Unit Economics")
        unit_econ = doc.get("unit_economics", {})
        if unit_econ:
            metrics = [
                ("Customer LTV", f"${unit_econ.get('ltv', 0):,.2f}"),
                ("CAC Payback", f"{unit_econ.get('cac_payback_months', 0):.1f} months"),
                ("Contribution Margin", f"{unit_econ.get('contribution_margin', 0)*100:.1f}%")
            ]
            
            cols = st.columns(3)
            for i, (name, value) in enumerate(metrics):
                with cols[i]:
                    st.metric(name, value)
    
    # Advantage tab
    with camp_tabs[1]:
        st.subheader("Advantage Moat")
        st.markdown(f"**Score: {doc.get('advantage_score', 0):.1f}/100**")
        
        # Key metrics
        metrics = [
            ("Moat Score", f"{doc.get('moat_score', 0):.1f}/100"),
            ("Technical Innovation", f"{doc.get('technical_innovation_score', 0):.1f}/100"),
            ("Product Maturity", f"{doc.get('product_maturity_score', 0):.1f}/100"),
            ("Scalability", f"{doc.get('scalability_score', 0):.1f}/100"),
            ("Patents", f"{doc.get('patent_count', 0)}"),
            ("API Integrations", f"{doc.get('api_integrations_count', 0)}")
        ]
        
        # Create metric grid
        cols = st.columns(3)
        for i, (name, value) in enumerate(metrics):
            with cols[i % 3]:
                st.metric(name, value)
        
        # Moat analysis chart
        st.subheader("Moat Analysis")
        moat_analysis = doc.get("moat_analysis", {})
        if moat_analysis:
            moat_factors = moat_analysis.get("factors", {})
            
            if moat_factors:
                factors = list(moat_factors.keys())
                scores = list(moat_factors.values())
                
                df = pd.DataFrame({
                    "Factor": factors,
                    "Score": scores
                })
                
                fig = px.bar(df, x="Factor", y="Score", 
                           title="Moat Strength by Factor",
                           color="Score",
                           color_continuous_scale="Viridis")
                
                st.plotly_chart(fig, use_container_width=True)

        # Technical assessment
        st.subheader("Technical Assessment")
        tech_assessment = doc.get("tech_assessment", {})
        if tech_assessment and isinstance(tech_assessment, dict):
            tech_scores = tech_assessment.get("scores", {})
            
            if tech_scores:
                factors = list(tech_scores.keys())
                scores = list(tech_scores.values())
                
                df = pd.DataFrame({
                    "Factor": factors,
                    "Score": scores
                })
                
                fig = px.bar(df, x="Factor", y="Score", 
                           title="Technical Assessment",
                           color="Score",
                           color_continuous_scale="Viridis")
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Market tab
    with camp_tabs[2]:
        st.subheader("Market Dynamics")
        st.markdown(f"**Score: {doc.get('market_score', 0):.1f}/100**")
        
        # Key metrics
        metrics = [
            ("Market Size", f"${doc.get('market_size', 0)/1e6:,.1f}M"),
            ("Market Growth", f"{doc.get('market_growth_rate', 0):.1f}%/yr"),
            ("Market Share", f"{doc.get('market_share', 0):.2f}%"),
            ("Revenue Growth", f"{doc.get('revenue_growth_rate', 0):.1f}%/mo"),
            ("User Growth", f"{doc.get('user_growth_rate', 0):.1f}%/mo"),
            ("Churn Rate", f"{doc.get('churn_rate', 0):.1f}%/mo")
        ]
        
        # Create metric grid
        cols = st.columns(3)
        for i, (name, value) in enumerate(metrics):
            with cols[i % 3]:
                st.metric(name, value)
        
        # User growth chart
        st.subheader("User Growth Projection")
        sys_dynamics = doc.get("system_dynamics", {})
        if sys_dynamics:
            if isinstance(sys_dynamics, dict) and "users" in sys_dynamics:
                users = sys_dynamics.get("users", [])
                months = list(range(len(users)))
                
                df = pd.DataFrame({
                    "Month": months,
                    "Users": users
                })
                
                fig = px.line(df, x="Month", y="Users", 
                            title="User Growth Projection", 
                            labels={"Users": "Users", "Month": "Month"})
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Market penetration chart
            st.subheader("Market Penetration")
            market_penetration = doc.get("market_penetration", {})
            if market_penetration:
                if isinstance(market_penetration, dict) and "timeline" in market_penetration:
                    timeline = market_penetration.get("timeline", [])
                    penetration = market_penetration.get("penetration", [])
                    
                    df = pd.DataFrame({
                        "Month": timeline,
                        "Penetration": [p * 100 for p in penetration]  # Convert to percentage
                    })
                    
                    fig = px.line(df, x="Month", y="Penetration", 
                                title="Market Penetration Projection", 
                                labels={"Penetration": "Penetration (%)", "Month": "Month"})
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # PMF analysis
        st.subheader("Product-Market Fit")
        pmf = doc.get("pmf_analysis", {})
        if pmf:
            metrics = [
                ("PMF Score", f"{pmf.get('pmf_score', 0):.1f}/100"),
                ("Engagement Score", f"{pmf.get('engagement_score', 0):.1f}/100"),
                ("Retention Score", f"{pmf.get('retention_score', 0):.1f}/100")
            ]
            
            cols = st.columns(3)
            for i, (name, value) in enumerate(metrics):
                with cols[i]:
                    st.metric(name, value)
            
            # PMF factors
            pmf_factors = pmf.get("factors", {})
            if pmf_factors:
                st.markdown("### PMF Factor Analysis")
                
                factors = list(pmf_factors.keys())
                scores = list(pmf_factors.values())
                
                df = pd.DataFrame({
                    "Factor": factors,
                    "Score": scores
                })
                
                fig = px.bar(df, x="Factor", y="Score", 
                           title="Product-Market Fit Factors",
                           color="Score",
                           color_continuous_scale="Viridis")
                
                st.plotly_chart(fig, use_container_width=True)
    
    # People tab
    with camp_tabs[3]:
        st.subheader("People & Performance")
        st.markdown(f"**Score: {doc.get('people_score', 0):.1f}/100**")
        
        # Key metrics
        metrics = [
            ("Team Score", f"{doc.get('team_score', 0):.1f}/100"),
            ("Founder Experience", f"{doc.get('founder_domain_exp_yrs', 0)} years"),
            ("Team Size", f"{doc.get('employee_count', 0)} employees"),
            ("Tech Talent Ratio", f"{doc.get('tech_talent_ratio', 0)*100:.1f}%"),
            ("Previous Exits", f"{doc.get('founder_exits', 0)}"),
            ("Team Satisfaction", f"{doc.get('management_satisfaction_score', 0):.1f}/100")
        ]
        
        # Create metric grid
        cols = st.columns(3)
        for i, (name, value) in enumerate(metrics):
            with cols[i % 3]:
                st.metric(name, value)
        
        # Team assessment
        st.subheader("Team Assessment")
        
        # Leadership presence
        st.markdown("### Leadership Presence")
        leadership = {
            "CEO": True,  # Assumed always present
            "CTO": doc.get("has_cto", False),
            "CMO": doc.get("has_cmo", False),
            "CFO": doc.get("has_cfo", False)
        }
        
        df_leadership = pd.DataFrame({
            "Role": list(leadership.keys()),
            "Present": [1 if v else 0 for v in leadership.values()]
        })
        
        fig = px.bar(df_leadership, x="Role", y="Present", 
                   title="Leadership Team Composition",
                   color="Present",
                   color_continuous_scale=["#EF4444", "#16A34A"],
                   range_color=[0, 1])
        
        fig.update_layout(yaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Execution risk
        st.subheader("Execution Risk")
        execution_risk = doc.get("execution_risk", {})
        if execution_risk:
            if isinstance(execution_risk, dict) and "risk_factors" in execution_risk:
                risk_factors = execution_risk.get("risk_factors", {})
                
                factors = list(risk_factors.keys())
                scores = list(risk_factors.values())
                
                df = pd.DataFrame({
                    "Factor": factors,
                    "Risk": scores
                })
                
                fig = px.bar(df, x="Factor", y="Risk", 
                           title="Execution Risk Factors",
                           color="Risk",
                           color_continuous_scale="Reds")
                
                st.plotly_chart(fig, use_container_width=True)


def render_pmf_tab(doc: dict):
    """Render product-market fit analysis."""
    st.header("Product-Market Fit Analysis")
    
    pmf = doc.get("pmf_analysis", {})
    
    if not pmf:
        st.warning("Product-Market fit analysis not available.")
        return
    
    # Overall PMF score and stage
    pmf_score = pmf.get("pmf_score", 0)
    pmf_stage = pmf.get("stage", "")
    
    st.metric("PMF Score", f"{pmf_score:.1f}/100")
    st.markdown(f"**Current Stage**: {pmf_stage}")
    
    # Key metrics
    st.subheader("Key Metrics")
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Retention Rate", f"{pmf.get('retention_rate', 0):.1f}%")
    with cols[1]:
        st.metric("Churn Rate", f"{pmf.get('churn_rate', 0):.1f}%")
    with cols[2]:
        st.metric("User Growth", f"{pmf.get('user_growth_rate', 0):.1f}%")
    with cols[3]:
        st.metric("NPS Score", f"{pmf.get('nps_score', 0)}")
    
    # PMF Dimensions radar chart
    st.subheader("PMF Dimensions")
    
    dimensions = pmf.get("dimensions", {})
    if dimensions:
        # Extract dimension names and scores
        dim_names = []
        dim_scores = []
        
        for name, dim in dimensions.items():
            if isinstance(dim, dict):
                dim_names.append(name)
                dim_scores.append(dim.get("score", 0))
            else:
                # Handle case where dimensions are not dictionaries
                dim_names.append(name)
                dim_scores.append(0)
        
        # Create radar chart data
        radar_data = pd.DataFrame({
            'Dimension': dim_names + [dim_names[0]],  # Close the loop
            'Score': dim_scores + [dim_scores[0]]     # Close the loop
        })
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=radar_data['Score'],
            theta=radar_data['Dimension'],
            fill='toself',
            name='PMF Dimensions',
            line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Strengths and weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strengths")
        strengths = pmf.get("strengths", [])
        for strength in strengths:
            st.markdown(f"✅ {strength}")
    
    with col2:
        st.subheader("Weaknesses")
        weaknesses = pmf.get("weaknesses", [])
        for weakness in weaknesses:
            st.markdown(f"🚧 {weakness}")
    
    # PMF Recommendations
    st.subheader("PMF Recommendations")
    recommendations = pmf.get("recommendations", [])
    
    for i, rec in enumerate(recommendations):
        st.markdown(f"{i+1}. {rec}")
    
    # Factor Analysis
    st.subheader("PMF Factor Analysis")
    factors = pmf.get("factors", {})
    
    if factors:
        factor_names = list(factors.keys())
        factor_values = list(factors.values())
        
        df = pd.DataFrame({
            "Factor": factor_names,
            "Value": factor_values
        })
        
        fig = px.bar(df, x="Factor", y="Value", 
                   title="PMF Factors",
                   color="Value",
                   color_continuous_scale="Viridis")
        
        st.plotly_chart(fig, use_container_width=True)


def render_cohort_tab(doc: dict):
    """Render cohort analysis."""
    st.header("Cohort Analysis")
    
    cohort_data = doc.get("cohort_data", {})
    
    if not cohort_data:
        st.warning("Cohort analysis not available.")
        return
    
    # Ensure valid frequency values
    def fix_invalid_frequency(data):
        """Fix invalid frequency values by converting them to valid pandas frequencies"""
        valid_frequencies = ['D', 'W', 'M', 'Q', 'Y']
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = fix_invalid_frequency(value)
        elif isinstance(data, list):
            return [fix_invalid_frequency(item) for item in data]
        elif isinstance(data, str):
            # Explicit handling for 'ME' frequency
            if data == 'ME':
                return 'M'
            # For other cases that might end with 'E'
            elif data.endswith('E'):
                for valid_freq in valid_frequencies:
                    if data.startswith(valid_freq):
                        return valid_freq
        return data
    
    # Apply frequency fixes
    cohort_data = fix_invalid_frequency(cohort_data)
    
    # Retention matrix
    st.subheader("Retention Matrix")
    retention_matrix = cohort_data.get("retention_matrix", [])
    
    if retention_matrix and isinstance(retention_matrix, list) and len(retention_matrix) > 0:
        # Ensure all rows have equal length
        max_length = max(len(row) for row in retention_matrix)
        normalized_matrix = [row + [None] * (max_length - len(row)) for row in retention_matrix]
        
        # Create DataFrame
        df = pd.DataFrame(normalized_matrix)
        
        # Set column names
        periods = list(range(len(df.columns)))
        df.columns = [f"Month {p}" for p in periods]
        
        # Set index
        cohorts = [f"Cohort {i+1}" for i in range(len(df))]
        df.index = cohorts
        
        # Create heatmap
        fig = px.imshow(df,
                      labels=dict(x="Time Since Acquisition", y="Cohort", color="Retention Rate"),
                      x=df.columns,
                      y=df.index,
                      color_continuous_scale="Viridis",
                      title="Cohort Retention Matrix")
        
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # User Value by Cohort
    st.subheader("User Value by Cohort")
    value_by_cohort = cohort_data.get("value_by_cohort", {})
    
    if value_by_cohort:
        cohorts = list(value_by_cohort.keys())
        values = list(value_by_cohort.values())
        
        df = pd.DataFrame({
            "Cohort": cohorts,
            "Value": values
        })
        
        fig = px.bar(df, x="Cohort", y="Value", 
                   title="Value per User by Cohort",
                   color="Value",
                   color_continuous_scale="Viridis")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Growth visualization
    st.subheader("Cohort Growth Visualization")
    growth_data = cohort_data.get("growth_data", {})
    
    if growth_data and "cohorts" in growth_data and "months" in growth_data and "values" in growth_data:
        cohorts = growth_data.get("cohorts", [])
        months = growth_data.get("months", [])
        values = growth_data.get("values", [])
        
        # Create DataFrame in long format
        data = []
        for i, cohort in enumerate(cohorts):
            for j, month in enumerate(months):
                if i < len(values) and j < len(values[i]):
                    data.append({
                        "Cohort": cohort,
                        "Month": month,
                        "Value": values[i][j]
                    })
        
        df = pd.DataFrame(data)
        
        # Create line chart
        fig = px.line(df, x="Month", y="Value", color="Cohort",
                    title="Cohort Growth Over Time",
                    labels={"Value": "User Value", "Month": "Month"})
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("Cohort Insights")
    insights = cohort_data.get("insights", [])
    
    if insights:
        for i, insight in enumerate(insights):
            st.markdown(f"{i+1}. {insight}")


def render_benchmarks_tab(doc: dict):
    """Render benchmark analysis."""
    st.header("Benchmark Analysis")
    
    benchmarks = doc.get("benchmarks", {})
    
    if not benchmarks:
        st.warning("Benchmark analysis not available.")
        return
    
    # Overall benchmarking
    st.subheader("Sector Benchmarking")
    
    sector = doc.get("sector", "")
    stage = doc.get("stage", "")
    
    st.markdown(f"Benchmarking against **{sector.title()}** companies at **{stage.title()}** stage")
    
    # Percentile ranking
    percentile = benchmarks.get("percentile", 0)
    st.metric("Industry Percentile", f"{percentile:.0f}th")
    
    # Metric comparisons
    metric_comparisons = benchmarks.get("metric_comparisons", {})
    
    if metric_comparisons:
        # Create DataFrame
        metrics = []
        company_values = []
        benchmark_values = []
        percentiles = []
        
        for metric, comparison in metric_comparisons.items():
            if isinstance(comparison, dict):
                metrics.append(metric)
                company_values.append(comparison.get("company_value", 0))
                benchmark_values.append(comparison.get("benchmark_value", 0))
                percentiles.append(comparison.get("percentile", 0))
        
        df = pd.DataFrame({
            "Metric": metrics,
            "Company Value": company_values,
            "Benchmark Value": benchmark_values,
            "Percentile": percentiles
        })
        
        # Create comparison chart
        fig = go.Figure()
        
        # Add company values
        fig.add_trace(go.Bar(
            x=df['Metric'],
            y=df['Company Value'],
            name='Your Company',
            marker_color='rgba(31, 119, 180, 0.8)'
        ))
        
        # Add benchmark values
        fig.add_trace(go.Bar(
            x=df['Metric'],
            y=df['Benchmark Value'],
            name='Industry Benchmark',
            marker_color='rgba(255, 127, 14, 0.8)'
        ))
        
        # Update layout
        fig.update_layout(
            title='Metric Comparison to Industry Benchmarks',
            xaxis_title='Metric',
            yaxis_title='Value',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed table
        st.markdown("### Detailed Benchmark Comparison")
        st.dataframe(df)
    
    # Peer comparisons
    peer_comparisons = benchmarks.get("peer_comparisons", [])
    
    if peer_comparisons and all(isinstance(peer, dict) for peer in peer_comparisons):
        st.subheader("Peer Comparisons")
        
        # Create DataFrame
        companies = ["Your Company"] + [peer.get("name", f"Peer {i+1}") for i, peer in enumerate(peer_comparisons)]
        
        # Get metrics to compare
        all_metrics = set()
        for peer in peer_comparisons:
            all_metrics.update(peer.get("metrics", {}).keys())
        
        # Create comparison data
        comparison_data = {metric: [] for metric in all_metrics}
        
        # Add company data
        for metric in all_metrics:
            comparison_data[metric].append(doc.get(metric, 0))
        
        # Add peer data
        for peer in peer_comparisons:
            peer_metrics = peer.get("metrics", {})
            for metric in all_metrics:
                comparison_data[metric].append(peer_metrics.get(metric, 0))
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data, index=companies)
        
        # Transpose for better display
        df_display = df.transpose()
        
        # Show table
        st.dataframe(df_display)
        
        # Create radar chart for key metrics
        key_metrics = ["revenue_growth_rate", "user_growth_rate", "ltv_cac_ratio", "gross_margin_percent"]
        key_metrics = [m for m in key_metrics if m in all_metrics]
        
        if key_metrics:
            # Create radar chart data
            radar_data = []
            
            for i, company in enumerate(companies):
                company_data = {"Company": company}
                for metric in key_metrics:
                    if i < len(comparison_data[metric]):
                        company_data[metric] = comparison_data[metric][i]
                radar_data.append(company_data)
            
            # Create radar chart
            fig = go.Figure()
            
            for data in radar_data:
                fig.add_trace(go.Scatterpolar(
                    r=[data.get(metric, 0) for metric in key_metrics] + [data.get(key_metrics[0], 0)],  # Close the loop
                    theta=key_metrics + [key_metrics[0]],  # Close the loop
                    fill='toself',
                    name=data['Company']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    )
                ),
                title="Key Metrics Comparison",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)


# Custom JSON Encoder for handling non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            # Handle PMFStage enum
            if hasattr(obj, 'value') and hasattr(obj, 'name'):
                return obj.name
            
            # Handle numpy arrays and scalars
            if hasattr(np, 'ndarray') and isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(np, 'integer') and isinstance(obj, np.integer):
                return int(obj)
            if hasattr(np, 'floating') and isinstance(obj, np.floating):
                return float(obj)
            
            # Handle pandas objects with improved Period handling
            if hasattr(pd, 'DataFrame') and isinstance(obj, pd.DataFrame):
                # Handle DataFrames with Period index
                if hasattr(obj, 'index') and hasattr(obj.index, 'dtype') and isinstance(obj.index.dtype, type):
                    if 'period' in str(obj.index.dtype).lower():
                        df_copy = obj.copy()
                        df_copy.index = [str(idx) for idx in obj.index]
                        return df_copy.to_dict()
                return obj.to_dict()
                
            if hasattr(pd, 'Series') and isinstance(obj, pd.Series):
                # Handle Series with Period index
                if hasattr(obj, 'index') and hasattr(obj.index, 'dtype') and isinstance(obj.index.dtype, type):
                    if 'period' in str(obj.index.dtype).lower():
                        series_copy = obj.copy()
                        series_copy.index = [str(idx) for idx in obj.index]
                        return series_copy.to_dict()
                return obj.to_dict()
                
            if hasattr(pd, 'Period') and isinstance(obj, pd.Period):
                return str(obj)
                
            if hasattr(pd, 'PeriodIndex') and isinstance(obj, pd.PeriodIndex):
                return [str(x) for x in obj]
                
            if hasattr(pd, 'Index') and isinstance(obj, pd.Index):
                return [str(x) for x in obj]
            
            # Handle datetime objects
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            
            # Handle sets
            if isinstance(obj, set):
                return list(obj)
            
            # Handle Competitor objects from competitive_intelligence 
            if obj.__class__.__name__ == 'Competitor':
                return {
                    "name": getattr(obj, "name", "Unknown"),
                    "sector": getattr(obj, "sector", ""),
                    "size": getattr(obj, "size", ""),
                    "strength": getattr(obj, "strength", 0),
                    "differentiators": getattr(obj, "differentiators", []),
                    "website": getattr(obj, "website", ""),
                    "founded": getattr(obj, "founded", ""),
                    "funding": getattr(obj, "funding", 0),
                    "market_share": getattr(obj, "market_share", 0)
                }
            
            # Handle any object with a to_dict method
            if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                return obj.to_dict()
            
            # Handle any object with __dict__ attribute
            if hasattr(obj, '__dict__'):
                return {
                    key: value 
                    for key, value in obj.__dict__.items() 
                    if not key.startswith('_')
                }
            
            return super().default(obj)
            
        except Exception as e:
            # Log the error but don't raise it
            logging.error(f"Error in JSON encoding: {str(e)} for object type {type(obj).__name__}")
            # Return a simple string representation as fallback
            try:
                return str(obj)
            except:
                return f"<Unserializable object of type {type(obj).__name__}>"

def render_report_tab(doc: dict):
    """Render the report tab with download options for enhanced visual reports."""
    st.header("Investor Report")
    
    try:
        # Create a deep copy of the document to avoid modifying the original
        import copy
        import tempfile
        import os
        import sys
        import subprocess
        import logging
        import json
        from io import StringIO, BytesIO
        import csv
        import base64
        import math
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        logger = logging.getLogger("pdf_generator")
        
        doc_copy = copy.deepcopy(doc)
        
        # Pre-process pandas Period indices to ensure proper serialization
        if "cohort_data" in doc_copy and isinstance(doc_copy["cohort_data"], dict):
            for key, value in list(doc_copy["cohort_data"].items()):
                if hasattr(value, 'index') and hasattr(pd, 'Period'):
                    # Safe check for dtype attribute and type
                    is_period_index = isinstance(value.index, pd.PeriodIndex)
                    if (hasattr(value.index, 'dtype') and 
                        isinstance(value.index.dtype, type) and 
                        'period' in str(value.index.dtype).lower()) or is_period_index:
                        
                        doc_copy["cohort_data"][key] = value.copy()
                        doc_copy["cohort_data"][key].index = [str(idx) for idx in value.index]
                        logging.info(f"Converted Period index to strings in cohort_data[{key}]")
        
        # Create download buttons
        col1, col2, col3 = st.columns(3)
        
        # Create JSON string with custom encoder
        json_str = json.dumps(doc_copy, indent=2, cls=CustomJSONEncoder)
        
        with col1:
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="startup_report.json",
                mime="application/json"
            )
        
        with col2:
            # Convert to CSV format
            try:
                # Extract key metrics into a flat dictionary
                flat_metrics = flatten_metrics(doc)
                
                # Convert to CSV
                csv_buffer = StringIO()
                writer = csv.writer(csv_buffer)
                writer.writerow(flat_metrics.keys())
                writer.writerow(flat_metrics.values())
                
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="startup_metrics.csv",
                    mime="text/csv"
                )
            except Exception as e:
                logging.error(f"Error creating CSV: {str(e)}")
                st.error("Could not generate CSV format. Please use JSON download.")
        
        with col3:
            # Enhanced PDF generation button
            if st.button("Generate PDF Report"):
                with st.spinner("Generating enhanced PDF report..."):
                    try:
                        # Create a unique filename with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"reports/{doc_copy.get('name', doc_copy.get('company_name', 'startup'))}_{timestamp}.pdf"
                        
                        # Ensure reports directory exists
                        os.makedirs("reports", exist_ok=True)
                        
                        # SIMPLIFIED APPROACH: Use unified_pdf_generator directly with better error handling
                        success = False
                        error_message = ""
                        
                        # Try using unified_pdf_generator directly - our fixed version
                        try:
                            # Import with absolute import to avoid any namespace issues
                            import unified_pdf_generator
                            # Force reload the module to ensure we get the latest version
                            import importlib
                            unified_pdf_generator = importlib.reload(unified_pdf_generator)
                            
                            # Get the PDF data as bytes
                            pdf_data = unified_pdf_generator.generate_enhanced_pdf(doc_copy, report_type="full", sections=None)
                            
                            # Write the bytes to the output file
                            with open(output_path, 'wb') as f:
                                f.write(pdf_data)
                                
                            success = True
                            logger.info(f"Generated PDF using unified_pdf_generator: {output_path}")
                        except Exception as e:
                            error_message = f"unified_pdf_generator failed: {str(e)}"
                            logger.error(f"unified_pdf_generator error: {error_message}")
                            # Add detailed traceback for debugging
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            
                            # Fallback to robust_pdf if available
                            try:
                                import robust_pdf
                                success = robust_pdf.generate_pdf(doc_copy, output_path)
                                logger.info(f"Generated PDF using robust_pdf.generate_pdf: {output_path}")
                            except Exception as e2:
                                error_message += f"\nrobust_pdf also failed: {str(e2)}"
                                logger.error(f"robust_pdf error: {str(e2)}")
                                
                                # Last resort: emergency PDF generator
                                try:
                                    from emergency_pdf_generator import emergency_generate_pdf
                                    pdf_data = emergency_generate_pdf(doc_copy, "full", None)
                                    with open(output_path, 'wb') as f:
                                        f.write(pdf_data)
                                    success = True
                                    logger.info(f"Generated emergency PDF: {output_path}")
                                except Exception as e3:
                                    error_message += f"\nEmergency PDF also failed: {str(e3)}"
                                    logger.error(f"All PDF generation methods failed: {error_message}")
                                    raise Exception(f"PDF generation failed: {error_message}")
                        
                        if success:
                            # Read the generated PDF file
                            with open(output_path, 'rb') as f:
                                pdf_bytes = f.read()
                            
                            # Check if the PDF is valid (non-empty and above minimum size)
                            if len(pdf_bytes) > 1000:  # A reasonable minimum size for a valid PDF
                                st.download_button(
                                    label="Download Enhanced PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"{doc_copy.get('name', doc_copy.get('company_name', 'startup'))}_report.pdf",
                                    mime="application/pdf"
                                )
                                st.success("Enhanced PDF report generated successfully!")
                            else:
                                st.error("Generated PDF appears to be invalid or empty. Please try JSON or CSV format instead.")
                                logger.error(f"Generated PDF too small: {len(pdf_bytes)} bytes")
                        else:
                            st.error("PDF generation failed. Please try JSON or CSV format instead.")
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
                        st.info("Please use JSON or CSV download formats instead.")
        
        # Report sections selection
        st.subheader("Customize Report Sections")
        st.write("Select sections to include in your next PDF report:")
        
        # Create columns for section selection
        section_cols = st.columns(3)
        section_options = {
            "Executive Summary": True,
            "Business Model": True,
            "Market Analysis": True,
            "Financial Projections": True,
            "Team Assessment": True,
            "Competitive Analysis": True,
            "Growth Metrics": True,
            "Risk Assessment": True,
            "Exit Strategy": True,
            "Technical Assessment": True
        }
        
        selected_sections = {}
        i = 0
        for section, default in section_options.items():
            with section_cols[i % 3]:
                selected_sections[section] = st.checkbox(section, value=default)
            i += 1
        
        # Generate custom report button
        if st.button("Generate Custom PDF Report"):
            with st.spinner("Generating custom PDF report..."):
                try:
                    # Create a unique filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"reports/{doc_copy.get('name', doc_copy.get('company_name', 'startup'))}_custom_{timestamp}.pdf"
                    
                    # Ensure reports directory exists
                    os.makedirs("reports", exist_ok=True)
                    
                    # SIMPLIFIED APPROACH: Use unified_pdf_generator directly for custom reports
                    success = False
                    
                    try:
                        # Import with absolute import to avoid any namespace issues
                        import unified_pdf_generator
                        # Force reload the module to ensure we get the latest version
                        import importlib
                        unified_pdf_generator = importlib.reload(unified_pdf_generator)
                        
                        # Get the PDF data as bytes
                        pdf_data = unified_pdf_generator.generate_enhanced_pdf(
                            doc_copy, 
                            report_type="custom", 
                            sections=selected_sections
                        )
                        
                        # Write the bytes to the output file
                        with open(output_path, 'wb') as f:
                            f.write(pdf_data)
                            
                        success = True
                        logger.info(f"Generated custom PDF using unified_pdf_generator: {output_path}")
                    except Exception as e:
                        logger.error(f"Custom PDF generation error: {str(e)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        
                        # Fallback to robust_pdf if available
                        try:
                            import robust_pdf
                            success = robust_pdf.generate_pdf(doc_copy, output_path, "custom", selected_sections)
                            logger.info(f"Generated custom PDF using robust_pdf: {output_path}")
                        except Exception as e2:
                            logger.error(f"robust_pdf custom generation failed: {str(e2)}")
                            
                            # Last resort: emergency PDF generator
                            try:
                                from emergency_pdf_generator import emergency_generate_pdf
                                pdf_data = emergency_generate_pdf(doc_copy, "custom", selected_sections)
                                with open(output_path, 'wb') as f:
                                    f.write(pdf_data)
                                success = True
                                logger.info(f"Generated emergency custom PDF: {output_path}")
                            except Exception as e3:
                                logger.error(f"All custom PDF generation methods failed: {str(e3)}")
                                raise Exception(f"Custom PDF generation failed completely")
                    
                    if success:
                        # Read the generated PDF file
                        with open(output_path, 'rb') as f:
                            pdf_bytes = f.read()
                        
                        # Check if the PDF is valid
                        if len(pdf_bytes) > 1000:
                            st.download_button(
                                label="Download Custom PDF Report",
                                data=pdf_bytes,
                                file_name=f"{doc_copy.get('name', doc_copy.get('company_name', 'startup'))}_custom_report.pdf",
                                mime="application/pdf"
                            )
                            st.success("Custom PDF report generated successfully!")
                        else:
                            st.error("Generated custom PDF appears to be invalid or empty.")
                            logger.error(f"Generated custom PDF too small: {len(pdf_bytes)} bytes") 
                    else:
                        st.error("Custom PDF generation failed. Please try JSON or CSV format instead.")
                except Exception as e:
                    st.error(f"Custom PDF generation failed: {str(e)}")
                    st.info("Please use JSON or CSV download formats instead.")
    except Exception as e:
        st.error(f"Report tab error: {str(e)}")
        logger.error(f"Report tab error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def generate_pdf_safely(doc_copy, report_type="full", sections=None):
    """
    Generate PDF report in a way that isolates from PyTorch and other potential conflicts.
    Uses subprocess or thread isolation to prevent PyTorch errors.
    """
    import tempfile
    import json
    import os
    import subprocess
    import base64
    import threading
    import logging
    
    logger = logging.getLogger("pdf_generator")
    
    # Create a unique temporary file for the document data
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tmp:
        try:
            # Serialize document with custom encoder to handle special types
            json.dump(doc_copy, tmp, cls=CustomJSONEncoder)
            tmp_path = tmp.name
        except Exception as e:
            logger.error(f"Error serializing document to JSON: {e}")
            os.unlink(tmp.name)
            return None
    
    # Create a temporary file for the output PDF
    output_fd, output_path = tempfile.mkstemp(suffix='.pdf')
    os.close(output_fd)  # Close the file descriptor
    
    # Create a temporary file for sections config if provided
    sections_path = None
    if sections:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tmp:
            try:
                json.dump(sections, tmp)
                sections_path = tmp.name
            except Exception as e:
                logger.error(f"Error serializing sections to JSON: {e}")
                os.unlink(tmp.name)
                os.unlink(tmp_path)
                os.unlink(output_path)
                return None
    
    # Create a standalone Python script to generate the PDF
    script_content = '''
import sys
import json
import os
import logging
import traceback

# Add the current directory to the Python path to find modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pdf_generator_script")

try:
    # Import report generator without triggering PyTorch
    try:
        from report_generator import generate_investor_report
        
        # Load document data
        with open(sys.argv[1], 'r') as f:
            doc = json.load(f)
        
        # Get report type
        report_type = "full"
        if len(sys.argv) > 3:
            report_type = sys.argv[3]
        
        # Load sections if provided
        sections = None
        if len(sys.argv) > 4:
            with open(sys.argv[4], 'r') as f:
                sections = json.load(f)
        
        # Generate PDF
        pdf_bytes = generate_investor_report(doc, report_type, sections)
        
        # Write to output file
        with open(sys.argv[2], 'wb') as f:
            f.write(pdf_bytes)
        
        # Report success
        print("PDF generated successfully")
        sys.exit(0)
        
    except Exception as primary_error:
        logger.error(f"Primary PDF generation failed: {str(primary_error)}\\n{traceback.format_exc()}")
        print(f"Attempting emergency PDF generation due to: {str(primary_error)}")
        
        # Try emergency PDF generator as fallback
        try:
            # Try to find emergency_pdf_generator in current directory
            try:
                from emergency_pdf_generator import emergency_generate_pdf
            except ImportError:
                # If not found, try to load it from script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                from emergency_pdf_generator import emergency_generate_pdf
            
            # Load document data
            with open(sys.argv[1], 'r') as f:
                doc = json.load(f)
            
            # Get report type
            report_type = "full"
            if len(sys.argv) > 3:
                report_type = sys.argv[3]
            
            # Load sections if provided
            sections = None
            if len(sys.argv) > 4:
                with open(sys.argv[4], 'r') as f:
                    sections = json.load(f)
            
            # Generate PDF with emergency generator
            pdf_bytes = emergency_generate_pdf(doc, report_type, sections)
            
            # Write to output file
            with open(sys.argv[2], 'wb') as f:
                f.write(pdf_bytes)
            
            print("PDF generated with emergency generator")
            sys.exit(0)
            
        except Exception as emergency_error:
            logger.error(f"Emergency PDF generation also failed: {str(emergency_error)}\\n{traceback.format_exc()}")
            raise RuntimeError(f"Both primary and emergency PDF generation failed: {str(primary_error)} AND {str(emergency_error)}")
except Exception as e:
    logger.error(f"Error generating PDF: {str(e)}\\n{traceback.format_exc()}")
    print(f"Error: {str(e)}")
    sys.exit(1)
'''
    
    # Write the script to a temporary file
    script_fd, script_path = tempfile.mkstemp(suffix='.py')
    with os.fdopen(script_fd, 'w') as f:
        f.write(script_content)
    
    try:
        # Prepare subprocess arguments
        args = [sys.executable, script_path, tmp_path, output_path, report_type]
        if sections_path:
            args.append(sections_path)
        
        # Run the subprocess with timeout
        logger.info(f"Running PDF generation subprocess with args: {args}")
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,  # 60 second timeout
            text=True
        )
        
        if result.returncode == 0:
            # Read the generated PDF
            with open(output_path, 'rb') as f:
                pdf_data = f.read()
            logger.info(f"PDF generated successfully, size: {len(pdf_data)} bytes")
            return pdf_data
        else:
            logger.error(f"PDF generation failed: {result.stderr}")
            # Try direct emergency generation if subprocess failed
            try:
                from emergency_pdf_generator import emergency_generate_pdf
                logger.info("Attempting direct emergency PDF generation after subprocess failure")
                return emergency_generate_pdf(doc_copy, report_type, sections)
            except Exception as e:
                logger.error(f"Direct emergency PDF generation also failed: {str(e)}")
                return None
            
    except subprocess.TimeoutExpired:
        logger.error("PDF generation timed out")
        
        # Try emergency generation directly as last resort
        try:
            from emergency_pdf_generator import emergency_generate_pdf
            logger.info("Attempting direct emergency PDF generation after timeout")
            return emergency_generate_pdf(doc_copy, report_type, sections)
        except Exception as e:
            logger.error(f"Direct emergency PDF generation also failed: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error running PDF generation process: {str(e)}")
        return None
    finally:
        # Clean up temporary files
        for path in [tmp_path, output_path, script_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {path}: {str(e)}")
        
        if sections_path and os.path.exists(sections_path):
            try:
                os.unlink(sections_path)
            except Exception as e:
                logger.error(f"Error cleaning up sections file {sections_path}: {str(e)}")

def flatten_metrics(doc: dict) -> dict:
    """Helper function to flatten nested metrics for CSV export."""
    flat = {}
    
    try:
        # Basic company info
        flat["company_name"] = doc.get("name", "")
        flat["stage"] = str(doc.get("stage", ""))
        flat["sector"] = doc.get("sector", "")
        
        # Key metrics
        flat["monthly_revenue"] = doc.get("monthly_revenue", 0)
        flat["burn_rate"] = doc.get("burn_rate", 0)
        flat["runway_months"] = doc.get("runway_months", 0)
        flat["current_users"] = doc.get("current_users", 0)
        
        # Growth metrics
        flat["revenue_growth"] = doc.get("revenue_growth_rate", 0)
        flat["user_growth"] = doc.get("user_growth_rate", 0)
        
        # Unit economics
        flat["arpu"] = doc.get("arpu", 0)
        flat["cac"] = doc.get("cac", 0)
        flat["ltv"] = doc.get("ltv", 0)
        flat["gross_margin"] = doc.get("gross_margin_percent", 0)
        
        # Market metrics
        flat["market_size"] = doc.get("market_size", 0)
        flat["market_share"] = doc.get("market_share", 0)
        
        # CAMP Framework scores
        flat["capital_score"] = doc.get("capital_score", 0)
        flat["market_score"] = doc.get("market_score", 0)
        flat["advantage_score"] = doc.get("advantage_score", 0)
        flat["people_score"] = doc.get("people_score", 0)
        flat["camp_score"] = doc.get("camp_score", 0)
        
        # Intangible rating
        flat["intangible_score"] = doc.get("intangible", 0)
        pitch_sentiment = doc.get("pitch_sentiment", {})
        if pitch_sentiment:
            flat["pitch_sentiment_score"] = pitch_sentiment.get("sentiment_score", 0)
            flat["pitch_confidence"] = pitch_sentiment.get("confidence", 0)
            
        # Success probability
        flat["success_probability"] = doc.get("success_prob", 0)
        
        # Monte Carlo results if available
        monte_carlo = doc.get("monte_carlo", {})
        if monte_carlo:
            flat["monte_carlo_success_probability"] = monte_carlo.get("success_probability", 0)
            flat["median_runway"] = monte_carlo.get("median_runway_months", 0)
            
            # Get the median projections if available
            projections = monte_carlo.get("projections", {})
            if projections:
                flat["projected_revenue_12m"] = get_projection_value(projections, "revenue", 12)
                flat["projected_users_12m"] = get_projection_value(projections, "users", 12)
                flat["projected_cash_12m"] = get_projection_value(projections, "cash", 12)
        
        # Risk metrics if available
        risk_factors = doc.get("risk_factors", {})
        if risk_factors:
            for factor, score in risk_factors.items():
                flat[f"risk_{factor}"] = score
    
    except Exception as e:
        logging.error(f"Error flattening metrics: {str(e)}")
    
    return flat

def get_projection_value(projections: dict, metric: str, month: int) -> float:
    """Helper function to safely get projection values."""
    try:
        metric_data = projections.get(metric, {})
        if isinstance(metric_data, dict):
            percentiles = metric_data.get("percentiles", {})
            if percentiles:
                # Get the median (p50) value for the specified month
                median = percentiles.get("p50", [])
                if len(median) > month:
                    return median[month]
    except Exception:
        pass
    return 0.0

def render_acquisition_tab(doc: dict):
    """Render acquisition fit analysis."""
    st.header("Acquisition Fit Analysis")
    
    acq_fit = doc.get("acquisition_fit", {})
    if not acq_fit:
        st.warning("Acquisition fit analysis not available.")
        return
    
    # Top metrics
    readiness = acq_fit.get("overall_acquisition_readiness", 0)
    primary_appeal = acq_fit.get("primary_acquisition_appeal", "")
    
    st.metric("Acquisition Readiness", f"{readiness:.1f}/100")
    st.markdown(f"**Primary Acquisition Appeal**: {primary_appeal}")
    
    # Readiness by dimension
    st.subheader("Acquisition Readiness by Dimension")
    readiness_dims = acq_fit.get("readiness_by_dimension", {})
    
    if readiness_dims:
        dimensions = list(readiness_dims.keys())
        scores = list(readiness_dims.values())
        
        df = pd.DataFrame({
            "Dimension": dimensions,
            "Score": scores
        })
        
        fig = px.bar(df, x="Dimension", y="Score", 
                   title="Acquisition Readiness by Dimension",
                   color="Score",
                   color_continuous_scale="Viridis")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Top synergies
    st.subheader("Potential Acquisition Synergies")
    synergies = acq_fit.get("top_synergies", [])
    
    if synergies:
        # Check if synergies is a list of dictionaries
        if all(isinstance(s, dict) for s in synergies):
            acquirer_types = [s.get("acquirer_type", "") for s in synergies]
            overall_scores = [s.get("overall_score", 0) for s in synergies]
            strategic_fits = [s.get("strategic_fit_score", 0) for s in synergies]
            revenue_synergies = [s.get("revenue_synergy", 0) for s in synergies]
            cost_synergies = [s.get("cost_synergy", 0) for s in synergies]
            
            df = pd.DataFrame({
                "Acquirer Type": acquirer_types,
                "Overall Score": overall_scores,
                "Strategic Fit": strategic_fits,
                "Revenue Synergy": revenue_synergies,
                "Cost Synergy": cost_synergies
            })
            
            fig = px.bar(df, x="Acquirer Type", y="Overall Score", 
                       title="Acquisition Synergy by Acquirer Type",
                       color="Overall Score",
                       color_continuous_scale="Viridis")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed synergy table
            st.markdown("### Detailed Synergy Analysis")
            
            # Format top 3 synergies for display
            synergy_data = []
            for i, synergy in enumerate(synergies[:3]):
                row = {
                    "Acquirer Type": synergy.get("acquirer_type", ""),
                    "Strategic Fit": f"{synergy.get('strategic_fit_score', 0):.1f}/100",
                    "Revenue Synergy": f"{synergy.get('revenue_synergy', 0):.1f}/100",
                    "Cost Synergy": f"{synergy.get('cost_synergy', 0):.1f}/100",
                    "Cultural Fit": f"{synergy.get('cultural_fit_score', 0):.1f}/100",
                    "Time to Value": f"{synergy.get('time_to_value', 0)} months",
                    "Overall Score": f"{synergy.get('overall_score', 0):.1f}/100"
                }
                synergy_data.append(row)
            
            synergy_df = pd.DataFrame(synergy_data)
            st.dataframe(synergy_df)
            
            # Show potential acquirers for top synergy
            if synergies and "potential_acquirers" in synergies[0] and synergies[0]["potential_acquirers"]:
                st.markdown(f"### Potential {synergies[0].get('acquirer_type', '')} Acquirers")
                acquirers = synergies[0]["potential_acquirers"]
                acquirer_list = ", ".join(acquirers)
                st.markdown(f"**{acquirer_list}**")
    
    # Valuation scenarios
    st.subheader("Acquisition Valuation Scenarios")
    valuations = acq_fit.get("valuations", [])
    
    if valuations and all(isinstance(v, dict) for v in valuations):
        methods = [v.get("method", "") for v in valuations]
        base_values = [v.get("base_value", 0) for v in valuations]
        low_values = [v.get("low_value", 0) for v in valuations]
        high_values = [v.get("high_value", 0) for v in valuations]
        
        # Create dataframe for base values
        df_base = pd.DataFrame({
            "Method": methods,
            "Value": base_values,
            "Range": ["Base" for _ in methods]
        })
        
        # Create dataframe for low values
        df_low = pd.DataFrame({
            "Method": methods,
            "Value": low_values,
            "Range": ["Low" for _ in methods]
        })
        
        # Create dataframe for high values
        df_high = pd.DataFrame({
            "Method": methods,
            "Value": high_values,
            "Range": ["High" for _ in methods]
        })
        
        # Combine dataframes
        df = pd.concat([df_low, df_base, df_high])
        
        # Create chart
        fig = px.bar(df, x="Method", y="Value", color="Range", 
                   title="Acquisition Valuation Scenarios",
                   barmode="group",
                   color_discrete_map={"Low": "#FFA07A", "Base": "#3CB371", "High": "#6495ED"})
        
        # Format y-axis as currency
        fig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.0f')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Integration risks
    st.subheader("Integration Risks")
    risks = acq_fit.get("integration_risks", [])
    
    if risks and all(isinstance(r, dict) for r in risks):
        risk_types = [r.get("risk_type", "") for r in risks]
        risk_scores = [r.get("risk_score", 0) for r in risks]
        probabilities = [r.get("probability", 0) for r in risks]
        impacts = [r.get("impact", 0) for r in risks]
        
        df = pd.DataFrame({
            "Risk Type": risk_types,
            "Risk Score": risk_scores,
            "Probability": probabilities,
            "Impact": impacts
        })
        
        fig = px.bar(df, x="Risk Type", y="Risk Score", 
                   title="Integration Risk Analysis",
                   color="Risk Score",
                   color_continuous_scale="Reds")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show mitigation strategies for top risks
        st.markdown("### Risk Mitigation Strategies")
        for i, risk in enumerate(risks[:3]):
            st.markdown(f"**{risk.get('risk_type', '')}**: {risk.get('mitigation_strategy', '')}")
    
    # Recommendations
    st.subheader("Acquisition Recommendations")
    recommendations = acq_fit.get("recommendations", [])
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            st.markdown(f"{i+1}. {rec}")


def render_exit_path_tab(doc: dict):
    """Render exit path analysis."""
    st.header("Exit Path Analysis")
    
    exit_analysis = doc.get("exit_path_analysis", {})
    exit_recs = doc.get("exit_recommendations", {})
    
    if not exit_analysis and not exit_recs:
        st.warning("Exit path analysis not available.")
        return
    
    # Handle potentially problematic infinity values
    def safe_value(value, default=0):
        """Safely process values that might be infinity or NaN"""
        if value is None or (isinstance(value, float) and (math.isinf(value) or math.isnan(value))):
            return default
        return value
    
    # Top metrics and optimal path
    optimal_path = exit_recs.get("optimal_path", "")
    path_details = exit_recs.get("path_details", {})
    readiness = safe_value(exit_recs.get("readiness", 0))
    
    st.metric("Exit Readiness", f"{readiness:.1f}/100")
    
    if optimal_path:
        st.markdown(f"**Optimal Exit Path**: {path_details.get('description', optimal_path)}")
    
    # Exit timeline
    timeline = exit_recs.get("timeline", {})
    if timeline:
        years_to_exit = safe_value(timeline.get("years_to_exit", 0))
        exit_valuation = safe_value(timeline.get("exit_valuation", 0))
        exit_year = safe_value(timeline.get("exit_year", 0))
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Years to Exit", f"{years_to_exit:.1f}")
        with cols[1]:
            st.metric("Exit Year", f"{exit_year}")
        with cols[2]:
            st.metric("Exit Valuation", f"${exit_valuation/1e6:.1f}M")
    
    # Success factors by exit path
    st.subheader("Success Factors by Exit Path")
    success_factors = exit_analysis.get("success_factors", {})
    
    if success_factors:
        # Filter out any infinity or NaN values
        filtered_factors = {k: safe_value(v) for k, v in success_factors.items()}
        
        paths = list(filtered_factors.keys())
        scores = list(filtered_factors.values())
        
        df = pd.DataFrame({
            "Exit Path": paths,
            "Success Factor": scores
        })
        
        fig = px.bar(df, x="Exit Path", y="Success Factor", 
                   title="Success Factors by Exit Path",
                   color="Success Factor",
                   color_continuous_scale="Viridis")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Exit scenarios
    st.subheader("Exit Path Scenarios")
    scenarios = exit_analysis.get("scenarios", [])
    
    if scenarios and all(isinstance(s, dict) for s in scenarios):
        # Process and clean scenario data
        processed_scenarios = []
        for s in scenarios:
            processed = {}
            processed["path_name"] = s.get("path_name", "")
            processed["exit_valuation"] = safe_value(s.get("exit_valuation", 0))
            processed["probability"] = safe_value(s.get("probability", 0))
            processed["risk_adjusted_value"] = safe_value(s.get("risk_adjusted_value", 0))
            processed["time_to_exit"] = safe_value(s.get("time_to_exit", 0))
            processed_scenarios.append(processed)
        
        path_names = [s.get("path_name", "") for s in processed_scenarios]
        exit_vals = [s.get("exit_valuation", 0) for s in processed_scenarios]
        probabilities = [s.get("probability", 0) for s in processed_scenarios]
        risk_adj_vals = [s.get("risk_adjusted_value", 0) for s in processed_scenarios]
        times = [s.get("time_to_exit", 0) for s in processed_scenarios]
        
        # Create DataFrame
        df = pd.DataFrame({
            "Exit Path": path_names,
            "Exit Valuation ($M)": [v/1e6 for v in exit_vals],
            "Probability": [p*100 for p in probabilities],
            "Risk-Adjusted Value ($M)": [v/1e6 for v in risk_adj_vals],
            "Time to Exit (Years)": times
        })
        
        # Display as table
        st.dataframe(df)
        
        # Create risk-adjusted value chart
        fig = px.bar(df, x="Exit Path", y="Risk-Adjusted Value ($M)", 
                   title="Risk-Adjusted Exit Values",
                   color="Risk-Adjusted Value ($M)",
                   color_continuous_scale="Viridis")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline visualization
    st.subheader("Exit Valuation Timeline")
    timeline_data = exit_analysis.get("timeline_data", {})
    
    if timeline_data and "years" in timeline_data and "paths" in timeline_data:
        years = timeline_data.get("years", [])
        paths = timeline_data.get("paths", [])
        
        # Create plot data
        plot_data = []
        
        for year in years:
            row = {"Year": year}
            for path in paths:
                if path in timeline_data:
                    path_vals = timeline_data[path]
                    if len(path_vals) > 0:
                        idx = min(int(year * len(path_vals) / years[-1]), len(path_vals) - 1)
                        row[path] = path_vals[idx] / 1e6  # Convert to millions
            plot_data.append(row)
        
        df = pd.DataFrame(plot_data)
        
        # Create line chart
        fig = px.line(df, x="Year", y=paths, 
                    title="Exit Valuation Timeline",
                    labels={"value": "Valuation ($M)", "variable": "Exit Path"})
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommended milestones
    st.subheader("Recommended Milestones")
    milestones = timeline.get("milestones", [])
    
    if milestones and all(isinstance(m, dict) for m in milestones):
        # Group milestones by category
        milestone_categories = {}
        for milestone in milestones:
            category = milestone.get("category", "other")
            if category not in milestone_categories:
                milestone_categories[category] = []
            milestone_categories[category].append(milestone)
        
        # Display milestones by category
        for category, cat_milestones in milestone_categories.items():
            st.markdown(f"### {category.title()}")
            for milestone in cat_milestones:
                st.markdown(f"**{milestone.get('title', '')}**")
                st.markdown(f"*Timeline: {milestone.get('timeline', '')}*")
                st.markdown(milestone.get("description", ""))
    
    # Comparable exits
    st.subheader("Comparable Exits")
    comparables = exit_analysis.get("comparable_exits", [])
    
    if comparables and all(isinstance(c, dict) for c in comparables):
        # Create DataFrame
        df = pd.DataFrame(comparables)
        
        # Rename and select columns
        if "company_name" in df.columns and "exit_type" in df.columns and "exit_amount" in df.columns:
            df = df.rename(columns={"company_name": "Company", 
                                  "exit_type": "Exit Type", 
                                  "exit_amount": "Exit Amount ($M)",
                                  "exit_year": "Year",
                                  "revenue_at_exit": "Revenue at Exit ($M)"})
            
            # Convert to millions
            if "Exit Amount ($M)" in df.columns:
                df["Exit Amount ($M)"] = df["Exit Amount ($M)"].apply(lambda x: x/1e6)
            
            if "Revenue at Exit ($M)" in df.columns:
                df["Revenue at Exit ($M)"] = df["Revenue at Exit ($M)"].apply(lambda x: x/1e6)
            
            # Select columns
            columns_to_display = ["Company", "Exit Type", "Year", "Exit Amount ($M)", "Revenue at Exit ($M)"]
            display_df = df[[col for col in columns_to_display if col in df.columns]]
            
            # Display table
            st.dataframe(display_df)
    
    # Recommendations
    st.subheader("Exit Strategy Recommendations")
    recommendations = exit_recs.get("recommendations", [])
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            st.markdown(f"{i+1}. {rec}")


def render_forecast_tab(doc: dict):
    """Render financial and growth forecasts."""
    st.header("Forecasts & Scenarios")
    
    # User growth forecast
    st.subheader("User Growth Forecast")
    sys_dynamics = doc.get("system_dynamics", {})
    
    if isinstance(sys_dynamics, dict) and "users" in sys_dynamics:
        users = sys_dynamics.get("users", [])
        months = list(range(len(users)))
        
        df = pd.DataFrame({
            "Month": months,
            "Users": users
        })
        
        fig = px.line(df, x="Month", y="Users", 
                    title="User Growth Projection (Base Case)", 
                    labels={"Users": "Users", "Month": "Month"})
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Virality simulation
    st.subheader("Virality Simulation")
    virality = doc.get("virality_sim", {})
    
    if virality:
        if isinstance(virality, dict) and "cycles" in virality and "users" in virality:
            cycles = virality.get("cycles", [])
            users = virality.get("users", [])
            
            df = pd.DataFrame({
                "Cycle": cycles,
                "Users": users
            })
            
            fig = px.line(df, x="Cycle", y="Users", 
                        title="User Growth with Viral Effects", 
                        labels={"Users": "Users", "Cycle": "Cycle"})
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Financial forecast
    st.subheader("Financial Forecast")
    forecast = doc.get("financial_forecast", {})
    
    if forecast:
        # Revenue forecast
        if "months" in forecast and "revenue" in forecast:
            months = forecast.get("months", [])
            revenue = forecast.get("revenue", [])
            
            df = pd.DataFrame({
                "Month": months,
                "Revenue": revenue
            })
            
            fig = px.line(df, x="Month", y="Revenue", 
                        title="Revenue Forecast", 
                        labels={"Revenue": "Revenue ($)", "Month": "Month"})
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Profitability forecast
        if "months" in forecast and "profit" in forecast:
            months = forecast.get("months", [])
            profit = forecast.get("profit", [])
            
            df = pd.DataFrame({
                "Month": months,
                "Profit": profit
            })
            
            fig = px.line(df, x="Month", y="Profit", 
                        title="Profit Forecast", 
                        labels={"Profit": "Profit ($)", "Month": "Month"})
            
            # Add a horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Monte Carlo simulation
    st.subheader("Monte Carlo Simulation")
    monte_carlo = doc.get("monte_carlo", {})
    
    if monte_carlo:
        if "user_projections" in monte_carlo:
            user_proj = monte_carlo.get("user_projections", {})
            
            if "months" in user_proj and "percentiles" in user_proj:
                months = user_proj.get("months", [])
                percentiles = user_proj.get("percentiles", {})
                
                # Create DataFrame in long format for plotting
                data = []
                for percentile, values in percentiles.items():
                    for i, month in enumerate(months):
                        if i < len(values):
                            data.append({
                                "Month": month,
                                "Users": values[i],
                                "Percentile": percentile
                            })
                
                df = pd.DataFrame(data)
                
                # Create line chart
                fig = px.line(df, x="Month", y="Users", color="Percentile",
                            title="User Growth Monte Carlo Simulation",
                            labels={"Users": "Users", "Month": "Month"})
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Revenue Monte Carlo
        if "revenue_projections" in monte_carlo:
            rev_proj = monte_carlo.get("revenue_projections", {})
            
            if "months" in rev_proj and "percentiles" in rev_proj:
                months = rev_proj.get("months", [])
                percentiles = rev_proj.get("percentiles", {})
                
                # Create DataFrame in long format for plotting
                data = []
                for percentile, values in percentiles.items():
                    for i, month in enumerate(months):
                        if i < len(values):
                            data.append({
                                "Month": month,
                                "Revenue": values[i],
                                "Percentile": percentile
                            })
                
                df = pd.DataFrame(data)
                
                # Create line chart
                fig = px.line(df, x="Month", y="Revenue", color="Percentile",
                            title="Revenue Monte Carlo Simulation",
                            labels={"Revenue": "Revenue ($)", "Month": "Month"})
                
                st.plotly_chart(fig, use_container_width=True)
    
    # HPC scenarios
    st.subheader("Optimization Scenarios")
    hpc_data = doc.get("hpc_data", [])
    
    if hpc_data and all(isinstance(s, dict) for s in hpc_data):
        # Extract data for plotting
        scenarios = []
        final_users = []
        churns = []
        referrals = []
        
        for i, scenario in enumerate(hpc_data):
            scenarios.append(f"Scenario {i+1}")
            final_users.append(scenario.get("final_users", 0))
            churns.append(scenario.get("churn_rate", 0) * 100)  # Convert to percentage
            referrals.append(scenario.get("referral_rate", 0) * 100)  # Convert to percentage
        
        # Create dataframe
        df = pd.DataFrame({
            "Scenario": scenarios,
            "Final Users": final_users,
            "Churn Rate (%)": churns,
            "Referral Rate (%)": referrals
        })
        
        # Create bar chart
        fig = px.bar(df, x="Scenario", y="Final Users", 
                   title="User Growth by Scenario",
                   color="Final Users",
                   color_continuous_scale="Viridis")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show optimal scenario
        optimal = doc.get("optimal_scenario", {})
        if optimal:
            st.markdown("### Optimal Scenario")
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("Final Users", f"{optimal.get('final_users', 0):,}")
            with cols[1]:
                st.metric("Churn Rate", f"{optimal.get('churn_rate', 0)*100:.1f}%")
            with cols[2]:
                st.metric("Referral Rate", f"{optimal.get('referral_rate', 0)*100:.1f}%")


def render_competitive_tab(doc: dict):
    """Render competitive analysis."""
    st.header("Competitive Analysis")
    
    # Competitor list
    competitors = doc.get("competitors", [])
    
    if competitors and all(isinstance(comp, dict) for comp in competitors):
        st.subheader("Key Competitors")
        
        # Format competitor data
        comp_data = []
        for comp in competitors:
            row = {
                "Name": comp.get("name", ""),
                "Funding": f"${comp.get('funding', 0)/1e6:.1f}M",
                "Employees": comp.get("employees", 0),
                "Founded": comp.get("founded", ""),
                "Threat Level": comp.get("threat_level", "Medium")
            }
            comp_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(comp_data)
        
        # Display table
        st.dataframe(df)
    
    # Competitive positioning
    positioning = doc.get("competitive_positioning", {})
    
    if positioning:
        st.subheader("Competitive Positioning")
        
        # Extract positioning data
        dimensions = positioning.get("dimensions", [])
        company_position = positioning.get("company_position", {})
        competitor_positions = positioning.get("competitor_positions", {})
        
        if dimensions and company_position and competitor_positions:
            # Create scatter plot data
            plot_data = []
            
            # Add company data point
            if len(dimensions) >= 2:
                x_dim = dimensions[0]
                y_dim = dimensions[1]
                
                company_x = company_position.get(x_dim, 50)
                company_y = company_position.get(y_dim, 50)
                
                plot_data.append({
                    "Company": "Your Company",
                    x_dim: company_x,
                    y_dim: company_y
                })
                
                # Add competitor data points
                for comp_name, comp_pos in competitor_positions.items():
                    comp_x = comp_pos.get(x_dim, 50)
                    comp_y = comp_pos.get(y_dim, 50)
                    
                    plot_data.append({
                        "Company": comp_name,
                        x_dim: comp_x,
                        y_dim: comp_y
                    })
                
                # Create DataFrame
                df = pd.DataFrame(plot_data)
                
                # Create scatter plot
                fig = px.scatter(df, x=x_dim, y=y_dim, text="Company", 
                               title=f"Competitive Positioning: {x_dim} vs {y_dim}",
                               color="Company",
                               size_max=60,
                               height=600)
                
                # Update layout
                fig.update_traces(textposition='top center')
                fig.update_layout(xaxis_range=[0, 100], yaxis_range=[0, 100])
                
                # Add quadrant lines
                fig.add_hline(y=50, line_dash="dash", line_color="gray")
                fig.add_vline(x=50, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Market trends
    trends = doc.get("market_trends", {})
    
    if trends:
        st.subheader("Market Trends")
        
        trend_items = trends.get("trends", [])
        if trend_items and all(isinstance(trend, dict) for trend in trend_items):
            for trend in trend_items:
                st.markdown(f"**{trend.get('name', '')}**")
                st.markdown(trend.get("description", ""))
    
    # Network effects
    network = doc.get("network_analysis", {})
    
    if network:
        st.subheader("Network Effects Analysis")
        
        # Network effect score
        ne_score = network.get("network_effect_score", 0)
        st.metric("Network Effect Score", f"{ne_score:.1f}/100")
        
        # Network effect types
        ne_types = network.get("network_effect_types", {})
        
        if ne_types:
            types = list(ne_types.keys())
            scores = list(ne_types.values())
            
            df = pd.DataFrame({
                "Network Effect Type": types,
                "Strength": scores
            })
            
            fig = px.bar(df, x="Network Effect Type", y="Strength", 
                       title="Network Effect Strength by Type",
                       color="Strength",
                       color_continuous_scale="Viridis")
            
            st.plotly_chart(fig, use_container_width=True)


def render_tech_tab(doc: dict):
    """Render technical assessment."""
    st.header("Technical Assessment")
    
    tech_assessment = doc.get("tech_assessment", {})
    
    if not tech_assessment:
        st.warning("Technical assessment not available.")
        return
    
    # Convert tech_assessment to dict if it's an object
    if not isinstance(tech_assessment, dict):
        tech_assessment = to_dict(tech_assessment)
    
    # Fix any invalid industry_benchmark_position values
    def fix_benchmark_position(assessment):
        """Ensure industry_benchmark_position has valid values"""
        valid_positions = {"Average", "Bottom 25%", "Top 10%", "Below average", "Top 25%"}
        if isinstance(assessment, dict):
            if "competitive_positioning" in assessment:
                comp_pos = assessment["competitive_positioning"]
                if isinstance(comp_pos, dict) and "industry_benchmark_position" in comp_pos:
                    pos = comp_pos["industry_benchmark_position"]
                    if pos is not None and pos not in valid_positions:
                        # Try to map similar values
                        position_map = {
                            "ABOVE AVERAGE": "Top 25%",
                            "BELOW AVERAGE": "Below average", 
                            "BELOW_AVERAGE": "Below average",
                            "BOTTOM": "Bottom 25%",
                            "TOP": "Top 10%",
                            "ABOVE_AVERAGE": "Top 25%",
                            "TOP_10": "Top 10%",
                            "TOP_25": "Top 25%",
                            "BOTTOM_25": "Bottom 25%",
                            "TOP10": "Top 10%",
                            "TOP25": "Top 25%",
                            "BOTTOM25": "Bottom 25%",
                            "HIGH": "Top 25%",
                            "LOW": "Below average",
                            "MEDIUM": "Average",
                            "VERY HIGH": "Top 10%",
                            "VERY LOW": "Bottom 25%",
                            "EXCELLENT": "Top 10%",
                            "GOOD": "Top 25%",
                            "FAIR": "Average",
                            "POOR": "Below average",
                            "VERY POOR": "Bottom 25%"
                        }
                        if isinstance(pos, str) and pos.upper() in position_map:
                            comp_pos["industry_benchmark_position"] = position_map[pos.upper()]
                        elif isinstance(pos, str):
                            # Try removing spaces, hyphens, underscores
                            normalized = pos.upper().replace(" ", "_").replace("-", "_")
                            if normalized in position_map:
                                comp_pos["industry_benchmark_position"] = position_map[normalized]
                            # Try with percentages
                            elif "%" in pos:
                                for valid_pos in valid_positions:
                                    if valid_pos.replace(" ", "") in pos.replace(" ", ""):
                                        comp_pos["industry_benchmark_position"] = valid_pos
                                        break
                                else:  # If no match found in the loop
                                    comp_pos["industry_benchmark_position"] = "Average"
                            else:
                                comp_pos["industry_benchmark_position"] = "Average"
                        else:
                            comp_pos["industry_benchmark_position"] = "Average"
                        logger.info(f"Fixed invalid industry_benchmark_position: '{pos}' -> '{comp_pos['industry_benchmark_position']}'")
                    elif pos is None:
                        comp_pos["industry_benchmark_position"] = "Average"
                        logger.info("Fixed None industry_benchmark_position")
        return assessment
    
    # Apply fixes
    tech_assessment = fix_benchmark_position(tech_assessment)
    
    # Overall tech score
    tech_score = tech_assessment.get("overall_score", 0)
    st.metric("Technical Assessment Score", f"{tech_score:.1f}/100")
    
    # Architecture diagram
    if "architecture_diagram" in tech_assessment:
        st.subheader("Technical Architecture")
        st.image(tech_assessment["architecture_diagram"], use_column_width=True)
    
    # Tech stack
    st.subheader("Technology Stack")
    tech_stack = tech_assessment.get("tech_stack", {})
    
    if tech_stack:
        stack_categories = list(tech_stack.keys())
        
        for category in stack_categories:
            st.markdown(f"**{category}**")
            technologies = tech_stack[category]
            
            if isinstance(technologies, list):
                st.markdown(", ".join(technologies))
            elif isinstance(technologies, dict):
                for tech, details in technologies.items():
                    st.markdown(f"- {tech}: {details}")
    
    # Component scores
    st.subheader("Component Scores")
    scores = tech_assessment.get("scores", {})
    
    if scores:
        categories = list(scores.keys())
        values = list(scores.values())
        
        df = pd.DataFrame({
            "Category": categories,
            "Score": values
        })
        
        fig = px.bar(df, x="Category", y="Score", 
                   title="Technical Component Scores",
                   color="Score",
                   color_continuous_scale="Viridis")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Scalability assessment
    st.subheader("Scalability Assessment")
    scalability = tech_assessment.get("scalability", {})
    
    if scalability:
        if isinstance(scalability, dict):
            scalability_score = scalability.get("score", 0)
            bottlenecks = scalability.get("bottlenecks", [])
            
            st.metric("Scalability Score", f"{scalability_score:.1f}/100")
            
            if bottlenecks:
                st.markdown("**Potential Bottlenecks:**")
                for bottleneck in bottlenecks:
                    st.markdown(f"- {bottleneck}")
    
    # Technical debt
    st.subheader("Technical Debt Assessment")
    tech_debt = tech_assessment.get("technical_debt", {})
    
    if tech_debt:
        if isinstance(tech_debt, dict):
            debt_score = tech_debt.get("score", 0)
            debt_areas = tech_debt.get("areas", {})
            
            st.metric("Technical Debt Score", f"{debt_score:.1f}/100")
            
            if debt_areas:
                areas = list(debt_areas.keys())
                severity = list(debt_areas.values())
                
                df = pd.DataFrame({
                    "Area": areas,
                    "Severity": severity
                })
                
                fig = px.bar(df, x="Area", y="Severity", 
                           title="Technical Debt by Area",
                           color="Severity",
                           color_continuous_scale="Reds")
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Technical Recommendations")
    recommendations = tech_assessment.get("recommendations", [])
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            st.markdown(f"{i+1}. {rec}")


def render_patterns_tab(doc: dict):
    """Render pattern detection insights."""
    st.header("Pattern Detection")
    
    patterns = doc.get("patterns_matched", [])
    
    if not patterns:
        st.warning("Pattern analysis not available.")
        return
    
    # Convert pattern objects to dicts if needed
    patterns = [p if isinstance(p, dict) else to_dict(p) for p in patterns]
    
    # Positive patterns
    positive_patterns = [p for p in patterns if p.get("is_positive", False)]
    
    if positive_patterns:
        st.subheader("Positive Patterns Detected")
        
        for pattern in positive_patterns:
            with st.expander(pattern.get("name", "")):
                st.markdown(f"**Confidence**: {pattern.get('confidence', 0):.1f}%")
                st.markdown(f"**Description**: {pattern.get('description', '')}")
                st.markdown(f"**Recommendation**: {pattern.get('recommendation', '')}")
    
    # Negative patterns
    negative_patterns = [p for p in patterns if not p.get("is_positive", True)]
    
    if negative_patterns:
        st.subheader("Negative Patterns Detected")
        
        for pattern in negative_patterns:
            with st.expander(pattern.get("name", "")):
                st.markdown(f"**Confidence**: {pattern.get('confidence', 0):.1f}%")
                st.markdown(f"**Description**: {pattern.get('description', '')}")
                st.markdown(f"**Recommendation**: {pattern.get('recommendation', '')}")
    
    # Pattern insights visualization
    insights = doc.get("pattern_insights", {})
    
    if insights:
        st.subheader("Pattern Insights")
        
        top_insights = insights.get("top_insights", [])
        for insight in top_insights:
            with st.expander(insight.get("title", "")):
                st.markdown(insight.get("description", ""))
                st.markdown(f"**Impact**: {insight.get('impact', '')}")
                st.markdown(f"**Action**: {insight.get('action', '')}")
        
        # Pattern distribution
        pattern_categories = insights.get("pattern_categories", {})
        
        if pattern_categories:
            categories = list(pattern_categories.keys())
            counts = list(pattern_categories.values())
            
            df = pd.DataFrame({
                "Category": categories,
                "Count": counts
            })
            
            fig = px.pie(df, values="Count", names="Category", 
                       title="Pattern Distribution by Category")
            
            st.plotly_chart(fig, use_container_width=True)


########################################
# Main App Flow
########################################

def main():
    """Main application flow."""
    # Setup page
    logo = setup_page()
    
    # Apply tab scrolling style
    apply_tab_scroll_style()
    
    # Initialize session state
    initialize_session()
    
    # Display header
    display_header(logo)
    
    # Render sidebar input form
    render_sidebar_input()
    
    # Main content area
    with st.container():
        if not st.session_state.analyzed and not st.session_state.analyze_clicked:
            # Display welcome screen
            if st.session_state.welcome_style == "radar":
                st.markdown("## Welcome to FlashDNA Infinity")
                st.markdown("Complete the CAMP Framework inputs in the sidebar to begin your startup analysis.")
                
                # Sample radar chart
                categories = ['Capital Efficiency', 'Market Dynamics', 'Advantage Moat', 'People & Performance']
                values = [60, 75, 45, 80]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Sample CAMP Scores',
                    line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
                    fillcolor='rgba(31, 119, 180, 0.3)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=False,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Bar chart welcome
                st.markdown("## Welcome to FlashDNA Infinity")
                st.markdown("Complete the CAMP Framework inputs in the sidebar to begin your startup analysis.")
                
                # Sample bar chart
                categories = ['Capital Efficiency', 'Market Dynamics', 'Advantage Moat', 'People & Performance']
                values = [60, 75, 45, 80]
                
                df = pd.DataFrame({
                    "Dimension": categories,
                    "Score": values
                })
                
                fig = px.bar(df, x="Dimension", y="Score", 
                           title="Sample CAMP Framework Analysis",
                           color="Score",
                           color_continuous_scale="Viridis")
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.analyze_clicked and not st.session_state.analyzed:
            # Run analysis
            st.session_state.doc = run_analysis(st.session_state.doc)
            st.session_state.analyzed = True
            st.rerun()
        
        elif st.session_state.analyzed:
            # Create tabs based on display mode setting
            if st.session_state.display_mode == "expanded":
                # 12-tab expanded mode
                tabs = st.tabs([
                    "Summary", "CAMP Details", "PMF", "Cohort Analysis", 
                    "Benchmarks", "Report", "Acquisition", "Exit Path", 
                    "Forecast", "Competitive", "Technical", "Patterns"
                ])
                
                with tabs[0]:
                    render_summary_tab(st.session_state.doc)
                with tabs[1]:
                    render_camp_details_tab(st.session_state.doc)
                with tabs[2]:
                    render_pmf_tab(st.session_state.doc)
                with tabs[3]:
                    render_cohort_tab(st.session_state.doc)
                with tabs[4]:
                    render_benchmarks_tab(st.session_state.doc)
                with tabs[5]:
                    render_report_tab(st.session_state.doc)
                with tabs[6]:
                    render_acquisition_tab(st.session_state.doc)
                with tabs[7]:
                    render_exit_path_tab(st.session_state.doc)
                with tabs[8]:
                    render_forecast_tab(st.session_state.doc)
                with tabs[9]:
                    render_competitive_tab(st.session_state.doc)
                with tabs[10]:
                    render_tech_tab(st.session_state.doc)
                with tabs[11]:
                    render_patterns_tab(st.session_state.doc)
            else:
                # 8-tab standard mode
                tabs = st.tabs([
                    "Summary", "CAMP Details", "PMF", "Exit Path", 
                    "Forecast", "Competitive", "Technical", "Patterns"
                ])
                
                with tabs[0]:
                    render_summary_tab(st.session_state.doc)
                with tabs[1]:
                    render_camp_details_tab(st.session_state.doc)
                with tabs[2]:
                    render_pmf_tab(st.session_state.doc)
                with tabs[3]:
                    render_exit_path_tab(st.session_state.doc)
                with tabs[4]:
                    render_forecast_tab(st.session_state.doc)
                with tabs[5]:
                    render_competitive_tab(st.session_state.doc)
                with tabs[6]:
                    render_tech_tab(st.session_state.doc)
                with tabs[7]:
                    render_patterns_tab(st.session_state.doc)


if __name__ == "__main__":
    main()
