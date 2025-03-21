######################################################
# analysis_flow.py - Debug Logging Enabled
######################################################

import streamlit as st
import os
import time
import pickle
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
from streamlit_option_menu import option_menu

# Import core analysis modules
from advanced_ml import evaluate_startup
# We call intangible_api instead of intangible_llm
from intangible_api import compute_intangible_llm

from domain_expansions import apply_domain_expansions
from team_moat import compute_team_depth_score, compute_moat_score, evaluate_team_execution_risk
from pattern_detector import detect_patterns, generate_pattern_insights
from system_dynamics import system_dynamics_sim, virality_sim, calculate_growth_metrics
from sir_model import sir_viral_adoption, calculate_market_penetration
from hpc_scenario import find_optimal_scenario
from financial_modeling import scenario_runway, calculate_unit_economics, forecast_financials, calculate_valuation_metrics
from competitive_intelligence import CompetitiveIntelligence
from monte_carlo import MonteCarloSimulator, SimulationResult
from pitch_sentiment import PitchAnalyzer
from ml_assessment import StartupAssessmentModel
from product_market_fit import ProductMarketFitAnalyzer, PMFMetrics
from technical_due_diligence import TechnicalDueDiligence, TechnicalAssessment
from cohort_analysis import CohortAnalyzer
from network_analysis import NetworkEffectAnalyzer
from benchmarking import BenchmarkEngine, BenchmarkResult
from report_generator import generate_investor_report
from utils import create_placeholder, extract_text_from_pdf

######################################################
# 1) Setup logging with DEBUG level
######################################################
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis_flow")

######################################################
# 2) Page Setup & Common Functions
######################################################

def setup_page():
    """Setup page configuration and branding."""
    st.set_page_config(
        page_title="FlashDNA Infinity – Advanced Startup Analysis",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Attempt to apply custom CSS if style.css is present
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Try to load a logo, or else create a placeholder
    try:
        logo = Image.open(os.path.join("static", "logo.png"))
    except:
        logo = create_placeholder(150, 80, "FlashDNA")
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
    """Display app header with logo and title."""
    c1, c2 = st.columns([1, 5])
    with c1:
        st.image(logo, width=150)
    with c2:
        st.markdown("<h1 class='main-header'>FlashDNA Infinity</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Advanced Startup Analysis</p>", unsafe_allow_html=True)

def initialize_session():
    """Initialize session state variables."""
    if "doc" not in st.session_state:
        st.session_state.doc = {}
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False

def load_xgb_model():
    """Load the trained XGBoost model for startup success prediction."""
    try:
        with open("model_xgb.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("model_xgb.pkl not found => please train or provide a model.")
        return None
    except Exception as e:
        st.error(f"Error => {str(e)}")
        return None

######################################################
# 3) Sidebar Input
######################################################

def render_sidebar_input():
    """Render multi-tabbed form in the sidebar for user input."""
    with st.sidebar:
        st.subheader("Startup Information")
        menu = option_menu(
            "Input Sections",
            ["Basic Info", "Financial", "Team", "Market", "Advanced", "PDF"],
            icons=["info-circle", "cash-coin", "people", "graph-up", "gear", "file-earmark-pdf"],
            menu_icon="list",
            default_index=0,
            orientation="horizontal"
        )

        doc = {}
        # Basic Info
        if menu == "Basic Info":
            doc["name"] = st.text_input("Startup Name", "NewCo")
            doc["stage"] = st.selectbox("Stage", ["pre-seed","seed","series-a","series-b","growth","other"])
            doc["sector"] = st.selectbox("Sector", ["fintech","biotech","saas","ai","ecommerce","marketplace","crypto","other"])

        # Financial Info
        elif menu == "Financial":
            doc["monthly_revenue"] = st.number_input("Monthly Revenue($)", 0.0, 1e9, 50000.0)
            doc["burn_rate"] = st.number_input("Burn Rate($)", 0.0, 1e9, 30000.0)
            doc["current_cash"] = st.number_input("Current Cash($)", 0.0, 1e9, 500000.0)
            doc["ltv_cac_ratio"] = st.slider("LTV:CAC", 0.0, 10.0, 2.5)
            doc["avg_revenue_per_user"] = st.number_input("ARPU($/mo)", 0.0,1e6,100.0)
            doc["customer_acquisition_cost"] = st.number_input("CAC($)", 0.0,1e6,300.0)
            doc["gross_margin_percent"] = st.slider("Gross Margin(%)", 0, 100, 70)
            doc["revenue_growth_rate"] = st.slider("Revenue Growth Rate(%)", 0, 200, 15)/100.0

        # Team Info
        elif menu == "Team":
            doc["founder_exits"] = st.number_input("Previous Founder Exits", 0, 10, 0)
            doc["founder_domain_exp_yrs"] = st.number_input("Founder Domain Exp(yrs)",0,30,5)
            doc["founder_diversity_score"] = st.slider("Diversity(0..100)",0,100,50)
            doc["employee_count"] = st.number_input("Employee Count", 0, 10000, 10)
            doc["has_cto"] = st.checkbox("Has CTO",True)
            doc["has_cmo"] = st.checkbox("Has CMO",False)
            doc["has_cfo"] = st.checkbox("Has CFO",False)
            doc["tech_talent_ratio"] = st.slider("Tech Talent Ratio",0.0,1.0,0.4)

        # Market Info
        elif menu == "Market":
            doc["market_size"] = st.number_input("Market Size($)", 0.0,1e12,50e6)
            doc["market_growth_rate"] = st.slider("Market Growth Rate(%)",0,200,10)/100.0
            doc["market_share"] = st.slider("Market Share(%)",0.0,100.0,0.5)/100.0
            doc["churn_rate"] = st.slider("Churn Rate(%)",0.0,100.0,5.0)/100.0
            doc["referral_rate"] = st.slider("Referral Rate(%)",0.0,50.0,2.0)/100.0
            doc["current_users"] = st.number_input("Current Users",0,1_000_000_000,1000)
            doc["user_growth_rate"] = st.slider("User Growth Rate(%)",0,200,10)/100.0
            doc["viral_coefficient"] = st.slider("Viral Coefficient",0.0,3.0,1.2)

        # Advanced
        elif menu == "Advanced":
            st.write("Sector-specific expansions")
            sector = st.session_state.doc.get("sector","saas").lower()
            if sector=="fintech":
                doc["licenses_count"]= st.number_input("Licenses Count",0,50,2)
                doc["default_rate"]= st.slider("Default Rate(%)",0.0,100.0,2.0)/100.0
            elif sector in ["biotech","healthtech"]:
                doc["clinical_phase"]= st.selectbox("Clinical Phase",[0,1,2,3,4])
                doc["patent_count"]= st.number_input("Patents",0,100,1)
            elif sector=="ai":
                doc["data_volume_tb"]= st.number_input("Data Volume(TB)",0.0,1e9,100.0,step=1.0)
                doc["patent_count"]= st.number_input("Patents",0,500,2)
            elif sector=="crypto":
                doc["token_utility_score"]= st.number_input("Token Utility(0..100)",0,100,50)
                doc["decentralization_factor"]= st.slider("Decentralization(0..1)",0.0,1.0,0.5)
            
            st.write("Moat & Brand expansions")
            doc["category_leadership_score"]= st.slider("Category Leadership(0..100)",0,100,50)
            doc["technical_innovation_score"]= st.slider("Tech Innovation(0..100)",0,100,50)
            doc["channel_partner_count"]= st.number_input("Channel Partners",0,500,5)

            # Technical assessment
            with st.expander("Technical Assessment Data"):
                doc["architecture_type"]= st.selectbox("Architecture Type",
                                                      ["Monolith","Microservice","Serverless","Hybrid","Other"])
                doc["test_coverage"] = st.slider("Test Coverage(%)",0,100,60)
                doc["has_code_reviews"] = st.checkbox("Has Code Reviews", True)
                doc["has_documentation"] = st.checkbox("Has Documentation", True)
                doc["open_bugs"] = st.number_input("Open Bugs", 0, 1000, 20)
                doc["has_ci_cd"] = st.checkbox("Has CI/CD", True)

                # Tech stack
                st.write("Tech stack (minimal):")
                tech_stack=[]
                for i in range(3):
                    col1,col2= st.columns(2)
                    with col1:
                        tech_name= st.text_input(f"Tech #{i+1}", value="" if i>0 else "Python")
                    with col2:
                        tech_cat= st.selectbox(f"Category #{i+1}",
                                               ["language","backend","frontend","database","infrastructure"],
                                               index=0 if i==0 else (3 if i==1 else 4))
                    if tech_name:
                        tech_stack.append({"name": tech_name, "category": tech_cat})
                if tech_stack:
                    doc["tech_stack"]= tech_stack

        # PDF
        elif menu == "PDF":
            st.subheader("Pitch Deck PDF Extraction")
            uploaded = st.file_uploader("Upload PDF", type=["pdf"])
            if uploaded:
                st.success(f"Uploaded => {uploaded.name}")
                pdf_text = extract_text_from_pdf(uploaded)
                st.text_area("Extracted PDF Text", pdf_text, height=200)
                doc["pitch_deck_text"] = pdf_text

        # Store doc in session
        for k,v in doc.items():
            st.session_state.doc[k]= v

        # Analysis button
        if st.button("Analyze Startup => Run Analysis", type="primary"):
            st.session_state.analyze_clicked= True

######################################################
# 4) Main Analysis Workflow
######################################################

def run_analysis(doc: dict)-> dict:
    """
    Perform the comprehensive startup analysis, skipping HPC synergy references.
    Replaced intangible => intangible_api usage with debug logs.
    """
    with st.spinner("Running advanced analysis..."):
        prog = st.progress(0)
        logger.debug("[AnalysisFlow] Starting run_analysis...")

        # 1) domain expansions
        expansions = apply_domain_expansions(doc)
        doc.update(expansions)
        prog.progress(10)

        # 2) intangible from intangible_api
        pitch_len = len(doc.get("pitch_deck_text",""))
        logger.debug(f"[AnalysisFlow] pitch_deck_text length => {pitch_len}")
        if doc.get("pitch_deck_text"):
            intangible = compute_intangible_llm(doc)
            doc["intangible"] = intangible
            logger.debug(f"[AnalysisFlow] intangible from intangible_api => {intangible}")
        else:
            doc["intangible"] = 50.0
            logger.debug("[AnalysisFlow] No pitch text => intangible => 50.0 fallback")

        prog.progress(20)

        # 3) team + moat
        doc["team_score"] = compute_team_depth_score(doc)
        doc["moat_score"] = compute_moat_score(doc)
        doc["execution_risk"] = evaluate_team_execution_risk(doc)
        prog.progress(30)

        # 4) Evaluate with XGB
        model = load_xgb_model()
        if model:
            ev = evaluate_startup(doc, model)
            doc.update(ev)
        else:
            doc["success_prob"] = 50.0
            doc["flashdna_score"] = 50.0
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
        doc["system_dynamics"]= sys_res
        doc["growth_metrics"]= calculate_growth_metrics(sys_res)

        doc["virality_sim"]= virality_sim(
            user_initial= doc.get("current_users",1000),
            k_factor= doc.get("viral_coefficient",0.2),
            conversion_rate= doc.get("conversion_rate",0.1),
            cycles=12
        )
        prog.progress(50)

        # 6) SIR
        S,I,R= sir_viral_adoption(
            S0= doc.get("market_size",5e7) - doc.get("current_users",1000),
            I0= doc.get("current_users",1000),
            R0= 0,
            beta= 0.001,
            gamma= doc.get("churn_rate",0.05),
            steps=24
        )
        doc["sir_data"]= (S,I,R)
        doc["market_penetration"]= calculate_market_penetration(doc)
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
        doc["unit_economics"]= ue

        fc= forecast_financials(doc)
        doc["financial_forecast"]= fc

        val= calculate_valuation_metrics(doc)
        doc["valuation_metrics"]= val
        prog.progress(80)

        # 8) Competitive Intelligence
        try:
            ci= CompetitiveIntelligence()
            comps= ci.get_competitors(doc.get("name",""), doc.get("sector","saas"))
            doc["competitors"]= comps
            doc["competitive_positioning"]= ci.competitive_positioning_analysis(doc, comps)
            doc["market_trends"]= ci.market_trends_analysis(doc.get("sector",""))
            doc["moat_analysis"]= ci.competitive_moat_analysis(doc)
        except Exception as e:
            logger.error(f"Competitive Intelligence => {e}")

        prog.progress(85)

        # 9) Monte Carlo
        try:
            mc= MonteCarloSimulator()
            mcres= mc.run_simulation(doc)
            doc["monte_carlo"]= mcres
        except Exception as e:
            logger.error(f"MonteCarlo => {e}")
            doc["monte_carlo"]= {}

        prog.progress(88)

        # 10) pitch sentiment
        try:
            if doc.get("pitch_deck_text"):
                analyzer= PitchAnalyzer()
                doc["pitch_sentiment"]= analyzer.analyze_pitch(doc["pitch_deck_text"])
            else:
                doc["pitch_sentiment"]= {}
        except Exception as e:
            logger.error(f"Pitch Sentiment => {e}")
            doc["pitch_sentiment"]= {}

        prog.progress(90)

        # 11) ML Assessment
        try:
            assess_model= StartupAssessmentModel()
            asses= assess_model.assess_startup(doc)
            doc["ml_assessment"]= asses
        except Exception as e:
            logger.error(f"ML Assessment => {e}")
            doc["ml_assessment"]= {}

        prog.progress(92)

        # 12) PMF
        try:
            pmf= ProductMarketFitAnalyzer()
            pmfa= pmf.analyze_pmf(doc)
            doc["pmf_analysis"]= pmfa
        except Exception as e:
            logger.error(f"PMF => {e}")
            doc["pmf_analysis"]= {}

        prog.progress(94)

        # 13) Tech Due Diligence
        try:
            tdd= TechnicalDueDiligence()
            tch= tdd.assess_technical_architecture(doc)
            doc["tech_assessment"]= tch
        except Exception as e:
            logger.error(f"TechDD => {e}")
            doc["tech_assessment"]= {}

        prog.progress(96)

        # 14) Cohort
        try:
            c_an= CohortAnalyzer()
            doc["cohort_data"]= c_an._generate_dummy_cohort_data(6)
        except Exception as e:
            logger.error(f"Cohort => {e}")
            doc["cohort_data"]= {}

        # 15) Network
        try:
            n_an= NetworkEffectAnalyzer()
            doc["network_analysis"]= n_an.analyze_network_effects(company_data=doc)
        except Exception as e:
            logger.error(f"Network => {e}")
            doc["network_analysis"]= {}

        prog.progress(98)

        # 16) Benchmarking
        try:
            be= BenchmarkEngine()
            doc["benchmarks"]= be.benchmark_startup(doc)
        except Exception as e:
            logger.error(f"Benchmark => {e}")
            doc["benchmarks"]= {}

        # Patterns
        pats= detect_patterns(doc)
        doc["patterns_matched"]= pats
        doc["pattern_insights"]= generate_pattern_insights(pats)

        prog.progress(100)
        time.sleep(0.5)
        prog.empty()

    return doc

######################################################
# 5) Tabs / UI Sections
######################################################

def show_overview(doc):
    """Display executive overview with key metrics."""
    st.title("Executive Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("FlashDNA Score", f"{doc.get('flashdna_score', 50):.1f}")
    with col2:
        st.metric("Success Probability", f"{doc.get('success_prob', 50):.1f}%")
    with col3:
        st.metric("Intangible(LLM)", f"{doc.get('intangible', 50):.1f}")
    with col4:
        runw = doc.get("runway_months", -1)
        if isinstance(runw, int) and runw >= 9999:
            st.metric("Runway", "∞")
        else:
            st.metric("Runway", f"{runw} months")

    st.subheader("Startup Profile")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.markdown(f"""
        **Name**: {doc.get('name', 'N/A')}  
        **Stage**: {doc.get('stage', 'N/A')}  
        **Sector**: {doc.get('sector', 'N/A')}
        """)
    with metric_cols[1]:
        st.markdown(f"""
        **Monthly Revenue**: ${doc.get('monthly_revenue', 0):,.0f}  
        **Burn Rate**: ${doc.get('burn_rate', 0):,.0f}  
        **Current Cash**: ${doc.get('current_cash', 0):,.0f}
        """)
    with metric_cols[2]:
        st.markdown(f"""
        **Team Score**: {doc.get('team_score', 0):.1f}/100  
        **Moat Score**: {doc.get('moat_score', 0):.1f}/100  
        **Users**: {doc.get('current_users', 0):,}
        """)

    # Pattern insights
    insights= doc.get("pattern_insights",[])
    if insights:
        st.subheader("Key Insights")
        for insight in insights:
            st.info(insight)

def show_growth(doc):
    """User Growth: system dynamics, virality, SIR model."""
    st.title("User Growth")
    arr= doc.get("system_dynamics",[])
    if arr:
        st.subheader("System Dynamics")
        months= [f"M{i+1}" for i in range(len(arr))]
        fig= px.line(x=months, y=arr, labels={"x":"Month","y":"Active Users"}, title="Growth Over Time")
        st.plotly_chart(fig,use_container_width=True)

        gm= doc.get("growth_metrics",{})
        if gm:
            c1,c2,c3,c4= st.columns(4)
            with c1: st.metric("Initial Users", f"{gm.get('initial_users',0):,.0f}")
            with c2: st.metric("Final Users", f"{gm.get('final_users',0):,.0f}")
            with c3: st.metric("Multiple", f"{gm.get('growth_multiple',1):.2f}x")
            with c4: st.metric("MoM Growth", f"{gm.get('avg_mom_growth_rate',0)*100:.1f}%")

    vsim= doc.get("virality_sim",{})
    if vsim:
        st.subheader("Virality Simulation")
        days= vsim.get("days",[])
        users= vsim.get("users",[])
        if users and days:
            fig2= px.line(x=days, y=users, title="Viral Growth Simulation (K-factor)",
                          labels={"x":"Day","y":"Users"})
            st.plotly_chart(fig2, use_container_width=True)

            is_viral= vsim.get('is_viral',False)
            st.markdown(f"""
            <div style="padding:15px; border-radius:5px; background-color: {"#e6f7ff" if is_viral else "#fff3cd"}; margin-bottom:20px;">
                <h4 style="margin-top:0;">{"🚀 Viral Growth Potential" if is_viral else "📉 Sub-viral Growth"}</h4>
                <p>K-factor: {vsim.get('growth_multiple',1):.2f}x ({'>1 => strong viral loop' if is_viral else '<1 => sub-viral'})</p>
                {f"<p>Time to 10x: {vsim.get('time_to_10x','N/A')} days</p>" if vsim.get('time_to_10x') else ""}
                {f"<p>Time to 100x: {vsim.get('time_to_100x','N/A')} days</p>" if vsim.get('time_to_100x') else ""}
                <p>Final user count: {vsim.get('final_users',0):,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

    # SIR
    sir_data= doc.get("sir_data",None)
    if sir_data and len(sir_data)==3:
        st.subheader("SIR Viral Adoption Model")
        S,I,R= sir_data
        periods= list(range(len(S)))
        import pandas as pd
        sir_df= pd.DataFrame({"Period":periods, "Potential Users":S,"Active Users":I,"Churned Users":R})
        fig3= px.line(sir_df,x='Period', y=['Potential Users','Active Users','Churned Users'],
                      title="SIR Viral Adoption Model",
                      labels={"value":"Users","variable":"Group"})
        st.plotly_chart(fig3,use_container_width=True)
        # Market penetration
        mp= doc.get("market_penetration",{})
        if mp:
            st.subheader("Market Penetration")
            c1,c2,c3= st.columns(3)
            with c1:
                st.metric("Penetration(%)", f"{mp.get('market_penetration_percentage',0):.1f}%")
            with c2:
                st.metric("Peak Users", f"{mp.get('peak_active_users',0):,.0f}")
            with c3:
                st.metric("Peak Time", f"{mp.get('peak_time',0)}")

def show_scenarios(doc):
    """Scenario analysis and custom scenario input."""
    st.title("Scenario Analysis")
    hpc_data= doc.get("hpc_data",[])
    if hpc_data:
        df= pd.DataFrame(hpc_data)
        if {"churn","referral","final_users"} <= set(df.columns):
            fig= go.Figure(data=[go.Scatter3d(
                x=df["churn"], y=df["referral"], z=df["final_users"],
                mode="markers",
                marker=dict(size=6,color=df["final_users"],colorscale="Viridis",opacity=0.8)
            )])
            fig.update_layout(
                scene=dict(
                    xaxis_title="Churn",
                    yaxis_title="Referral",
                    zaxis_title="Final Users"
                ),
                height=600,
                title="3D: Churn vs Referral vs Final Users"
            )
            st.plotly_chart(fig,use_container_width=True)

            # Optimal scenario
            st.subheader("Optimal Scenario")
            opt= doc.get("optimal_scenario",{})
            if opt:
                c1,c2,c3= st.columns(3)
                with c1: st.metric("Churn", f"{opt.get('churn',0)*100:.2f}%")
                with c2: st.metric("Referral", f"{opt.get('referral',0)*100:.2f}%")
                with c3: st.metric("Final Users", f"{opt.get('final_users',0):,.0f}")
                
                # Custom scenario widget
                st.subheader("Custom Scenario")
                cu1, cu2= st.columns(2)
                with cu1:
                    custom_churn= st.slider("Churn Rate",0.01,0.30,doc.get("churn_rate",0.05), format="%.2f")
                with cu2:
                    custom_ref= st.slider("Referral Rate",0.01,0.20,doc.get("referral_rate",0.02), format="%.2f")
                
                if st.button("Run Custom Scenario"):
                    with st.spinner("Simulating custom scenario..."):
                        from system_dynamics import system_dynamics_sim
                        custom_result= system_dynamics_sim(
                            user_initial= doc.get("current_users",1000),
                            months=24,
                            marketing_spend= doc.get("burn_rate",30000)*0.4,
                            referral_rate= custom_ref,
                            churn_rate= custom_churn
                        )
                        # Plot
                        mth= [f"M{i+1}" for i in range(len(custom_result))]
                        fig_cus= px.line(x=mth, y=custom_result,
                                         labels={"x":"Month","y":"Users"},
                                         title=f"Churn={custom_churn:.2%}, Referral={custom_ref:.2%}")
                        st.plotly_chart(fig_cus, use_container_width=True)

                        # Growth
                        if len(custom_result)>1:
                            growth= ((custom_result[-1]/ custom_result[0])**(1/(len(custom_result)-1))-1)*100
                        else:
                            growth= 0
                        colA, colB, colC= st.columns(3)
                        with colA: st.metric("Initial", f"{custom_result[0]:,.0f}")
                        with colB: st.metric("Final", f"{custom_result[-1]:,.0f}")
                        with colC: st.metric("MoM Growth", f"{growth:.2f}%")
            else:
                st.write("No optimal scenario found.")
        else:
            st.warning("Scenario data missing columns => churn/referral/final_users")
    else:
        st.info("No scenario data => run analysis first.")

def show_financial(doc):
    """Financial health & projections."""
    st.title("Financial Health & Projections")
    runway= doc.get("runway_months",-1)
    if runway>=9999:
        runway_txt= "∞"
    else:
        runway_txt= f"{runway}"
    c1,c2,c3= st.columns(3)
    c1.metric("Runway", runway_txt)
    c2.metric("Burn Rate", f"${doc.get('burn_rate',0):,.0f}")
    c3.metric("Monthly Revenue", f"${doc.get('monthly_revenue',0):,.0f}")

    # Cash flow
    cf= doc.get("cash_flow",[])
    if cf:
        st.subheader("Cash Flow Projection")
        fig= go.Figure(go.Scatter(
            x=list(range(1,len(cf)+1)),
            y= cf, mode='lines+markers', fill='tozeroy',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgb(0,176,246)', width=2)
        ))
        fig.update_layout(title="Cash Flow Projection",
                          xaxis_title="Month",
                          yaxis_title="Cash($)",
                          hovermode="x unified")
        st.plotly_chart(fig,use_container_width=True)
        if min(cf)<0:
            crossing= next((i+1 for i,cash in enumerate(cf) if cash<0), None)
            if crossing:
                st.warning(f"⚠️ Cash crosses zero in month {crossing} => consider fundraising or cost cuts.")

    # Unit economics
    ue= doc.get("unit_economics",{})
    if ue:
        st.subheader("Unit Economics")
        colA, colB, colC= st.columns(3)
        with colA:
            st.metric("ARPU", f"${ue.get('arpu',0):.2f}/mo")
            st.metric("CAC", f"${ue.get('cac',0):.2f}")
        with colB:
            st.metric("LTV", f"${ue.get('ltv',0):.2f}")
            st.metric("LTV:CAC", f"{ue.get('ltv_cac_ratio',0):.2f}x")
        with colC:
            pay= ue.get("cac_payback_months",0)
            st.metric("CAC Payback", f"{pay:.1f} mo")
            gm= ue.get("gross_margin",0)*100
            st.metric("Gross Margin", f"{gm:.1f}%")

        ratio= ue.get("ltv_cac_ratio",0)
        if ratio>0:
            st.subheader("LTV:CAC Visualization")
            ltv= ue.get("ltv",0)
            cac= ue.get("cac",0)

            fig_ltv_cac= go.Figure()
            fig_ltv_cac.add_trace(go.Bar(
                y=["Metric"], x=[cac], name="CAC", orientation='h',
                marker=dict(color='#FF6B6B')
            ))
            fig_ltv_cac.add_trace(go.Bar(
                y=["Metric"], x=[ltv - cac], name="LTV remainder", orientation='h',
                marker=dict(color='#4ECDC4')
            ))
            fig_ltv_cac.update_layout(
                barmode='stack',
                title=f"LTV:CAC = {ratio:.2f}x",
                height=150,
                margin=dict(l=20,r=20,t=40,b=20),
                xaxis=dict(title="Value ($)")
            )
            st.plotly_chart(fig_ltv_cac,use_container_width=True)
            if ratio<1:
                st.error("⚠️ LTV:CAC <1 => unsustainable unit economics.")
            elif ratio<3:
                st.warning("⚠️ LTV:CAC <3 => borderline efficiency.")
            else:
                st.success("✅ LTV:CAC >3 => healthy unit economics.")

    # 5-year forecast
    fc= doc.get("financial_forecast",{})
    if fc and "annual" in fc and fc["annual"].get("revenue"):
        st.subheader("5-Year Financial Forecast")
        yrs= range(1, len(fc["annual"]["revenue"])+1)
        df= pd.DataFrame({
            "Year": [f"Year {y}" for y in yrs],
            "Revenue": fc["annual"]["revenue"],
            "Costs": fc["annual"]["costs"],
            "Profit": fc["annual"]["profit"]
        })
        fig2= go.Figure()
        fig2.add_trace(go.Bar(x=df["Year"], y=df["Revenue"], name="Revenue", marker_color="#66BB6A"))
        fig2.add_trace(go.Bar(x=df["Year"], y=df["Costs"], name="Costs", marker_color="#EF5350"))
        fig2.add_trace(go.Scatter(x=df["Year"], y=df["Profit"], mode='lines+markers', name="Profit", line=dict(color="#1E88E5")))
        fig2.update_layout(title="5-Year Forecast", barmode='group')
        st.plotly_chart(fig2,use_container_width=True)

        mets= fc.get("metrics",{})
        if mets:
            st.subheader("Forecast Metrics")
            cols= st.columns(3)
            with cols[0]:
                pm= mets.get("profitable_month",-1)
                if pm>0: st.metric("Profitable Month", f"{pm}")
                else: st.metric("Profitable Month","N/A")
            with cols[1]:
                py= mets.get("profitable_year",-1)
                if py>0: st.metric("Profitable Year", f"Year {py}")
                else: st.metric("Profitable Year","N/A")
            with cols[2]:
                cagr= mets.get("cagr",0)
                st.metric("Revenue CAGR", f"{cagr*100:.1f}%")

        val= doc.get("valuation_metrics",{})
        if val:
            st.subheader("Valuation Metrics")
            cA,cB,cC= st.columns(3)
            with cA:
                st.metric("Revenue Multiple", f"{val.get('revenue_multiple',0):.1f}x")
                st.metric("ARR Valuation", f"${val.get('arr_valuation',0)/1e6:.1f}M")
            with cB:
                st.metric("Rule of 40", f"{val.get('rule_of_40_score',0):.1f}")
                st.metric("Forward ARR", f"${val.get('forward_arr',0)/1e6:.1f}M")
            with cC:
                st.metric("Growth Rate", f"{val.get('annual_growth_rate',0)*100:.1f}%")
                st.metric("Berkus Value", f"${val.get('berkhus_valuation',0)/1e6:.1f}M")

            st.markdown(f"""
            <div style="padding:15px; border-radius:5px; background-color:#e8f4f8; margin-top:15px;">
                <h4 style="margin-top:0;">Valuation Justification</h4>
                <p>{val.get('justification','N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No forecast data available.")

def show_team(doc):
    """Team & Execution Risk overview."""
    st.title("Team & Execution")
    team_score= doc.get("team_score",50)
    moat= doc.get("moat_score",0)
    intangible= doc.get("intangible",50)

    col1,col2,col3= st.columns(3)
    with col1:
        st.metric("Team Depth", f"{team_score:.1f}/100")
    with col2:
        st.metric("Moat Score", f"{moat:.1f}/100")
    with col3:
        st.metric("Intangible(LLM)", f"{intangible:.1f}/100")

    # Execution risk
    exrisk= doc.get("execution_risk",{})
    if exrisk:
        risk_score= exrisk.get("execution_risk_score", 0.5)
        st.subheader("Execution Risk")

        fig= go.Figure(go.Indicator(
            mode="gauge+number",
            value= risk_score*100,
            title={'text':"Execution Risk"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"darkblue"},
                'steps':[
                    {'range':[0,33],'color':"green"},
                    {'range':[33,66],'color':"yellow"},
                    {'range':[66,100],'color':"red"}
                ],
                'threshold':{
                    'line':{'color':"red",'width':4},
                    'thickness':0.75,
                    'value': risk_score*100
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig,use_container_width=True)

        # risk factors
        factors= exrisk.get("risk_factors",{})
        if factors:
            st.subheader("Risk Factors")
            rf_list= []
            for k,v in factors.items():
                rf_list.append({"Factor": k.replace('_',' ').title(),"Risk": v})
            df= pd.DataFrame(rf_list)
            fig2= px.bar(df, x="Risk", y="Factor", orientation='h',
                         title="Risk Factor Breakdown",
                         color="Risk", color_continuous_scale=["green","yellow","red"],
                         range_color=[0,1])
            st.plotly_chart(fig2,use_container_width=True)

    # Team details
    st.subheader("Team Metrics")
    colA,colB= st.columns(2)
    with colA:
        st.markdown(f"""
        - **Founder Exits**: {doc.get('founder_exits',0)}
        - **Domain Exp(yrs)**: {doc.get('founder_domain_exp_yrs',0)}
        - **Diversity**: {doc.get('founder_diversity_score',0)}/100
        - **Employees**: {doc.get('employee_count',0)}
        """)
    with colB:
        c_lev= []
        if doc.get("has_cto",False): c_lev.append("CTO")
        if doc.get("has_cmo",False): c_lev.append("CMO")
        if doc.get("has_cfo",False): c_lev.append("CFO")
        c_lev_txt= ", ".join(c_lev) if c_lev else "None"
        st.markdown(f"""
        - **C-level**: {c_lev_txt}
        - **Tech Talent Ratio**: {doc.get('tech_talent_ratio',0):.2f}
        """)

    # Moat analysis
    moat_analysis= doc.get("moat_analysis",{})
    if moat_analysis:
        st.subheader("Competitive Moat Analysis")
        msc= moat_analysis.get("moat_scores",{})
        if msc:
            cats= list(msc.keys())
            vals= list(msc.values())
            fig3= go.Figure()
            fig3.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill='toself', name='Moat'
            ))
            fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])),
                               showlegend=False)
            st.plotly_chart(fig3,use_container_width=True)
        # strong vs weak
        colx,coly= st.columns(2)
        with colx:
            st.markdown("#### Strongest Moats")
            sm= moat_analysis.get("strongest_moats",[])
            for s in sm:
                if isinstance(s, tuple) and len(s)>=2:
                    st.markdown(f"- **{s[0]}** => {s[1]:.1f}")
        with coly:
            st.markdown("#### Weakest Moats")
            wm= moat_analysis.get("weakest_moats",[])
            for w in wm:
                if isinstance(w, tuple) and len(w)>=2:
                    st.markdown(f"- **{w[0]}** => {w[1]:.1f}")
        recs= moat_analysis.get("recommendations",[])
        if recs:
            st.subheader("Moat Recommendations")
            for rec in recs:
                st.markdown(f"- {rec}")

def show_market(doc):
    """Market & sector analysis, competitor landscape."""
    st.title("Market & Sector")
    mt= doc.get("market_trends",{})
    if mt:
        st.subheader("Market Overview")
        st.write(mt.get("overview","No overview"))
        tam= mt.get("tam",0)
        sam= mt.get("sam",0)
        som= mt.get("som",0)

        if tam>0:
            st.subheader("Market Size")
            c1,c2,c3= st.columns(3)
            with c1: st.metric("TAM", f"${tam/1e9:.1f}B")
            with c2: st.metric("SAM", f"${sam/1e9:.1f}B")
            with c3: st.metric("SOM", f"${som/1e9:.1f}B")

            fig= go.Funnel(
                y=["TAM","SAM","SOM"],
                x=[tam/1e9, sam/1e9, som/1e9],
                textinfo="value+percent initial",
                marker={"color":["#0D47A1","#1976D2","#64B5F6"]}
            )
            st.plotly_chart(go.Figure(fig),use_container_width=True)
        
        # Market growth
        if mt.get("market_size",[]):
            st.subheader("Market Growth Trend")
            ms= pd.DataFrame(mt["market_size"])
            fig2= px.line(ms, x="year", y="size", title=f"Market Growth (CAGR={mt.get('cagr',0):.1f}%)",
                          labels={"year":"Year","size":"Market Size($B)"})
            st.plotly_chart(fig2,use_container_width=True)

        # Key trends
        if mt.get("trends",[]):
            st.subheader("Key Trends")
            tr_df= pd.DataFrame(mt["trends"])
            fig3= px.bar(tr_df, y="name", x="impact_score",
                         orientation='h',
                         labels={"name":"Trend","impact_score":"Impact"},
                         color="impact_score", color_continuous_scale= px.colors.sequential.Blues,
                         title="Market Trends")
            fig3.update_layout(yaxis_categoryorder='total ascending')
            st.plotly_chart(fig3,use_container_width=True)
            st.markdown("### Trend Descriptions")
            for t in mt["trends"]:
                st.markdown(f"- **{t.get('name','')}** => {t.get('description','')}")
        
        if mt.get("expansion_opportunities",[]):
            st.subheader("Expansion Opportunities")
            for e in mt["expansion_opportunities"]:
                st.markdown(f"- **{e.get('market','')}**: {e.get('opportunity','')}")

    else:
        st.info("No market trend data found.")
    
    st.subheader("Competitive Landscape")
    comps= doc.get("competitors",[])
    if comps:
        st.write(f"Found {len(comps)} sector competitors.")
        cdata= []
        for c in comps:
            cdata.append({
                "name": c.get("name","Unknown"),
                "market_share": c.get("market_share",0)*100,
                "growth_rate": c.get("growth_rate",0)*100,
                "funding": c.get("total_funding",0)/ 1e6,
                "founded": c.get("founded_year",2020),
                "employees": c.get("estimated_employees",0)
            })
        df= pd.DataFrame(cdata)
        fig4= px.scatter(df, x="market_share", y="growth_rate", size="funding",
                         hover_name="name", color="market_share",
                         labels={"market_share":"Market Share (%)","growth_rate":"Growth Rate(%)","funding":"Funding($M)"},
                         title="Competitor Landscape")
        st.plotly_chart(fig4,use_container_width=True)

        # competitive_positioning
        cp= doc.get("competitive_positioning",{})
        if cp:
            st.subheader("Competitive Positioning")
            st.write(f"**Position** => {cp.get('position','N/A')}")
            dims= cp.get("dimensions",[])
            cs= cp.get("company_scores",{})
            avg= cp.get("average_scores",{})
            if dims and cs and avg:
                radar_data= []
                for d in dims:
                    radar_data.append({
                        "Dimension": d,
                        "Your Score": cs.get(d,0),
                        "Industry Median": avg.get(d,0)
                    })
                rdf= pd.DataFrame(radar_data)
                fig5= go.Figure()
                fig5.add_trace(go.Scatterpolar(
                    r=rdf["Your Score"], theta=rdf["Dimension"],
                    fill='toself', name='Your Company'
                ))
                fig5.add_trace(go.Scatterpolar(
                    r=rdf["Industry Median"], theta=rdf["Dimension"],
                    fill='toself', name='Industry Median'
                ))
                fig5.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0,10])
                    ),
                    showlegend=True,
                    title="Company vs Industry"
                )
                st.plotly_chart(fig5,use_container_width=True)

            colA,colB= st.columns(2)
            with colA:
                adv= cp.get("advantages",[])
                if adv:
                    st.markdown("#### Advantages")
                    for a in adv:
                        st.markdown(f"- **{a['dimension']}** => {a['description']}")
            with colB:
                dis= cp.get("disadvantages",[])
                if dis:
                    st.markdown("#### Disadvantages")
                    for d in dis:
                        st.markdown(f"- **{d['dimension']}** => {d['description']}")
    else:
        st.info("No competitor data => run analysis.")

def show_cohort(doc):
    """Cohort analysis results."""
    st.title("Cohort Analysis")
    cdata= doc.get("cohort_data",{})
    if cdata and hasattr(cdata, 'retention'):
        st.subheader("Retention Heatmap")
        ret= cdata.retention
        if isinstance(ret, pd.DataFrame):
            ret_copy= ret.copy()
            ret_copy.index= ret_copy.index.astype(str)
            ret_copy.columns= ret_copy.columns.astype(str)

            fig= go.Figure(data= go.Heatmap(
                z= ret_copy.values,
                x= ret_copy.columns,
                y= ret_copy.index,
                colorscale="Blues",
                text= ret_copy.values,
                texttemplate="%{text:.1f}%",
                hoverongaps=False
            ))
            fig.update_layout(title="Cohort Retention(%)",
                              xaxis_title="Periods",
                              yaxis_title="Cohort")
            st.plotly_chart(fig,use_container_width=True)
        
        # LTV
        if hasattr(cdata, 'ltv'):
            st.subheader("LTV by Cohort")
            ltv= cdata.ltv
            ltv_copy= ltv.copy()
            ltv_copy.index= ltv_copy.index.astype(str)
            ltv_copy.columns= ltv_copy.columns.astype(str)

            fig2= go.Figure()
            for i, cohort in enumerate(ltv_copy.index):
                fig2.add_trace(go.Scatter(
                    x= ltv_copy.columns,
                    y= ltv_copy.iloc[i],
                    mode='lines+markers',
                    name= cohort
                ))
            fig2.update_layout(
                title="LTV Progression by Cohort",
                xaxis_title="Period",
                yaxis_title="Cumulative LTV($)"
            )
            st.plotly_chart(fig2,use_container_width=True)

        summary= getattr(cdata,'summary',{})
        if summary:
            st.subheader("Cohort Summary")
            cA, cB, cC= st.columns(3)
            with cA:
                st.metric("Avg Cohort Growth", f"{summary.get('avg_cohort_growth',0):.1f}%")
            with cB:
                st.metric("Retention Improve", f"{summary.get('retention_improvement',0):.1f}%")
            with cC:
                st.metric("LTV 3mo Trend", f"{summary.get('ltv_3month_trend',0):.1f}%")

            avg_ret= summary.get('avg_retention_by_period')
            if isinstance(avg_ret, pd.Series):
                st.subheader("Average Retention Curve")
                df_ret= pd.DataFrame({
                    "Period": avg_ret.index.astype(str),
                    "Retention": avg_ret.values
                })
                fig3= px.line(df_ret, x='Period', y='Retention',
                              markers=True,
                              title="Retention Curve(Avg)")
                st.plotly_chart(fig3,use_container_width=True)
    else:
        st.info("No cohort data => run analysis.")

def show_network(doc):
    """Network Effects Analysis."""
    st.title("Network Effects")
    net= doc.get("network_analysis",{})
    if net:
        st.subheader("Network Strength")
        ns= net.get("network_strength_score",0)
        fig= go.Figure(go.Indicator(
            mode="gauge+number",
            value= ns,
            title={'text':"Network Strength"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"darkblue"},
                'steps':[
                    {'range':[0,30],'color':"#EF5350"},
                    {'range':[30,70],'color':"#FFCA28"},
                    {'range':[70,100],'color':"#66BB6A"}
                ],
                'threshold':{
                    'line':{'color':"red",'width':4},
                    'thickness':0.75,
                    'value': ns
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig,use_container_width=True)

        # Additional metrics
        c1,c2,c3= st.columns(3)
        with c1: st.metric("Type", net.get("network_type","N/A"))
        with c2: st.metric("Viral Coeff", f"{net.get('viral_coefficient',0):.2f}")
        with c3: st.metric("Defensible", "Yes" if net.get("defensible_network",False) else "No")

        st.subheader("Network Insights")
        colA,colB= st.columns(2)
        with colA:
            st.markdown("#### Predictions")
            for p in net.get("predictions",[]):
                st.markdown(f"- {p}")
        with colB:
            st.markdown("#### Recommendations")
            for r in net.get("recommendations",[]):
                st.markdown(f"- {r}")
    else:
        st.info("No network analysis => run analysis.")

def show_competitive(doc):
    """Dedicated tab for listing raw competitor data."""
    st.title("Competitive Intelligence")
    comps= doc.get("competitors",[])
    if comps:
        st.write(f"**Found {len(comps)} competitors** => see Market & 'competitive_positioning'")
        st.json(comps)
    else:
        st.info("No competitor data => analysis incomplete.")

def show_monte_carlo(doc):
    """Monte Carlo simulation results."""
    st.title("Monte Carlo Risk")
    mcres= doc.get("monte_carlo", None)
    if not mcres:
        st.info("No Monte Carlo data => run analysis.")
        return
    if isinstance(mcres, SimulationResult):
        sp= mcres.success_probability
        st.metric("Success Probability(Avg)", f"{sp:.1f}%")
        # percentiles
        st.subheader("Outcome Percentiles")
        if mcres.percentiles:
            metrics_to_show= ["success_probability","runway","final_users","ltv_cac_ratio"]
            percentiles= mcres.percentiles
            table_data= []
            for m in metrics_to_show:
                if m in percentiles:
                    row={"Metric": m.replace('_',' ').title()}
                    for p in [10,25,50,75,90]:
                        row[f"P{p}"]= percentiles[m].get(p,"N/A")
                    table_data.append(row)
            if table_data:
                st.table(pd.DataFrame(table_data))

        # sensitivity
        if mcres.sensitivity:
            st.subheader("Parameter Sensitivity")
            items= []
            for param, corr in mcres.sensitivity.items():
                items.append({"Parameter": param.title(),"Correlation": corr})
            sdf= pd.DataFrame(items).sort_values("Correlation", key=abs, ascending=False)
            fig= px.bar(sdf, y="Parameter", x="Correlation", orientation='h',
                        color="Correlation",
                        color_continuous_scale= px.colors.diverging.RdBu,
                        range_color=[-1,1],
                        title="Sensitivity => Success Probability")
            st.plotly_chart(fig,use_container_width=True)

        # distribution
        df= mcres.scenarios
        if not df.empty and "success_probability" in df.columns:
            st.subheader("Success Probability Distribution")
            fig2= px.histogram(df, x="success_probability", nbins=20,
                               title="Distribution => success_probability",
                               color_discrete_sequence=["royalblue"])
            mean_val= sp
            fig2.add_vline(x= mean_val, line_dash="dash", line_color="red",
                           annotation_text=f"Mean={mean_val:.1f}%")
            st.plotly_chart(fig2,use_container_width=True)

            other_metrics= ["final_users","runway","ltv_cac_ratio","year3_revenue"]
            choices= [m for m in other_metrics if m in df.columns]
            if choices:
                pick= st.selectbox("View distribution for:", choices)
                if pick in df.columns:
                    st.subheader(f"{pick.title()} Distribution")
                    fig3= px.histogram(df, x=pick, nbins=20,
                                       title=f"Distribution => {pick}",
                                       color_discrete_sequence=["green"])
                    median_val= df[pick].median()
                    fig3.add_vline(x=median_val, line_dash="dash", line_color="red",
                                   annotation_text=f"Median={median_val:,.1f}")
                    st.plotly_chart(fig3,use_container_width=True)
    else:
        if isinstance(mcres,dict):
            sp= mcres.get("success_probability",50)
            st.metric("Average Success Probability", f"{sp:.1f}%")
            st.json(mcres)
        else:
            st.warning("Unknown MonteCarlo format => no HPC synergy references")

def show_tech_dd(doc):
    """Technical due diligence."""
    st.title("Technical Due Diligence")
    tch= doc.get("tech_assessment", None)
    if isinstance(tch, TechnicalAssessment):
        st.subheader("Scores")
        c1,c2,c3,c4= st.columns(4)
        with c1: st.metric("Overall", f"{tch.overall_score*100:.1f}/100")
        with c2: st.metric("Architecture", f"{tch.architecture_score*100:.1f}/100")
        with c3: st.metric("Scalability", f"{tch.scalability_score*100:.1f}/100")
        with c4: st.metric("Tech Debt", f"{tch.tech_debt_score*100:.1f}/100")

        if tch.tech_stack:
            st.subheader("Tech Stack")
            stack_data= []
            for item in tch.tech_stack:
                stack_data.append({
                    "Name": item.name,
                    "Category": item.category,
                    "Maturity": item.maturity,
                    "Scalability": item.scalability,
                    "Adoption": item.market_adoption,
                    "Expertise": item.expertise_required
                })
            df= pd.DataFrame(stack_data)
            fig= go.Figure()
            metrics= ["Maturity","Scalability","Adoption","Expertise"]
            for m in metrics:
                fig.add_trace(go.Bar(x=df["Name"], y=df[m], name=m))
            fig.update_layout(barmode='group', title="Stack Metrics", xaxis_title="Tech", yaxis_title="Score(0..1)")
            st.plotly_chart(fig,use_container_width=True)

        risk= tch.risk_assessment
        if risk:
            st.subheader("Risk Assessment")
            rd= []
            for rt, rv in risk.items():
                if rt!="overall_risk":
                    rd.append({"Risk Type": rt.replace('_',' ').title(),"Level": rv})
            rdf= pd.DataFrame(rd)
            fig2= px.bar(rdf, y="Risk Type", x="Level",
                         orientation='h',
                         color="Level", color_continuous_scale= px.colors.sequential.Reds,
                         title="Tech Risk(0..1)")
            st.plotly_chart(fig2,use_container_width=True)
            orisk= risk.get("overall_risk",0.5)
            st.metric("Overall Risk", f"{orisk*100:.1f}%")

        colA,colB= st.columns(2)
        with colA:
            st.subheader("Strengths")
            for s in tch.strengths:
                st.markdown(f"- {s}")
        with colB:
            st.subheader("Weaknesses")
            for w in tch.weaknesses:
                st.markdown(f"- {w}")

        st.subheader("Recommendations")
        for r in tch.recommendations:
            st.markdown(f"- {r}")
    elif isinstance(tch, dict):
        st.json(tch)
    else:
        st.info("No technical due diligence => run analysis.")

def show_pmf(doc):
    """Product-Market Fit analysis."""
    st.title("Product-Market Fit")
    pmfa= doc.get("pmf_analysis", None)
    if isinstance(pmfa, PMFMetrics):
        st.subheader("PMF Score")
        fig= go.Figure(go.Indicator(
            mode="gauge+number",
            value= pmfa.score,
            title={'text': f"Stage => {pmfa.stage}"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"darkblue"},
                'steps':[
                    {'range':[0,40],'color':"#ffcccb"},
                    {'range':[40,65],'color':"#ffffe0"},
                    {'range':[65,80],'color':"#90ee90"},
                    {'range':[80,100],'color':"#32cd32"}
                ],
                'threshold': {
                    'line':{'color':"red",'width':4},
                    'thickness':0.75,
                    'value': pmfa.score
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig,use_container_width=True)

        data= [
            {"Component":"Retention","Score": pmfa.retention_score},
            {"Component":"Engagement","Score": pmfa.engagement_score},
            {"Component":"Growth","Score": pmfa.growth_score},
            {"Component":"NPS","Score": pmfa.nps_score},
            {"Component":"Qualitative","Score": pmfa.qualitative_score}
        ]
        df= pd.DataFrame(data)
        fig2= px.bar(df, x="Score", y="Component",
                     orientation='h',
                     color="Score", color_continuous_scale= px.colors.sequential.Viridis,
                     range_x=[0,100],
                     title="PMF Components")
        st.plotly_chart(fig2,use_container_width=True)

        colA,colB= st.columns(2)
        with colA:
            st.subheader("Strengths")
            for s in pmfa.strengths:
                st.markdown(f"- {s}")
        with colB:
            st.subheader("Weaknesses")
            for w in pmfa.weaknesses:
                st.markdown(f"- {w}")

        st.subheader("Recommendations")
        for r in pmfa.recommendations:
            st.markdown(f"- {r}")
    elif isinstance(pmfa, dict):
        st.json(pmfa)
    else:
        st.info("No PMF => run analysis.")

def show_pitch_sentiment(doc):
    """Pitch deck sentiment analysis."""
    st.title("Pitch Sentiment Analysis")
    ps= doc.get("pitch_sentiment", {})
    if ps:
        if "overall_sentiment" in ps and isinstance(ps["overall_sentiment"], dict):
            overall= ps["overall_sentiment"]
            score= overall.get("score",0.0)
            category= overall.get("category","neutral")
            fig= go.Figure(go.Indicator(
                mode="gauge+number",
                value= score,
                title={'text': f"Overall => {category}"},
                gauge={
                    'axis':{'range':[-1,1]},
                    'bar':{'color':"royalblue"},
                    'steps':[
                        {'range':[-1,-0.3],'color':"firebrick"},
                        {'range':[-0.3,0.3],'color':"gold"},
                        {'range':[0.3,1],'color':"forestgreen"}
                    ],
                    'threshold': {
                        'line':{'color':"black",'width':4},
                        'thickness':0.75,
                        'value': score
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig,use_container_width=True)

        cat_sents= ps.get("category_sentiments",{})
        if cat_sents:
            st.subheader("Category Sentiments")
            items= []
            for cat, val in cat_sents.items():
                items.append({"Category": cat, "Score": val.get("score",0.0)})
            df= pd.DataFrame(items)
            fig2= px.bar(df, x="Score", y="Category", orientation='h',
                         range_x=[-1,1],
                         title="Sentiment by Category")
            st.plotly_chart(fig2,use_container_width=True)

        if ps.get("improvement_suggestions",[]):
            st.subheader("Improvements")
            for sug in ps["improvement_suggestions"]:
                st.markdown(f"- {sug}")
    else:
        st.info("No pitch sentiment => add pitch deck & re-run.")

def show_benchmarking(doc):
    """Industry benchmark comparisons."""
    st.title("Benchmarking vs Industry")
    bench= doc.get("benchmarks", None)
    if isinstance(bench, BenchmarkResult):
        st.markdown(f"### {bench.performance_summary}")
        st.subheader("Your Metrics vs Industry")
        c_m= bench.company_metrics
        i_b= bench.industry_benchmarks
        perc= bench.percentiles

        metric_names= list(c_m.keys())
        cvals= list(c_m.values())
        industry_median= []
        for m in metric_names:
            if m in i_b and 'p50' in i_b[m]:
                industry_median.append(i_b[m]['p50'])
            else:
                industry_median.append(0)

        fig= go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=cvals, theta= metric_names,
            fill='toself', name='Company'
        ))
        fig.add_trace(go.Scatterpolar(
            r=industry_median, theta= metric_names,
            fill='toself', name='Industry Median'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            showlegend=True,
            title="Company vs Industry Median"
        )
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Percentile Rankings")
        items= []
        for m,pval in perc.items():
            items.append({"Metric": m.replace('_',' ').title(),"Percentile": pval})
        pdf= pd.DataFrame(items)
        fig2= px.bar(pdf, x="Percentile", y="Metric", orientation='h',
                     range_x=[0,100],
                     title="Percentile vs Industry",
                     color="Percentile", color_continuous_scale="Blues")
        fig2.add_vline(x=50, line_dash="dash", line_color="red")
        st.plotly_chart(fig2,use_container_width=True)

        st.subheader("Recommendations")
        for r in bench.recommendations:
            st.markdown(f"- {r}")
    elif isinstance(bench,dict):
        st.json(bench)
    else:
        st.info("No benchmark => run analysis.")

def show_patterns(doc):
    """Detected patterns & insights."""
    st.title("Patterns & Insights")
    pats= doc.get("patterns_matched",[])
    if pats:
        st.subheader("Detected Patterns")
        cols= st.columns(2)
        for i,p in enumerate(pats):
            idx= i%2
            with cols[idx]:
                st.markdown(f"""
                <div style="padding:15px; border-radius:5px; background-color:#e8f4f8; margin-bottom:15px;">
                    <h4 style="margin:0; color:#2471A3;">✅ {p.get('name','Pattern')}</h4>
                    <p style="margin-bottom:0;">{p.get('description','')}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No patterns matched.")
    st.subheader("Pattern Insights")
    pi= doc.get("pattern_insights",[])
    if pi:
        for ins in pi:
            st.markdown(f"""
            <div style="padding:15px; border-radius:5px; background-color:#FEF9E7; margin-bottom:15px; border-left:4px solid #F1C40F;">
                <p style="margin:0;">{ins}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No pattern insights.")

def show_report(doc):
    """Generate PDF report."""
    st.title("Export Report")
    c1,c2= st.columns(2)
    with c1:
        incl_scen= st.checkbox("Include Scenario Analysis", True)
        incl_pats= st.checkbox("Include Patterns", True)
        incl_growth= st.checkbox("Include Growth Projections", True)
        incl_recs= st.checkbox("Include Recommendations", True)
    with c2:
        incl_fin= st.checkbox("Include Financials", True)
        incl_market= st.checkbox("Include Market Analysis", True)
        incl_team= st.checkbox("Include Team Analysis", True)
        incl_tech= st.checkbox("Include Technical Analysis", True)

    st.subheader("Report Style Options")
    report_style= st.radio("Report Style", ["Standard","Visual (with charts)","Comprehensive"], horizontal=True)
    include_appendix= st.checkbox("Include Data Appendix", False)

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF..."):
            try:
                pdf_data= generate_investor_report(
                    doc,
                    system_dyn= doc.get("system_dynamics",[]),
                    sir_data= doc.get("sir_data", None),
                    hpc_data= doc.get("hpc_data",[] if incl_scen else None) if incl_scen else None,
                    patterns_matched= doc.get("patterns_matched",[] if incl_pats else None),
                    cash_flow= doc.get("cash_flow",[]),
                    unit_economics= doc.get("unit_economics",{}),
                    include_recommendations= incl_recs,
                    include_financials= incl_fin,
                    include_market= incl_market,
                    include_team= incl_team,
                    include_tech= incl_tech,
                    include_appendix= include_appendix
                )
                fname= f"FlashDNA_{doc.get('name','Startup')}_{datetime.now().strftime('%Y%m%d')}.pdf"
                st.download_button("Download PDF Report", data=pdf_data, file_name=fname, mime="application/pdf")
                st.success(f"Report generated successfully: {fname}")
            except Exception as e:
                st.error(f"Error generating report => {str(e)}")

######################################################
# 6) Main App
######################################################

def main():
    """Main Streamlit app entry point."""
    logo= setup_page()
    apply_tab_scroll_style()
    initialize_session()
    display_header(logo)
    render_sidebar_input()

    # if user clicked analyze
    if st.session_state.analyze_clicked:
        doc= st.session_state.doc
        doc= run_analysis(doc)
        st.session_state.doc= doc
        st.session_state.analyzed= True
        st.session_state.analyze_clicked= False

    if st.session_state.analyzed:
        doc= st.session_state.doc
        tab_names= [
            "Executive Overview",
            "User Growth",
            "Scenario Analysis",
            "Financial Health",
            "Team & Execution",
            "Market & Sector",
            "Cohort Analysis",
            "Network Effects",
            "Competitive Intel",
            "Monte Carlo Risk",
            "Technical Diligence",
            "PMF",
            "Pitch Sentiment",
            "Benchmark",
            "Patterns & Insights",
            "Export Report"
        ]
        tabs= st.tabs(tab_names)

        with tabs[0]:
            show_overview(doc)
        with tabs[1]:
            show_growth(doc)
        with tabs[2]:
            show_scenarios(doc)
        with tabs[3]:
            show_financial(doc)
        with tabs[4]:
            show_team(doc)
        with tabs[5]:
            show_market(doc)
        with tabs[6]:
            show_cohort(doc)
        with tabs[7]:
            show_network(doc)
        with tabs[8]:
            show_competitive(doc)
        with tabs[9]:
            show_monte_carlo(doc)
        with tabs[10]:
            show_tech_dd(doc)
        with tabs[11]:
            show_pmf(doc)
        with tabs[12]:
            show_pitch_sentiment(doc)
        with tabs[13]:
            show_benchmarking(doc)
        with tabs[14]:
            show_patterns(doc)
        with tabs[15]:
            show_report(doc)
    else:
        st.markdown("""
        # Welcome to FlashDNA Infinity

        This advanced platform analyzes startups across multiple dimensions:
        - **Success Prediction** with ML
        - **Growth Projections** (system dynamics, viral)
        - **Financial analysis** (runway, unit economics)
        - **Team & Execution** risk
        - **Market & Competitive** intelligence
        - **Technical Diligence**
        - **Pitch analysis** with LLM intangible rating (via PDF extraction)
        - **Benchmarking** vs. industry
        - **PDF Report** generator

        Fill out the **sidebar** and click "Analyze Startup => Run Analysis" to begin.
        """)

if __name__=="__main__":
    main()
