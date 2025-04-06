from typing import Dict, List

# CAMP Framework categories for metrics
CAMP_CATEGORIES = {
    "capital": {
        "name": "Capital Efficiency",
        "weight": 0.30,
        "description": "Financial health, sustainability, and capital deployment efficiency",
        "metrics": [
            "monthly_revenue",
            "annual_recurring_revenue",
            "lifetime_value_ltv",
            "gross_margin_percent",
            "operating_margin_percent",
            "burn_rate",
            "runway_months",
            "cash_on_hand_million",
            "debt_ratio",
            "ltv_cac_ratio",
            "customer_acquisition_cost",
            "avg_revenue_per_user"
        ]
    },
    "market": {
        "name": "Market Dynamics",
        "weight": 0.25,
        "description": "Market opportunity, growth potential, and competitive positioning",
        "metrics": [
            "market_size",
            "market_growth_rate",
            "market_share",
            "user_growth_rate",
            "viral_coefficient",
            "category_leadership_score",
            "revenue_growth_rate",
            "net_retention_rate",
            "churn_rate",
            "community_growth_rate",
            "referral_rate",
            "current_users",
            "conversion_rate"
        ]
    },
    "advantage": {
        "name": "Advantage Moat",
        "weight": 0.20,
        "description": "Competitive advantages and defensibility",
        "metrics": [
            "patent_count",
            "technical_innovation_score",
            "product_maturity_score",
            "api_integrations_count",
            "scalability_score",
            "technical_debt_score",
            "product_security_score",
            "business_model_strength",
            "category_leadership_score",
            "data_moat_strength",
            "algorithm_uniqueness",
            "moat_score"
        ]
    },
    "people": {
        "name": "People & Performance",
        "weight": 0.25,
        "description": "Team strength, experience, and execution ability",
        "metrics": [
            "founder_exits",
            "founder_domain_exp_yrs",
            "employee_count",
            "founder_diversity_score",
            "management_satisfaction_score",
            "tech_talent_ratio",
            "has_cto",
            "has_cmo",
            "has_cfo",
            "employee_turnover_rate",
            "nps_score",
            "support_ticket_sla_percent",
            "support_ticket_volume",
            "team_score"
        ]
    },
    "intangible": {
        "name": "Intangible Factor",
        "weight": "±0.20",
        "description": "AI-derived assessment of qualitative factors from pitch materials",
        "metrics": [
            "intangible",
            "pitch_sentiment"
        ]
    }
}

# Original named metrics list (preserved for backward compatibility)
NAMED_METRICS_50 = [
    "monthly_revenue",
    "annual_recurring_revenue",
    "lifetime_value_ltv",
    "gross_margin_percent",
    "operating_margin_percent",
    "burn_rate",
    "runway_months",
    "cash_on_hand_million",
    "debt_ratio",
    "financing_round_count",

    "monthly_active_users",
    "daily_active_users",
    "user_growth_rate",
    "churn_rate",
    "churn_cohort_6mo",
    "activation_rate",
    "conversion_rate",
    "repeat_purchase_rate",
    "referral_rate",
    "session_frequency",

    "product_maturity_score",
    "product_security_score",
    "technical_debt_score",
    "release_frequency",
    "api_integrations_count",
    "uptime_percent",
    "scalability_score",
    "esg_sustainability_score",
    "patent_count",
    "technical_innovation_score",

    "employee_count",
    "employee_turnover_rate",
    "founder_diversity_score",
    "management_satisfaction_score",
    "tech_talent_ratio",
    "founder_exits",
    "founder_domain_exp_yrs",
    "founder_network_reach",
    "board_experience_score",
    "hiring_velocity_score",

    "nps_score",
    "customer_acquisition_cost",
    "roi_on_ad_spend",
    "lead_conversion_percent",
    "organic_traffic_share",
    "channel_partner_count",
    "support_ticket_volume",
    "support_ticket_sla_percent",
    "investor_interest_score",
    "category_leadership_score",

    "business_model_strength",
    "market_size",
    "market_growth_rate",
    "market_share",
    "viral_coefficient",
    "revenue_growth_rate",
    "net_retention_rate",
    "community_growth_rate",
    "upsell_rate",
    "ltv_cac_ratio",

    "default_rate",
    "licenses_count",
    "clinical_phase"
]

# Business models list
BUSINESS_MODELS = [
    "SaaS",
    "Marketplace",
    "E-commerce",
    "Consumer",
    "Enterprise",
    "Hardware",
    "Biotech",
    "Fintech",
    "AI/ML",
    "Crypto/Blockchain",
    "Media",
    "EdTech"
]

# Startup stages
STARTUP_STAGES = [
    "Pre-seed",
    "Seed",
    "Series A",
    "Series B",
    "Series C",
    "Series D+",
    "Growth",
    "Pre-IPO"
]

# PMF stages reference
PMF_STAGES = {
    "pre-PMF": {"min": 0, "max": 50, "color": "#f44336", "label": "Pre-PMF"},
    "early-PMF": {"min": 50, "max": 65, "color": "#ff9800", "label": "Early PMF"},
    "PMF": {"min": 65, "max": 80, "color": "#8bc34a", "label": "Product-Market Fit"},
    "scaling": {"min": 80, "max": 100, "color": "#4caf50", "label": "Scaling"}
}

# CAMP scoring thresholds
CAMP_SCORE_BANDS = {
    "outstanding": {"min": 80, "max": 100, "color": "#4caf50", "label": "Outstanding"},
    "strong": {"min": 65, "max": 80, "color": "#8bc34a", "label": "Strong"},
    "average": {"min": 50, "max": 65, "color": "#ff9800", "label": "Average"},
    "below-average": {"min": 0, "max": 50, "color": "#f44336", "label": "Below Average"}
}
