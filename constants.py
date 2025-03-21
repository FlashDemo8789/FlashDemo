from typing import Dict

# HPC synergy BFS–free + “NEW(UI)” => large list of metrics
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

# HPC synergy BFS–free references
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

# Example reference for PMF stages used in product_market_fit
PMF_STAGES = {
    "pre-PMF": {"min": 0, "max": 50, "color": "#f44336", "label": "Pre-PMF"},
    "early-PMF": {"min": 50, "max": 65, "color": "#ff9800", "label": "Early PMF"},
    "PMF": {"min": 65, "max": 80, "color": "#8bc34a", "label": "Product-Market Fit"},
    "scaling": {"min": 80, "max": 100, "color": "#4caf50", "label": "Scaling"}
}
