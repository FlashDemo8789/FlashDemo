from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Core pattern definitions with criteria, descriptions, and recommendations
PATTERNS = [
    {
        "name": "Experienced Founding Team",
        "criteria": [
            {"metric": "founder_domain_exp_yrs", "op": ">", "value": 5},
            {"metric": "founder_exits", "op": ">", "value": 0}
        ],
        "description": "Founding team with >5 years domain experience and a prior exit",
        "recommendation": "Leverage founder experience in pitch materials and investor discussions",
        "impact": "high",
        "sector_relevance": ["all"]
    },
    {
        "name": "High Referral Growth",
        "criteria": [
            {"metric": "referral_rate", "op": ">", "value": 0.03}
        ],
        "description": "Referral rate above 3% => strong WOM network effect",
        "recommendation": "Invest in referral programs and virality mechanisms",
        "impact": "high",
        "sector_relevance": ["all"]
    },
    {
        "name": "Large Market & Growth",
        "criteria": [
            {"metric": "market_size", "op": ">", "value": 50000000},
            {"metric": "market_growth_rate", "op": ">", "value": 0.1}
        ],
        "description": "Big market (>50M) with >10% annual growth => robust potential",
        "recommendation": "Focus marketing on high-growth market segments",
        "impact": "medium",
        "sector_relevance": ["all"]
    },
    {
        "name": "Efficient Go-To-Market",
        "criteria": [
            {"metric": "ltv_cac_ratio", "op": ">", "value": 3},
            {"metric": "customer_acquisition_cost", "op": "<", "value": 5000}
        ],
        "description": "Healthy LTV:CAC ratio with moderate CAC => efficient GTM",
        "recommendation": "Double down on successful acquisition channels",
        "impact": "high",
        "sector_relevance": ["all"]
    },
    {
        "name": "Cash Efficient",
        "criteria": [
            {"metric": "runway_months", "op": ">", "value": 18},
            {"metric": "burn_rate", "op": "<", "value": 100000}
        ],
        "description": "Long runway + controlled burn => strong capital efficiency",
        "recommendation": "Highlight capital efficiency in investor presentations",
        "impact": "medium",
        "sector_relevance": ["all"]
    },
    # SaaS specific patterns
    {
        "name": "SaaS Quick Payback",
        "criteria": [
            {"metric": "cac_payback_months", "op": "<", "value": 12},
            {"metric": "gross_margin_percent", "op": ">", "value": 70}
        ],
        "description": "CAC payback <12 months with healthy margins => unit economics favoring rapid scaling",
        "recommendation": "Accelerate customer acquisition spending with confidence in quick returns",
        "impact": "high",
        "sector_relevance": ["saas", "software"]
    },
    {
        "name": "SaaS Net Expansion",
        "criteria": [
            {"metric": "net_retention_rate", "op": ">", "value": 1.1},
            {"metric": "upsell_rate", "op": ">", "value": 0.1}
        ],
        "description": "Net retention >110% with upsell opportunities => negative net churn",
        "recommendation": "Develop strong customer success and upsell programs",
        "impact": "high",
        "sector_relevance": ["saas", "software"]
    },
    # Marketplace specific patterns
    {
        "name": "Marketplace Liquidity",
        "criteria": [
            {"metric": "liquidity_score", "op": ">", "value": 50},
            {"metric": "monthly_active_users", "op": ">", "value": 10000}
        ],
        "description": "Strong marketplace liquidity with growing user base => network effects taking hold",
        "recommendation": "Focus on balancing supply and demand sides of marketplace",
        "impact": "high",
        "sector_relevance": ["marketplace", "platform"]
    },
    # Fintech specific patterns
    {
        "name": "Fintech Regulation Ready",
        "criteria": [
            {"metric": "licenses_count", "op": ">", "value": 2},
            {"metric": "compliance_index", "op": ">", "value": 70}
        ],
        "description": "Well-prepared for regulatory requirements with necessary licenses",
        "recommendation": "Highlight regulatory readiness in investor presentations",
        "impact": "medium",
        "sector_relevance": ["fintech", "banking"]
    },
    {
        "name": "Fintech Risk Management",
        "criteria": [
            {"metric": "default_rate", "op": "<", "value": 0.02},
            {"metric": "fraud_risk_factor", "op": "<", "value": 0.03}
        ],
        "description": "Strong risk management with low default and fraud rates",
        "recommendation": "Emphasize risk management capabilities in investor discussions",
        "impact": "high",
        "sector_relevance": ["fintech", "banking"]
    },
    # AI/ML specific patterns
    {
        "name": "AI Data Advantage",
        "criteria": [
            {"metric": "data_moat_strength", "op": ">", "value": 70},
            {"metric": "data_volume_tb", "op": ">", "value": 10}
        ],
        "description": "Strong data advantage with significant proprietary data assets",
        "recommendation": "Protect and expand proprietary data collection",
        "impact": "high",
        "sector_relevance": ["ai", "ml"]
    },
    {
        "name": "AI IP Protection",
        "criteria": [
            {"metric": "patent_count", "op": ">", "value": 2},
            {"metric": "technical_innovation_score", "op": ">", "value": 70}
        ],
        "description": "Protected AI innovations with strong IP portfolio",
        "recommendation": "Continue patenting core algorithmic innovations",
        "impact": "medium",
        "sector_relevance": ["ai", "ml"]
    },
    # Biotech specific patterns
    {
        "name": "Biotech Clinical Progress",
        "criteria": [
            {"metric": "clinical_phase", "op": ">", "value": 1}
        ],
        "description": "Advanced clinical phase reducing development risk",
        "recommendation": "Focus resources on advancing clinical trials",
        "impact": "high",
        "sector_relevance": ["biotech", "healthtech"]
    },
    # E-commerce specific patterns
    {
        "name": "E-commerce Repeat Purchases",
        "criteria": [
            {"metric": "repeat_purchase_rate", "op": ">", "value": 0.3}
        ],
        "description": "Strong customer loyalty with high repeat purchase rate",
        "recommendation": "Develop retention marketing programs",
        "impact": "high",
        "sector_relevance": ["ecommerce", "retail"]
    },
    # Crypto/Blockchain specific patterns
    {
        "name": "Crypto Token Utility",
        "criteria": [
            {"metric": "token_utility_score", "op": ">", "value": 70},
            {"metric": "decentralization_factor", "op": ">", "value": 0.6}
        ],
        "description": "High token utility with genuine decentralization",
        "recommendation": "Focus on expanding token use cases within ecosystem",
        "impact": "high",
        "sector_relevance": ["crypto", "blockchain"]
    },
    # Team quality patterns
    {
        "name": "Strong Technical Leadership",
        "criteria": [
            {"metric": "tech_talent_ratio", "op": ">", "value": 0.4},
            {"metric": "has_cto", "op": "==", "value": True}
        ],
        "description": "Strong technical leadership with experienced CTO",
        "recommendation": "Highlight technical leadership in investor discussions",
        "impact": "medium",
        "sector_relevance": ["all"]
    },
    {
        "name": "Diverse Founding Team",
        "criteria": [
            {"metric": "founder_diversity_score", "op": ">", "value": 70}
        ],
        "description": "Diverse founding team with varied perspectives",
        "recommendation": "Leverage diversity as a strength in hiring and decision-making",
        "impact": "medium",
        "sector_relevance": ["all"]
    },
    # Growth patterns
    {
        "name": "Viral Growth",
        "criteria": [
            {"metric": "viral_coefficient", "op": ">", "value": 1.0}
        ],
        "description": "Viral coefficient >1 indicating exponential organic growth potential",
        "recommendation": "Optimize onboarding for virality and word-of-mouth",
        "impact": "high",
        "sector_relevance": ["all"]
    },
    {
        "name": "Rapid User Growth",
        "criteria": [
            {"metric": "user_growth_rate", "op": ">", "value": 0.15}
        ],
        "description": "User base growing >15% monthly indicating product-market fit",
        "recommendation": "Ensure infrastructure can scale with rapid growth",
        "impact": "high",
        "sector_relevance": ["all"]
    }
]

def meets_criterion(value, operator, threshold):
    """Check if a value meets a criterion based on the specified operator."""
    if operator == ">": return value > threshold
    elif operator == ">=": return value >= threshold
    elif operator == "<": return value < threshold
    elif operator == "<=": return value <= threshold
    elif operator == "==": return value == threshold
    return False

def detect_patterns(doc: dict) -> List[dict]:
    """
    Detect which patterns match the startup data.
    
    Args:
        doc: Dictionary containing startup metrics and data
        
    Returns:
        List of matched pattern dictionaries
    """
    matched = []
    
    # Get sector for sector-specific patterns
    sector = doc.get("sector", "").lower()
    
    # Check each pattern
    for pattern in PATTERNS:
        # Skip sector-specific patterns that don't match this startup's sector
        sector_relevance = pattern.get("sector_relevance", ["all"])
        if "all" not in sector_relevance and sector not in sector_relevance:
            continue
            
        # Check if all criteria are met
        all_criteria_met = True
        for criterion in pattern["criteria"]:
            metric = criterion["metric"]
            operator = criterion["op"]
            threshold = criterion["value"]
            
            # Get actual value from doc
            actual_value = doc.get(metric, None)
            
            # Skip if metric is missing
            if actual_value is None:
                all_criteria_met = False
                break
                
            # Check if criterion is met
            if not meets_criterion(actual_value, operator, threshold):
                all_criteria_met = False
                break
                
        # If all criteria are met, add pattern to matched list
        if all_criteria_met:
            matched.append(pattern)
    
    return matched

def get_unmatched_patterns(doc: dict, matched_patterns: List[dict]) -> List[dict]:
    """
    Get patterns that were not matched but are relevant to the startup's sector.
    
    Args:
        doc: Dictionary containing startup metrics and data
        matched_patterns: List of patterns that were already matched
        
    Returns:
        List of unmatched but relevant pattern dictionaries
    """
    sector = doc.get("sector", "").lower()
    matched_names = [p["name"] for p in matched_patterns]
    
    unmatched = []
    for pattern in PATTERNS:
        # Skip if already matched
        if pattern["name"] in matched_names:
            continue
            
        # Check if relevant to this sector
        sector_relevance = pattern.get("sector_relevance", ["all"])
        if "all" in sector_relevance or sector in sector_relevance:
            unmatched.append(pattern)
    
    return unmatched

def analyze_missing_criteria(doc: dict, pattern: dict) -> List[dict]:
    """
    Analyze which specific criteria are missing for an unmatched pattern.
    
    Args:
        doc: Dictionary containing startup metrics and data
        pattern: Pattern dictionary to analyze
        
    Returns:
        List of missing criteria with details
    """
    missing_criteria = []
    
    for criterion in pattern["criteria"]:
        metric = criterion["metric"]
        operator = criterion["op"]
        threshold = criterion["value"]
        
        # Get actual value from doc
        actual_value = doc.get(metric, None)
        
        # Check if metric is missing or criterion is not met
        if actual_value is None:
            missing_criteria.append({
                "metric": metric,
                "issue": "missing",
                "recommendation": f"Start tracking {metric}"
            })
        elif not meets_criterion(actual_value, operator, threshold):
            # Calculate gap to threshold
            if operator in [">", ">="]:
                gap = threshold - actual_value
                missing_criteria.append({
                    "metric": metric,
                    "issue": "below_threshold",
                    "actual": actual_value,
                    "threshold": threshold,
                    "gap": gap,
                    "recommendation": f"Increase {metric} by at least {gap}"
                })
            elif operator in ["<", "<="]:
                gap = actual_value - threshold
                missing_criteria.append({
                    "metric": metric,
                    "issue": "above_threshold",
                    "actual": actual_value,
                    "threshold": threshold,
                    "gap": gap,
                    "recommendation": f"Decrease {metric} by at least {gap}"
                })
            else:
                missing_criteria.append({
                    "metric": metric,
                    "issue": "not_equal",
                    "actual": actual_value,
                    "threshold": threshold,
                    "recommendation": f"Change {metric} to match {threshold}"
                })
    
    return missing_criteria

def generate_pattern_insights(matched_patterns: List[dict]) -> List[str]:
    """
    Generate insights based on matched patterns.
    
    Args:
        matched_patterns: List of matched pattern dictionaries
        
    Returns:
        List of insight strings
    """
    insights = []
    
    if not matched_patterns:
        insights.append("No strong patterns detected. Focus on improving key metrics to establish pattern matches.")
        return insights
    
    # Get pattern names
    names = [p["name"] for p in matched_patterns]
    
    # Generate insight for multiple patterns
    if len(names) >= 3:
        high_impact = [p for p in matched_patterns if p.get("impact", "") == "high"]
        if high_impact:
            high_impact_names = [p["name"] for p in high_impact]
            insights.append(f"Multiple high-impact patterns detected: {', '.join(high_impact_names)} => robust fundamentals with strong growth potential.")
        else:
            insights.append(f"Multiple patterns detected: {', '.join(names[:3])} and more => solid foundation with diverse strengths.")
    elif len(names) == 2:
        insights.append(f"Two complementary patterns detected: {names[0]} and {names[1]} => building momentum with clear strengths.")
    elif len(names) == 1:
        insights.append(f"One key pattern detected: {names[0]} => build on this strength while developing other areas.")
    
    # Generate insights for specific pattern combinations
    if any(p["name"] == "High Referral Growth" for p in matched_patterns) and \
       any(p["name"] == "Large Market & Growth" for p in matched_patterns):
        insights.append("High referral combined with large market => strong possibility of viral expansion in a big TAM.")
    
    if any(p["name"] == "Experienced Founding Team" for p in matched_patterns) and \
       any(p["name"] == "Cash Efficient" for p in matched_patterns):
        insights.append("Experienced team with cash efficiency => strong execution capability with runway to prove model.")
    
    if any(p["name"] == "Efficient Go-To-Market" for p in matched_patterns) and \
       any(p["name"] == "Rapid User Growth" for p in matched_patterns):
        insights.append("Efficient GTM with rapid growth => scalable acquisition model is working, potential for capital-efficient growth.")
    
    if any(p["name"] == "SaaS Net Expansion" for p in matched_patterns) and \
       any(p["name"] == "SaaS Quick Payback" for p in matched_patterns):
        insights.append("SaaS quick payback with net expansion => exceptional unit economics favoring aggressive growth investment.")
    
    # Add sector-specific insights
    sector_patterns = {
        "saas": ["SaaS Net Expansion", "SaaS Quick Payback"],
        "marketplace": ["Marketplace Liquidity"],
        "fintech": ["Fintech Regulation Ready", "Fintech Risk Management"],
        "ai": ["AI Data Advantage", "AI IP Protection"],
        "biotech": ["Biotech Clinical Progress"],
        "ecommerce": ["E-commerce Repeat Purchases"],
        "crypto": ["Crypto Token Utility"]
    }
    
    for sector, patterns in sector_patterns.items():
        sector_matched = [p["name"] for p in matched_patterns if p["name"] in patterns]
        if sector_matched:
            insights.append(f"Strong {sector.capitalize()} patterns: {', '.join(sector_matched)} => competitive advantage in core {sector} success factors.")
    
    return insights

def generate_pattern_recommendations(doc: dict, matched_patterns: List[dict], unmatched_patterns: List[dict] = None) -> List[dict]:
    """
    Generate actionable recommendations based on matched and unmatched patterns.
    
    Args:
        doc: Dictionary containing startup metrics and data
        matched_patterns: List of matched pattern dictionaries
        unmatched_patterns: List of unmatched pattern dictionaries (optional)
        
    Returns:
        List of recommendation dictionaries with priority levels
    """
    recommendations = []
    
    # First, add recommendations from matched patterns
    for pattern in matched_patterns:
        if "recommendation" in pattern:
            recommendations.append({
                "text": pattern["recommendation"],
                "source": f"Based on '{pattern['name']}' pattern",
                "priority": "medium"  # Build on strengths
            })
    
    # If unmatched patterns provided, analyze them
    if unmatched_patterns is None:
        unmatched_patterns = get_unmatched_patterns(doc, matched_patterns)
    
    # Focus on high-impact unmatched patterns
    high_impact_unmatched = [p for p in unmatched_patterns if p.get("impact", "") == "high"]
    
    # Limit to top 3 high-impact unmatched patterns
    for pattern in high_impact_unmatched[:3]:
        # Analyze missing criteria
        missing = analyze_missing_criteria(doc, pattern)
        
        # Add recommendations for each missing criterion
        for criterion in missing:
            if "recommendation" in criterion:
                recommendations.append({
                    "text": criterion["recommendation"],
                    "source": f"To achieve '{pattern['name']}' pattern",
                    "priority": "high"  # Address gaps in high-impact patterns
                })
    
    # Add general recommendations based on startup stage
    stage = doc.get("stage", "").lower()
    sector = doc.get("sector", "").lower()
    
    stage_recommendations = {
        "pre-seed": [
            "Focus on product-market fit before scaling growth",
            "Minimize burn rate while validating core assumptions",
            "Build minimum viable product with core functionality only"
        ],
        "seed": [
            "Establish repeatable sales process with clear unit economics",
            "Focus on a narrow market segment for initial traction",
            "Optimize onboarding to improve activation metrics"
        ],
        "series-a": [
            "Build scalable acquisition channels with predictable CAC",
            "Develop customer success function to improve retention",
            "Standardize core processes for team scale-up"
        ],
        "series-b": [
            "Optimize unit economics to demonstrate path to profitability",
            "Expand to adjacent market segments strategically",
            "Strengthen management team for scale"
        ],
        "growth": [
            "Focus on operational efficiency and margin improvement",
            "Expand internationally in high-potential markets",
            "Develop secondary revenue streams"
        ]
    }
    
    if stage in stage_recommendations:
        for rec in stage_recommendations[stage][:2]:  # Add top 2 recommendations for stage
            recommendations.append({
                "text": rec,
                "source": f"Based on {stage} stage best practices",
                "priority": "medium"
            })
    
    # Add sector-specific recommendations
    sector_recommendations = {
        "saas": [
            "Focus on reducing churn through improved product experience",
            "Develop strong customer success and expansion revenue motion",
            "Optimize pricing tiers for different customer segments"
        ],
        "marketplace": [
            "Focus on liquidity in core segments before expanding",
            "Balance supply and demand sides of marketplace",
            "Reduce transaction friction to improve conversion"
        ],
        "fintech": [
            "Secure necessary regulatory approvals early",
            "Implement robust risk management and compliance",
            "Focus on building trust through security and transparency"
        ],
        "ai": [
            "Secure proprietary data sources to strengthen competitive moat",
            "Establish clear ROI metrics for enterprise customers",
            "Develop explainability frameworks for complex models"
        ],
        "biotech": [
            "Focus resources on advancing clinical progress",
            "Strengthen IP portfolio through patents",
            "Establish strategic partnerships for distribution and validation"
        ],
        "ecommerce": [
            "Optimize logistics and fulfillment for cost and speed",
            "Improve customer retention with post-purchase engagement",
            "Reduce CAC through diversified acquisition channels"
        ],
        "crypto": [
            "Develop clear token utility and value capture mechanisms",
            "Strengthen security and audit practices",
            "Focus on regulatory compliance and transparency"
        ]
    }
    
    if sector in sector_recommendations:
        for rec in sector_recommendations[sector][:2]:  # Add top 2 recommendations for sector
            recommendations.append({
                "text": rec,
                "source": f"Based on {sector} sector best practices",
                "priority": "medium"
            })
    
    # Sort recommendations by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order[x["priority"]])
    
    return recommendations

def get_sector_recommendations(doc: dict) -> list:
    """
    Generate sector-specific recommendations based on startup metrics.
    
    Args:
        doc: Dictionary containing startup data
        
    Returns:
        List of recommendation strings
    """
    sector = doc.get("sector", "other").lower()
    stage = doc.get("stage", "seed").lower()
    recs = []
    
    if sector == "fintech":
        recs.append("Focus on regulatory compliance and security certifications")
        if doc.get("licenses_count", 0) < 2:
            recs.append("Secure additional financial licenses to reduce regulatory risk")
        if doc.get("default_rate", 0) > 0.05:
            recs.append("Implement stronger risk assessment models to reduce default rates")
    
    elif sector in ["biotech", "healthtech"]:
        recs.append("Accelerate clinical progress while strengthening IP portfolio")
        if doc.get("patent_count", 0) < 3:
            recs.append("Prioritize patent applications for core technology")
        if doc.get("clinical_phase", 0) < 2:
            recs.append("Focus resources on advancing to Phase 2 trials")
    
    elif sector == "saas":
        recs.append("Focus on reducing churn and increasing expansion revenue")
        if doc.get("net_retention_rate", 1.0) < 1.1:
            recs.append("Implement upsell/cross-sell strategy to boost net retention above 110%")
        if doc.get("churn_rate", 0.05) > 0.03:
            recs.append("Develop customer success program to reduce monthly churn below 3%")
    
    elif sector == "marketplace":
        recs.append("Prioritize liquidity in core segments before expanding")
        if doc.get("session_frequency", 0) < 3:
            recs.append("Increase engagement through gamification and retention hooks")
    
    elif sector in ["crypto", "blockchain"]:
        recs.append("Clarify token utility and regulatory compliance approach")
        recs.append("Develop cross-chain compatibility to maximize market reach")
    
    elif sector == "ai":
        recs.append("Secure proprietary data sources to strengthen competitive moat")
        recs.append("Demonstrate clear ROI metrics for enterprise customers")
    
    # Add ecommerce recommendations
    elif sector in ["ecommerce", "retail"]:
        recs.append("Optimize logistics and fulfillment to improve margins")
        if doc.get("repeat_purchase_rate", 0) < 0.2:
            recs.append("Implement customer loyalty program to boost repeat purchases")
        recs.append("Diversify acquisition channels to reduce CAC")
    
    # Stage-specific
    if stage in ["pre-seed", "seed"]:
        recs.append("Focus on product-market fit before scaling go-to-market")
    elif stage == "series-a":
        recs.append("Develop scalable acquisition channels with predictable CAC")
    elif stage in ["series-b", "series-c", "growth"]:
        recs.append("Optimize unit economics to demonstrate path to profitability")
    
    return recs[:5]  # Return top 5 recommendations

def evaluate_pattern_strength(doc: dict) -> dict:
    """
    Evaluate the overall pattern strength profile of the startup.
    
    Args:
        doc: Dictionary containing startup data
        
    Returns:
        Dictionary with pattern strength metrics
    """
    matched_patterns = detect_patterns(doc)
    
    # Count patterns by impact level
    high_impact = sum(1 for p in matched_patterns if p.get("impact", "") == "high")
    medium_impact = sum(1 for p in matched_patterns if p.get("impact", "") == "medium")
    low_impact = sum(1 for p in matched_patterns if p.get("impact", "") == "low")
    
    # Calculate pattern coverage score (0-100)
    total_relevant_patterns = len(get_unmatched_patterns(doc, [])) 
    pattern_coverage = (len(matched_patterns) / max(1, total_relevant_patterns)) * 100 if total_relevant_patterns > 0 else 0
    
    # Calculate weighted score
    weighted_score = (high_impact * 5 + medium_impact * 3 + low_impact * 1) * 5
    
    # Get sector coverage
    sector = doc.get("sector", "").lower()
    sector_patterns = [p for p in matched_patterns if sector in p.get("sector_relevance", [])]
    sector_pattern_ratio = len(sector_patterns) / max(1, len(matched_patterns)) if matched_patterns else 0
    
    strength_profile = {
        "total_patterns": len(matched_patterns),
        "high_impact_patterns": high_impact,
        "medium_impact_patterns": medium_impact,
        "low_impact_patterns": low_impact,
        "pattern_coverage": pattern_coverage,
        "weighted_score": weighted_score,
        "sector_pattern_ratio": sector_pattern_ratio,
        "overall_assessment": ""
    }
    
    # Generate overall assessment
    if weighted_score > 75:
        strength_profile["overall_assessment"] = "Exceptional pattern strength with multiple high-impact indicators"
    elif weighted_score > 50:
        strength_profile["overall_assessment"] = "Strong pattern profile with good coverage of key success factors"
    elif weighted_score > 25:
        strength_profile["overall_assessment"] = "Moderate pattern strength with some key indicators present"
    else:
        strength_profile["overall_assessment"] = "Limited pattern matches, focus on developing core success factors"
    
    return strength_profile