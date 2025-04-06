from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import copy
import time
from functools import lru_cache

# Set up proper logging
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
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 90
    },
    {
        "name": "High Referral Growth",
        "criteria": [
            {"metric": "referral_rate", "op": ">", "value": 0.03}
        ],
        "description": "Referral rate above 3% => strong WOM network effect",
        "recommendation": "Invest in referral programs and virality mechanisms",
        "impact": "high",
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 85
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
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 80
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
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 88
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
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 85
    },
    # SaaS specific patterns
    {
        "name": "SaaS Quick Payback",
        "criteria": [
            {"metric": "unit_economics.cac_payback_months", "op": "<", "value": 12},
            {"metric": "gross_margin_percent", "op": ">", "value": 70}
        ],
        "description": "CAC payback <12 months with healthy margins => unit economics favoring rapid scaling",
        "recommendation": "Accelerate customer acquisition spending with confidence in quick returns",
        "impact": "high",
        "sector_relevance": ["saas", "software"],
        "is_positive": True,
        "confidence": 90
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
        "sector_relevance": ["saas", "software"],
        "is_positive": True,
        "confidence": 92
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
        "sector_relevance": ["marketplace", "platform"],
        "is_positive": True,
        "confidence": 87
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
        "sector_relevance": ["fintech", "banking"],
        "is_positive": True,
        "confidence": 83
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
        "sector_relevance": ["fintech", "banking"],
        "is_positive": True,
        "confidence": 88
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
        "sector_relevance": ["ai", "ml"],
        "is_positive": True,
        "confidence": 91
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
        "sector_relevance": ["ai", "ml"],
        "is_positive": True,
        "confidence": 84
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
        "sector_relevance": ["biotech", "healthtech"],
        "is_positive": True,
        "confidence": 92
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
        "sector_relevance": ["ecommerce", "retail"],
        "is_positive": True,
        "confidence": 86
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
        "sector_relevance": ["crypto", "blockchain"],
        "is_positive": True,
        "confidence": 85
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
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 82
    },
    {
        "name": "Diverse Founding Team",
        "criteria": [
            {"metric": "founder_diversity_score", "op": ">", "value": 70}
        ],
        "description": "Diverse founding team with varied perspectives",
        "recommendation": "Leverage diversity as a strength in hiring and decision-making",
        "impact": "medium",
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 80
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
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 95
    },
    {
        "name": "Rapid User Growth",
        "criteria": [
            {"metric": "user_growth_rate", "op": ">", "value": 0.15}
        ],
        "description": "User base growing >15% monthly indicating product-market fit",
        "recommendation": "Ensure infrastructure can scale with rapid growth",
        "impact": "high",
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 88
    },
    # Negative patterns
    {
        "name": "High Burn Rate",
        "criteria": [
            {"metric": "burn_rate", "op": ">", "value": 200000},
            {"metric": "runway_months", "op": "<", "value": 12}
        ],
        "description": "Excessive burn rate with limited runway creates significant financial risk",
        "recommendation": "Implement cost-cutting measures and focus on core growth drivers",
        "impact": "high",
        "sector_relevance": ["all"],
        "is_positive": False,
        "confidence": 90
    },
    {
        "name": "High Churn",
        "criteria": [
            {"metric": "churn_rate", "op": ">", "value": 0.08}
        ],
        "description": "Monthly churn above 8% indicates potential product-market fit issues",
        "recommendation": "Conduct churn analysis and improve core product experience",
        "impact": "high",
        "sector_relevance": ["all"],
        "is_positive": False,
        "confidence": 85
    },
    {
        "name": "CAC Inefficiency",
        "criteria": [
            {"metric": "ltv_cac_ratio", "op": "<", "value": 2},
            {"metric": "customer_acquisition_cost", "op": ">", "value": 0}
        ],
        "description": "Poor LTV:CAC ratio indicates unsustainable acquisition economics",
        "recommendation": "Optimize acquisition channels and improve customer monetization",
        "impact": "high",
        "sector_relevance": ["all"],
        "is_positive": False,
        "confidence": 88
    },
    # Fallback patterns - these have easy-to-meet criteria to ensure at least some patterns are detected
    {
        "name": "Early Stage Venture",
        "criteria": [
            {"metric": "monthly_revenue", "op": ">=", "value": 0}
        ],
        "description": "Early stage venture with opportunity to build strong foundations",
        "recommendation": "Focus on product-market fit and building core metrics",
        "impact": "medium",
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 70
    },
    {
        "name": "Market Opportunity",
        "criteria": [
            {"metric": "market_size", "op": ">", "value": 1000000}
        ],
        "description": "Addressing a market with significant potential",
        "recommendation": "Validate product-market fit through customer feedback",
        "impact": "medium",
        "sector_relevance": ["all"],
        "is_positive": True,
        "confidence": 75
    }
]

# Dictionary of valid operators for pattern criteria
VALID_OPERATORS = {
    ">": lambda x, y: x > y,
    ">=": lambda x, y: x >= y,
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y,
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "in": lambda x, y: x in y,
    "contains": lambda x, y: y in x if isinstance(x, (list, str, tuple)) else False,
}

class PatternValidator:
    """Validates pattern definitions for consistency and completeness."""

    @staticmethod
    def validate_patterns(patterns: List[Dict]) -> List[str]:
        """
        Validate all patterns for required fields and correct data types.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            List of validation error messages, empty list if no errors
        """
        errors = []
        
        for i, pattern in enumerate(patterns):
            # Check required fields
            required_fields = ["name", "criteria", "description", "impact", "sector_relevance"]
            for field in required_fields:
                if field not in pattern:
                    errors.append(f"Pattern {i}: Missing required field '{field}'")
            
            # Validate criteria
            if "criteria" in pattern and isinstance(pattern["criteria"], list):
                for j, criterion in enumerate(pattern["criteria"]):
                    if not isinstance(criterion, dict):
                        errors.append(f"Pattern {i}, criterion {j}: Not a dictionary")
                        continue
                    
                    # Check required criterion fields
                    criterion_fields = ["metric", "op", "value"]
                    for field in criterion_fields:
                        if field not in criterion:
                            errors.append(f"Pattern {i}, criterion {j}: Missing required field '{field}'")
                    
                    # Validate operator
                    if "op" in criterion and criterion["op"] not in VALID_OPERATORS:
                        errors.append(f"Pattern {i}, criterion {j}: Invalid operator '{criterion['op']}'")
            elif "criteria" in pattern:
                errors.append(f"Pattern {i}: 'criteria' must be a list")
            
            # Validate impact
            if "impact" in pattern and pattern["impact"] not in ["high", "medium", "low"]:
                errors.append(f"Pattern {i}: Invalid impact level '{pattern['impact']}', must be 'high', 'medium', or 'low'")
            
            # Validate sector_relevance
            if "sector_relevance" in pattern and not isinstance(pattern["sector_relevance"], list):
                errors.append(f"Pattern {i}: 'sector_relevance' must be a list")
                
            # Validate confidence
            if "confidence" in pattern:
                if not isinstance(pattern["confidence"], (int, float)):
                    errors.append(f"Pattern {i}: 'confidence' must be a number")
                elif pattern["confidence"] < 0 or pattern["confidence"] > 100:
                    errors.append(f"Pattern {i}: 'confidence' must be between 0 and 100")
            
            # Validate is_positive
            if "is_positive" in pattern and not isinstance(pattern["is_positive"], bool):
                errors.append(f"Pattern {i}: 'is_positive' must be a boolean")
        
        return errors

def get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """
    Extract a value from a nested dictionary using dot notation.
    
    Args:
        data: Dictionary to extract value from
        path: Path to value using dot notation (e.g., "unit_economics.cac")
        
    Returns:
        The value if found, None otherwise
    """
    if not path:
        return None
        
    parts = path.split('.')
    current = data
    
    try:
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    except Exception as e:
        logger.debug(f"Error extracting nested value for path '{path}': {e}")
        return None

def safe_type_conversion(value: Any, target_type: Any) -> Tuple[Any, bool]:
    """
    Safely convert a value to a target type.
    
    Args:
        value: Value to convert
        target_type: Type to convert to (e.g., int, float, bool)
        
    Returns:
        Tuple of (converted_value, success)
    """
    if isinstance(value, target_type):
        return value, True
    
    try:
        if isinstance(target_type, (int, float)):
            # For numeric types, handle special string cases
            if isinstance(value, str):
                value = value.strip().lower()
                if value == "true":
                    value = 1
                elif value == "false":
                    value = 0
                elif value == "":
                    return None, False
            
            if target_type == int:
                return int(float(value)), True
            else:  # float
                return float(value), True
        elif target_type == bool:
            if isinstance(value, str):
                value = value.strip().lower()
                if value in ("true", "yes", "1", "t", "y"):
                    return True, True
                elif value in ("false", "no", "0", "f", "n"):
                    return False, True
            elif isinstance(value, (int, float)):
                return bool(value), True
        
        # For other types, use direct conversion
        return target_type(value), True
    except (ValueError, TypeError) as e:
        logger.debug(f"Type conversion error: {e}, value={value}, target_type={target_type}")
        return None, False

def meets_criterion(doc: Dict[str, Any], criterion: Dict[str, Any]) -> bool:
    """
    Check if a document meets a criterion based on the specified operator.
    
    Args:
        doc: Dictionary containing startup metrics and data
        criterion: Criterion dictionary with metric, operator, and value
        
    Returns:
        True if the criterion is met, False otherwise
    """
    metric = criterion.get("metric", "")
    operator = criterion.get("op", "")
    threshold = criterion.get("value")
    
    # Validate inputs
    if not metric or not operator or threshold is None:
        logger.warning(f"Invalid criterion: {criterion}")
        return False
    
    # Check if operator is valid
    if operator not in VALID_OPERATORS:
        logger.warning(f"Unknown operator '{operator}' in criterion: {criterion}")
        return False
    
    # Get the actual value from the document
    actual_value = get_nested_value(doc, metric)
    
    # If value is None, the criterion is not met
    if actual_value is None:
        return False
    
    # Ensure types are compatible for comparison
    if isinstance(threshold, (int, float)) and not isinstance(actual_value, (int, float)):
        actual_value, success = safe_type_conversion(actual_value, type(threshold))
        if not success:
            return False
    
    # Handle boolean values
    if isinstance(threshold, bool) and not isinstance(actual_value, bool):
        actual_value, success = safe_type_conversion(actual_value, bool)
        if not success:
            return False
    
    # Handle list/sequence comparisons
    if operator in ("in", "contains") and not isinstance(actual_value, type(threshold)) and not (
            isinstance(actual_value, (list, tuple, str)) and isinstance(threshold, (list, tuple, str))):
        return False
    
    # Apply the operator
    try:
        return VALID_OPERATORS[operator](actual_value, threshold)
    except Exception as e:
        logger.warning(f"Error evaluating criterion {criterion}: {e}")
        return False

def calculate_pattern_match_confidence(doc: Dict[str, Any], pattern: Dict[str, Any], matched_criteria: List[Dict[str, Any]]) -> int:
    """
    Calculate a confidence score for a pattern match based on how strongly criteria are met.
    
    Args:
        doc: Dictionary containing startup metrics and data
        pattern: Pattern dictionary
        matched_criteria: List of criteria that matched
        
    Returns:
        Confidence score (0-100)
    """
    # Start with the base confidence from the pattern
    base_confidence = pattern.get("confidence", 80)
    
    # If no criteria matched, return 0
    if not matched_criteria:
        return 0
    
    # Calculate criterion strength scores
    criterion_scores = []
    for criterion in matched_criteria:
        metric = criterion.get("metric", "")
        operator = criterion.get("op", "")
        threshold = criterion.get("value")
        
        value = get_nested_value(doc, metric)
        if value is None:
            criterion_scores.append(0)
            continue
        
        # Calculate how strongly the criterion is met
        strength = 1.0  # Default strength
        
        # For numeric comparisons, calculate strength based on how far beyond threshold
        if isinstance(value, (int, float)) and isinstance(threshold, (int, float)):
            if operator in (">", ">="):
                # For greater than, strength increases as value increases above threshold
                if value > threshold:
                    strength = min(2.0, 1.0 + (value - threshold) / max(1, abs(threshold)) * 0.5)
            elif operator in ("<", "<="):
                # For less than, strength increases as value decreases below threshold
                if value < threshold:
                    strength = min(2.0, 1.0 + (threshold - value) / max(1, abs(threshold)) * 0.5)
        
        criterion_scores.append(strength)
    
    # Calculate average strength
    avg_strength = sum(criterion_scores) / len(criterion_scores) if criterion_scores else 0
    
    # Adjust confidence based on strength
    adjusted_confidence = int(base_confidence * avg_strength)
    
    # Ensure confidence is within bounds
    return max(50, min(100, adjusted_confidence))

@lru_cache(maxsize=128)
def get_relevant_patterns(sector: str) -> List[Dict[str, Any]]:
    """
    Get patterns relevant to a specific sector.
    
    Args:
        sector: Sector to get patterns for
        
    Returns:
        List of relevant pattern dictionaries
    """
    sector = sector.lower()
    return [
        pattern for pattern in PATTERNS
        if "all" in pattern.get("sector_relevance", []) or sector in pattern.get("sector_relevance", [])
    ]

def detect_patterns(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect which patterns match the startup data.
    
    Args:
        doc: Dictionary containing startup metrics and data
        
    Returns:
        List of matched pattern dictionaries
    """
    start_time = time.time()
    matched = []
    
    try:
        # Validate input
        if not isinstance(doc, dict):
            logger.error(f"Invalid input: doc must be a dictionary, got {type(doc).__name__}")
            return []
        
        # Get sector for sector-specific patterns
        sector = str(doc.get("sector", "")).lower()
        
        # Get relevant patterns for this sector
        relevant_patterns = get_relevant_patterns(sector)
        logger.debug(f"Found {len(relevant_patterns)} relevant patterns for sector '{sector}'")
        
        # Check each pattern
        for pattern in relevant_patterns:
            # Check if all criteria are met
            criteria = pattern.get("criteria", [])
            matched_criteria = []
            
            for criterion in criteria:
                if meets_criterion(doc, criterion):
                    matched_criteria.append(criterion)
                else:
                    matched_criteria = []  # Reset if any criterion fails
                    break
                    
            # If all criteria are met, add pattern to matched list
            if len(matched_criteria) == len(criteria):
                # Create a deep copy of the pattern to avoid modifying the original
                matched_pattern = copy.deepcopy(pattern)
                
                # Calculate confidence based on strength of match
                confidence = calculate_pattern_match_confidence(doc, pattern, matched_criteria)
                matched_pattern["confidence"] = confidence
                
                matched.append(matched_pattern)
        
        # Ensure we have at least a few patterns - use fallback patterns if needed
        if len(matched) == 0:
            logger.info("No patterns matched, adding fallback patterns")
            # Add fallback patterns that should match almost any startup
            for pattern in relevant_patterns:
                if pattern["name"] in ["Early Stage Venture", "Market Opportunity"]:
                    matched_pattern = copy.deepcopy(pattern)
                    matched.append(matched_pattern)
        
        logger.info(f"Detected {len(matched)} patterns in {time.time() - start_time:.2f}s")
        return matched
        
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}", exc_info=True)
        # Return fallback pattern in case of error
        return [{
            "name": "Startup Analysis",
            "description": "Basic startup analysis with limited pattern matching",
            "recommendation": "Focus on improving key metrics to match success patterns",
            "impact": "medium",
            "is_positive": True,
            "confidence": 50
        }]

def get_unmatched_patterns(doc: Dict[str, Any], matched_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get patterns that were not matched but are relevant to the startup's sector.
    
    Args:
        doc: Dictionary containing startup metrics and data
        matched_patterns: List of patterns that were already matched
        
    Returns:
        List of unmatched but relevant pattern dictionaries
    """
    try:
        sector = str(doc.get("sector", "")).lower()
        matched_names = {p["name"] for p in matched_patterns}
        
        # Get relevant patterns for this sector
        relevant_patterns = get_relevant_patterns(sector)
        
        # Filter out already matched patterns
        unmatched = [
            pattern for pattern in relevant_patterns
            if pattern["name"] not in matched_names
        ]
        
        return unmatched
    except Exception as e:
        logger.error(f"Error getting unmatched patterns: {e}", exc_info=True)
        return []

def analyze_missing_criteria(doc: Dict[str, Any], pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyze which specific criteria are missing for an unmatched pattern.
    
    Args:
        doc: Dictionary containing startup metrics and data
        pattern: Pattern dictionary to analyze
        
    Returns:
        List of missing criteria with details
    """
    try:
        missing_criteria = []
        
        for criterion in pattern.get("criteria", []):
            metric = criterion.get("metric", "")
            operator = criterion.get("op", "")
            threshold = criterion.get("value")
            
            # Get actual value from doc
            actual_value = get_nested_value(doc, metric)
            
            # Check if metric is missing or criterion is not met
            if actual_value is None:
                missing_criteria.append({
                    "metric": metric,
                    "issue": "missing",
                    "recommendation": f"Start tracking {metric}"
                })
            elif not meets_criterion(doc, criterion):
                # Calculate gap to threshold for numeric values
                if isinstance(actual_value, (int, float)) and isinstance(threshold, (int, float)):
                    if operator in [">", ">="]:
                        gap = threshold - actual_value
                        missing_criteria.append({
                            "metric": metric,
                            "issue": "below_threshold",
                            "actual": actual_value,
                            "threshold": threshold,
                            "gap": gap,
                            "recommendation": f"Increase {metric} by at least {gap:.2f}"
                        })
                    elif operator in ["<", "<="]:
                        gap = actual_value - threshold
                        missing_criteria.append({
                            "metric": metric,
                            "issue": "above_threshold",
                            "actual": actual_value,
                            "threshold": threshold,
                            "gap": gap,
                            "recommendation": f"Decrease {metric} by at least {gap:.2f}"
                        })
                else:
                    missing_criteria.append({
                        "metric": metric,
                        "issue": "not_met",
                        "actual": actual_value,
                        "threshold": threshold,
                        "recommendation": f"Address gap in {metric} to meet criterion"
                    })
        
        return missing_criteria
    except Exception as e:
        logger.error(f"Error analyzing missing criteria: {e}", exc_info=True)
        return []

def generate_pattern_insights(matched_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate insights based on matched patterns.
    
    Args:
        matched_patterns: List of matched pattern dictionaries
        
    Returns:
        Dictionary with insights and pattern analysis
    """
    try:
        insights = []
        
        if not matched_patterns:
            insights.append({
                "title": "No Strong Patterns Detected",
                "description": "No strong patterns detected. Focus on improving key metrics to establish pattern matches.",
                "impact": "High",
                "action": "Review metrics and focus on achieving core success patterns."
            })
            return {
                "top_insights": insights,
                "pattern_categories": {}
            }
        
        # Sort patterns by confidence score
        sorted_patterns = sorted(matched_patterns, key=lambda p: p.get("confidence", 0), reverse=True)
        
        # Get pattern names
        names = [p["name"] for p in sorted_patterns]
        
        # Generate insight for multiple patterns
        if len(names) >= 3:
            high_impact = [p for p in sorted_patterns if p.get("impact", "") == "high"]
            if high_impact:
                high_impact_names = [p["name"] for p in high_impact[:3]]
                insights.append({
                    "title": "Multiple High-Impact Patterns Detected",
                    "description": f"Multiple high-impact patterns detected: {', '.join(high_impact_names)} => robust fundamentals with strong growth potential.",
                    "impact": "High",
                    "action": "Continue to build on these strengths and leverage them in funding discussions."
                })
            else:
                insights.append({
                    "title": "Multiple Patterns Detected",
                    "description": f"Multiple patterns detected: {', '.join(names[:3])} and more => solid foundation with diverse strengths.",
                    "impact": "Medium",
                    "action": "Consolidate existing strengths while addressing remaining gaps."
                })
        elif len(names) == 2:
            insights.append({
                "title": "Two Complementary Patterns",
                "description": f"Two complementary patterns detected: {names[0]} and {names[1]} => building momentum with clear strengths.",
                "impact": "Medium",
                "action": "Focus on leveraging these strengths while developing other areas."
            })
        elif len(names) == 1:
            insights.append({
                "title": "Single Key Pattern",
                "description": f"One key pattern detected: {names[0]} => build on this strength while developing other areas.",
                "impact": "Medium",
                "action": f"Leverage the {names[0]} pattern while working on other key success factors."
            })
        
        # Generate insights for specific pattern combinations
        pattern_names = {p["name"] for p in matched_patterns}
        
        # Synergistic pattern combinations
        pattern_combos = [
            {
                "patterns": ["High Referral Growth", "Large Market & Growth"],
                "title": "Viral Growth in Large Market",
                "description": "High referral combined with large market => strong possibility of viral expansion in a big TAM.",
                "impact": "High",
                "action": "Double down on referral mechanisms and focus marketing on high-growth segments."
            },
            {
                "patterns": ["Experienced Founding Team", "Cash Efficient"],
                "title": "Experienced Team with Capital Efficiency",
                "description": "Experienced team with cash efficiency => strong execution capability with runway to prove model.",
                "impact": "High",
                "action": "Leverage founder experience to maintain capital efficiency while pursuing growth."
            },
            {
                "patterns": ["Efficient Go-To-Market", "Rapid User Growth"],
                "title": "Scaling with Efficient Acquisition",
                "description": "Efficient GTM with rapid growth => scalable acquisition model is working, potential for capital-efficient growth.",
                "impact": "High",
                "action": "Increase marketing spend confidently based on proven acquisition efficiency."
            },
            {
                "patterns": ["SaaS Net Expansion", "SaaS Quick Payback"],
                "title": "Exceptional SaaS Unit Economics",
                "description": "SaaS quick payback with net expansion => exceptional unit economics favoring aggressive growth investment.",
                "impact": "High",
                "action": "Consider aggressive growth investment backed by strong unit economics."
            }
        ]
        
        # Check for matching combinations
        for combo in pattern_combos:
            required_patterns = set(combo["patterns"])
            if required_patterns.issubset(pattern_names):
                insights.append({
                    "title": combo["title"],
                    "description": combo["description"],
                    "impact": combo["impact"],
                    "action": combo["action"]
                })
        
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
        
        # Create pattern categories for visualization
        pattern_categories = {}
        
        # Count patterns by positive/negative
        positive_patterns = [p for p in matched_patterns if p.get("is_positive", True)]
        negative_patterns = [p for p in matched_patterns if not p.get("is_positive", True)]
        
        pattern_categories["positive"] = len(positive_patterns)
        if negative_patterns:
            pattern_categories["negative"] = len(negative_patterns)
        
        # Count patterns by impact
        for impact in ["high", "medium", "low"]:
            count = sum(1 for p in matched_patterns if p.get("impact", "") == impact)
            if count > 0:
                pattern_categories[impact] = count
        
        # Add sector category
        sector = None
        for s, patterns in sector_patterns.items():
            sector_matched = [p["name"] for p in matched_patterns if p["name"] in patterns]
            if sector_matched:
                sector = s
                insights.append({
                    "title": f"Strong {s.capitalize()} Patterns",
                    "description": f"Strong {s.capitalize()} patterns: {', '.join(sector_matched)} => competitive advantage in core {s} success factors.",
                    "impact": "High",
                    "action": f"Leverage your core {s} strengths in investor discussions and competitive positioning."
                })
        
        # Add sector to pattern categories if found
        if sector:
            pattern_categories["sector_specific"] = len([p for p in matched_patterns if p["name"] in sector_patterns.get(sector, [])])
        
        return {
            "top_insights": insights[:3],  # Limit to top 3 insights
            "pattern_categories": pattern_categories
        }
    except Exception as e:
        logger.error(f"Error generating pattern insights: {e}", exc_info=True)
        return {
            "top_insights": [{
                "title": "Pattern Analysis",
                "description": "Basic pattern analysis completed.",
                "impact": "Medium",
                "action": "Review metrics to improve key success factors."
            }],
            "pattern_categories": {}
        }

def generate_pattern_recommendations(doc: Dict[str, Any], matched_patterns: List[Dict[str, Any]], unmatched_patterns: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Generate actionable recommendations based on matched and unmatched patterns.
    
    Args:
        doc: Dictionary containing startup metrics and data
        matched_patterns: List of matched pattern dictionaries
        unmatched_patterns: List of unmatched pattern dictionaries (optional)
        
    Returns:
        List of recommendation dictionaries with priority levels
    """
    try:
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
        stage = str(doc.get("stage", "")).lower()
        sector = str(doc.get("sector", "")).lower()
        
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
        recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))
        
        # Deduplicate recommendations
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            text = rec.get("text", "")
            if text and text not in seen:
                seen.add(text)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    except Exception as e:
        logger.error(f"Error generating pattern recommendations: {e}", exc_info=True)
        return [{
            "text": "Focus on improving key metrics to establish pattern matches",
            "source": "Default recommendation",
            "priority": "medium"
        }]

def get_sector_recommendations(doc: Dict[str, Any]) -> List[str]:
    """
    Generate sector-specific recommendations based on startup metrics.
    
    Args:
        doc: Dictionary containing startup data
        
    Returns:
        List of recommendation strings
    """
    try:
        sector = str(doc.get("sector", "other")).lower()
        stage = str(doc.get("stage", "seed")).lower()
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
        
        # Ensure we have at least some recommendations
        if not recs:
            recs.append("Focus on core metrics for your sector to establish pattern matches")
            recs.append("Validate product-market fit through customer feedback")
        
        return recs[:5]  # Return top 5 recommendations
    except Exception as e:
        logger.error(f"Error generating sector recommendations: {e}", exc_info=True)
        return ["Focus on improving key metrics for your sector"]

def evaluate_pattern_strength(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the overall pattern strength profile of the startup.
    
    Args:
        doc: Dictionary containing startup data
        
    Returns:
        Dictionary with pattern strength metrics
    """
    try:
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
        sector = str(doc.get("sector", "")).lower()
        sector_patterns = [p for p in matched_patterns if sector in p.get("sector_relevance", [])]
        sector_pattern_ratio = len(sector_patterns) / max(1, len(matched_patterns)) if matched_patterns else 0
        
        # Filter out fallback patterns for scoring
        fallback_patterns = ["Early Stage Venture", "Market Opportunity"]
        strong_patterns = [p for p in matched_patterns if p["name"] not in fallback_patterns]
        
        # Calculate strength tier
        strength_tier = "weak"
        if weighted_score > 75:
            strength_tier = "exceptional"
        elif weighted_score > 50:
            strength_tier = "strong"
        elif weighted_score > 25:
            strength_tier = "moderate"
        
        # Count positive vs negative patterns
        positive_patterns = [p for p in matched_patterns if p.get("is_positive", True)]
        negative_patterns = [p for p in matched_patterns if not p.get("is_positive", True)]
        
        strength_profile = {
            "total_patterns": len(matched_patterns),
            "strong_patterns": len(strong_patterns),
            "high_impact_patterns": high_impact,
            "medium_impact_patterns": medium_impact,
            "low_impact_patterns": low_impact,
            "positive_patterns": len(positive_patterns),
            "negative_patterns": len(negative_patterns),
            "pattern_coverage": pattern_coverage,
            "weighted_score": weighted_score,
            "sector_pattern_ratio": sector_pattern_ratio,
            "strength_tier": strength_tier,
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
    except Exception as e:
        logger.error(f"Error evaluating pattern strength: {e}", exc_info=True)
        return {
            "total_patterns": 0,
            "strong_patterns": 0,
            "high_impact_patterns": 0,
            "medium_impact_patterns": 0,
            "low_impact_patterns": 0,
            "positive_patterns": 0,
            "negative_patterns": 0,
            "pattern_coverage": 0,
            "weighted_score": 0,
            "sector_pattern_ratio": 0,
            "strength_tier": "unknown",
            "overall_assessment": "Unable to evaluate pattern strength"
        }

# Run pattern validation on module load
validation_errors = PatternValidator.validate_patterns(PATTERNS)
if validation_errors:
    logger.warning(f"Pattern validation failed with {len(validation_errors)} errors:")
    for error in validation_errors:
        logger.warning(f"  {error}")
else:
    logger.info(f"Successfully validated {len(PATTERNS)} pattern definitions")
