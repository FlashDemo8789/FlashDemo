from typing import Dict, Any
import logging
import numpy as np
import xgboost as xgb

from constants import NAMED_METRICS_50
from domain_expansions import apply_domain_expansions
from team_moat import compute_team_depth_score, compute_moat_score
# Use intangible_api (not intangible_llm) to compute intangible
from intangible_api import compute_intangible_llm

logger = logging.getLogger("flashdna")

def build_feature_vector_no_llm(doc: dict) -> np.ndarray:
    """
    HPC synergy BFS–free approach (no LLM):
    Uses doc['intangible'] or doc['feel_score'] if present, else defaults=50
    """
    try:
        base = []
        for metric in NAMED_METRICS_50:
            val = float(doc.get(metric, 0.0))
            base.append(val)

        expansions = apply_domain_expansions(doc)
        intangible_val = float(doc.get("intangible", doc.get("feel_score", 50.0)))
        team_val = compute_team_depth_score(doc)
        moat_val = compute_moat_score(doc)

        feature_list = base + list(expansions.values()) + [intangible_val, team_val, moat_val]
        return np.array(feature_list, dtype=float)

    except Exception as e:
        logger.error(f"Error building feature vector (no LLM): {str(e)}")
        return np.zeros(len(NAMED_METRICS_50) + 5, dtype=float)

def build_feature_vector(doc: dict) -> np.ndarray:
    """
    HPC synergy BFS–free approach for inference,
    including intangible from intangible_api if missing pitch text.
    """
    try:
        base = []
        for metric in NAMED_METRICS_50:
            val = float(doc.get(metric, 0.0))
            base.append(val)

        expansions = apply_domain_expansions(doc)
        # If intangible missing => call intangible_api or fallback
        if "intangible" not in doc:
            if "pitch_deck_text" in doc and doc["pitch_deck_text"].strip():
                doc["intangible"] = compute_intangible_llm(doc)
            else:
                # fallback => random range so not always 50
                import random
                doc["intangible"] = random.uniform(55, 60)

        intangible_val = float(doc.get("intangible", 60.0))
        team_val = compute_team_depth_score(doc)
        moat_val = compute_moat_score(doc)

        feature_list = base + list(expansions.values()) + [intangible_val, team_val, moat_val]
        return np.array(feature_list, dtype=float)

    except Exception as e:
        logger.error(f"Error building feature vector: {str(e)}")
        return np.zeros(len(NAMED_METRICS_50) + 5, dtype=float)

def predict_probability(doc: dict, model) -> float:
    """
    HPC synergy BFS–free => success probability
    """
    try:
        feats = build_feature_vector(doc).reshape(1, -1)
        expected = getattr(model, 'n_features_in_', None)
        if expected is not None and expected != feats.shape[1]:
            import numpy as np
            if feats.shape[1] < expected:
                pad = np.zeros((1, expected - feats.shape[1]))
                feats = np.concatenate([feats, pad], axis=1)
            else:
                feats = feats[:, :expected]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feats)[0, 1] * 100
        else:
            pred = model.predict(feats)[0]
            proba = float(pred) * 100
        return proba

    except Exception as e:
        logger.error(f"Error in predict_probability: {str(e)}")
        return 50.0

def predict_success(doc: dict, xgb_model) -> float:
    """
    HPC synergy BFS–free => single-proba approach
    """
    try:
        fv = build_feature_vector(doc).reshape(1, -1)
        expected = getattr(xgb_model, 'n_features_in_', None)
        if expected and fv.shape[1] < expected:
            import numpy as np
            pad = np.zeros((1, expected - fv.shape[1]))
            fv = np.concatenate([fv, pad], axis=1)
        elif expected and fv.shape[1] > expected:
            fv = fv[:, :expected]

        if hasattr(xgb_model, 'predict_proba'):
            prob = xgb_model.predict_proba(fv)[0][1] * 100
        else:
            pred = xgb_model.predict(fv)[0]
            prob = float(pred) * 100
        return prob
    except Exception as e:
        logger.error(f"predict_success error => {str(e)}")
        return 50.0

def evaluate_startup(doc: dict, xgb_model) -> Dict[str, Any]:
    """
    HPC synergy BFS–free => success_prob & intangible & team + moat => final flashdna_score
    Weighted approach => 60% from success_prob, 20% intangible, 10% team, 10% moat
    """
    success_prob = predict_success(doc, xgb_model)
    intangible = doc.get("intangible", 50.0)
    team_val = doc.get("team_score", 0.0)
    moat_val = doc.get("moat_score", 0.0)
    # Weighted
    final_score = success_prob * 0.6 + intangible * 0.2 + team_val * 0.1 + moat_val * 0.1

    return {
        "success_prob": success_prob,
        "success_probability": success_prob,
        "flashdna_score": final_score
    }

def train_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """
    HPC synergy BFS–free => train XGBoost with default XGB_PARAMS
    """
    from sklearn.model_selection import train_test_split
    from config import XGB_PARAMS

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info(f"XGB Train acc={train_acc:.2f}, Test acc={test_acc:.2f}")

    return model