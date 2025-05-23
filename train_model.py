import os
import json
import pickle
import numpy as np
import logging
from config import XGB_PARAMS
from constants import NAMED_METRICS_50
from advanced_ml import build_feature_vector_no_llm, train_model

logger= logging.getLogger("train_model")
logging.basicConfig(level=logging.INFO)

def load_training_docs():
    """
    optionally load big_startups.json with pass/fail outcome,
    or fallback with multiple sample docs.
    """
    if os.path.exists("data/big_startups.json"):
        try:
            with open("data/big_startups.json","r",encoding="utf-8", errors="replace") as f:
                data= json.load(f)
            docs= data.get("startups",[])
            # only keep docs that have outcome= pass or fail
            pf_docs= [d for d in docs if d.get("outcome") in ["pass","fail"]]
            if pf_docs:
                logger.info(f"Loaded {len(pf_docs)} pass/fail docs from big_startups.json")
                return pf_docs
        except Exception as e:
            logger.error(f"Error loading big_startups.json => {str(e)}")

    # If no file or no pass/fail doc found => fallback
    logger.warning("Using minimal pass/fail sample => please expand for realism")

    data_docs= [
        {
            "outcome":"pass",
            "monthly_revenue":100000,
            "user_growth_rate":0.15,
            "burn_rate":60000,
            "churn_rate":0.03,
            "ltv_cac_ratio":3.5,
            "founder_exits":2,
            "founder_domain_exp_yrs":8,
            "pitch_deck_text": "We have a strong user base and well-defined product..."
        },
        {
            "outcome":"pass",
            "monthly_revenue":20000,
            "user_growth_rate":0.2,
            "burn_rate":10000,
            "churn_rate":0.02,
            "ltv_cac_ratio":4.0,
            "founder_exits":1,
            "founder_domain_exp_yrs":5,
            "pitch_deck_text": "Our solution addresses a big market with a strong team..."
        },
        {
            "outcome":"fail",
            "monthly_revenue":3000,
            "user_growth_rate":0.01,
            "burn_rate":20000,
            "churn_rate":0.15,
            "ltv_cac_ratio":1.2,
            "founder_exits":0,
            "founder_domain_exp_yrs":1
        },
        {
            "outcome":"fail",
            "monthly_revenue":500,
            "user_growth_rate":0.0,
            "burn_rate":5000,
            "churn_rate":0.2,
            "ltv_cac_ratio":0.8,
            "founder_exits":0,
            "founder_domain_exp_yrs":0
        }
    ]
    return data_docs

def load_training_data():
    """
    build X, y from pass/fail outcome
    """
    docs= load_training_docs()
    X_list= []
    y_list= []
    for d in docs:
        fv= build_feature_vector_no_llm(d)
        X_list.append(fv)
        outcome = d.get("outcome","fail")
        y_list.append(1 if outcome=="pass" else 0)
    return np.array(X_list), np.array(y_list,dtype=int)

def train_model_xgb():
    X,y= load_training_data()
    if len(set(y))<2:
        logger.error("Need >=2 classes => add more pass/fail docs.")
        return
    model= train_model(X,y)
    with open("model_xgb.pkl","wb") as f:
        pickle.dump(model,f)
    logger.info("Saved model_xgb.pkl => done")

if __name__=="__main__":
    train_model_xgb()