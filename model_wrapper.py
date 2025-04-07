import numpy as np
import logging

logger = logging.getLogger("model_wrapper")

class ModelWrapper:
    """
    A wrapper for XGBoost models that handles compatibility issues
    between different versions of XGBoost and provides fallbacks.
    """
    
    def __init__(self, model=None):
        self.model = model
        
    def predict_proba(self, X):
        """Wrapper for predict_proba that handles various model types."""
        try:
            # For models with predict_proba
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            
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
                    
        except Exception as e:
            logger.error(f"Error in predict_proba: {str(e)}")
            # Safe fallback
            if len(X) == 1:
                return np.array([[0.5, 0.5]])
            else:
                return np.tile([0.5, 0.5], (len(X), 1))
                
    def predict(self, X):
        """Predict method that falls back to probabilistic output if needed."""
        try:
            # Try direct predict first
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            
            # Fall back to predict_proba
            probs = self.predict_proba(X)
            return probs[:, 1]
            
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            # Safe fallback
            return np.array([0.5] * X.shape[0])