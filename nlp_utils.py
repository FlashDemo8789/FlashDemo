import logging
import os
import sys
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger("nlp_utils")

# Try to detect if we're running in Streamlit Cloud
def is_streamlit_cloud():
    return os.environ.get("STREAMLIT_SHARING", "") == "true" or "/mount/src/" in os.getcwd()

# Initialize NLP - attempt multiple paths to get spaCy working
def initialize_nlp():
    """Initialize NLP pipeline with fallbacks"""
    try:
        import spacy
        
        # Check if model is already loaded in this environment
        try:
            # Try direct loading - this should work if model was installed via requirements.txt
            nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded en_core_web_sm model")
            return nlp
        except OSError:
            # Model not installed directly, try alternatives
            logger.warning("Could not load spaCy model directly, trying alternatives...")
            
            # Try to find the model in the repository
            try:
                # Check if model exists in various potential locations
                possible_paths = [
                    os.path.join("models", "en_core_web_sm"),
                    os.path.join("static", "models", "en_core_web_sm"),
                    "en_core_web_sm"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"Loading spaCy model from path: {path}")
                        nlp = spacy.load(path)
                        return nlp
                
                # If we're in Streamlit Cloud, we shouldn't try downloading
                if is_streamlit_cloud():
                    logger.warning("Running in Streamlit Cloud - skipping download attempt")
                    raise ImportError("Cannot load spaCy model in Streamlit Cloud environment")
                
                # Not in Streamlit Cloud, could try downloading (but will likely fail in restricted envs)
                logger.info("Attempting to download spaCy model (may fail in restricted environments)")
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                return nlp
                
            except Exception as e:
                logger.error(f"Failed to load/download spaCy model: {str(e)}")
                raise ImportError("Cannot load spaCy model through any method")
    
    except (ImportError, Exception) as e:
        logger.error(f"Error initializing spaCy: {str(e)}")
        
        # Return a fallback implementation
        return FallbackNLP()


class FallbackNLP:
    """
    Fallback NLP implementation when spaCy is not available
    Provides minimal compatible interface for basic NLP tasks
    """
    def __init__(self):
        logger.info("Using fallback NLP implementation (no spaCy)")
        
        # Load minimal English stopwords
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 
            'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
            'against', 'between', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
            'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
            'too', 'very', 'can', 'will', 'just', 'should', 'now'
        }
    
    def __call__(self, text):
        """Process text and return a document-like object"""
        return FallbackDoc(text, self.stopwords)
    
    def pipe(self, texts, **kwargs):
        """Process multiple texts"""
        for text in texts:
            yield self(text)


class FallbackDoc:
    """Simple document-like object returned by FallbackNLP"""
    
    def __init__(self, text, stopwords):
        self.text = text
        self.stopwords = stopwords
        
        # Basic tokenization
        self.tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Create basic token objects
        self._tokens = [FallbackToken(t, i, self) for i, t in enumerate(self.tokens)]
        
        # Create sentences with basic splitting
        self._sents = []
        for sent_text in re.split(r'[.!?]+', text):
            if sent_text.strip():
                self._sents.append(FallbackSpan(sent_text, self))
    
    def __len__(self):
        return len(self.tokens)
    
    def __iter__(self):
        """Iterate through tokens"""
        return iter(self._tokens)
    
    @property
    def ents(self):
        """Return empty entities for compatibility"""
        return []
    
    @property
    def noun_chunks(self):
        """Return empty noun chunks for compatibility"""
        return []
    
    @property
    def sents(self):
        """Return basic sentences"""
        return self._sents
    
    def similarity(self, other):
        """Very basic similarity based on word overlap"""
        if hasattr(other, 'text'):
            other_tokens = set(re.findall(r'\b\w+\b', other.text.lower()))
        else:
            other_tokens = set(re.findall(r'\b\w+\b', str(other).lower()))
        
        my_tokens = set(self.tokens)
        
        # Remove stopwords from both
        my_tokens = {t for t in my_tokens if t not in self.stopwords}
        other_tokens = {t for t in other_tokens if t not in self.stopwords}
        
        # Calculate Jaccard similarity
        if not my_tokens or not other_tokens:
            return 0.0
            
        intersection = len(my_tokens.intersection(other_tokens))
        union = len(my_tokens.union(other_tokens))
        
        return intersection / union if union > 0 else 0.0


class FallbackToken:
    """Simple token-like object for the fallback implementation"""
    
    def __init__(self, text, idx, doc):
        self.text = text.lower()
        self.i = idx
        self.doc = doc
        self._is_stop = text.lower() in doc.stopwords
    
    @property
    def is_stop(self):
        return self._is_stop
    
    @property
    def lemma_(self):
        """Very basic lemmatization - just lowercase the word"""
        return self.text
    
    @property
    def pos_(self):
        """Default part of speech - unknown"""
        return "X"


class FallbackSpan:
    """Simple span-like object for sentences"""
    
    def __init__(self, text, doc):
        self.text = text.strip()
        self.doc = doc
    
    def __str__(self):
        return self.text


# Create an exported NLP object that other modules can import
nlp = initialize_nlp()