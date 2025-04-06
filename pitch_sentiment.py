import os
import logging
import re
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import random
import time

# Try to load nltk for advanced sentiment analysis
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Try to load spacy for entity extraction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Try to load environment variables
try:
    from dotenv import load_dotenv
    for env_file in ['.env', '.env.local', '/app/.env']:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            break
except ImportError:
    pass  # dotenv not available

# Configure logging
logger = logging.getLogger("pitch_sentiment")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

# Constants
MAX_TEXT_LENGTH = 10000
MIN_CATEGORY_CONFIDENCE = 0.6
DEFAULT_CONFIDENCE = 0.75

# Default NLP models
DEFAULT_SENTIMENT_MODEL = "vader"
DEFAULT_NER_MODEL = "en_core_web_sm"

# Pitch categories as enum for better type safety
class PitchCategory(Enum):
    TEAM = "team"
    PRODUCT = "product"
    MARKET = "market"
    BUSINESS_MODEL = "business_model"
    FINANCIALS = "financials"
    COMPETITION = "competition"
    VISION = "vision"
    TRACTION = "traction"
    GENERAL = "general"

@dataclass
class CategorySentiment:
    """Sentiment analysis results for a specific pitch category."""
    category: str
    score: float  # -1 to 1 scale
    confidence: float  # 0 to 1 scale
    text_samples: List[str]  # Key text snippets that influenced the score

@dataclass
class SentimentResult:
    """Overall sentiment analysis results for a pitch deck."""
    sentiment_score: float  # -1 to 1 scale
    confidence: float  # 0 to 1 scale
    category_sentiments: Dict[str, CategorySentiment]
    key_phrases: List[Dict[str, Any]]  # Key phrases with sentiment information
    raw_scores: Dict[str, float]  # Raw scoring data
    analysis_method: str  # Method used for analysis

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert enums to their values if present in category_sentiments
        if result.get('category_sentiments'):
            result['category_sentiments'] = {
                k.value if hasattr(k, 'value') else k: v 
                for k, v in result['category_sentiments'].items()
            }
        return result

class PitchAnalyzer:
    """
    Advanced analyzer for startup pitch decks that performs sentiment analysis
    and extracts key insights.
    
    This analyzer uses a combination of rule-based and ML-based techniques
    to provide robust sentiment analysis even in resource-constrained environments.
    """
    
    def __init__(self, sentiment_model: str = DEFAULT_SENTIMENT_MODEL, 
                 ner_model: str = DEFAULT_NER_MODEL,
                 load_nltk_resources: bool = True):
        """
        Initialize the pitch analyzer with specified models.
        
        Args:
            sentiment_model: Name of the sentiment model to use (default: vader)
            ner_model: Name of the NER model to use for entity extraction
            load_nltk_resources: Whether to download required NLTK resources
        """
        self.sentiment_model = sentiment_model
        self.ner_model = ner_model
        self.nlp = None
        self.sid = None
        self.initialized = False
        
        # Try to initialize the models
        try:
            self._initialize_nlp(load_nltk_resources)
            self.initialized = True
        except Exception as e:
            logger.warning(f"Could not initialize NLP models: {e}. Will use fallback methods.")
        
        # Category recognition patterns
        self.category_patterns = self._get_category_patterns()
        
    def _initialize_nlp(self, load_nltk_resources: bool) -> None:
        """Initialize NLP components."""
        # Initialize sentiment analyzer
        if self.sentiment_model == "vader" and NLTK_AVAILABLE:
            try:
                # Download VADER lexicon if needed and requested
                if load_nltk_resources:
                    try:
                        nltk.data.find('vader_lexicon.zip')
                    except LookupError:
                        nltk.download('vader_lexicon', quiet=True)
                        nltk.download('punkt', quiet=True)
                
                self.sid = SentimentIntensityAnalyzer()
                logger.info("Initialized VADER sentiment analyzer")
            except Exception as e:
                logger.error(f"Failed to initialize VADER: {e}")
                self.sid = None
        
        # Initialize spaCy for NER if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.ner_model)
                logger.info(f"Loaded spaCy model: {self.ner_model}")
            except OSError:
                # If model not found, try to download it (this may still fail)
                if load_nltk_resources:
                    logger.info(f"Downloading spaCy model: {self.ner_model}")
                    try:
                        spacy.cli.download(self.ner_model)
                        self.nlp = spacy.load(self.ner_model)
                    except Exception as e:
                        logger.warning(f"Failed to download spaCy model: {e}")
            except Exception as e:
                logger.warning(f"Error loading spaCy model: {e}")
    
    def _get_category_patterns(self) -> Dict[PitchCategory, List[str]]:
        """Define regex patterns for identifying pitch categories."""
        return {
            PitchCategory.TEAM: [
                r'\bteam\b', r'\bfounder', r'\bco-founder', r'\bceo\b', r'\bcto\b', 
                r'\bexperience\b', r'(\bour\s+team\b)', r'(\bthe\s+team\b)',
                r'\bbackground\b', r'\bleadership\b', r'\bmanagement\b'
            ],
            PitchCategory.PRODUCT: [
                r'\bproduct\b', r'\bsolution\b', r'\btechnology\b', r'\bplatform\b',
                r'\bfeatures\b', r'\bservice\b', r'\binnovation\b', r'\bapp\b',
                r'\bsoftware\b', r'\bhardware\b', r'\bprototype\b'
            ],
            PitchCategory.MARKET: [
                r'\bmarket\b', r'\bindustry\b', r'\bsegment\b', r'\btam\b', 
                r'\bsam\b', r'\bsom\b', r'\bmarket\s+size\b', r'\bopportunity\b',
                r'\btrend\b', r'\bdemand\b', r'\bcustomer\b'
            ],
            PitchCategory.BUSINESS_MODEL: [
                r'\bbusiness\s+model\b', r'\brevenue\s+model\b', r'\bmonetization\b',
                r'\bpricing\b', r'\bsubscription\b', r'\btransaction\b', r'\bunit\s+economics\b',
                r'\bmargins\b', r'\bcac\b', r'\bltv\b', r'\bcost\s+structure\b'
            ],
            PitchCategory.FINANCIALS: [
                r'\bfinancial', r'\bprojection', r'\bforecast', r'\brevenue', 
                r'\bprofit', r'\bebitda\b', r'\bburn\s+rate\b', r'\bcash\s+flow\b',
                r'\binvestment\b', r'\bfunding\b', r'\bvaluation\b'
            ],
            PitchCategory.COMPETITION: [
                r'\bcompetit', r'\brival', r'\blandscape\b', r'\balternative', 
                r'\bversus\b', r'\bvs\.?\b', r'\bmarket\s+leader\b', r'\bincumbent\b',
                r'\bmoat\b', r'\bdifferent', r'\bunique\s+value'
            ],
            PitchCategory.VISION: [
                r'\bvision\b', r'\bmission\b', r'\blong[\s-]term\b', r'\bfuture\b',
                r'\bgoal\b', r'\baspirat', r'\bpurpose\b', r'\bimpact\b',
                r'\bchang', r'\btransform'
            ],
            PitchCategory.TRACTION: [
                r'\btraction\b', r'\bgrowth\b', r'\bmilestone\b', r'\bachiev', 
                r'\bmetric\b', r'\bkpi\b', r'\bcustomer\s+acquisition\b', r'\bretention\b',
                r'\bprogress\b', r'\bmomentum\b', r'\bengagement\b'
            ]
        }
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of a pitch deck text.
        
        Args:
            text: The pitch deck text to analyze
            
        Returns:
            SentimentResult object containing overall and category-specific sentiment
        """
        logger.info("Starting pitch sentiment analysis")
        start_time = time.time()
        
        # Prepare text
        if not text:
            logger.warning("Empty text provided for sentiment analysis")
            return self._get_default_result()
        
        # Truncate text if needed
        text = self._preprocess_text(text)
        
        # Determine analysis method based on available resources
        if self.initialized and self.sid:
            logger.info("Using VADER for sentiment analysis")
            result = self._analyze_with_vader(text)
        else:
            logger.info("Using fallback sentiment analysis")
            result = self._analyze_with_fallback(text)
        
        logger.info(f"Sentiment analysis completed in {time.time() - start_time:.2f}s")
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and prepare text for analysis."""
        if len(text) > MAX_TEXT_LENGTH:
            logger.info(f"Truncating text from {len(text)} to {MAX_TEXT_LENGTH} characters")
            text = text[:MAX_TEXT_LENGTH]
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common PDF artifacts
        text = re.sub(r'\f', ' ', text)  # Form feed characters
        
        return text
    
    def _analyze_with_vader(self, text: str) -> SentimentResult:
        """Analyze sentiment using VADER."""
        # Split text into sentences for more detailed analysis
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback if NLTK tokenization fails
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Get overall sentiment
        overall_score = self.sid.polarity_scores(text)
        compound_score = overall_score['compound']
        
        # Categorize sentences and get category sentiment
        category_sentiments = self._analyze_categories(sentences)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(sentences)
        
        return SentimentResult(
            sentiment_score=compound_score,
            confidence=DEFAULT_CONFIDENCE,
            category_sentiments=category_sentiments,
            key_phrases=key_phrases,
            raw_scores=overall_score,
            analysis_method="vader"
        )
    
    def _analyze_with_fallback(self, text: str) -> SentimentResult:
        """Analyze sentiment using a fallback rule-based approach."""
        # Use a simple lexicon-based approach
        positive_words = [
            'excellent', 'great', 'good', 'positive', 'promising', 'innovative', 
            'growth', 'profitable', 'success', 'opportunity', 'leading', 'best',
            'unique', 'advantage', 'efficient', 'strong', 'robust', 'scalable',
            'proven', 'experienced', 'qualified', 'expert', 'revolutionary',
            'disruptive', 'proprietary', 'patented', 'exclusive', 'competitive',
            'superior', 'advanced', 'cutting-edge', 'market-leading', 'trusted'
        ]
        
        negative_words = [
            'bad', 'poor', 'negative', 'risk', 'challenge', 'difficult', 'problem',
            'weakness', 'threat', 'competitor', 'lose', 'loss', 'expense', 'costly',
            'failure', 'fail', 'uncertain', 'decline', 'decrease', 'limited',
            'restriction', 'constraint', 'concern', 'issue', 'doubt', 'delay',
            'complex', 'complicated', 'dangerous', 'unpredictable', 'volatile'
        ]
        
        # Count occurrences (case-insensitive)
        text_lower = text.lower()
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        total_count = positive_count + negative_count
        if total_count == 0:
            score = 0.0
        else:
            score = (positive_count - negative_count) / (positive_count + negative_count)
        
        # Get rough category sentiment
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        category_sentiments = self._analyze_categories_fallback(sentences, positive_words, negative_words)
        
        # Simple key phrase extraction
        key_phrases = []
        for sentence in sentences[:15]:  # Limit to first 15 sentences for efficiency
            for word in positive_words + negative_words:
                if word in sentence.lower():
                    sentiment = 'positive' if word in positive_words else 'negative'
                    if len(sentence) > 20 and len(sentence) < 200:
                        key_phrases.append({
                            'text': sentence,
                            'sentiment': sentiment,
                            'score': 0.8 if sentiment == 'positive' else -0.8
                        })
                    break
        
        # Take unique phrases
        seen_phrases = set()
        unique_phrases = []
        for phrase in key_phrases:
            if phrase['text'] not in seen_phrases and len(unique_phrases) < 5:
                seen_phrases.add(phrase['text'])
                unique_phrases.append(phrase)
        
        return SentimentResult(
            sentiment_score=score,
            confidence=0.5,  # Lower confidence for fallback method
            category_sentiments=category_sentiments,
            key_phrases=unique_phrases,
            raw_scores={'pos': positive_count, 'neg': negative_count, 'total': total_count},
            analysis_method="lexicon_fallback"
        )
    
    def _analyze_categories(self, sentences: List[str]) -> Dict[str, CategorySentiment]:
        """Analyze sentiment for each pitch category."""
        # Initialize category data
        category_data = {cat: {'sentences': [], 'scores': []} for cat in PitchCategory}
        
        # Assign sentences to categories and get their sentiment
        for sentence in sentences:
            assigned = False
            for category, patterns in self.category_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        score = self.sid.polarity_scores(sentence)
                        category_data[category]['sentences'].append(sentence)
                        category_data[category]['scores'].append(score['compound'])
                        assigned = True
                        break
                if assigned:
                    break
            
            # If not assigned to any specific category, add to general
            if not assigned:
                score = self.sid.polarity_scores(sentence)
                category_data[PitchCategory.GENERAL]['sentences'].append(sentence)
                category_data[PitchCategory.GENERAL]['scores'].append(score['compound'])
        
        # Compile results for each category
        results = {}
        for category, data in category_data.items():
            if not data['scores']:
                continue
                
            avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
            
            # Find representative text samples (most positive and negative)
            samples = []
            if data['sentences']:
                # Sort sentence-score pairs by score
                pairs = sorted(zip(data['sentences'], data['scores']), key=lambda x: x[1])
                
                # Get most negative if available
                if pairs and pairs[0][1] < 0:
                    samples.append(pairs[0][0])
                
                # Get most positive if available and different from negative
                if pairs and pairs[-1][1] > 0 and (not samples or pairs[-1][0] != samples[0]):
                    samples.append(pairs[-1][0])
            
            # Calculate confidence based on number of sentences
            confidence = min(1.0, len(data['sentences']) / 5 * MIN_CATEGORY_CONFIDENCE)
            
            # Only include categories with enough data
            if confidence >= MIN_CATEGORY_CONFIDENCE / 2:
                results[category.value] = CategorySentiment(
                    category=category.value,
                    score=avg_score,
                    confidence=confidence,
                    text_samples=samples[:2]  # Limit to 2 examples max
                )
        
        return results
    
    def _analyze_categories_fallback(self, sentences: List[str], 
                                    positive_words: List[str],
                                    negative_words: List[str]) -> Dict[str, CategorySentiment]:
        """Fallback method for category sentiment analysis when NLTK is not available."""
        # Initialize category data
        category_data = {cat.value: {'sentences': [], 'positive': 0, 'negative': 0} 
                        for cat in PitchCategory}
        
        # Assign sentences to categories
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for positive/negative sentiment
            pos_count = sum(sentence_lower.count(word) for word in positive_words)
            neg_count = sum(sentence_lower.count(word) for word in negative_words)
            
            # Assign to categories
            assigned = False
            for category, patterns in self.category_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        cat_key = category.value
                        category_data[cat_key]['sentences'].append(sentence)
                        category_data[cat_key]['positive'] += pos_count
                        category_data[cat_key]['negative'] += neg_count
                        assigned = True
                        break
                if assigned:
                    break
            
            # If not assigned, add to general
            if not assigned:
                category_data[PitchCategory.GENERAL.value]['sentences'].append(sentence)
                category_data[PitchCategory.GENERAL.value]['positive'] += pos_count
                category_data[PitchCategory.GENERAL.value]['negative'] += neg_count
        
        # Compile results
        results = {}
        for category, data in category_data.items():
            if not data['sentences']:
                continue
                
            total = data['positive'] + data['negative']
            if total == 0:
                score = 0.0
            else:
                score = (data['positive'] - data['negative']) / (data['positive'] + data['negative'])
            
            # Calculate confidence based on number of sentences
            confidence = min(0.8, len(data['sentences']) / 5 * MIN_CATEGORY_CONFIDENCE)
            
            # Only include categories with enough data
            if confidence >= MIN_CATEGORY_CONFIDENCE / 2:
                results[category] = CategorySentiment(
                    category=category,
                    score=score,
                    confidence=confidence,
                    text_samples=data['sentences'][:2]  # Take first two sentences as samples
                )
        
        return results
    
    def _extract_key_phrases(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Extract key phrases with sentiment information."""
        key_phrases = []
        
        # Sort sentences by sentiment intensity
        scored_sentences = []
        for sentence in sentences:
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            score = self.sid.polarity_scores(sentence)
            abs_score = abs(score['compound'])
            
            # Only consider sentences with strong sentiment
            if abs_score > 0.3:
                scored_sentences.append((sentence, score['compound'], abs_score))
        
        # Sort by absolute score (intensity) in descending order
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # Take top phrases, ensuring mix of positive and negative
        positive_phrases = [s for s in scored_sentences if s[1] > 0][:3]
        negative_phrases = [s for s in scored_sentences if s[1] < 0][:2]
        
        # Combine and convert to required format
        for sentence, score, _ in positive_phrases + negative_phrases:
            key_phrases.append({
                'text': sentence,
                'sentiment': 'positive' if score > 0 else 'negative',
                'score': score
            })
        
        return key_phrases
    
    def _get_default_result(self) -> SentimentResult:
        """Return default result when analysis fails."""
        return SentimentResult(
            sentiment_score=0.0,
            confidence=0.1,
            category_sentiments={},
            key_phrases=[],
            raw_scores={},
            analysis_method="default"
        )
    
    def extract_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extract key metrics from pitch text.
        
        This is a supplementary function that extracts numerical metrics like funding amounts,
        user counts, revenue figures, etc. from the pitch text.
        """
        metrics = {
            'funding_amount': None,
            'user_count': None,
            'revenue': None,
            'growth_rate': None,
            'market_size': None
        }
        
        if not text:
            return metrics
        
        # Extract patterns for each metric
        funding_patterns = [
            r'\$\s*(\d+(?:\.\d+)?)\s*(?:million|m)',
            r'raised\s*\$\s*(\d+(?:\.\d+)?)',
            r'funding\s*of\s*\$\s*(\d+(?:\.\d+)?)',
            r'investment\s*of\s*\$\s*(\d+(?:\.\d+)?)'
        ]
        
        user_patterns = [
            r'(\d+(?:,\d+)?)\s*(?:users|customers|clients)',
            r'user\s*base\s*of\s*(\d+(?:,\d+)?)',
            r'serving\s*(\d+(?:,\d+)?)\s*(?:users|customers)'
        ]
        
        revenue_patterns = [
            r'revenue\s*of\s*\$\s*(\d+(?:\.\d+)?)',
            r'\$\s*(\d+(?:\.\d+)?)\s*(?:in revenue|annual revenue)',
            r'arr\s*of\s*\$\s*(\d+(?:\.\d+)?)'
        ]
        
        growth_patterns = [
            r'(\d+(?:\.\d+)?)%\s*(?:growth|increase)',
            r'growing\s*(?:at|by)\s*(\d+(?:\.\d+)?)%',
            r'growth\s*rate\s*of\s*(\d+(?:\.\d+)?)%'
        ]
        
        market_patterns = [
            r'market\s*size\s*of\s*\$\s*(\d+(?:\.\d+)?)',
            r'market\s*worth\s*\$\s*(\d+(?:\.\d+)?)',
            r'tam\s*of\s*\$\s*(\d+(?:\.\d+)?)'
        ]
        
        pattern_sets = [
            (funding_patterns, 'funding_amount'),
            (user_patterns, 'user_count'),
            (revenue_patterns, 'revenue'),
            (growth_patterns, 'growth_rate'),
            (market_patterns, 'market_size')
        ]
        
        for patterns, metric_key in pattern_sets:
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Convert to float and remove commas
                    try:
                        value = float(matches[0].replace(',', ''))
                        metrics[metric_key] = value
                        break
                    except (ValueError, IndexError):
                        continue
        
        return metrics
    
    def analyze_overall_sentiment(self, doc: dict) -> Dict[str, Any]:
        """
        Analyze overall sentiment from a document containing pitch text.
        
        This is a convenience method for CAMP framework integration that
        analyzes the pitch and returns a structured result with both
        overall sentiment and category breakdowns.
        
        Args:
            doc: Document dictionary containing pitch_deck_text field
            
        Returns:
            Dict containing sentiment analysis results
        """
        pitch_text = doc.get("pitch_deck_text", "")
        
        if not pitch_text:
            logger.warning("No pitch text found in document")
            return {
                "overall_sentiment": {"score": 0, "confidence": 0.1},
                "category_sentiments": {},
                "key_phrases": []
            }
        
        try:
            # Analyze sentiment
            result = self.analyze_sentiment(pitch_text)
            
            # Extract structured result
            response = {
                "overall_sentiment": {
                    "score": result.sentiment_score,
                    "confidence": result.confidence
                },
                "category_sentiments": {},
                "key_phrases": result.key_phrases
            }
            
            # Convert category sentiments to proper format
            for cat_key, cat_sentiment in result.category_sentiments.items():
                response["category_sentiments"][cat_key] = {
                    "score": cat_sentiment.score,
                    "confidence": cat_sentiment.confidence,
                    "examples": cat_sentiment.text_samples
                }
            
            # Add any extracted metrics
            response["metrics"] = self.extract_metrics(pitch_text)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                "overall_sentiment": {"score": 0, "confidence": 0.1},
                "category_sentiments": {},
                "key_phrases": [],
                "error": str(e)
            }