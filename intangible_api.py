import os
import logging
import random
import re
import json
import openai

logger = logging.getLogger("intangible_api")
logger.setLevel(logging.DEBUG)

OPENAI_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MAX_PITCH_CHARS = 4000

def compute_intangible_llm(doc: dict) -> float:
    pitch_text = doc.get("pitch_deck_text", "") or ""
    pitch_text = pitch_text.strip()

    pitch_sent = doc.get("pitch_sentiment", {})

    logger.debug(f"[Intangible] Starting intangible for doc: {doc.get('name','N/A')}, length={len(pitch_text)}")

    if not pitch_text:
        logger.info("[Intangible] No pitch text => fallback triggered.")
        fallback_score = _investor_fallback(doc, pitch_sent)
        logger.info(f"[Intangible] Fallback => {fallback_score:.2f}")
        return fallback_score

    if len(pitch_text) > MAX_PITCH_CHARS:
        logger.info(f"[Intangible] Truncating from {len(pitch_text)} to {MAX_PITCH_CHARS} chars.")
        pitch_text = pitch_text[:MAX_PITCH_CHARS] + "\n[...Truncated...]"

    try:
        score = _call_deepseek_api_chat(pitch_text)
        logger.info(f"[Intangible] DeepSeek => {score:.2f}")
        return float(score)
    except Exception as e:
        logger.error(f"[Intangible] DeepSeek error => {e}")
        fallback_score = _investor_fallback(doc, pitch_sent)
        logger.info(f"[Intangible] Fallback => {fallback_score:.2f}")
        return fallback_score

def _call_deepseek_api_chat(pitch_text: str) -> float:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("[Intangible] Missing DEEPSEEK_API_KEY environment variable")

    openai.api_key = DEEPSEEK_API_KEY
    openai.api_base = OPENAI_BASE_URL

    system_prompt = (
        "You are an intangible rating assistant. Return JSON with one key 'score' in [0..100]."
        "No triple backticks, just raw JSON."
    )
    user_prompt = (
        f"Startup pitch:\n{pitch_text}\n"
        "Reply with JSON only, e.g. {\"score\": 55.0}"
    )

    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        max_tokens=200,
        temperature=0.0
    )

    content = response.choices[0].message.content.strip()
    logger.debug(f"[Intangible] Raw model response => {content}")

    # Remove triple-backtick fences if present
    fenced = re.match(r"^(?:json)?\s*(\{.*\})\s*$", content, re.DOTALL)
    if fenced:
        content = fenced.group(1).strip()

    try:
        data = json.loads(content)
        raw_score = data.get("score", 50)
    except json.JSONDecodeError:
        logger.warning("[Intangible] Model didn't return valid JSON => fallback to 50")
        raw_score = 50

    if not (0 <= raw_score <= 100):
        logger.warning(f"[Intangible] Out-of-range => {raw_score}, clamping.")
        raw_score = max(0, min(100, raw_score))

    return float(raw_score)

def _investor_fallback(doc: dict, pitch_sent: dict) -> float:
    base = 50.0
    founder_exits = doc.get("founder_exits", 0)
    domain_exp = doc.get("founder_domain_exp_yrs", 0)

    if founder_exits >= 1:
        base += 5
    elif domain_exp >= 5:
        base += 3
    if domain_exp < 1 and founder_exits == 0:
        base -= 5

    base = _combine_with_sentiment(base, pitch_sent)
    final = max(0, min(100, base))
    if abs(final - 50) < 0.01:
        final += random.uniform(-5, 5)
    return float(final)

def _combine_with_sentiment(score: float, pitch_sent: dict) -> float:
    if not pitch_sent or "overall_sentiment" not in pitch_sent:
        return score

    overall = pitch_sent["overall_sentiment"]
    sentiment_val = float(overall.get("score", 0.0))
    if sentiment_val > 0.3:
        score += (sentiment_val * 5.0)
    elif sentiment_val < -0.3:
        score += (sentiment_val * 5.0)

    if len(pitch_sent.get("category_sentiments", {})) < 5:
        score -= 3
    return score