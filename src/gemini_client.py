"""
Gemini API client wrapper for all LLM calls in the pipeline.
Uses Gemini Flash via the Google AI REST API with a circuit breaker for network failures.
"""
import os
import json
import time
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Circuit breaker: if API fails consecutive times, disable to keep pipeline ultra-responsive
_consecutive_failures = 0
_max_failures = 2

def gemini_generate(prompt: str, temperature: float = 0.0, max_tokens: int = 256, retries: int = 1) -> str:
    """
    Call Gemini Flash API with circuit breaker.
    """
    global _consecutive_failures
    if _consecutive_failures >= _max_failures:
        return ""
        
    url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                _consecutive_failures = 0
                return text.strip()
            else:
                _consecutive_failures += 1
                return ""
        except Exception:
            _consecutive_failures += 1
            return ""
    
    return ""

def gemini_generate_json(prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> dict:
    """
    Call Gemini and parse the response as JSON.
    """
    text = gemini_generate(prompt, temperature=temperature, max_tokens=max_tokens)
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    return {}
