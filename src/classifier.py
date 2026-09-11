"""
Intent classifier using Gemini Flash few-shot prompting with
a calibrated TF-IDF + Logistic Regression fallback trained on TRAIN data.
"""
import os
import json
import numpy as np
from taxonomy import INTENT_NAMES
from gemini_client import gemini_generate_json

SYSTEM_PROMPT = """You are an expert customer service triage agent for AmazonHelp on Twitter.
Classify the incoming customer tweet into exactly ONE of these 7 intents:
- DELIVERY_STATUS_DELAY: Missing packages, tracking, late deliveries, shipping status
- RETURN_REFUND_EXCHANGE: Returns, refunds, replacements, damaged/defective goods
- PAYMENT_BILLING_PROMO: Charges, payment failures, gift cards, promos, invoices
- ACCOUNT_SECURITY_ACCESS: Login, password reset, account closure, 2FA, hacked accounts
- ORDER_CHANGE_CANCEL: Cancel order, change shipping address, modify order items
- PRODUCT_TECH_DIGITAL: Prime Video, Kindle, Fire TV, Echo, Alexa, app issues, digital content
- FEEDBACK_SERVICE_COMPLAINT: Dissatisfaction with service, agents, delivery experience

Respond ONLY with a JSON object: {"intent": "INTENT_NAME", "confidence": 0.0-1.0, "reasoning": "brief reason"}
"""

# Calibrated ML model trained on TRAIN only
_ml_classifier = None
_vectorizer = None

def _get_ml_classifier():
    global _ml_classifier, _vectorizer
    if _ml_classifier is not None:
        return _ml_classifier, _vectorizer
        
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from baselines import load_data, assign_pseudo_labels
    
    train_threads, _ = load_data()
    train_data = assign_pseudo_labels(train_threads)
    
    X_train = [t[0] for t in train_data]
    y_train = [t[1] for t in train_data]
    
    _vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_vec = _vectorizer.fit_transform(X_train)
    
    _ml_classifier = LogisticRegression(max_iter=1000, C=1.5, class_weight="balanced", random_state=42)
    _ml_classifier.fit(X_vec, y_train)
    return _ml_classifier, _vectorizer

def classify_tweet(text: str) -> dict:
    """
    Classify a customer tweet into one of 7 intents.
    Tries Gemini Flash first; falls back to calibrated ML classifier on API timeout/failure.
    """
    # 1. Try Gemini Flash
    prompt = f'{SYSTEM_PROMPT}\n\nCustomer: "{text}"\n'
    result = gemini_generate_json(prompt, temperature=0.0, max_tokens=120)
    
    if isinstance(result, dict) and "intent" in result:
        intent = result["intent"].strip().upper()
        if intent in INTENT_NAMES:
            conf = float(result.get("confidence", 0.85))
            return {
                "intent": intent,
                "confidence": max(0.0, min(1.0, conf)),
                "reasoning": result.get("reasoning", "Gemini Flash classification")
            }
        for valid in INTENT_NAMES:
            if valid in intent or intent in valid:
                return {
                    "intent": valid,
                    "confidence": 0.80,
                    "reasoning": "Gemini Flash fuzzy matched"
                }

    # 2. Calibrated ML Classifier fallback (trained on TRAIN)
    clf, vec = _get_ml_classifier()
    vec_text = vec.transform([text])
    probs = clf.predict_proba(vec_text)[0]
    best_idx = np.argmax(probs)
    pred_intent = clf.classes_[best_idx]
    confidence = float(probs[best_idx])
    
    # Confidence calibration based on margin
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]
    calibrated_conf = min(0.98, max(0.40, confidence * 0.7 + margin * 0.3))
    
    return {
        "intent": str(pred_intent),
        "confidence": float(round(calibrated_conf, 3)),
        "reasoning": f"Calibrated ML prediction (margin: {margin:.2f})"
    }

if __name__ == "__main__":
    test_queries = [
        "Where is my package? It was supposed to arrive yesterday!",
        "I was charged twice on my credit card for the same order",
        "Your delivery driver threw the box over my fence and it broke",
        "How do I cancel my order before it ships?",
        "Prime Video is showing error 5004 when I try to stream",
        "I hate your customer service, absolute worst experience",
        "My account got hacked and someone placed orders on my behalf"
    ]
    for q in test_queries:
        res = classify_tweet(q)
        print(f"Q: {q}")
        print(f"A: {res}\n")
