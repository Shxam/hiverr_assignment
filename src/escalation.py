"""
Escalation policy + deterministic quality gate + unified merge node.

Escalation Policy (Decision A):
  Combines classifier confidence + heuristic risk signals + retrieval match quality.
  
Quality Gate (Decision B):
  Deterministic checks: non-empty, reasonable length, unsupported-claim detection, safety.

Final Decision:
  ESCALATE if A == escalate OR B == fail
  AUTO-SEND only if both clear.
"""
import re

# Heuristic risk-signal phrases (labeled as such, not a learned model)
# NOTE: Known false-positive pattern: "charge" can appear in innocent contexts like
# "free of charge" or "in charge of". This is documented as a limitation.
RISK_PHRASES = [
    "lawyer", "attorney", "legal action", "court", "sue", "police",
    "chargeback", "fraud", "scam", "bbb", "consumer protection",
    "stolen", "unauthorized", "identity theft",
    "furious", "disgusted", "unacceptable", "worst ever",
    "supervisor", "manager", "escalate"
]

UNSAFE_PATTERNS = [
    r"\b(kill|death threat|bomb|weapon)\b",
    r"\b(self[- ]?harm|suicide)\b"
]

PROMISE_TOKENS = [
    "refund", "replacement", "compensation", "credit", "coupon",
    "free", "reimburse", "discount", "within .* days",
    "guarantee", "promise", "we will send"
]

def escalation_policy(classification: dict, retrieved_examples: list, confidence_threshold: float = 0.75) -> dict:
    """
    Decision A: Should this be escalated based on confidence, risk signals, and retrieval quality?
    
    Returns: {"decision": "escalate"|"auto-send", "reason": str, "signals": list}
    """
    signals = []
    intent = classification.get("intent", "")
    confidence = classification.get("confidence", 0.0)
    text = classification.get("original_text", "").lower()
    
    # 1. Low classifier confidence
    if confidence < confidence_threshold:
        signals.append(f"low_confidence ({confidence:.2f} < {confidence_threshold})")
    
    # 2. Risk phrase detection (heuristic signals, NOT a risk model)
    matched_risk = [rp for rp in RISK_PHRASES if rp in text]
    if matched_risk:
        signals.append(f"risk_phrases: {matched_risk}")
    
    # 3. Retrieval match quality
    if retrieved_examples:
        best_sim = max(ex["similarity"] for ex in retrieved_examples)
        if best_sim < 0.15:
            signals.append(f"weak_retrieval (best_sim={best_sim:.3f})")
    else:
        signals.append("no_retrieval_results")
    
    if signals:
        return {
            "decision": "escalate",
            "reason": "; ".join(signals),
            "signals": signals
        }
    
    return {
        "decision": "auto-send",
        "reason": f"Confidence {confidence:.2f} above threshold; no risk signals; adequate retrieval match",
        "signals": []
    }

def quality_gate(reply_text: str, retrieved_examples: list) -> dict:
    """
    Decision B: Deterministic quality checks on the generated reply.
    
    Checks:
    1. Response non-empty, reasonable length
    2. Unsupported-claim check (promises not in retrieved context)
    3. No unsafe content
    
    Returns: {"decision": "pass"|"fail", "reason": str, "checks": dict}
    """
    checks = {}
    
    # Check 1: Non-empty and reasonable length
    if not reply_text or len(reply_text.strip()) < 10:
        checks["empty_or_short"] = True
        return {"decision": "fail", "reason": "Reply is empty or too short", "checks": checks}
    
    if len(reply_text) > 1000:
        checks["too_long"] = True
        return {"decision": "fail", "reason": "Reply exceeds reasonable length (>1000 chars)", "checks": checks}
    
    checks["length_ok"] = True
    
    # Check 2: Unsupported-claim detection
    # If reply contains action/promise tokens, verify they appear in retrieved context
    reply_lower = reply_text.lower()
    retrieved_text = " ".join(
        ex["support_reply"].lower() for ex in retrieved_examples
    ) if retrieved_examples else ""
    
    unsupported_claims = []
    for token in PROMISE_TOKENS:
        if re.search(token, reply_lower):
            if not re.search(token, retrieved_text):
                unsupported_claims.append(token)
    
    if unsupported_claims:
        checks["unsupported_claims"] = unsupported_claims
        return {
            "decision": "fail",
            "reason": f"Unsupported claims detected: {unsupported_claims}. Not evidenced in retrieved context.",
            "checks": checks
        }
    
    checks["claims_grounded"] = True
    
    # Check 3: Safety check
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, reply_lower):
            checks["unsafe_content"] = True
            return {"decision": "fail", "reason": f"Unsafe content detected matching: {pattern}", "checks": checks}
    
    checks["safety_ok"] = True
    
    return {"decision": "pass", "reason": "All quality checks passed", "checks": checks}

def final_decision(escalation_result: dict, gate_result: dict) -> dict:
    """
    Unified merge node:
    ESCALATE if escalation_policy says escalate OR quality_gate says fail.
    AUTO-SEND only if BOTH clear.
    """
    reasons = []
    
    if escalation_result["decision"] == "escalate":
        reasons.append(f"Escalation policy: {escalation_result['reason']}")
    
    if gate_result["decision"] == "fail":
        reasons.append(f"Quality gate: {gate_result['reason']}")
    
    if reasons:
        return {
            "action": "ESCALATE",
            "reason": " | ".join(reasons),
            "escalation_policy": escalation_result,
            "quality_gate": gate_result
        }
    
    return {
        "action": "AUTO-SEND",
        "reason": f"Both cleared: {escalation_result['reason']}; {gate_result['reason']}",
        "escalation_policy": escalation_result,
        "quality_gate": gate_result
    }

if __name__ == "__main__":
    # Test case: normal query
    cls = {"intent": "DELIVERY_STATUS_DELAY", "confidence": 0.92, "original_text": "Where is my package?"}
    retrieved = [{"support_reply": "We can help track that!", "similarity": 0.6, "inbound_text": "lost package"}]
    reply = "We're sorry for the delay! Please DM us your order number so we can look into it. ^AI"
    
    esc = escalation_policy(cls, retrieved)
    gate = quality_gate(reply, retrieved)
    decision = final_decision(esc, gate)
    print("Normal case:", decision)
    
    # Test case: risky query
    cls2 = {"intent": "FEEDBACK_SERVICE_COMPLAINT", "confidence": 0.55, "original_text": "I will sue you, this is fraud!"}
    esc2 = escalation_policy(cls2, [])
    gate2 = quality_gate(reply, [])
    decision2 = final_decision(esc2, gate2)
    print("\nRisky case:", decision2)
