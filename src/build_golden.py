import os
import json
import re
import random
from taxonomy import INTENTS, INTENT_NAMES

THREADS_FILE = os.path.join("data", "processed", "threads.jsonl")
GOLDEN_IDS_FILE = os.path.join("data", "processed", "golden_ids.json")
GOLDEN_OUTPUT = os.path.join("data", "golden", "golden.jsonl")

def build_golden_set():
    os.makedirs(os.path.dirname(GOLDEN_OUTPUT), exist_ok=True)
    with open(GOLDEN_IDS_FILE, "r", encoding="utf-8") as f:
        golden_ids = set(json.load(f))
        
    golden_threads = []
    with open(THREADS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t["thread_id"] in golden_ids:
                golden_threads.append(t)
                
    print(f"Loaded {len(golden_threads)} candidates from golden pool.")
    random.seed(1337)
    random.shuffle(golden_threads)
    
    intent_keywords = {
        "DELIVERY_STATUS_DELAY": ["track", "where is", "delivery", "late", "arrive", "shipped", "carrier", "scorpio", "fedex", "ups", "transit", "courier", "post", "dispatch"],
        "RETURN_REFUND_EXCHANGE": ["refund", "return", "send back", "exchange", "replacement", "damaged", "broken", "money back", "defective", "wrong item"],
        "PAYMENT_BILLING_PROMO": ["charged", "charge", "card", "bill", "billing", "promo", "voucher", "payment", "gift card", "bank", "invoice", "double charge", "cost"],
        "ACCOUNT_SECURITY_ACCESS": ["password", "login", "locked", "account", "close account", "2fa", "otp", "sign in", "hacked", "access", "credentials"],
        "ORDER_CHANGE_CANCEL": ["cancel", "change address", "cancellation", "ordered by mistake", "cancel order", "modify order", "remove item"],
        "PRODUCT_TECH_DIGITAL": ["kindle", "prime video", "fire tv", "firestick", "app", "echo", "alexa", "audible", "download", "ebook", "audio", "movie", "sync", "screen"],
        "FEEDBACK_SERVICE_COMPLAINT": ["rude", "worst", "pissed", "useless", "terrible", "disgusted", "horrible", "awful", "service", "drop the ball", "scam", "unacceptable"]
    }
    
    risk_phrases = ["lawyer", "attorney", "legal", "court", "sue", "police", "chargeback", "fraud", "scam", "bbb", "consumer protection", "stolen", "unacceptable", "furious"]
    
    buckets = {intent: [] for intent in INTENT_NAMES}
    high_risk_pool = []
    ambiguous_pool = []
    
    for t in golden_threads:
        text = t["inbound_text"].lower()
        matches = []
        for intent, kws in intent_keywords.items():
            if any(kw in text for kw in kws):
                matches.append(intent)
                
        is_high_risk = any(rp in text for rp in risk_phrases)
        if is_high_risk:
            high_risk_pool.append((t, matches[0] if matches else "FEEDBACK_SERVICE_COMPLAINT"))
        elif len(matches) > 1:
            ambiguous_pool.append((t, matches[0]))
        elif len(matches) == 1:
            buckets[matches[0]].append(t)
        else:
            # Fallback based on text heuristics
            buckets["FEEDBACK_SERVICE_COMPLAINT"].append(t)
            
    sampled_records = []
    
    # 1. Core stratified sampling: take up to 14 per bucket
    for intent, items in buckets.items():
        take = items[:14]
        for item in take:
            sampled_records.append({
                "thread_id": item["thread_id"],
                "inbound_text": item["inbound_text"],
                "support_reply": item["support_reply"],
                "intent": intent,
                "needs_human": False,
                "golden_reason": f"Standard inquiry regarding {intent.lower().replace('_', ' ')}.",
                "difficulty": "clear"
            })
            
    # 2. Ambiguous pool (22 examples)
    for item, primary_intent in ambiguous_pool[:22]:
        text = item["inbound_text"].lower()
        needs_human = any(rp in text for rp in risk_phrases) or "close" in text or "supervisor" in text
        sampled_records.append({
            "thread_id": item["thread_id"],
            "inbound_text": item["inbound_text"],
            "support_reply": item["support_reply"],
            "intent": primary_intent,
            "needs_human": needs_human,
            "golden_reason": f"Overlapping intent inquiry requiring contextual resolution; primary focus is {primary_intent}.",
            "difficulty": "ambiguous"
        })
        
    # 3. High risk escalation cases (18 examples)
    for item, primary_intent in high_risk_pool[:18]:
        sampled_records.append({
            "thread_id": item["thread_id"],
            "inbound_text": item["inbound_text"],
            "support_reply": item["support_reply"],
            "intent": primary_intent,
            "needs_human": True,
            "golden_reason": "Escalation triggered by strong legal/fraud risk phrases or severe service dissatisfaction.",
            "difficulty": "clear"
        })
        
    # 4. Fill exactly up to 150 from remaining golden pool
    used_ids = {r["thread_id"] for r in sampled_records}
    remaining = [t for t in golden_threads if t["thread_id"] not in used_ids]
    
    needed = 150 - len(sampled_records)
    print(f"Sampling {needed} from natural random pool to reach exactly 150...")
    
    for item in remaining[:needed]:
        text = item["inbound_text"].lower()
        assigned_intent = "FEEDBACK_SERVICE_COMPLAINT"
        for intent, kws in intent_keywords.items():
            if any(kw in text for kw in kws):
                assigned_intent = intent
                break
        needs_human = any(rp in text for rp in risk_phrases)
        sampled_records.append({
            "thread_id": item["thread_id"],
            "inbound_text": item["inbound_text"],
            "support_reply": item["support_reply"],
            "intent": assigned_intent,
            "needs_human": needs_human,
            "golden_reason": f"Natural-distribution case classified under {assigned_intent}.",
            "difficulty": "clear"
        })
        
    assert len(sampled_records) == 150, f"Expected 150, got {len(sampled_records)}"
    
    # Final schema verification
    for r in sampled_records:
        assert r["intent"] in INTENTS, f"Invalid intent: {r['intent']}"
        assert isinstance(r["needs_human"], bool)
        assert r["difficulty"] in ["clear", "ambiguous"]
        assert len(r["golden_reason"]) > 5
        
    with open(GOLDEN_OUTPUT, "w", encoding="utf-8") as out:
        for r in sampled_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Successfully saved exactly 150 validated golden records to {GOLDEN_OUTPUT}")
    print("SETTING ASIDE: In accordance with the Golden Rule, this file is untouched until Task 13.")

if __name__ == "__main__":
    build_golden_set()
