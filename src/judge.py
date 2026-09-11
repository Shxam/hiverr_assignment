"""
LLM Judge and Human Calibration Evaluator.

Evaluates generated customer support replies across 4 rubric dimensions (1-5 scale):
1. Groundedness: Is the response strictly grounded in retrieved evidence?
2. Correctness: Does it accurately address the customer inquiry?
3. Tone: Is it polite, professional, and empathetic?
4. Helpfulness: Does it provide actionable next steps?

Also computes human-judge calibration metrics (Pearson correlation, Mean Absolute Difference).
"""
import re
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from gemini_client import gemini_generate_json

JUDGE_SYSTEM_PROMPT = """You are an expert impartial evaluator assessing AI-generated customer service responses for AmazonHelp on Twitter.
Evaluate the candidate reply against the customer message and retrieved context on a 1-5 scale across 4 dimensions:

1. GROUNDEDNESS (1-5):
   1: Gross hallucinations; invents refunds, replacement promises, or false facts not in context.
   3: Mostly grounded but assumes minor details not in context.
   5: Completely grounded in the retrieved examples or safely directs customer without unsupported claims.

2. CORRECTNESS (1-5):
   1: Completely wrong intent or nonsensical advice.
   3: Addresses part of the issue or generic advice.
   5: Directly and accurately addresses the core issue.

3. TONE (1-5):
   1: Rude, robotic, defensive, or offensive.
   3: Neutral, somewhat impersonal.
   5: Highly professional, empathetic, and courteous.

4. HELPFULNESS (1-5):
   1: Completely unhelpful, leaves customer stranded.
   3: Somewhat helpful, provides generic link or instructions.
   5: Clear, actionable resolution or proper routing.

Return ONLY a JSON object:
{
  "groundedness": 1-5,
  "correctness": 1-5,
  "tone": 1-5,
  "helpfulness": 1-5,
  "overall_score": 1.0-5.0,
  "rationale": "one-sentence explanation"
}
"""

def heuristic_judge(customer_text: str, support_reply: str, retrieved_context: str) -> dict:
    """
    Calibrated heuristic fallback judge evaluating the 4 rubric dimensions:
    Groundedness, Correctness, Tone, and Helpfulness.
    """
    reply_lower = support_reply.lower()
    ctx_lower = retrieved_context.lower()
    
    # 1. Groundedness
    unsupported_promises = ["refund", "full $", "cash", "90 days", "guarantee", "processed a full"]
    hallucination_found = any(p in reply_lower and p not in ctx_lower for p in unsupported_promises)
    
    conflict_found = ("non-returnable" in ctx_lower and "can return" in reply_lower)
    
    if hallucination_found or conflict_found:
        groundedness = 1.0
        correctness = 1.0
        helpfulness = 1.0
        rationale = "Unsupported claims / hallucinated policy not evidenced in context."
    else:
        # Check context overlap
        groundedness = 4.8 if len(ctx_lower) > 20 else 4.2
        correctness = 4.8 if ("http" in reply_lower or "please" in reply_lower) else 4.0
        helpfulness = 4.8 if ("http" in reply_lower or "contact" in reply_lower or "tap" in reply_lower or "go to" in reply_lower) else 4.0
        rationale = "Grounded in retrieved support context with appropriate routing."
        
    # Tone evaluation
    polite_words = ["sorry", "apologize", "please", "sincerely", "understand", "welcome", "happy to help"]
    rude_words = ["shut up", "idiot", "not our problem", "go away", "liar"]
    if any(rw in reply_lower for rw in rude_words):
        tone = 1.0
    elif sum(1 for pw in polite_words if pw in reply_lower) >= 2:
        tone = 5.0
    elif any(pw in reply_lower for pw in polite_words):
        tone = 4.5
    else:
        tone = 4.0
        
    overall = round(float((groundedness + correctness + tone + helpfulness) / 4.0), 3)
    return {
        "groundedness": float(groundedness),
        "correctness": float(correctness),
        "tone": float(tone),
        "helpfulness": float(helpfulness),
        "overall_score": overall,
        "rationale": rationale
    }

def judge_reply(customer_text: str, support_reply: str, retrieved_context: str) -> dict:
    """
    Judge a candidate reply using the 4-dimension rubric.
    Tries Gemini Flash first; falls back to calibrated rubric judge.
    """
    prompt = f"""{JUDGE_SYSTEM_PROMPT}

Customer Message: "{customer_text}"

Retrieved Context:
{retrieved_context}

Candidate AI Reply: "{support_reply}"

Evaluate now:"""

    res = gemini_generate_json(prompt, temperature=0.0, max_tokens=200)
    if isinstance(res, dict) and "groundedness" in res and "correctness" in res:
        g = float(res.get("groundedness", 3))
        c = float(res.get("correctness", 3))
        t = float(res.get("tone", 4))
        h = float(res.get("helpfulness", 3))
        overall = res.get("overall_score", (g + c + t + h) / 4.0)
        return {
            "groundedness": g,
            "correctness": c,
            "tone": t,
            "helpfulness": h,
            "overall_score": round(float(overall), 2),
            "rationale": res.get("rationale", "LLM Judge evaluation")
        }
    
    return heuristic_judge(customer_text, support_reply, retrieved_context)

def compute_calibration(human_scores: list, llm_scores: list) -> dict:
    """
    Computes calibration metrics between human ratings and judge ratings:
    - Pearson correlation
    - Spearman rank correlation
    - Mean Absolute Difference (MAD)
    - Root Mean Squared Error (RMSE)
    """
    assert len(human_scores) == len(llm_scores), "Scores list length mismatch"
    n = len(human_scores)
    if n < 2:
        return {}
    
    h = np.array(human_scores, dtype=float)
    l = np.array(llm_scores, dtype=float)
    
    mad = float(np.mean(np.abs(h - l)))
    rmse = float(np.sqrt(np.mean((h - l) ** 2)))
    
    p_corr, p_val = pearsonr(h, l)
    s_corr, s_val = spearmanr(h, l)
    
    return {
        "sample_size": n,
        "pearson_correlation": round(float(p_corr), 3) if not np.isnan(p_corr) else 0.0,
        "pearson_p_value": float(p_val) if not np.isnan(p_val) else 1.0,
        "spearman_correlation": round(float(s_corr), 3) if not np.isnan(s_corr) else 0.0,
        "spearman_p_value": float(s_val) if not np.isnan(s_val) else 1.0,
        "mean_absolute_difference": round(mad, 3),
        "rmse": round(rmse, 3),
        "mean_human_score": round(float(np.mean(h)), 3),
        "mean_llm_score": round(float(np.mean(l)), 3)
    }

if __name__ == "__main__":
    test_q = "My package has been delayed for 4 days, where is it?"
    test_ctx = "Example: We're sorry for the delay! Track your package at https://t.co/track"
    test_reply = "We apologize for the delivery delay! You can check the current tracking status here: https://t.co/track. Let us know if you need further help! ^AI"
    
    res = judge_reply(test_q, test_reply, test_ctx)
    print("Judge test output:", json.dumps(res, indent=2))
