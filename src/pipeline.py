"""
Unified End-to-End Inference and Evaluation Pipeline.

Orchestrates:
1. Intent Classification (classifier.py)
2. Historical Case Retrieval (retriever.py)
3. Grounded Reply Generation (generator.py)
4. Escalation Policy - Decision A (escalation.py)
5. Deterministic Quality Gate - Decision B (escalation.py)
6. Unified Merge Node - Final Decision (escalation.py)
7. LLM Judge Evaluation (judge.py)
"""
import os
import json
import time
import numpy as np
from tqdm import tqdm
from taxonomy import INTENT_NAMES
from classifier import classify_tweet
from retriever import HistoricalRetriever
from generator import generate_reply
from escalation import escalation_policy, quality_gate, final_decision
from judge import judge_reply

THREADS_FILE = os.path.join("data", "processed", "threads.jsonl")
DEV_IDS_FILE = os.path.join("data", "processed", "dev_ids.json")
GOLDEN_FILE = os.path.join("data", "golden", "golden.jsonl")
GOLDEN_RESULTS_FILE = os.path.join("eval", "golden_results.json")

class SupportPipeline:
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold
        print("Initializing Support Pipeline...")
        self.retriever = HistoricalRetriever()
        print("Retriever initialized successfully.")

    def process_thread(self, thread: dict, run_judge: bool = True) -> dict:
        """
        Processes a single conversation thread end-to-end through the full pipeline.
        """
        thread_id = thread.get("thread_id", "unknown")
        inbound_text = thread.get("inbound_text", "")
        true_intent = thread.get("intent", None)
        needs_human = thread.get("needs_human", None)

        # Step 1: Classify intent
        cls_result = classify_tweet(inbound_text)
        pred_intent = cls_result.get("intent", "FEEDBACK_SERVICE_COMPLAINT")
        confidence = cls_result.get("confidence", 0.7)
        cls_result["original_text"] = inbound_text

        # Step 2: Retrieve similar historical resolved threads (TRAIN+DEV only)
        retrieved = self.retriever.retrieve(inbound_text, top_k=3)
        retrieved_context_str = "\n".join(
            f"Case {i+1}: {r['inbound_text']} -> {r['support_reply']}"
            for i, r in enumerate(retrieved)
        )

        # Step 3: Escalation Policy (Decision A)
        esc_result = escalation_policy(cls_result, retrieved, self.confidence_threshold)

        # Step 4: Generate grounded reply
        reply = generate_reply(inbound_text, pred_intent, retrieved)

        # Step 5: Deterministic Quality Gate (Decision B)
        gate_result = quality_gate(reply, retrieved)

        # Step 6: Unified Merge Node (Final Decision)
        merged = final_decision(esc_result, gate_result)

        # Step 7: LLM Judge Rubric Evaluation
        judge_scores = {}
        if run_judge:
            judge_scores = judge_reply(inbound_text, reply, retrieved_context_str)

        return {
            "thread_id": thread_id,
            "inbound_text": inbound_text,
            "true_intent": true_intent,
            "needs_human": needs_human,
            "difficulty": thread.get("difficulty", "clear"),
            "predicted_intent": pred_intent,
            "confidence": confidence,
            "intent_reasoning": cls_result.get("reasoning", ""),
            "retrieved_top_k": retrieved,
            "draft_reply": reply,
            "escalation_decision": esc_result["decision"],
            "escalation_reason": esc_result["reason"],
            "quality_gate_decision": gate_result["decision"],
            "quality_gate_reason": gate_result["reason"],
            "final_action": merged["action"],
            "final_reason": merged["reason"],
            "judge_scores": judge_scores
        }

def run_golden_evaluation():
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    
    # Assert data isolation
    with open(os.path.join("data", "processed", "train_ids.json"), "r") as f:
        train_ids = set(json.load(f))
    with open(os.path.join("data", "processed", "dev_ids.json"), "r") as f:
        dev_ids = set(json.load(f))
    with open(os.path.join("data", "processed", "golden_ids.json"), "r") as f:
        golden_ids = set(json.load(f))
    assert not (golden_ids & train_ids or golden_ids & dev_ids), "Data leakage detected!"
    
    pipeline = SupportPipeline(confidence_threshold=0.75)
    
    golden_threads = []
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            golden_threads.append(json.loads(line))
            
    print(f"\n=======================================================")
    print(f"RUNNING FROZEN ONE-SHOT GOLDEN EVALUATION (N={len(golden_threads)})")
    print(f"=======================================================\n")
    
    detailed_results = []
    y_true_intent = []
    y_pred_intent = []
    
    y_true_needs_human = []
    y_pred_escalate = []
    
    groundedness_scores = []
    correctness_scores = []
    tone_scores = []
    helpfulness_scores = []
    overall_judge_scores = []
    
    start_time = time.time()
    
    for thread in tqdm(golden_threads, desc="Evaluating Golden Set"):
        res = pipeline.process_thread(thread, run_judge=True)
        detailed_results.append(res)
        
        y_true_intent.append(res["true_intent"])
        y_pred_intent.append(res["predicted_intent"])
        
        true_needs_human = bool(res["needs_human"])
        pred_escalated = (res["final_action"] == "ESCALATE")
        
        y_true_needs_human.append(true_needs_human)
        y_pred_escalate.append(pred_escalated)
        
        j = res["judge_scores"]
        if j:
            groundedness_scores.append(j.get("groundedness", 3.0))
            correctness_scores.append(j.get("correctness", 3.0))
            tone_scores.append(j.get("tone", 4.0))
            helpfulness_scores.append(j.get("helpfulness", 3.0))
            overall_judge_scores.append(j.get("overall_score", 3.25))
            
    elapsed = time.time() - start_time
    
    # 1. Intent Metrics
    intent_acc = accuracy_score(y_true_intent, y_pred_intent)
    intent_macro_f1 = f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)
    intent_macro_p = precision_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)
    intent_macro_r = recall_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)
    intent_report = classification_report(y_true_intent, y_pred_intent, output_dict=True, zero_division=0)
    
    # 2. Escalation & Coverage Metrics
    total_cases = len(detailed_results)
    num_escalated = sum(y_pred_escalate)
    num_auto_send = total_cases - num_escalated
    
    auto_send_coverage = num_auto_send / total_cases
    escalation_rate = num_escalated / total_cases
    
    # Escalation Precision: Of cases flagged ESCALATE, how many truly needed human?
    # Escalation Recall: Of cases that truly needed human, how many were flagged ESCALATE?
    tp_esc = sum(1 for yt, yp in zip(y_true_needs_human, y_pred_escalate) if yt and yp)
    fp_esc = sum(1 for yt, yp in zip(y_true_needs_human, y_pred_escalate) if not yt and yp)
    fn_esc = sum(1 for yt, yp in zip(y_true_needs_human, y_pred_escalate) if yt and not yp)
    tn_esc = sum(1 for yt, yp in zip(y_true_needs_human, y_pred_escalate) if not yt and not yp)
    
    esc_precision = tp_esc / (tp_esc + fp_esc) if (tp_esc + fp_esc) > 0 else 0.0
    esc_recall = tp_esc / (tp_esc + fn_esc) if (tp_esc + fn_esc) > 0 else 0.0
    esc_f1 = 2 * (esc_precision * esc_recall) / (esc_precision + esc_recall) if (esc_precision + esc_recall) > 0 else 0.0
    
    # Safety on Auto-Send: Did any auto-send thread actually need a human?
    escaped_to_autosend = sum(1 for yt, yp in zip(y_true_needs_human, y_pred_escalate) if yt and not yp)
    autosend_safety_rate = 1.0 - (escaped_to_autosend / num_auto_send) if num_auto_send > 0 else 1.0
    
    # 3. Judge Metrics
    judge_summary = {
        "mean_groundedness": round(float(np.mean(groundedness_scores)), 3),
        "mean_correctness": round(float(np.mean(correctness_scores)), 3),
        "mean_tone": round(float(np.mean(tone_scores)), 3),
        "mean_helpfulness": round(float(np.mean(helpfulness_scores)), 3),
        "mean_overall_score": round(float(np.mean(overall_judge_scores)), 3)
    }
    
    summary = {
        "dataset": "Golden Evaluation Set (data/golden/golden.jsonl)",
        "sample_size": total_cases,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_pipeline_config": {
            "confidence_threshold": pipeline.confidence_threshold,
            "retrieval_pool_size": len(pipeline.retriever.threads),
            "guardrail_checks": ["length", "unsupported_claims", "safety"]
        },
        "intent_classification_metrics": {
            "accuracy": round(float(intent_acc), 4),
            "macro_f1": round(float(intent_macro_f1), 4),
            "macro_precision": round(float(intent_macro_p), 4),
            "macro_recall": round(float(intent_macro_r), 4),
            "per_class": intent_report
        },
        "escalation_and_coverage_metrics": {
            "total_cases": total_cases,
            "auto_send_count": num_auto_send,
            "auto_send_coverage": round(float(auto_send_coverage), 4),
            "escalated_count": num_escalated,
            "escalation_rate": round(float(escalation_rate), 4),
            "escalation_precision": round(float(esc_precision), 4),
            "escalation_recall": round(float(esc_recall), 4),
            "escalation_f1": round(float(esc_f1), 4),
            "escaped_to_autosend_count": escaped_to_autosend,
            "autosend_safety_precision": round(float(autosend_safety_rate), 4)
        },
        "judge_rubric_metrics": judge_summary,
        "runtime_seconds": round(elapsed, 2)
    }
    
    full_output = {
        "summary": summary,
        "detailed_results": detailed_results
    }
    
    os.makedirs(os.path.dirname(GOLDEN_RESULTS_FILE), exist_ok=True)
    with open(GOLDEN_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
        
    print(f"\nSaved final golden results to {GOLDEN_RESULTS_FILE}")
    print("\n================ SUMMARY RESULTS ================")
    print(f"Intent Macro-F1:       {summary['intent_classification_metrics']['macro_f1']:.4f}")
    print(f"Intent Accuracy:       {summary['intent_classification_metrics']['accuracy']:.4f}")
    print(f"Auto-Send Coverage:    {summary['escalation_and_coverage_metrics']['auto_send_coverage']*100:.1f}% ({num_auto_send}/{total_cases})")
    print(f"Escalation Recall:     {summary['escalation_and_coverage_metrics']['escalation_recall']*100:.1f}%")
    print(f"Escalation Precision:  {summary['escalation_and_coverage_metrics']['escalation_precision']*100:.1f}%")
    print(f"Auto-Send Safety:      {summary['escalation_and_coverage_metrics']['autosend_safety_precision']*100:.1f}%")
    print(f"Judge Groundedness:    {judge_summary['mean_groundedness']:.2f} / 5.0")
    print(f"Judge Correctness:     {judge_summary['mean_correctness']:.2f} / 5.0")
    print(f"Judge Tone:            {judge_summary['mean_tone']:.2f} / 5.0")
    print(f"Judge Helpfulness:     {judge_summary['mean_helpfulness']:.2f} / 5.0")
    print(f"Judge Overall:         {judge_summary['mean_overall_score']:.2f} / 5.0")
    print(f"Runtime:               {elapsed:.1f}s")
    print("=================================================\n")

if __name__ == "__main__":
    run_golden_evaluation()

