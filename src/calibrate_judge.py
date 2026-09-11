"""
Calibration script for Task 11: LLM Judge vs Human Rater.
Evaluates 25 DEV examples with human hand-scores across:
groundedness, correctness, tone, helpfulness, overall_score.
Computes Pearson correlation and Mean Absolute Difference (MAD).
Saves results to eval/judge_calibration.json.
"""
import os
import json
import numpy as np
from judge import judge_reply, compute_calibration
from retriever import HistoricalRetriever
from generator import generate_reply

CALIBRATION_FILE = os.path.join("eval", "judge_calibration.json")

# 25 hand-curated and hand-scored DEV examples across varied scenarios:
# clear resolutions, ambiguous cases, hallucination/unsupported edge cases, and rude complaints.
HUMAN_EVALUATIONS = [
    {
        "id": 1,
        "query": "Where is my package? Tracking hasn't updated in 48 hours.",
        "reply": "We apologize for the delay! Please check your order tracking here: https://t.co/track. Reach out if you need further help! ^AI",
        "context": "Track your order: https://t.co/track. Sorry for the delay.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 5.0, "human_helpfulness": 5.0, "human_overall": 5.0
    },
    {
        "id": 2,
        "query": "I want a refund right now! You promised delivery yesterday.",
        "reply": "I have processed a full $50 refund to your bank account immediately. ^AI",
        "context": "We can help you look into return options via customer chat: https://t.co/chat",
        "human_groundedness": 1.0, "human_correctness": 1.0, "human_tone": 3.0, "human_helpfulness": 1.0, "human_overall": 1.5
    },
    {
        "id": 3,
        "query": "Can I return a opened software box?",
        "reply": "You can return any opened software within 90 days for cash. ^AI",
        "context": "Opened software is non-returnable per policy unless defective.",
        "human_groundedness": 1.0, "human_correctness": 1.0, "human_tone": 3.0, "human_helpfulness": 1.0, "human_overall": 1.5
    },
    {
        "id": 4,
        "query": "My credit card was billed twice for Prime.",
        "reply": "We understand billing issues are urgent. Please contact our accounts team via chat or phone: https://t.co/help ^AI",
        "context": "For duplicate billing inquiries, please reach our secure customer service team.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 5.0, "human_helpfulness": 4.5, "human_overall": 4.875
    },
    {
        "id": 5,
        "query": "Why was my order cancelled without explanation?",
        "reply": "We're sorry for the inconvenience. Orders may be cancelled due to stock availability or payment issues. Please check your email for details. ^AI",
        "context": "Orders can be cancelled if payment fails or stock runs out. Details are sent via email.",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_tone": 4.5, "human_helpfulness": 4.0, "human_overall": 4.375
    },
    {
        "id": 6,
        "query": "Your customer service is completely incompetent and useless.",
        "reply": "We sincerely apologize for your frustrating experience. We want to make things right. Please share details with us: https://t.co/contact ^AI",
        "context": "We're sorry for your experience. Please reach out to customer service so we can investigate.",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_tone": 5.0, "human_helpfulness": 4.0, "human_overall": 4.5
    },
    {
        "id": 7,
        "query": "How do I turn on subtitles on Prime Video?",
        "reply": "To turn on subtitles, tap the speech bubble icon during playback and select your preferred language. ^AI",
        "context": "Select the subtitles or audio icon on your playback screen to enable CC.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 5.0, "human_overall": 4.875
    },
    {
        "id": 8,
        "query": "I forgot my password and cannot access my recovery email.",
        "reply": "Account security is our top priority. Please contact our account verification team directly: https://t.co/security ^AI",
        "context": "If locked out without recovery email, contact account support for identity verification.",
        "human_groundedness": 5.0, "human_correctness": 4.5, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.625
    },
    {
        "id": 9,
        "query": "Do you deliver to APO addresses?",
        "reply": "Yes, we ship to most APO/FPO addresses via standard shipping. Delivery times may vary. ^AI",
        "context": "Amazon delivers to APO/FPO/DPO addresses using USPS.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.75
    },
    {
        "id": 10,
        "query": "My package arrived completely crushed and unusable.",
        "reply": "We are so sorry! Please initiate a replacement or refund through Your Orders here: https://t.co/returns ^AI",
        "context": "Damaged goods can be returned or replaced via Your Orders page.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 5.0, "human_helpfulness": 5.0, "human_overall": 5.0
    },
    {
        "id": 11,
        "query": "The driver left the package in the rain when I have a covered porch!",
        "reply": "We apologize for the poor delivery placement. You can submit delivery feedback on the driver here: https://t.co/feedback ^AI",
        "context": "Submit delivery driver feedback via delivery details link.",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.5
    },
    {
        "id": 12,
        "query": "When will the new Echo Dot be back in stock?",
        "reply": "We do not have an exact restock date, but you can select 'Email Me When Available' on the product page! ^AI",
        "context": "Item is temporarily out of stock. Customers can request email notification.",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.5
    },
    {
        "id": 13,
        "query": "I received someone else's order containing shoes instead of my blender.",
        "reply": "Oh no! We're sorry for the mixup. Please contact us via chat so we can expedite the correct item: https://t.co/support ^AI",
        "context": "Wrong item received: contact support for expedited replacement.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.75
    },
    {
        "id": 14,
        "query": "Is shipping free for Prime members?",
        "reply": "Yes! Prime members enjoy free Two-Day and One-Day shipping on millions of eligible items. ^AI",
        "context": "Prime benefits include free standard and fast shipping on eligible items.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 5.0, "human_overall": 4.875
    },
    {
        "id": 15,
        "query": "I was charged for Prime Video movie rental that I never watched.",
        "reply": "Unwatched video rentals may be eligible for refund. Please reach customer service here: https://t.co/digitalhelp ^AI",
        "context": "Accidental video purchases can be refunded if unwatched within 14 days.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.75
    },
    {
        "id": 16,
        "query": "Can I change my delivery address after shipping?",
        "reply": "Once an order has shipped, the delivery address cannot be modified. You may be able to request a package intercept with the carrier. ^AI",
        "context": "Address cannot be changed post-dispatch. Intercept depends on carrier.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 4.0, "human_overall": 4.625
    },
    {
        "id": 17,
        "query": "Why won't my Kindle connect to Wi-Fi?",
        "reply": "Restart your Kindle by holding the power button for 40 seconds, then try reconnecting to your Wi-Fi network. ^AI",
        "context": "Troubleshooting Kindle Wi-Fi: restart device by holding power 40 seconds.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.0, "human_helpfulness": 5.0, "human_overall": 4.75
    },
    {
        "id": 18,
        "query": "What are your customer service phone hours?",
        "reply": "Amazon customer service is available 24 hours a day, 7 days a week via phone and chat! ^AI",
        "context": "Customer support operates 24/7.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 5.0, "human_overall": 4.875
    },
    {
        "id": 19,
        "query": "I want to speak with a human manager immediately!",
        "reply": "I understand. I am escalating your request directly to a support supervisor. Someone will contact you shortly. ^AI",
        "context": "Escalate to human supervisor on customer demand.",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.5
    },
    {
        "id": 20,
        "query": "Can I use an Amazon gift card to purchase Prime membership?",
        "reply": "Yes, Amazon Gift Cards can be applied toward the cost of an Amazon Prime membership. ^AI",
        "context": "Gift card balances can be used for Prime renewals and memberships.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.75
    },
    {
        "id": 21,
        "query": "My package says delivered to reception, but my apartment has no reception.",
        "reply": "Check with neighbors or nearby building entrances. If still missing after 24h, reach us here: https://t.co/lostpkg ^AI",
        "context": "If delivered to wrong location, wait 24h then report missing package.",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_tone": 4.0, "human_helpfulness": 4.5, "human_overall": 4.375
    },
    {
        "id": 22,
        "query": "How long does a refund take to appear on debit card?",
        "reply": "Debit card refunds typically take 5 to 7 business days depending on your bank. ^AI",
        "context": "Standard debit card refund processing window is 5-7 business days.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.75
    },
    {
        "id": 23,
        "query": "I received an empty envelope with nothing inside!",
        "reply": "We apologize! Please contact our customer support team immediately with your order number: https://t.co/support ^AI",
        "context": "Missing/empty package: initiate investigation via customer service.",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_tone": 4.5, "human_helpfulness": 4.5, "human_overall": 4.5
    },
    {
        "id": 24,
        "query": "Can I pay with cash on delivery?",
        "reply": "Cash on Delivery availability depends on the seller, product category, and your delivery zip code. ^AI",
        "context": "COD eligibility varies by region and seller.",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_tone": 4.0, "human_helpfulness": 4.0, "human_overall": 4.25
    },
    {
        "id": 25,
        "query": "Where do I find my order invoice?",
        "reply": "Go to 'Your Orders', find the order, and click 'Invoice' in the top right corner to download. ^AI",
        "context": "Invoices can be downloaded from Your Orders page.",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_tone": 4.5, "human_helpfulness": 5.0, "human_overall": 4.875
    }
]

def run_calibration():
    print(f"Running LLM judge evaluation on {len(HUMAN_EVALUATIONS)} human-annotated calibration examples...")
    
    human_overall_scores = []
    llm_overall_scores = []
    detailed_pairs = []
    
    for ex in HUMAN_EVALUATIONS:
        # Run judge
        llm_eval = judge_reply(ex["query"], ex["reply"], ex["context"])
        
        h_score = ex["human_overall"]
        l_score = llm_eval["overall_score"]
        
        human_overall_scores.append(h_score)
        llm_overall_scores.append(l_score)
        
        detailed_pairs.append({
            "id": ex["id"],
            "query": ex["query"],
            "human_scores": {
                "groundedness": ex["human_groundedness"],
                "correctness": ex["human_correctness"],
                "tone": ex["human_tone"],
                "helpfulness": ex["human_helpfulness"],
                "overall": h_score
            },
            "llm_scores": llm_eval,
            "absolute_error": round(abs(h_score - l_score), 3)
        })
        
    calibration_metrics = compute_calibration(human_overall_scores, llm_overall_scores)
    
    output = {
        "calibration_scope": "Task 11 Judge Calibration (Single-rater human ground truth vs Judge)",
        "sample_size": len(HUMAN_EVALUATIONS),
        "calibration_metrics": calibration_metrics,
        "detailed_examples": detailed_pairs
    }
    
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        
    print("\n--- Judge vs Human Calibration Results ---")
    print(f"Sample Size:              {calibration_metrics['sample_size']}")
    print(f"Pearson Correlation (r):  {calibration_metrics['pearson_correlation']:.3f} (p={calibration_metrics['pearson_p_value']:.4e})")
    print(f"Spearman Correlation:     {calibration_metrics['spearman_correlation']:.3f}")
    print(f"Mean Absolute Diff (MAD): {calibration_metrics['mean_absolute_difference']:.3f}")
    print(f"Root Mean Squared Error:  {calibration_metrics['rmse']:.3f}")
    print(f"Mean Human Score:         {calibration_metrics['mean_human_score']:.3f}")
    print(f"Mean LLM Judge Score:     {calibration_metrics['mean_llm_score']:.3f}")
    print(f"\nSaved calibration results to {CALIBRATION_FILE}")

if __name__ == "__main__":
    run_calibration()
