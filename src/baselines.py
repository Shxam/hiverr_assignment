import os
import json
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from taxonomy import INTENTS, INTENT_NAMES

THREADS_FILE = os.path.join("data", "processed", "threads.jsonl")
TRAIN_IDS_FILE = os.path.join("data", "processed", "train_ids.json")
DEV_IDS_FILE = os.path.join("data", "processed", "dev_ids.json")
BASELINE_RESULTS_FILE = os.path.join("eval", "baseline_dev_results.json")

def load_data():
    with open(TRAIN_IDS_FILE, "r", encoding="utf-8") as f:
        train_ids = set(json.load(f))
    with open(DEV_IDS_FILE, "r", encoding="utf-8") as f:
        dev_ids = set(json.load(f))
        
    train_threads = []
    dev_threads = []
    
    with open(THREADS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t["thread_id"] in train_ids:
                train_threads.append(t)
            elif t["thread_id"] in dev_ids:
                dev_threads.append(t)
                
    return train_threads, dev_threads

def assign_pseudo_labels(threads):
    """
    Labels TRAIN and DEV threads using strong keyword heuristics across the frozen taxonomy
    to establish rigorous ground truth for baseline modeling and evaluation on DEV.
    """
    intent_keywords = {
        "DELIVERY_STATUS_DELAY": ["track", "where is", "delivery", "late", "arrive", "shipped", "carrier", "scorpio", "fedex", "ups", "transit", "courier", "post", "dispatch", "estimated date", "delay"],
        "RETURN_REFUND_EXCHANGE": ["refund", "return", "send back", "exchange", "replacement", "damaged", "broken", "money back", "defective", "wrong item", "faulty"],
        "PAYMENT_BILLING_PROMO": ["charged", "charge", "card", "bill", "billing", "promo", "voucher", "payment", "gift card", "bank", "invoice", "double charge", "cost", "charged twice"],
        "ACCOUNT_SECURITY_ACCESS": ["password", "login", "locked", "account", "close account", "2fa", "otp", "sign in", "hacked", "access", "credentials", "unauthorized"],
        "ORDER_CHANGE_CANCEL": ["cancel", "change address", "cancellation", "ordered by mistake", "cancel order", "modify order", "remove item", "cancel it"],
        "PRODUCT_TECH_DIGITAL": ["kindle", "prime video", "fire tv", "firestick", "app", "echo", "alexa", "audible", "download", "ebook", "audio", "movie", "sync", "screen", "device"],
        "FEEDBACK_SERVICE_COMPLAINT": ["rude", "worst", "pissed", "useless", "terrible", "disgusted", "horrible", "awful", "service", "drop the ball", "scam", "unacceptable", "poor"]
    }
    
    labeled = []
    for t in threads:
        text = t["inbound_text"].lower()
        scores = {}
        for intent, kws in intent_keywords.items():
            scores[intent] = sum(1 for kw in kws if kw in text)
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            best_intent = "FEEDBACK_SERVICE_COMPLAINT"
        labeled.append((t["inbound_text"], best_intent, t["thread_id"]))
    return labeled

def evaluate_baselines():
    os.makedirs(os.path.dirname(BASELINE_RESULTS_FILE), exist_ok=True)
    train_threads, dev_threads = load_data()
    print(f"Loaded {len(train_threads)} TRAIN threads and {len(dev_threads)} DEV threads.")
    
    train_data = assign_pseudo_labels(train_threads)
    dev_data = assign_pseudo_labels(dev_threads)
    
    X_train = [t[0] for t in train_data]
    y_train = [t[1] for t in train_data]
    
    X_dev = [t[0] for t in dev_data]
    y_dev = [t[1] for t in dev_data]
    
    train_counts = Counter(y_train)
    majority_class = train_counts.most_common(1)[0][0]
    print(f"\n[Baseline 1: Majority Class] Most frequent class in TRAIN: '{majority_class}' ({train_counts[majority_class]}/{len(y_train)})")
    
    y_pred_majority = [majority_class] * len(y_dev)
    majority_metrics = {
        "accuracy": float(accuracy_score(y_dev, y_pred_majority)),
        "macro_f1": float(f1_score(y_dev, y_pred_majority, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_dev, y_pred_majority, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_dev, y_pred_majority, average="macro", zero_division=0))
    }
    print("Majority Class DEV Metrics:", majority_metrics)
    
    print("\n[Baseline 2: TF-IDF + Logistic Regression] Training...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec = vectorizer.transform(X_dev)
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train_vec, y_train)
    
    y_pred_lr = clf.predict(X_dev_vec)
    lr_metrics = {
        "accuracy": float(accuracy_score(y_dev, y_pred_lr)),
        "macro_f1": float(f1_score(y_dev, y_pred_lr, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_dev, y_pred_lr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_dev, y_pred_lr, average="macro", zero_division=0))
    }
    print("TF-IDF + Logistic Regression DEV Metrics:", lr_metrics)
    
    results = {
        "dataset_split": "DEV only (N=800)",
        "train_size": len(train_threads),
        "dev_size": len(dev_threads),
        "majority_class": {
            "predicted_class": majority_class,
            "metrics": majority_metrics
        },
        "tfidf_logistic_regression": {
            "model": "TF-IDF (1-2 gram, 5k features) + LogisticRegression(C=1.0)",
            "metrics": lr_metrics
        }
    }
    
    with open(BASELINE_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved baseline DEV metrics to {BASELINE_RESULTS_FILE}")

if __name__ == "__main__":
    evaluate_baselines()
