import os
import json
import random

THREADS_FILE = os.path.join("data", "processed", "threads.jsonl")
TRAIN_IDS_FILE = os.path.join("data", "processed", "train_ids.json")
DEV_IDS_FILE = os.path.join("data", "processed", "dev_ids.json")
GOLDEN_IDS_FILE = os.path.join("data", "processed", "golden_ids.json")

def split_threads(seed=42):
    random.seed(seed)
    with open(THREADS_FILE, "r", encoding="utf-8") as f:
        threads = [json.loads(line) for line in f]
        
    thread_ids = [t["thread_id"] for t in threads]
    random.shuffle(thread_ids)
    
    n = len(thread_ids)
    n_train = int(n * 0.60)
    n_dev = int(n * 0.20)
    
    train_ids = thread_ids[:n_train]
    dev_ids = thread_ids[n_train:n_train + n_dev]
    golden_ids = thread_ids[n_train + n_dev:]
    
    # Critical data leakage assertion:
    assert not (set(golden_ids) & set(train_ids) | set(golden_ids) & set(dev_ids)), "Leakage between GOLDEN and TRAIN/DEV!"
    assert not (set(train_ids) & set(dev_ids)), "Leakage between TRAIN and DEV!"
    
    with open(TRAIN_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)
    with open(DEV_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(dev_ids, f, indent=2)
    with open(GOLDEN_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(golden_ids, f, indent=2)
        
    print(f"Total threads: {n}")
    print(f"Train: {len(train_ids)} ({len(train_ids)/n*100:.1f}%)")
    print(f"Dev  : {len(dev_ids)} ({len(dev_ids)/n*100:.1f}%)")
    print(f"Golden: {len(golden_ids)} ({len(golden_ids)/n*100:.1f}%)")
    print("Assertion passed: No ID overlap between splits.")

if __name__ == "__main__":
    split_threads()
