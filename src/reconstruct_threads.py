import os
import csv
import json
import re
from tqdm import tqdm

RAW_CSV = os.path.join("data", "raw", "twcs.csv")
OUTPUT_JSONL = os.path.join("data", "processed", "threads.jsonl")
TARGET_BRAND = "AmazonHelp"
MAX_THREADS = 4000

def is_clean_english(text):
    """
    Ensures text is primarily English: high ASCII letter ratio, low foreign non-ASCII.
    """
    letters = re.findall(r'[a-zA-Z]', text)
    if len(letters) < 15:
        return False
    non_ascii = len(re.findall(r'[^\x00-\x7F]', text))
    if non_ascii > 2:
        return False
    # Check for common German, Spanish, French, Italian words
    foreign_words = {' beim ', ' ich ', ' und ', ' der ', ' die ', ' das ', ' pedido ', ' entrega ', ' para ', ' avec ', ' vous ', ' sono '}
    text_lower = f" {text.lower()} "
    if any(w in text_lower for w in foreign_words):
        return False
    return True

def reconstruct_brand_threads():
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    print(f"Indexing brand replies for {TARGET_BRAND} from {RAW_CSV}...")
    
    brand_replies = []
    needed_tweet_ids = set()
    
    with open(RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Scanning replies", total=2811774):
            if row["author_id"] == TARGET_BRAND and row["inbound"] == "False":
                parent_id = row["in_response_to_tweet_id"].strip()
                if parent_id and parent_id != "\\N":
                    brand_text = row["text"].strip()
                    if is_clean_english(brand_text):
                        brand_replies.append({
                            "reply_id": row["tweet_id"],
                            "parent_id": parent_id,
                            "brand_text": brand_text,
                            "created_at": row["created_at"]
                        })
                        needed_tweet_ids.add(parent_id)
            if len(brand_replies) >= 40000:
                break
                
    print(f"Collected {len(brand_replies)} English replies for {TARGET_BRAND}. Needed parent IDs: {len(needed_tweet_ids)}")
    
    print("Resolving customer inbound tweets...")
    parent_tweets = {}
    with open(RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Resolving inbounds", total=2811774):
            tid = row["tweet_id"].strip()
            if tid in needed_tweet_ids and row["inbound"] == "True":
                inbound_text = row["text"].strip()
                if is_clean_english(inbound_text):
                    parent_tweets[tid] = row
            if len(parent_tweets) >= len(needed_tweet_ids):
                break
                
    print(f"Resolved {len(parent_tweets)} valid English customer inbound tweets.")
    
    valid_threads = []
    seen_inbounds = set()
    
    for reply in brand_replies:
        pid = reply["parent_id"]
        if pid not in parent_tweets:
            continue
            
        parent = parent_tweets[pid]
        inbound_text = parent["text"].strip()
        reply_text = reply["brand_text"].strip()
        
        # Deduplication
        norm_inbound = re.sub(r'\s+', ' ', inbound_text.lower())
        if norm_inbound in seen_inbounds:
            continue
        seen_inbounds.add(norm_inbound)
        
        thread = {
            "thread_id": f"amzn_{pid}_{reply['reply_id']}",
            "parent_tweet_id": pid,
            "reply_tweet_id": reply["reply_id"],
            "company": TARGET_BRAND,
            "inbound_text": inbound_text,
            "support_reply": reply_text,
            "created_at": parent.get("created_at", ""),
            "conversation": f"Customer: {inbound_text}\nSupport: {reply_text}"
        }
        valid_threads.append(thread)
        if len(valid_threads) >= MAX_THREADS:
            break
            
    print(f"Total reconstructed clean English threads: {len(valid_threads)}")
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for t in valid_threads:
            out.write(json.dumps(t, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(valid_threads)} threads to {OUTPUT_JSONL}")

if __name__ == "__main__":
    reconstruct_brand_threads()
