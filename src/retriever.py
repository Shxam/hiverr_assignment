"""
TF-IDF cosine similarity retriever over TRAIN+DEV historical threads.
Retrieves top-k most similar resolved conversations for grounding reply generation.
GOLDEN ids are strictly excluded from the retrieval index.
"""
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THREADS_FILE = os.path.join("data", "processed", "threads.jsonl")
TRAIN_IDS_FILE = os.path.join("data", "processed", "train_ids.json")
DEV_IDS_FILE = os.path.join("data", "processed", "dev_ids.json")
GOLDEN_IDS_FILE = os.path.join("data", "processed", "golden_ids.json")

class HistoricalRetriever:
    def __init__(self):
        self.threads = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()
    
    def _build_index(self):
        with open(TRAIN_IDS_FILE, "r", encoding="utf-8") as f:
            train_ids = set(json.load(f))
        with open(DEV_IDS_FILE, "r", encoding="utf-8") as f:
            dev_ids = set(json.load(f))
        with open(GOLDEN_IDS_FILE, "r", encoding="utf-8") as f:
            golden_ids = set(json.load(f))
        
        allowed_ids = train_ids | dev_ids
        
        # Critical assertion: GOLDEN must never appear in retrieval pool
        assert not (allowed_ids & golden_ids), "GOLDEN ids leaked into retrieval pool!"
        
        with open(THREADS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                if t["thread_id"] in allowed_ids:
                    self.threads.append(t)
        
        print(f"Retriever index: {len(self.threads)} threads (TRAIN+DEV only, GOLDEN excluded)")
        
        texts = [t["inbound_text"] for t in self.threads]
        self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query_text: str, top_k: int = 3) -> list:
        """
        Returns top-k most similar historical threads with similarity scores.
        """
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "thread_id": self.threads[idx]["thread_id"],
                "inbound_text": self.threads[idx]["inbound_text"],
                "support_reply": self.threads[idx]["support_reply"],
                "similarity": float(sims[idx])
            })
        return results

if __name__ == "__main__":
    retriever = HistoricalRetriever()
    test_queries = [
        "Where is my package? It was supposed to arrive yesterday!",
        "I want a refund for the damaged item I received",
        "How do I cancel my Prime subscription?"
    ]
    for q in test_queries:
        results = retriever.retrieve(q, top_k=3)
        print(f"\nQuery: {q}")
        for i, r in enumerate(results):
            print(f"  [{i+1}] sim={r['similarity']:.3f} | Customer: {r['inbound_text'][:80]}...")
            print(f"       Support: {r['support_reply'][:80]}...")
