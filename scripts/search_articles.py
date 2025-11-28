import os
import re
import math
import json
import logging
import sys
from collections import defaultdict
from typing import List, Dict, Any

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    filename="search.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# Tokenize helper
# ----------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

# ----------------------------
# TF, DF, IDF
# ----------------------------
def compute_tf(tokens: List[str]) -> Dict[str, float]:
    counts = defaultdict(int)
    for token in tokens:
        counts[token] += 1
    length = max(1, len(tokens))
    return {token: counts[token] / length for token in counts}

def compute_df(doc_tokens: List[List[str]]) -> Dict[str, float]:
    df = defaultdict(int)
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] += 1
    return df

def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = compute_tf(tokens)
    return {t: tf[t] * idf.get(t, 0.0) for t in tf}

# ----------------------------
# Cosine similarity
# ----------------------------
def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return dot / (na * nb + 1e-12)

# ----------------------------
# Load processed corpus
# ----------------------------
def load_processed(processed_dir="../data/processed") -> List[Dict[str, Any]]:
    files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError("No processed corpus found. Run preprocess_articles.py first.")
    files.sort(reverse=True)
    latest = os.path.join(processed_dir, files[0])
    with open(latest, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    logging.info(f"Loaded processed corpus: {latest} with {len(corpus)} docs")
    return corpus  # corpus is a list of dicts

# ----------------------------
# Search function
# ----------------------------
def search_corpus(query: str, corpus: List[Dict[str, Any]], k: int = 5):
    docs = [{"id": d["id"], "tokens": d.get("tokens", [])} for d in corpus]
    doc_tokens = [d["tokens"] for d in docs]
    df = compute_df(doc_tokens)
    n_docs = len(doc_tokens)
    idf = {t: math.log((n_docs + 1) / (df[t] + 0.5)) + 1 for t in df}
    doc_vecs = [tfidf_vector(d["tokens"], idf) for d in docs]

    q_tokens = tokenize(query)
    q_vec = tfidf_vector(q_tokens, idf)

    scored = [(cosine(q_vec, v), i) for i, v in enumerate(doc_vecs)]
    scored.sort(reverse=True)

    results = []
    for score, idx in scored[:k]:
        results.append({
            "id": docs[idx]["id"],
            "score": score,
            "snippet": " ".join(docs[idx]["tokens"][:30])  # preview first 30 tokens
        })
    return results

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/search_articles.py \"your query\" [top_k]")
        sys.exit(1)

    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    corpus = load_processed()
    results = search_corpus(query, corpus, k=k)

    print(f"\nTop {k} results for query: {query}\n")
    for r in results:
        print(f"Doc: {r['id']} | Score: {r['score']:.4f}")
        print(f"Snippet: {r['snippet']}...\n")
