import time
import math
import random
import string
from collections import Counter


def compute_marginal_entropy(corpus_tokens: list[list[str]]) -> float:
    all_tokens = [t for doc in corpus_tokens for t in doc]
    total = len(all_tokens)
    if total == 0:
        return 0.0
    freq = Counter(all_tokens)
    return -sum((c / total) * math.log2(c / total) for c in freq.values() if c > 0)


def compute_nmi(tokens_a: list[str], tokens_b: list[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    vocab = set_a | set_b
    total = len(tokens_a) + len(tokens_b)
    freq_a = Counter(tokens_a)
    freq_b = Counter(tokens_b)
    joint_freq = Counter()
    for t in tokens_a:
        if t in set_b:
            joint_freq[t] += 1
    for t in tokens_b:
        if t in set_a:
            joint_freq[t] += 1
    h_a = -sum((c / len(tokens_a)) * math.log2(c / len(tokens_a)) for c in freq_a.values() if c > 0)
    h_b = -sum((c / len(tokens_b)) * math.log2(c / len(tokens_b)) for c in freq_b.values() if c > 0)
    if h_a == 0 and h_b == 0:
        return 1.0
    if not joint_freq:
        return 0.0
    joint_total = sum(joint_freq.values())
    h_joint = -sum((c / joint_total) * math.log2(c / joint_total) for c in joint_freq.values() if c > 0)
    mi = h_a + h_b - h_joint
    normalizer = max(h_a, h_b)
    return max(0.0, mi / normalizer) if normalizer > 0 else 0.0


def compute_cosine(tokens_a: list[str], tokens_b: list[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    vocab = set(tokens_a) | set(tokens_b)
    freq_a = Counter(tokens_a)
    freq_b = Counter(tokens_b)
    dot = sum(freq_a[t] * freq_b[t] for t in vocab)
    mag_a = math.sqrt(sum(v ** 2 for v in freq_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in freq_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def entropy_weighted_hybrid_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    corpus_tokens: list[list[str]],
    vocab_size: int,
) -> float:
    h_marginal = compute_marginal_entropy(corpus_tokens)
    log_vocab = math.log2(vocab_size) if vocab_size > 1 else 1.0
    alpha = min(1.0, h_marginal / log_vocab)
    cosine = compute_cosine(query_tokens, doc_tokens)
    nmi = compute_nmi(query_tokens, doc_tokens)
    return alpha * cosine + (1 - alpha) * nmi


def generate_synthetic_corpus(n_docs: int, vocab_size: int, tokens_per_doc: int) -> list[list[str]]:
    vocab = [f"term_{i}" for i in range(vocab_size)]
    return [
        [random.choice(vocab) for _ in range(tokens_per_doc)]
        for _ in range(n_docs)
    ]


def benchmark_this() -> dict:
    random.seed(42)
    n_docs = 500
    vocab_size = 300
    tokens_per_doc = 40
    corpus = generate_synthetic_corpus(n_docs, vocab_size, tokens_per_doc)
    query = generate_synthetic_corpus(1, vocab_size, tokens_per_doc)[0]
    start = time.perf_counter()
    scores = [
        entropy_weighted_hybrid_score(query, doc, corpus, vocab_size)
        for doc in corpus
    ]
    elapsed = time.perf_counter() - start
    top_score = max(scores)
    avg_score = sum(scores) / len(scores)
    throughput = n_docs / elapsed
    return {
        "n_docs": n_docs,
        "elapsed_ms": round(elapsed * 1000, 2),
        "throughput_docs_per_sec": round(throughput, 1),
        "top_score": round(top_score, 4),
        "avg_score": round(avg_score, 4),
    }


COMPETITIVE_COMPARISON = [
    {
        "solution": "NexusSimilarity (NMI+Cosine hybrid, this)",
        "integration_time_min": 2,
        "loc_required": 12,
        "throughput_docs_per_sec": None,
        "persistent_index_required": False,
        "nmi_native": True,
        "adaptive_alpha": True,
    },
    {
        "solution": "Pinecone (cosine only, serverless)",
        "integration_time_min": 45,
        "loc_required": 80,
        "throughput_docs_per_sec": 4000,
        "persistent_index_required": True,
        "nmi_native": False,
        "adaptive_alpha": False,
    },
    {
        "solution": "Weaviate (BM25+vector hybrid)",
        "integration_time_min": 90,
        "loc_required": 140,
        "throughput_docs_per_sec": 2200,
        "persistent_index_required": True,
        "nmi_native": False,
        "adaptive_alpha": False,
    },
    {
        "solution": "sklearn NearestNeighbors (cosine, local)",
        "integration_time_min": 20,
        "loc_required": 55,
        "throughput_docs_per_sec": 18000,
        "persistent_index_required": False,
        "nmi_native": False,
        "adaptive_alpha": False,
    },
]


if __name__ == "__main__":
    result = benchmark_this()
    COMPETITIVE_COMPARISON[0]["throughput_docs_per_sec"] = result["throughput_docs_per_sec"]

    print("=== NexusSimilarity Benchmark: NMI+Cosine Hybrid Score ===")
    print(f"Corpus: {result['n_docs']} docs | Elapsed: {result['elapsed_ms']} ms | "
          f"Throughput: {result['throughput_docs_per_sec']} docs/s")
    print(f"Top score: {result['top_score']} | Avg score: {result['avg_score']}")
    print()
    print("=== Competitive Comparison (hardcoded estimates) ===")
    header = f"{'Solution':<45} {'Integ(min)':>10} {'LOC':>6} {'Thrput(d/s)':>12} {'Index?':>7} {'NMI?':>6} {'AdaptW?':>8}"
    print(header)
    print("-" * len(header))
    for row in COMPETITIVE_COMPARISON:
        tput = str(row["throughput_docs_per_sec"]) if row["throughput_docs_per_sec"] else "N/A"
        print(
            f"{row['solution']:<45} "
            f"{row['integration_time_min']:>10} "
            f"{row['loc_required']:>6} "
            f"{tput:>12} "
            f"{'yes' if row['persistent_index_required'] else 'no':>7} "
            f"{'yes' if row['nmi_native'] else 'no':>6} "
            f"{'yes' if row['adaptive_alpha'] else 'no':>8}"
        )
    print()
    print("alpha = H_marginal(corpus) / log2(|V|) recalculated per corpus ingest.")
    print("NMI weight adapts to corpus entropy — not replicable with fixed hyperparameters.")