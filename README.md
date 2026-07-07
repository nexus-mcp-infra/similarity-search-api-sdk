# Similarity Search API

Statistical significance built into every similarity score.

---

## Install

```bash
pip install similarity-search-client
```

---

## Search in 3 lines

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")
results = client.search(query="transformer attention mechanism", corpus=my_docs, top_k=10)
```

Every result includes a composite score, a p-value, and the raw cosine and NMI signals — no extra calls, no config.

---

## The problem with cosine similarity

Every vector database returns cosine scores. None of them tell you whether those scores mean anything.

Two embeddings can share a cosine of 0.91 because they describe the same concept — or because they both live in a dense region of the embedding space that your model overweights. You cannot distinguish those cases from the score alone. You act on a false positive, surface irrelevant results, and your downstream precision degrades silently.

**Similarity Search API solves this at the score level, not at the infra level.**

---

## How it works

Each similarity score is a composite of two independent signals:

**Cosine similarity** measures geometric proximity in the embedding space — fast, standard, familiar.

**Normalized Mutual Information (NMI)** measures informational dependence between the activation distributions of the two embeddings. Given an embedding vector of dimension `D`, the API partitions it into `k = sqrt(D)` segments and builds an empirical magnitude distribution per segment using Freedman-Diaconis binning. NMI is then:

```
NMI(P, Q) = MI(P, Q) / sqrt(H(P) * H(Q))
```

where `H` is Shannon entropy. NMI detects shared structure that survives across the distributional shape of the embedding — not just its direction.

**The composite score** fuses both signals:

```
S = alpha * cosine + (1 - alpha) * NMI
```

`alpha` defaults to `0.6` and is adjustable per call. The p-value is computed via chi-squared test of independence between P and Q, then Bonferroni-corrected for `m = corpus_size` comparisons. A result with `p < 0.05` is statistically significant given your actual corpus — not just geometrically close in abstract space.

---

## What you get back

```python
{
  "id": "doc_8f3a1c",
  "composite_score": 0.847,
  "cosine": 0.912,
  "nmi": 0.741,
  "p_value": 0.003,
  "significant": true,
  "alpha_used": 0.6
}
```

`significant: true` means the similarity is unlikely to be explained by embedding-space density alone, at your corpus size. You can filter, rank, or gate on it directly.

---

## Why not build this yourself

| What you'd have to build | Why it's harder than it looks |
|---|---|
| Freedman-Diaconis binning per embedding segment | Bin width depends on IQR per segment, not global IQR — naively applied it produces degenerate histograms on sparse dims |
| Corpus-level Bonferroni correction | Requires tracking corpus size per query context, not per index — most infra doesn't expose this |
| NMI stability on short vectors | Below ~128 dims, entropy estimates are biased; requires sample-size correction (Miller-Madow or similar) |
| Calibrated alpha tuning | Optimal alpha shifts with domain; exposing it as a per-call param without breaking score monotonicity requires careful normalization |
| Zero-infra operation | Running this correctly without a persistent index means on-demand histogram construction — non-trivial to make fast without precomputation strategies |

None of this is impossible. Each piece individually is a few days of work. Together, with edge cases, correctness tests, and latency constraints, it's a project — and it's not your core product.

---

## Designed for your scale

No vector database to provision. No namespace to pay for whether you use it or not. No minimum corpus size.

Pass your corpus inline, or reference a corpus you've registered. Pay per search operation. Works correctly on 50 documents or 500,000.

---

## Quickstart

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")

# Single query against an inline corpus
results = client.search(
    query="attention is all you need",
    corpus=["transformer paper", "BERT architecture", "recurrent networks", "CNN for text"],
    top_k=3,
    alpha=0.6,            # composite weight, cosine vs NMI
    significance=0.05     # filter to statistically significant results only
)

for r in results:
    print(r.composite_score, r.p_value, r.significant, r.text)
```

```python
# Batch: compare all pairs in a list
pairs = client.compare_pairs(
    items=["doc A", "doc B", "doc C"],
    alpha=0.7
)
# Returns N*(N-1)/2 scored pairs, each with composite score and p-value
```

---

## Authentication

All requests require an API key passed via header or client constructor:

```python
client = SimilarityClient(api_key="YOUR_API_KEY")
```

```bash
curl https://api.similarity-search.io/v1/search \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "neural retrieval", "corpus": ["dense retrieval", "BM25", "sparse vectors"], "top_k": 2}'
```

Keys are scoped to your account. Rotate them from the dashboard at any time without downtime.

---

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/v1/search` | Query against inline or registered corpus, returns ranked results with composite scores and p-values |
| `POST` | `/v1/compare` | Score a single pair of texts, returns full signal breakdown |
| `POST` | `/v1/batch_compare` | Score all pairs in a list, returns N*(N-1)/2 results |
| `POST` | `/v1/corpus` | Register a corpus by ID for repeated queries without re-sending documents |
| `GET` | `/v1/corpus/{id}` | Retrieve metadata for a registered corpus |

---

## Language support

- **Python** — `pip install similarity-search-client`
- **Node.js** — `npm install @similarity-search/client`
- **HTTP** — any language, full REST API, JSON in and out

---

## Requirements

- Python 3.8+ for the client library
- No local GPU, no local model, no vector index
- Embeddings are computed server-side — send raw text

---

## Links

- [API Reference](https://docs.similarity-search.io/api)
- [Composite Score Deep Dive](https://docs.similarity-search.io/nmi-explained)
- [Alpha Tuning Guide](https://docs.similarity-search.io/alpha-calibration)
- [Status](https://status.similarity-search.io)
- [Support](mailto:support@similarity-search.io)

---

## Pricing

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |

No base fee. No storage fee. No minimum commitment. You pay for computation, not for parking vectors you queried once.