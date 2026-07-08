# Similarity Search API

Stateless semantic similarity over HTTP — no vector database, no infrastructure, no setup.

---

## Install

```bash
pip install similarity-search-sdk
```

---

## Quickstart

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="sk_test_xxxxxxxxxxxxxxxx")
results = client.search(query="neural network optimization", corpus=my_documents, top_k=5)
print(results[0].score, results[0].text)
```

---

## Why not build it yourself?

**The honest answer:** you can. You can wire up cosine similarity in 10 lines of NumPy. What you cannot wire up in a weekend is the part that makes it accurate.

Every similarity API you've seen — Pinecone, Weaviate, pgvector — solves a storage problem. This solves a **measurement problem**.

### What the incumbents miss

Cosine similarity measures geometric angle between embedding vectors. It's fast, it's differentiable, and it's wrong in a specific, predictable way: it treats all dimensions as equally informative. In a corpus of 50,000 product descriptions, most embedding dimensions carry near-zero discriminative signal. Cosine penalizes you for that noise and you pay for it in precision.

Normalized Mutual Information (NMI) captures statistical dependence between items without assuming linearity or geometric structure. It sees cluster density that cosine misses. But NMI alone doesn't generalize across semantically sparse corpora — it over-weights coincidental co-occurrence.

The fix is composing them. The score this API returns is:

```
score = alpha * cosine(q, d) + (1 - alpha) * NMI_normalized(q, d)
```

Where `alpha` is not a parameter you tune. It's computed per-request from the marginal entropy of your corpus:

```
alpha = H(corpus) / (H(corpus) + H_max)
```

`H(corpus)` is the Shannon entropy of the term distribution across your submitted documents, calculated via `src/math/information` at request time — specifically `H(X)`, `H(Y)`, and the joint `H(X,Y)` needed to derive NMI. The complexity is O(n log n) on corpus size, which is why this runs as a stateless call rather than a precomputed index.

**What this means in practice:**

- High-entropy corpus (semantically dispersed documents, many topics) -> `alpha` approaches 1.0 -> cosine dominates, because geometric spread is the real signal
- Low-entropy corpus (dense cluster, narrow domain) -> `alpha` drops toward 0.5 -> NMI dominates, because mutual information between co-occurring terms reveals structure that cosine flattens

You don't configure this. You don't know your corpus entropy ahead of time. The API measures it on every call and calibrates automatically.

---

## What you skip

| Without this API | With this API |
|---|---|
| Spin up Pinecone, configure index dimensions, manage upserts, pay for idle replicas | One HTTP POST with your documents and query |
| Implement NMI from scratch, debug joint entropy estimation, handle zero-probability terms | Handled inside every request |
| Tune a weighting parameter between geometric and information-theoretic metrics per domain | Alpha computed automatically from your corpus entropy |
| Maintain a persistent vector store for a corpus that changes weekly | Stateless — send the corpus every time, or cache it yourself |

The setup cost for a managed vector database makes sense at 10M+ documents with a dedicated ML team. Below 500k items and without a dedicated infrastructure function, you are paying for complexity you don't need.

---

## API surface

```
POST /v1/search          # Query a corpus, returns ranked results with composite scores
POST /v1/score           # Score a single (query, document) pair — returns alpha, cosine, NMI components
POST /v1/entropy         # Compute H(corpus) — useful for debugging calibration behavior
GET  /v1/health          # Liveness check
```

Four endpoints. No state. No index. No SDK required — curl works.

---

## curl example

```bash
curl https://api.similarity-search.io/v1/search \
  -H "Authorization: Bearer sk_test_xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer attention mechanism",
    "corpus": ["Self-attention in NLP...", "Convolutional filters for images...", "..."],
    "top_k": 3
  }'
```

```json
{
  "results": [
    { "index": 0, "score": 0.912, "cosine": 0.887, "nmi": 0.961, "alpha": 0.731 },
    { "index": 2, "score": 0.741, "cosine": 0.803, "nmi": 0.634, "alpha": 0.731 }
  ],
  "corpus_entropy": 3.847,
  "alpha": 0.731,
  "latency_ms": 43
}
```

The response returns `alpha` and both component scores so you can audit exactly why a document ranked where it did.

---

## Limits

- Corpus size: up to 500,000 items per request
- Document length: up to 8,192 tokens per item
- Query length: up to 512 tokens
- Latency: O(n log n) on corpus size — benchmark in `/v1/health` response includes current node throughput

---

## Authentication

All requests require a Bearer token in the `Authorization` header. Keys are scoped to an account and rate-limited per tier. No key is embedded in the SDK by default — pass it explicitly or via the `SIMILARITY_SEARCH_API_KEY` environment variable.

```python
import os
from similarity_search import SimilarityClient

client = SimilarityClient(api_key=os.environ["SIMILARITY_SEARCH_API_KEY"])
```

---

## Language support

| Language | Package |
|---|---|
| Python 3.9+ | `pip install similarity-search-sdk` |
| Node.js 18+ | `npm install @similarity-search/sdk` |
| HTTP (any) | REST — no SDK needed |

---

## Status and support

- Status page: status.similarity-search.io
- API reference: docs.similarity-search.io
- Support: support@similarity-search.io

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