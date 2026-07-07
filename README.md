# Similarity Search API

Semantic similarity ranking without the infrastructure. Send vectors or text, get ranked scores back. No database, no index, no state.

---

## Install

```bash
pip install similarity-search-sdk
```

## Quickstart

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")
results = client.rank(query="machine learning for time series", corpus=["LSTM forecasting", "image segmentation with CNNs", "gradient boosting on tabular data"])
print(results[0])  # -> {"text": "LSTM forecasting", "score": 0.91, "rank": 1}
```

---

## Why not FAISS, Pinecone, or a cosine wrapper?

**The short answer:** those tools solve a different problem — storing and retrieving millions of vectors at scale. If you need semantic ranking for a batch that exists right now, you're paying the full infrastructure tax for a problem that doesn't require it.

**The real answer:** standard cosine similarity fails silently on skewed distributions. If your corpus has three near-identical documents and one outlier, cosine ranks them confidently — but that confidence is an artifact of the geometry, not the semantics.

This API runs **NMI-cosine weighted fusion** on every request. The fusion weight is computed from the actual entropy of the corpus you send:

```
w_nmi = H(corpus) / (H(corpus) + H_baseline)
score  = w_nmi * NMI(query, doc) + (1 - w_nmi) * cosine(query, doc)
```

When your corpus is concentrated (low entropy), the ranker leans harder on mutual information. When it's diffuse (high entropy), it becomes more conservative. This calibration happens in-request, against the full corpus you sent — which means it can never be replicated by a system that indexed your vectors offline and lost the marginal distribution.

---

## What problem does this solve exactly?

| Situation | What you'd normally do | What you do here |
|---|---|---|
| MVP with < 5k queries/day | Spin up Pinecone, write ingestion pipeline, manage index | POST your corpus + query, get ranked results |
| One-off semantic comparison | Install FAISS, write embedding glue code, no REST layer | One HTTP call, no local GPU/CPU allocation |
| Skewed domain corpus (legal, medical, code) | Tune cosine thresholds manually per domain | Entropy-adaptive weight adjusts automatically |
| Stateless serverless function | Vector DBs require persistent connections | Fully stateless — each request is self-contained |

---

## Endpoints

```
POST /v1/rank          # text or vectors -> ranked scores
POST /v1/rank/batch    # multiple queries against same corpus in one call
GET  /v1/health        # latency + entropy diagnostics for last request
```

All endpoints are stateless. Nothing you send is stored. The corpus lives only for the duration of the computation.

---

## Request shape

```json
{
  "query": "transformer architecture for NLP",
  "corpus": [
    "attention mechanisms in deep learning",
    "convolutional filters for image recognition",
    "BERT fine-tuning on downstream tasks"
  ],
  "mode": "text",
  "top_k": 3
}
```

`mode: "text"` — the API embeds internally before ranking.
`mode: "vector"` — send raw float arrays, skip embedding, get scores faster.

---

## Response shape

```json
{
  "query_entropy": 0.43,
  "corpus_entropy": 1.82,
  "w_nmi": 0.71,
  "results": [
    {"rank": 1, "text": "attention mechanisms in deep learning", "score": 0.94, "nmi": 0.89, "cosine": 0.96},
    {"rank": 2, "text": "BERT fine-tuning on downstream tasks",  "score": 0.81, "nmi": 0.78, "cosine": 0.83},
    {"rank": 3, "text": "convolutional filters for image recognition", "score": 0.41, "nmi": 0.32, "cosine": 0.47}
  ]
}
```

`w_nmi` is returned on every response so you can audit exactly how the ranker weighted the two signals for your specific corpus.

---

## Authentication

```bash
curl -X POST https://api.similaritysearch.io/v1/rank \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "corpus": ["...", "..."], "mode": "text", "top_k": 5}'
```

Get your key at [similaritysearch.io/dashboard](https://similaritysearch.io/dashboard).

---

## Limits

| Parameter | Limit |
|---|---|
| `corpus` items per request | 10,000 |
| vector dimensions (`mode: vector`) | 128 – 4,096 |
| request body | 10 MB |
| `top_k` | 1 – 1,000 |
| concurrent requests per key | 50 |

---

## Why stateless is a feature, not a limitation

A vector database keeps your data to make future queries faster. This API doesn't keep anything — which means:

- No data residency concerns. Your corpus never touches a persistent store.
- No index drift. The ranking is always computed against exactly what you sent.
- No warm-up. Cold starts don't exist for a service with no index to load.
- The entropy calculation is exact, not approximated from a stale index snapshot.

For use cases where freshness and correctness matter more than sub-millisecond recall at billion-vector scale, stateless is the right architecture.

---

## SDK reference

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY", timeout=30)

# Text mode
result = client.rank(query="...", corpus=[...], top_k=10)

# Vector mode
result = client.rank_vectors(query_vec=[0.1, 0.4, ...], corpus_vecs=[[...], [...]], top_k=5)

# Batch mode
results = client.rank_batch(queries=["...", "..."], corpus=[...], top_k=3)

# Diagnostics
diag = client.health()
print(diag["last_request_entropy"])
```

Full SDK source: [github.com/similarity-search/sdk-python](https://github.com/similarity-search/sdk-python)

---

## Stack

Python 3.11 · FastAPI · NumPy · SciPy · No vector database · No GPU required at client side

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