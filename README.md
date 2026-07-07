# Similarity Search API

Semantic similarity over mixed-data records — stateless, per-call, no index required.

---

## Install

```bash
pip install nexus-similarity
```

---

## 30 seconds to your first result

```python
from nexus_similarity import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")
result = client.compare(query={"category": "electronics", "price": 299.99, "brand": "sony"}, candidates=[{"category": "electronics", "price": 279.00, "brand": "sony"}, {"category": "appliances", "price": 310.00, "brand": "lg"}])
print(result.ranked[0].score, result.ranked[0].confidence_interval)  # 0.91, (0.87, 0.94)
```

---

## What you get back

```json
{
  "ranked": [
    {
      "index": 0,
      "score": 0.91,
      "confidence_interval": [0.87, 0.94],
      "feature_weights": {"categorical": 0.63, "continuous": 0.37}
    },
    {
      "index": 1,
      "score": 0.44,
      "confidence_interval": [0.38, 0.51],
      "feature_weights": {"categorical": 0.63, "continuous": 0.37}
    }
  ],
  "schema_inferred": {"category": "categorical", "price": "continuous", "brand": "categorical"},
  "bootstrap_n": 500
}
```

---

## Why not build this yourself

Every team that reaches this problem makes the same three mistakes — in order.

**Mistake 1: Using a vector DB for small-scale similarity.**
Pinecone, Weaviate, and Qdrant are excellent at 10M+ vectors with persistent indices. For catalogs under 100k items, you pay for infrastructure you don't need, write ingestion pipelines you have to maintain, and wake up to sync failures when your source data changes. A stateless per-call API has no index to go stale.

**Mistake 2: Mixing categorical and continuous features naively.**
Cosine similarity on embeddings works for text and dense vectors. It produces meaningless results when your records contain a mix of categorical fields (brand, category, status) and continuous ones (price, latitude, duration). The standard fix — one-hot encoding everything and running cosine — distorts the geometry of categorical relationships. NMI (Normalized Mutual Information) measures dependency between categorical features without imposing a false metric structure. The right answer is to use both, weighted by what your data actually contains.

**Mistake 3: Getting a distance back, not a decision.**
Raw cosine scores have no calibration. A score of 0.74 could mean "very similar" or "barely related" depending on the domain and the data distribution. Without a confidence interval grounded in bootstrap resampling, you're making ranking decisions on point estimates that could swing significantly with minor input variation. This API returns 500-resample bootstrap CIs on every call — you know not just the score but how much to trust it.

---

## How the score is computed

The hybrid score is not a fixed weighted average. On each call, the API:

1. **Infers feature types** from the input — no schema declaration required. Fields are classified as categorical or continuous using entropy estimation from `src/math/information`.
2. **Computes NMI** (normalized by joint entropy) across categorical feature pairs between query and each candidate.
3. **Computes Cosine similarity** over continuous feature vectors, with per-feature z-score normalization.
4. **Calibrates the NMI/Cosine blend weight** dynamically: if 70% of your features are categorical, the score reflects that — mathematically, not by convention. The weight `w` for the categorical component is `w = n_cat / (n_cat + n_cont)` adjusted by a mutual-information-derived confidence term, not a hyperparameter you tune.
5. **Runs bootstrap CI** with n=500 resamples via `src/math/statistics` on the fused score distribution to return the 90% confidence interval.

The same endpoint behaves differently on a product catalog than on a user-profile dataset — because the math responds to the data, not to a config file.

---

## When to use this API

**Use it when:**
- Your records have mixed categorical + continuous fields and you want a single comparable score
- You need similarity on demand without maintaining a persistent index
- You want confidence intervals on similarity scores to drive ranking or deduplication decisions
- Your dataset is under 500k records and standing up a vector DB is operationally disproportionate

**Do not use it when:**
- You need approximate nearest-neighbor search over millions of vectors at sub-10ms latency — use a vector DB
- Your data is pure unstructured text with no categorical structure — a plain embedding + cosine pipeline is sufficient and cheaper
- You need to persist, version, or diff an index over time — this API is stateless by design and stores nothing

---

## Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| `POST` | `/v1/compare` | Score one query against N candidates, return ranked list with CIs |
| `POST` | `/v1/batch` | Score M queries against N candidates in a single call |
| `GET`  | `/v1/schema` | Return the inferred feature type map for a given record set, without scoring |

---

## Authentication

All requests require a bearer token in the `Authorization` header:

```bash
curl -X POST https://api.nexus.ai/v1/compare \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": {...}, "candidates": [...]}'
```

API keys are scoped per project. Rotate them from the dashboard without downtime.

---

## Errors

The API returns standard HTTP status codes with a structured error body:

```json
{
  "error": {
    "code": "FEATURE_TYPE_INFERENCE_FAILED",
    "message": "Field 'timestamp' could not be classified as categorical or continuous. Provide a value with at least 2 distinct states or a numeric type.",
    "field": "timestamp"
  }
}
```

| Code | Meaning |
|------|---------|
| `400` | Malformed request — missing `query`, empty `candidates`, or unparseable field values |
| `401` | Invalid or missing API key |
| `422` | Records contain fewer than 2 comparable fields after type inference |
| `429` | Rate limit exceeded |
| `500` | Internal error — retryable, idempotent |

---

## Python SDK reference

```python
from nexus_similarity import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY", timeout=10)

# Single comparison
result = client.compare(query={...}, candidates=[...])

# Batch
result = client.batch(queries=[...], candidates=[...])

# Schema inspection without scoring
schema = client.infer_schema(records=[...])
```

Full SDK source: [github.com/nexus-ai/nexus-similarity-python](https://github.com/nexus-ai/nexus-similarity-python)

---

## Stack

Python 3.11 — FastAPI — Uvicorn — ClickHouse (score distribution logging for flywheel recalibration)

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