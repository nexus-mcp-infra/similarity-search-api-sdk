# Similarity Search API

Score, rank, and decompose similarity across mixed-feature records — without spinning up a vector database.

---

## The problem

Comparing records with mixed feature types (categorical + continuous) requires combining two fundamentally different measures: **Normalized Mutual Information** for categorical features (which captures statistical dependence, not geometric distance) and **Cosine similarity** for continuous features (embeddings, numerics). No existing SDK does this in a single coherent scorer.

The typical workaround:

```python
# What you're doing today
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Separate preprocessing pipelines per feature type
cat_score = normalized_mutual_info_score(record_a["category"], record_b["category"])
cont_score = cosine_similarity([record_a["embedding"]], [record_b["embedding"]])[0][0]

# Arbitrary manual blend -- no principled weight derivation
score = 0.5 * cat_score + 0.5 * cont_score
```

The weight `0.5` is a guess. The normalization between NMI and Cosine spaces is not equivalent. And this still does not rank, calibrate, or decompose.

---

## What this API does

The Similarity Search API exposes a **stateless hybrid scorer** that fuses NMI and Cosine in a single normalized scalar:

```
S = w_cat * NMI_norm + w_cont * Cosine
```

where `w_cat` and `w_cont` are derived automatically from the feature-type composition of your records — not hardcoded. Each call is a pure function: no index to provision, no cluster to manage, no storage billed.

Five atomic operations, no side effects:

| Operation | Route |
|---|---|
| Score two records | `POST /v1/similarity/pairwise-score` |
| Rank a candidate set | `POST /v1/similarity/rank-candidates` |
| Decompose a score by feature | `POST /v1/similarity/feature-decomposition` |
| Calibrate categorical weight | `POST /v1/similarity/calibrate-weight` |
| Benchmark hybrid vs. Cosine-only | `POST /v1/similarity/benchmark-hybrid-vs-cosine` |

Base URL: `https://similarity-search-api.railway.app`

---

## Quickstart

All endpoints accept and return JSON. Call them directly over HTTP — no package to install.

```bash
curl -X POST https://similarity-search-api.railway.app/v1/similarity/pairwise-score \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "record_a": {
      "category": "electronics",
      "brand": "sony",
      "embedding": [0.12, 0.87, 0.34, 0.56]
    },
    "record_b": {
      "category": "electronics",
      "brand": "panasonic",
      "embedding": [0.15, 0.80, 0.38, 0.51]
    }
  }'
```

```json
{
  "hybrid_score": 0.847,
  "categorical_weight": 0.333,
  "continuous_weight": 0.667
}
```

---

## Endpoints

### `POST /v1/similarity/pairwise-score`

Computes a weighted hybrid similarity score between two individual records. NMI is applied per categorical feature; Cosine is applied per continuous feature. The result is a single normalized scalar in `[0, 1]`.

**Use when:** you need one mathematically coherent similarity value between exactly two mixed-type records, with no index or state.

**Do not use when:** comparing more than two records at once (use `/rank-candidates`), or when all features are purely continuous (Cosine alone is sufficient and cheaper).

```bash
curl -X POST https://similarity-search-api.railway.app/v1/similarity/pairwise-score \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "record_a": {
      "job_title": "software engineer",
      "seniority": "senior",
      "skills_vector": [0.91, 0.45, 0.77]
    },
    "record_b": {
      "job_title": "backend engineer",
      "seniority": "senior",
      "skills_vector": [0.88, 0.50, 0.72]
    }
  }'
```

```json
{
  "hybrid_score": 0.913,
  "categorical_weight": 0.4,
  "continuous_weight": 0.6
}
```

---

### `POST /v1/similarity/rank-candidates`

Scores a query record against a list of candidate records using the same NMI-Cosine hybrid scorer, and returns candidates sorted by descending hybrid score.

**Use when:** performing nearest-neighbor retrieval over a small-to-medium candidate set (up to 2,000 records) in serverless pipelines, CI validation, or ad-hoc retrieval where provisioning a persistent vector index is not justified.

**Do not use when:** your candidate set exceeds 2,000 records (latency degrades linearly); or when you only need the top-1 score between exactly two records (use `/pairwise-score` instead).

```bash
curl -X POST https://similarity-search-api.railway.app/v1/similarity/rank-candidates \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "query": {
      "category": "shoes",
      "style": "running",
      "embedding": [0.33, 0.71, 0.55]
    },
    "candidates": [
      {"id": "prod_001", "category": "shoes", "style": "trail", "embedding": [0.30, 0.68, 0.60]},
      {"id": "prod_002", "category": "apparel", "style": "running", "embedding": [0.40, 0.65, 0.50]},
      {"id": "prod_003", "category": "shoes", "style": "running", "embedding": [0.34, 0.72, 0.54]}
    ],
    "top_k": 2
  }'
```

```json
{
  "ranked_candidates": [
    {"id": "prod_003", "hybrid_score": 0.981},
    {"id": "prod_001", "hybrid_score": 0.762}
  ]
}
```

---

### `POST /v1/similarity/feature-decomposition`

Returns the per-feature contribution to the hybrid score: individual NMI values for each categorical feature, individual Cosine components for each continuous feature, plus the aggregated hybrid scalar.

**Use when:** you need interpretability — debugging why two records received a particular score, validating feature engineering choices, or generating audit trails.

**Do not use when:** running high-throughput scoring loops where only the final scalar matters. Decomposition overhead is approximately 3x versus `/pairwise-score`.

```bash
curl -X POST https://similarity-search-api.railway.app/v1/similarity/feature-decomposition \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "record_a": {
      "industry": "fintech",
      "stage": "series_a",
      "revenue_vector": [0.60, 0.45, 0.80]
    },
    "record_b": {
      "industry": "fintech",
      "stage": "seed",
      "revenue_vector": [0.55, 0.50, 0.78]
    }
  }'
```

```json
{
  "hybrid_score": 0.791,
  "categorical_contributions": {
    "industry": {"nmi": 1.0, "weight": 0.2},
    "stage":    {"nmi": 0.0, "weight": 0.2}
  },
  "continuous_contributions": {
    "revenue_vector": {"cosine": 0.997, "weight": 0.6}
  }
}
```

---

### `POST /v1/similarity/calibrate-weight`

Given a labeled dataset of record pairs with ground-truth similarity judgments (`0.0` to `1.0`), estimates the `categorical_weight` value that minimizes mean squared error between hybrid scores and ground-truth labels via grid search over `[0.0, 1.0]` in configurable steps.

**Use when:** you have a gold-standard evaluation set and want to calibrate the categorical/continuous blend before deploying scoring at scale.

**Do not use when:** you have fewer than 20 labeled pairs (results will be unreliable); or as a substitute for proper cross-validation on large datasets (this is a single-pass grid search, not regularized regression).

```bash
curl -X POST https://similarity-search-api.railway.app/v1/similarity/calibrate-weight \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "labeled_pairs": [
      {
        "record_a": {"genre": "jazz", "tempo_vector": [0.6, 0.4]},
        "record_b": {"genre": "jazz", "tempo_vector": [0.7, 0.3]},
        "ground_truth": 0.85
      },
      {
        "record_a": {"genre": "classical", "tempo_vector": [0.2, 0.9]},
        "record_b": {"genre": "jazz",      "tempo_vector": [0.6, 0.4]},
        "ground_truth": 0.20
      }
    ],
    "grid_steps": 20
  }'
```

```json
{
  "optimal_categorical_weight": 0.45,
  "mse_at_optimal": 0.0031,
  "grid_search_steps": 20
}
```

---

### `POST /v1/similarity/benchmark-hybrid-vs-cosine`

Runs a head-to-head benchmark on a provided labeled pair set, comparing the hybrid NMI-Cosine scorer against a pure-Cosine baseline. Returns MSE, Spearman rank correlation, and mean absolute error for both methods, plus the relative lift of the hybrid over baseline.

**Use when:** you need a reproducible metric demonstrating the superiority of hybrid scoring on a specific mixed-feature dataset — publishable in an evaluation report or model card.

**Do not use as** a general-purpose benchmarking framework for arbitrary models. This endpoint is scoped exclusively to the NMI-Cosine hybrid vs. Cosine-only comparison defined in this API.

```bash
curl -X POST https://similarity-search-api.railway.app/v1/similarity/benchmark-hybrid-vs-cosine \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "labeled_pairs": [
      {
        "record_a": {"color": "red",  "price_vector": [0.8, 0.2]},
        "record_b": {"color": "red",  "price_vector": [0.75, 0.25]},
        "ground_truth": 0.90
      },
      {
        "record_a": {"color": "blue", "price_vector": [0.3, 0.7]},
        "record_b": {"color": "red",  "price_vector": [0.8, 0.2]},
        "ground_truth": 0.15
      }
    ],
    "categorical_weight": 0.45
  }'
```

```json
{
  "hybrid": {
    "mse": 0.0041,
    "spearman_correlation": 0.94,
    "mean_absolute_error": 0.048
  },
  "cosine_baseline": {
    "mse": 0.0189,
    "spearman_correlation": 0.71,
    "mean_absolute_error": 0.121
  },
  "relative_lift": {
    "mse_reduction_pct": 78.3,
    "spearman_gain": 0.23
  }
}
```

---

## Using the Python SDK

The SDK is not published to PyPI. Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/similarity-search-api
cd similarity-search-api
pip install -r requirements.txt
```

The `Client` class maps directly to the five endpoints above. It handles HTTP transport via `httpx`, raises typed exceptions for auth failures (`SimilaritySearchAuthError`), validation rejections (`SimilaritySearchValidationError`), rate limits (`SimilaritySearchRateLimitError`), and server errors (`SimilaritySearchAPIError`), and enforces a default timeout of 30 seconds. The base URL and timeout are configurable at construction time.

```python
from client import Client, SimilaritySearchAuthError, SimilaritySearchValidationError

client = Client(api_key="YOUR_API_KEY")

result = client.hybrid_similarity_score(
    query={
        "category": "electronics",
        "brand": "sony",
        "embedding": [0.12, 0.87, 0.34, 0.56]
    },
    corpus=[
        {"category": "electronics", "brand": "panasonic", "embedding": [0.15, 0.80, 0.38, 0.51]},
        {"category": "appliances",  "brand": "sony",      "embedding": [0.10, 0.60, 0.44, 0.70]},
    ],
    top_k=1
)
```

Exception hierarchy:

```
SimilaritySearchAuthError          -- 401: invalid or expired api_key
SimilaritySearchValidationError    -- 422: payload rejected by the API
SimilaritySearchRateLimitError     -- 429: rate limit exceeded, check Retry-After header
SimilaritySearchAPIError           -- 4xx/5xx: all other API-level errors
```

---

## Why not Pinecone / Weaviate / a vector DB?

Those tools are right when you need persistent, indexed search over millions of vectors with sub-10ms latency at query time. This API is right when:

- Your candidate set fits within a single request (up to 2,000 records for ranking)
- You need **no persistent state** — each call is a pure function, nothing stored
- Your records have **mixed feature types** and pure Cosine over concatenated embeddings loses information in the categorical dimensions
- You are in a **serverless or CI context** where provisioning an index per run is wasteful

The benchmark endpoint exists precisely to let you measure and report the gap.

---

## Stateless design

Every endpoint is a pure function of its inputs. There is no index to provision, no session to maintain, no stored state between calls. This makes the API:

- Safe to call in parallel without coordination
- Suitable for ephemeral environments (Lambda, Cloud Run, GitHub Actions)
- Auditable: given the same inputs, the same score is always returned

---

## HTTP errors

| Status | Meaning |
|---|---|
| `401` | Missing or invalid authorization header |
| `422` | Payload failed schema validation — check the `detail` field |
| `429` | Rate limit exceeded — respect the `Retry-After` header |
| `5xx` | Server error — safe to retry with exponential backoff |

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