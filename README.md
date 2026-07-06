# Similarity Search API

Stateless similarity scoring that fuses NMI and cosine into a single calibrated score — no database, no indexing, no infrastructure to manage.

---

## Install

```bash
pip install similarity-search-client
```

## Quickstart

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")
result = client.compare(query=[0.8, 1, "premium"], corpus=[[0.3, 0, "basic"], [0.9, 1, "premium"]])
print(result.scores)  # [0.12, 0.97] — composite NMI+Cosine, entropy-weighted per call
```

---

## Why not just use cosine similarity?

Cosine similarity measures directional alignment in dense vector spaces. It works well when all your dimensions are continuous and roughly uniformly distributed. It fails silently when they are not.

Consider a dataset where one dimension encodes user tier (`free`, `pro`, `enterprise`) and another encodes session duration as a float. Cosine treats both as coordinates in the same Euclidean space. NMI treats them as random variables and captures the statistical dependency between them regardless of scale or distribution shape.

**The problem**: no existing API gives you both, fused, without requiring you to stand up a database first.

---

## The gap this fills

Every major vector search provider — Pinecone, Weaviate, Qdrant, pgvector — is built around the same assumption: you want to store vectors and query against a persistent index. That is the right architecture for retrieval at scale. It is the wrong tool when you need to compare two payloads right now, on the fly, without upserting anything.

| Requirement | Pinecone | Faiss | **Similarity Search API** |
|---|---|---|---|
| Stateless, per-call comparison | No | No | Yes |
| Works on mixed feature types | No | No | Yes |
| No indexing before first query | No | No | Yes |
| NMI + cosine fused score | No | No | Yes |
| Zero infrastructure to provision | No | No | Yes |

---

## How the score works

The composite score is not a fixed linear blend. The weight of each dimension is computed from its own marginal entropy in the payload you send:

```
w_i = H(X_i) / sum_j( H(X_j) )

score = sum_i( w_i * fusion(NMI_i, cosine_i) )
```

`H(X_i)` is the marginal Shannon entropy of dimension `i` across your corpus vectors. Dimensions with low entropy (nearly constant across candidates) contribute little to the final score. Dimensions with high variance drive it. No hyperparameter to tune. No pre-trained model to load. The weighting is specific to the distribution of the input you send in that call.

This means two calls with differently distributed corpora will produce different weightings from the same query vector — which is exactly correct behavior, and not reproducible with any single standard metric.

---

## What NMI catches that cosine misses

```python
# Two items that cosine scores as dissimilar (different magnitude, angle)
# but NMI scores as highly dependent (same underlying pattern)
query  = [1.0, 0.1, "enterprise", 42]
item_a = [9.0, 0.9, "enterprise", 38]  # scaled version of query

# Cosine: low score (different magnitudes distort angle)
# NMI+Cosine composite: high score (dependency structure is preserved)
```

Categorical dimensions encoded as integers are the clearest failure case for pure cosine. A tier encoded as `[1, 2, 3]` is treated as a continuous axis — the distance between `free` and `enterprise` is 2, not "maximum categorical distance." NMI captures the mutual information between the query's tier and each candidate's tier without making any linearity assumption.

---

## Endpoints

### `POST /v1/compare`

Compare a single query vector against a corpus of candidate vectors. Returns a ranked list of composite NMI+Cosine scores, one per candidate.

```bash
curl -X POST https://api.similarity-search.io/v1/compare \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.8, 1, "premium", 0.33],
    "corpus": [
      [0.3, 0, "basic", 0.10],
      [0.9, 1, "premium", 0.31],
      [0.5, 1, "pro", 0.28]
    ]
  }'
```

```json
{
  "scores": [0.11, 0.96, 0.54],
  "weights": [0.18, 0.41, 0.29, 0.12],
  "top_k": [
    {"index": 1, "score": 0.96},
    {"index": 2, "score": 0.54},
    {"index": 0, "score": 0.11}
  ]
}
```

`weights` is the entropy-derived weight vector computed from your corpus for this call. It is returned so you can inspect which dimensions drove the score — this is your explainability layer, built in.

### `POST /v1/batch`

Submit multiple query vectors in a single call. Each query is scored independently against the shared corpus.

```bash
curl -X POST https://api.similarity-search.io/v1/batch \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      [0.8, 1, "premium", 0.33],
      [0.2, 0, "basic", 0.05]
    ],
    "corpus": [
      [0.3, 0, "basic", 0.10],
      [0.9, 1, "premium", 0.31]
    ]
  }'
```

---

## Error handling

The client raises typed exceptions, not raw HTTP errors:

```python
from similarity_search import SimilarityClient, DimensionMismatchError, AuthenticationError

client = SimilarityClient(api_key="YOUR_API_KEY")

try:
    result = client.compare(query=[1, 2], corpus=[[1, 2, 3]])
except DimensionMismatchError as e:
    print(e.message)       # "Query has 2 dimensions; corpus item 0 has 3"
    print(e.offending_index)  # 0
except AuthenticationError:
    print("Check your API key")
```

All 4xx errors include a machine-readable `error.code` and a human-readable `error.message`. No generic 400s.

---

## Constraints

| Parameter | Limit |
|---|---|
| Dimensions per vector | 2 – 2048 |
| Corpus size per call (`/v1/compare`) | 1 – 10,000 vectors |
| Queries per call (`/v1/batch`) | 1 – 500 |
| Supported dimension types | `float`, `int`, `string` (categorical) |

---

## Python SDK reference

```python
from similarity_search import SimilarityClient

client = SimilarityClient(
    api_key="YOUR_API_KEY",
    timeout=10.0,   # seconds
    retries=3
)

# Single query
result = client.compare(
    query=[...],
    corpus=[[...], [...]],
    top_k=5             # optional, returns full ranking if omitted
)
result.scores           # list[float], one per corpus item
result.weights          # list[float], entropy-derived per dimension
result.top_k            # list[{index, score}], sorted descending

# Batch
batch_result = client.batch(
    queries=[[...], [...]],
    corpus=[[...], [...]]
)
batch_result.results    # list[CompareResult], one per query
```

---

## Why build on this instead of rolling your own

Implementing NMI correctly for mixed-type vectors (continuous + categorical in the same payload) requires binning continuous dimensions, estimating marginal and joint probability distributions, handling zero-entropy edge cases without division errors, and then combining the result with cosine in a way that is numerically stable across different scales. That is roughly 600–900 lines of numpy before you touch HTTP, auth, rate limiting, or error handling.

This API gives you a single POST call and a typed score back in under 100ms. The math runs on our infrastructure. You do not maintain it, version it, or debug it when a new distribution of inputs breaks your entropy binning heuristic.

---

## Authentication

All requests require a Bearer token in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

Keys are scoped per environment. Rotate them from the dashboard without downtime — in-flight requests on the old key complete normally during a 60-second grace window.

---

## Languages

| Language | Status |
|---|---|
| Python | Available (`pip install similarity-search-client`) |
| TypeScript / Node | Available (`npm install similarity-search-client`) |
| Go | In progress |
| REST (any language) | Always available |

---

## Support

- Docs: [docs.similarity-search.io](https://docs.similarity-search.io)
- Status: [status.similarity-search.io](https://status.similarity-search.io)
- Issues: [github.com/similarity-search/api/issues](https://github.com/similarity-search/api/issues)
- Email: support@similarity-search.io

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