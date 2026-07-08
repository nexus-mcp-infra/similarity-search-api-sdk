# Similarity Search API

Stateless vector similarity with NMI+Cosine fusion. No index setup. No persistent storage. One call.

---

## Install

```bash
pip install similarity-search-sdk
```

## Query in 3 lines

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")
results = client.rank(corpus=my_vectors, query=query_vector, top_k=10)
```

`results` is a ranked list of `{index, score, nmi, cosine}` objects. No index to build. No storage to provision.

---

## The problem with every other solution

Every major vector search provider — Pinecone, Weaviate, Qdrant, pgvector — assumes your corpus lives in a persistent index. That means:

- **You configure before you query.** HNSW or IVF index setup blocks you from getting a useful result until the infrastructure is provisioned and populated. For a team prototyping on a corpus that changes per request, that's the wrong abstraction entirely.
- **You pay for storage you don't need.** If your corpus is ephemeral — user-uploaded documents, session context, per-request candidate sets — you're paying for vector storage that holds data you never intended to keep.
- **You write glue code for hybrid scoring.** Cosine similarity misses items that are statistically dependent but geometrically distant. NMI catches those. Getting both in one score requires estimating joint distributions over continuous vectors — non-trivial to implement correctly, and nothing exposes it as a primitive.

This API takes your corpus and query in the request body, computes the score, and returns a ranking. State lives in the caller.

---

## Why NMI+Cosine instead of cosine alone

Cosine similarity measures geometric proximity in embedding space. It answers: *are these vectors pointing in the same direction?*

Normalized Mutual Information measures statistical dependence. It answers: *do these vectors share information structure, regardless of angular distance?*

On noisy or high-dimensional corpora, items can be semantically related but geometrically dispersed — cosine alone produces false negatives. NMI catches the dependency. The fusion score combines both:

```
score = w_nmi * NMI(query, item) + (1 - w_nmi) * cosine(query, item)
```

The weight `w_nmi` is not a static hyperparameter. It is derived from the inter-item variance of the corpus in each call: when variance is high (noisy corpus), NMI receives greater weight; when variance is low (homogeneous corpus), cosine dominates. The API adapts to the distribution it receives, not to a fixed blend you have to tune.

The NMI estimator uses KDE with a Gaussian kernel to approximate the joint distribution `p(x, y)` over continuous vectors. Bin-based discretization — the common shortcut — introduces resolution bias that makes NMI unstable at high dimensionality. The KDE approach eliminates that bias. It is the reason the score is stable across embedding models and corpus sizes.

---

## Complexity

| Operation | Complexity |
|---|---|
| Score computation per item | O(d) |
| Full corpus ranking | O(n·d) |
| Index setup required | None |
| Persistent state | None |

`n` = corpus size, `d` = vector dimension. Every call is independent.

---

## Full request shape

```python
results = client.rank(
    corpus=[[0.1, 0.4, ...], [0.9, 0.2, ...]],  # list of float vectors
    query=[0.3, 0.5, ...],                        # same dimensionality
    top_k=10,                                     # 1–1000
    min_score=0.0                                 # optional floor filter
)

# Each result:
# {
#   "index": 4,
#   "score": 0.871,
#   "nmi": 0.743,
#   "cosine": 0.912,
#   "nmi_weight": 0.38
# }
```

No schema registration. No collection name. No API version negotiation for the index format. Send vectors, get ranking.

---

## When to use this

**Use it when:**
- Your corpus changes per request (per-user, per-session, per-document)
- You need a ranking in a pipeline step without provisioning infrastructure
- You want a single score that captures both geometric and information-theoretic similarity
- You are prototyping and cannot afford index setup latency

**Do not use it when:**
- Your corpus is static, large (>100k items), and queried at high QPS — a persistent HNSW index will outperform a per-call computation at that scale
- You need approximate nearest neighbor guarantees with sub-millisecond latency on millions of vectors

---

## Authentication

```bash
curl -X POST https://api.similaritysearch.io/v1/rank \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"corpus": [[...], [...]], "query": [...], "top_k": 5}'
```

Get your key at [similaritysearch.io/dashboard](https://similaritysearch.io/dashboard).

---

## Errors

All errors return structured JSON:

```json
{
  "error": {
    "code": "dimension_mismatch",
    "message": "Query vector has dimension 768; corpus item at index 3 has dimension 512.",
    "request_id": "req_a8f3c1"
  }
}
```

| Code | Meaning |
|---|---|
| `dimension_mismatch` | Corpus and query vectors are not the same dimension |
| `corpus_too_large` | Corpus exceeds per-call item limit |
| `invalid_vector` | Non-numeric or empty vector in corpus |
| `unauthorized` | Missing or invalid API key |
| `rate_limited` | Request rate exceeded; retry after the indicated delay |

---

## Built with

Python 3.11 · FastAPI · NumPy · SciPy

---

[Docs](https://similaritysearch.io/docs) · [Dashboard](https://similaritysearch.io/dashboard) · [Status](https://status.similaritysearch.io) · [support@similaritysearch.io](mailto:support@similaritysearch.io)

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