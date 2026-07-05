# Similarity Search API

Find what's similar. No infrastructure required.

```bash
pip install similarity-search-sdk
```

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="sk_live_...")
results = client.search(query=product, collection=catalog, top_k=10)
```

Your top-10 results, ranked by Information-Geometric Similarity, in one round-trip.

---

## The problem with every other option

You have a collection of 800 items. You want to find the 10 most similar to a query. Here is what the current landscape forces you to do:

**Vector store route** — Spin up Pinecone, Weaviate, or Qdrant. Vectorize every item. Push embeddings to the index. Keep the index synchronized when the collection changes. Pay $70–300/month whether you run 10 queries or 10,000. Total time before your first result: 45–90 minutes, and that is if nothing breaks.

**Roll your own route** — Implement cosine similarity in NumPy. Realize cosine alone misses categorical structure. Add NMI. Debug the entropy estimator. Handle edge cases in sparse distributions. Write the fusion function. Tune the weights. Two days later you have something that works on your test fixture and fails silently on production data.

**Similarity Search API route** — One HTTP call. No index. No sync. No monthly commitment. First result in under 200 ms.

---

## When to use this

- Collections that change frequently and would be expensive to keep indexed
- Ad-hoc similarity on ephemeral datasets (search-before-you-commit workflows)
- Collections under 100k items where a full vector store is architectural overkill
- Any case where you are paying for infrastructure you use less than 10k times per month
- Items with mixed structure: numeric features alongside categorical fields (product catalogs, job listings, medical codes, content metadata)

## When **not** to use this

- Collections above 100k items where ANN indexes give you sub-linear query time you cannot match stateless
- Workloads requiring persistent vector state across sessions (recommendation engines with user history)
- Real-time streaming ingestion at high write volume

---

## Why the ranking is better

Every stateless similarity API you can find today uses cosine distance alone. Cosine is a geometric metric — it measures the angle between two vectors. It is excellent for dense continuous features and terrible for categorical distributions, because two items with completely different category distributions can have a small angle if their magnitudes are similar.

NMI (Normalized Mutual Information) measures how much knowing one item's category distribution reduces uncertainty about the other's. It captures structural overlap that cosine cannot see.

The problem is they are not directly composable — they live in different mathematical spaces. Cosine is bounded in [-1, 1] geometrically; NMI is bounded in [0, 1] information-theoretically. Naively averaging them produces scores that are neither interpretable nor well-calibrated.

**The Information-Geometric Similarity Score (IGS)** solves this with adaptive fusion:

```
IGS(a, b | d) = w_nmi(d) * NMI(a, b) + w_cos(d) * Cosine(a, b)

where w_nmi(d) + w_cos(d) = 1
and w_nmi(d) is a function of the categorical density of domain d,
estimated per-request using the Miller-Madow entropy corrector
to handle small-sample bias in sparse category distributions.
```

The weights are not hyperparameters you tune once and forget. They are estimated from the structure of each collection you pass in the request, then recalibrated offline every 24 hours using accepted/rejected ranking signals accumulated across all queries. The result: the fusion becomes more accurate the more the API is used, without any action required from you.

This is not reproducible by copying the endpoint interface. The moat is in the calibrated fusion function and the data flywheel behind the weight recalibration — not in the HTTP surface.

---

## Quickstart

```python
from similarity_search import SimilarityClient, Item

client = SimilarityClient(api_key="sk_live_...")

query = Item(
    id="product-001",
    features={"embedding": [0.12, 0.87, ...], "category": "electronics", "price_tier": "mid"}
)

collection = [
    Item(id="product-042", features={"embedding": [...], "category": "electronics", "price_tier": "mid"}),
    Item(id="product-107", features={"embedding": [...], "category": "accessories", "price_tier": "low"}),
    # ... up to 100k items
]

results = client.search(query=query, collection=collection, top_k=10)

for r in results:
    print(r.id, r.igs_score, r.nmi_component, r.cosine_component)
```

The response includes the decomposed score — `igs_score`, `nmi_component`, `cosine_component` — so your ranking is auditable. You can see exactly how much of the similarity is geometric versus distributional. No black box.

---

## API reference

### `POST /v1/search`

Find the top-k most similar items to a query within a collection passed in the request body. Stateless — nothing is stored.

**Request**

```json
{
  "query": {
    "id": "string",
    "features": { "field": "value or [float]" }
  },
  "collection": [
    { "id": "string", "features": { "field": "value or [float]" } }
  ],
  "top_k": 10,
  "weights": "adaptive"
}
```

`weights` accepts `"adaptive"` (default, uses IGS with Miller-Madow estimation) or `{"nmi": 0.4, "cosine": 0.6}` if you want to fix the fusion manually.

**Response**

```json
{
  "results": [
    {
      "id": "product-042",
      "rank": 1,
      "igs_score": 0.891,
      "nmi_component": 0.934,
      "cosine_component": 0.847,
      "w_nmi": 0.61,
      "w_cos": 0.39
    }
  ],
  "collection_size": 847,
  "latency_ms": 143,
  "miller_madow_correction_applied": true
}
```

### `POST /v1/score`

Compute the IGS score for a single pair. Useful for threshold checks without ranking a full collection.

### `GET /v1/health`

Returns API status and current model recalibration timestamp.

---

## Pricing

Pay per call. No monthly seat. No infrastructure.

| Volume | Price per call |
|---|---|
| 0 – 10k calls/month | $0.004 |
| 10k – 100k calls/month | $0.002 |
| 100k+ calls/month | $0.001 |

A team running 5,000 queries per month pays $20. The equivalent Pinecone starter plan costs $70/month before you write a single line of code.

---

## Error handling

The SDK raises typed exceptions. You do not need to parse error strings.

```python
from similarity_search.exceptions import (
    CollectionTooLargeError,   # collection exceeds 100k items
    FeatureShapeMismatchError, # query and collection embedding dims differ
    InsufficientFeaturesError, # NMI requires at least one categorical field
    AuthenticationError,       # invalid or missing API key
    RateLimitError,            # retry with exponential backoff
)
```

---

## Security

- TLS 1.3 on all endpoints
- API keys are scoped (read / write / admin) and revocable from the dashboard
- Request bodies are not logged or stored — stateless by design, not just by claim
- SOC 2 Type II audit in progress

---

## Get an API key

[dashboard.similaritysearch.dev](https://dashboard.similaritysearch.dev) — key in 30 seconds, no credit card required for the first 1,000 calls.

Questions: [support@similaritysearch.dev](mailto:support@similaritysearch.dev)