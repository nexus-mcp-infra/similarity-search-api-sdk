# Similarity Search API

Find the most relevant items in any collection — text, numeric, categorical, or mixed — with a single HTTP call. No vector database. No index setup. No infrastructure.

```bash
pip install similarity-search-sdk
```

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="sk_live_...")
results = client.search(query={"name": "espresso machine", "price": 89.99, "category": "kitchen"}, collection=products)
```

```json
[{"id": "prod_482", "score": 0.94}, {"id": "prod_219", "score": 0.87}]
```

---

## Why not cosine alone?

Cosine similarity assumes all features contribute equally and linearly. That assumption breaks the moment your data mixes text descriptions, numeric prices, and categorical labels in the same record — which is most real-world data.

This API uses **NMI-weighted cosine**: Normalized Mutual Information measures the statistical dependence between each feature dimension and the query target, including non-linear dependencies that cosine never sees. Features that actually predict relevance get higher weight. Features that are noise get downweighted automatically, per request.

```
sim(a, b) = cosine(w * a, w * b)

where w_i = NMI(X_i, Y)
      NMI estimated via Freedman-Diaconis binning (continuous features)
      or direct frequency counts (categorical features)
```

Complexity: **O(n * d * log d)** per call, where `n` = items in your collection and `d` = feature dimensions. No index amortization needed — the math is fast enough to be stateless.

---

## The problem with every alternative

| Approach | What you get | What you give up |
|---|---|---|
| Pinecone / Weaviate / Qdrant | Fast ANN at scale | Vector DB setup, index management, storage costs, vendor lock-in |
| scikit-learn cosine | Works locally | No NMI weighting, no API, you own the infra |
| Manual NMI + cosine | Full control | 3–5 days of entropy estimation code, binning edge cases, normalization bugs |
| **Similarity Search API** | NMI-weighted results, one HTTP call | Nothing — no infra, no index, no state |

**Configuring a full vector database to compare 500 products in a shopping cart is over-engineering.** NMI is not a native metric in any mainstream vector DB — every team that wants it today writes their own post-processing layer and maintains it forever. This API makes that layer a single `POST /search`.

---

## Zero infrastructure, zero lock-in

The API is **fully stateless**. Your data travels in the request body and nowhere else. There is no index to populate before your first query, no async ingestion pipeline, no schema migration when your product schema changes. Change your feature set between calls with no consequences.

This is the architectural difference: every competitor's business model is built on storing your vectors. Ours is built on computing the right answer faster than it takes to set up storage.

---

## Endpoint

```
POST https://api.similaritysearch.dev/v1/search
Authorization: Bearer sk_live_...
```

```json
{
  "query": {
    "description": "noise cancelling headphones",
    "price": 249.00,
    "brand": "Sony",
    "in_stock": true
  },
  "collection": [
    {"id": "p1", "description": "wireless earbuds", "price": 199.00, "brand": "Sony", "in_stock": true},
    {"id": "p2", "description": "gaming headset", "price": 89.00, "brand": "Razer", "in_stock": false}
  ],
  "top_k": 10
}
```

```json
{
  "results": [
    {"id": "p1", "score": 0.91, "rank": 1},
    {"id": "p2", "score": 0.43, "rank": 2}
  ],
  "meta": {
    "nmi_weights": {"description": 0.72, "price": 0.61, "brand": 0.88, "in_stock": 0.34},
    "latency_ms": 18,
    "items_scored": 2
  }
}
```

The `nmi_weights` field in every response shows you exactly which features drove the ranking. No black box.

---

## Pricing

| Volume | Price |
|---|---|
| First 10,000 calls/month | Free |
| 10,001 – 500,000 calls/month | $0.0008 per call |
| 500,001+ calls/month | $0.0004 per call |

Per call. No seat licenses. No storage fees. No minimum commitment.

---

## SDK

```bash
pip install similarity-search-sdk   # Python
npm install @similarity-search/sdk  # Node.js
```

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="sk_live_...")

# Search
results = client.search(query=my_product, collection=catalog, top_k=5)

# Inspect feature weights the API computed
print(results.meta.nmi_weights)
```

Full SDK reference -> [docs.similaritysearch.dev/sdk](https://docs.similaritysearch.dev/sdk)

---

## Get your API key

[docs.similaritysearch.dev](https://docs.similaritysearch.dev) -> Dashboard -> New Key

First key activates in 30 seconds. No credit card required for the free tier.