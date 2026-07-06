# Similarity Search API

Find the 10 most similar items to any JSON record — no embeddings, no index, no infrastructure.

```bash
pip install similarity-search-client
```

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")
results = client.search(query={"title": "noise-canceling headphones", "category": "electronics", "price": 149}, corpus=items)
```

That's it. No Pinecone pod. No embedding model. No orchestration layer.

---

## The problem with every other approach

Getting semantic search working looks simple until you actually do it:

1. Pick an embedding model and figure out how to run it
2. Stand up a vector database (Pinecone, Weaviate, Qdrant — pick your poison)
3. Write the ingestion pipeline to keep the index in sync
4. Write the query layer to hit both services and merge results
5. Pay for all of it before you know if it even works for your data

That's 45–90 minutes of setup before you can answer the question: *does similarity search actually solve my problem?* And if your workload is 500 queries/day, you're paying $70+/month for a Pinecone pod that's idle 99% of the time.

**This API collapses all of that into one HTTP call.**

---

## Why NMI + cosine, not just cosine

Every vector database scores similarity using cosine distance over embeddings. That works well when your data is pure text. It breaks when your records mix text, categories, and numerics — which is most real-world data.

Cosine over a TF-IDF or embedding vector has no way to express the fact that `category: "electronics"` and `category: "electronics"` is a *structurally stronger signal* than two documents that happen to share vocabulary. It treats all dimensions as continuous and linear.

**Normalized Mutual Information (NMI)** measures statistical dependence between features directly from their distributions — it captures non-linear relationships between categorical and ordinal fields that cosine simply cannot see. NMI ranges from 0 (independent) to 1 (perfectly dependent), and it requires no pre-trained model to compute.

The Similarity Search API fuses both signals in a single scoring step:

```
score(q, item) = alpha * cosine(q_vec, item_vec) + (1 - alpha) * NMI(q_features, item_features)
```

Default `alpha = 0.6` weights cosine slightly higher for mixed datasets. If your corpus is mostly categorical, send `alpha=0.3` in the request body. If it's mostly text, send `alpha=0.8`. No retraining. No model. Just one parameter.

---

## Quickstart

### Install

```bash
pip install similarity-search-client
```

### Search

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")

query = {
    "title": "wireless mechanical keyboard",
    "category": "peripherals",
    "price": 89,
    "brand": "keychron"
}

corpus = [
    {"title": "USB-C mechanical keyboard", "category": "peripherals", "price": 95, "brand": "keychron"},
    {"title": "gaming mouse", "category": "peripherals", "price": 59, "brand": "logitech"},
    {"title": "laptop stand", "category": "accessories", "price": 35, "brand": "nexstand"},
    # ... up to 10,000 items per call
]

results = client.search(query=query, corpus=corpus, top_k=5)

for r in results:
    print(r.rank, r.score, r.item["title"])
# 1  0.94  USB-C mechanical keyboard
# 2  0.61  gaming mouse
# 3  0.29  laptop stand
```

### Tune alpha for your data

```python
# Corpus is mostly categorical fields (enums, tags, labels)
results = client.search(query=query, corpus=corpus, alpha=0.3)

# Corpus is mostly free text
results = client.search(query=query, corpus=corpus, alpha=0.8)
```

### Raw HTTP

```bash
curl -X POST https://api.similaritysearch.io/v1/search \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {"title": "wireless keyboard", "category": "peripherals", "price": 89},
    "corpus": [...],
    "top_k": 10,
    "alpha": 0.6
  }'
```

Response:

```json
{
  "results": [
    {"rank": 1, "score": 0.94, "item": {"title": "USB-C mechanical keyboard", "category": "peripherals", "price": 95}},
    {"rank": 2, "score": 0.61, "item": {"title": "gaming mouse", "category": "peripherals", "price": 59}}
  ],
  "meta": {
    "corpus_size": 3,
    "top_k": 2,
    "alpha_used": 0.6,
    "latency_ms": 18
  }
}
```

---

## API reference

### `POST /v1/search`

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | `object` | yes | The reference item. Any JSON object with string, numeric, or boolean fields. |
| `corpus` | `array[object]` | yes | Items to rank. Max 10,000 per call. Must share at least one key with `query`. |
| `top_k` | `integer` | no | Number of results to return. Default `10`, max `1000`. |
| `alpha` | `float` | no | Cosine weight in [0.0, 1.0]. Default `0.6`. NMI weight is `1 - alpha`. |

### `GET /v1/health`

Returns `200 OK` when the service is available. No authentication required. Use this for uptime monitoring.

---

## What it does not do

- **No persistent index.** The corpus lives in the request. There is nothing to sync, nothing to update, nothing to delete.
- **No embedding model.** Numeric and text fields are vectorized with TF-IDF inside the scoring pipeline. You do not choose or host a model.
- **No state between calls.** Every request is fully independent. Horizontal scaling is implicit.
- **Not designed for corpora above 10,000 items.** For static corpora that large, a persistent index genuinely makes more sense — use Qdrant or Weaviate and accept the infrastructure cost. This API is optimized for the case where the corpus changes frequently, is small-to-medium, or you need to validate the approach before committing to index infrastructure.

---

## Authentication

All requests require a Bearer token in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

API keys are scoped to a project. Rotate them in the dashboard without downtime — old and new keys are valid during a 60-second overlap window.

---

## Errors

```json
{"error": {"code": "corpus_too_large", "message": "corpus must contain at most 10000 items, received 12400"}}
{"error": {"code": "alpha_out_of_range", "message": "alpha must be between 0.0 and 1.0, received 1.4"}}
{"error": {"code": "no_shared_keys", "message": "query and corpus items share no common keys; scoring is undefined"}}
{"error": {"code": "unauthorized", "message": "missing or invalid API key"}}
```

HTTP status codes follow standard semantics: `400` for malformed input, `401` for auth, `429` for rate limits, `500` for service errors.

---

## Compared to building it yourself

| | Build it yourself | Similarity Search API |
|---|---|---|
| Time to first result | 45–90 min | 3 min |
| Infrastructure to manage | Embedding model + vector DB + ingestion pipeline | Zero |
| Works on mixed data (text + categories + numerics) | Requires custom feature engineering | Native |
| Categorical dependency signal | Not captured by cosine | NMI, built-in |
| Cost at 500 queries/day | $70+/month (Pinecone Starter) | Pay per call |
| Stateless, no index drift | No | Yes |

---

## Language support

| Language | Install |
|---|---|
| Python | `pip install similarity-search-client` |
| Node.js | `npm install @similarity-search/client` |
| Go | `go get github.com/similarity-search/go-client` |
| Raw HTTP | Any language that can send JSON |

---

## Support

- Docs: [docs.similaritysearch.io](https://docs.similaritysearch.io)
- Status: [status.similaritysearch.io](https://status.similaritysearch.io)
- Issues: open a GitHub issue or email support@similaritysearch.io
- Response time: < 24h on business days

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