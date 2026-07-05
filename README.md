# Hybrid Similarity Search API

Stateless NMI+cosine scoring over arbitrary payloads. No indexes. No embeddings pipeline. No infrastructure.

---

## Install

```bash
pip install hybrid-similarity-client
```

## Quickstart

```python
from hybrid_similarity import SimilarityClient

client = SimilarityClient(api_key="sk-...")
results = client.search(query={"category": "electronics", "price": 299.0, "brand": "sony"}, corpus=product_catalog, top_k=10)
```

Every result includes `score`, `nmi_component`, `cosine_component`, and `dominant_signal` — no black box.

---

## Why not build this yourself

The hard part is not cosine similarity. The hard part is knowing *when* to use cosine versus NMI, and how to combine them without introducing a manually-tuned weight that breaks on every new dataset.

This API solves exactly that:

**The weight is derived from entropy, not from your intuition.**

For each feature in your payload, the API computes marginal entropy H at query time. Features where H < 1.5 bits (low-variability, categorical by behavior) route through Normalized Mutual Information. Features where H >= 1.5 bits route through cosine similarity. The relative weight is:

```
w_nmi = sum(H_categorical) / (sum(H_categorical) + sum(H_continuous))
```

This means a payload of `{category, subcategory, price, embedding_vector}` auto-calibrates to weight NMI heavily. A payload of `{title_embedding, description_embedding, click_rate}` auto-calibrates toward cosine. You never touch a weight parameter.

Replicating this requires: per-feature entropy estimation at runtime, NMI computation invariant to cardinality differences across fields, adaptive fusion without distributional leakage between components, and a stateless design that does all of this per-request without persisting any index. That is several engineering weeks, not an afternoon.

---

## What problem this replaces

| Approach | What you manage | Cost at < 100k items |
|---|---|---|
| Pinecone / Weaviate | Index provisioning, embedding pipeline, infra uptime | $70–$140/mo minimum |
| Custom NMI + cosine pipeline | Feature typing, weight tuning, data drift recalibration | 2–4 weeks engineering |
| **This API** | Nothing | Pay per call |

If your corpus fits in a request payload (under 10MB), you do not need a vector database. You need a search call that returns in under 200ms with an explainable score.

---

## Response anatomy

```json
{
  "results": [
    {
      "id": "item_0042",
      "score": 0.847,
      "nmi_component": 0.91,
      "cosine_component": 0.76,
      "dominant_signal": "nmi",
      "w_nmi": 0.63,
      "w_cosine": 0.37
    }
  ],
  "payload_profile": {
    "categorical_features": ["category", "brand"],
    "continuous_features": ["price"],
    "entropy_weights_computed_at": "request_time"
  }
}
```

`dominant_signal` tells you which component drove the match. If it says `nmi`, the items are similar because their categorical distributions align — not because their numeric values are close. That distinction matters for debugging recommendations, fraud signals, and catalog deduplication.

---

## Pricing

$0.004 per search call (query against corpus up to 10,000 items).
$0.018 per search call (corpus 10,001–100,000 items).
No base fee. No index storage. No minimum spend.

---

## Limits

- Corpus size: up to 100,000 items per call
- Payload fields per item: up to 64
- Field value types: `str`, `int`, `float`, `bool`, `list[str]`
- Latency SLA: p95 < 220ms at 10k corpus, p95 < 800ms at 100k corpus

---

## Get an API key

[dashboard.hybridsimilarity.dev](https://dashboard.hybridsimilarity.dev) -> New key -> copy `sk-...`