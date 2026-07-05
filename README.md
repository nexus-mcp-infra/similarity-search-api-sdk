# Similarity Search API

NMI-weighted cosine similarity in a single HTTP call. No vector database. No setup. No fine-tuning.

---

## Install

```bash
pip install similarity-search-sdk
```

## Search in 3 lines

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="sk_test_xxxxxxxxxxxxxxxx")
results = client.search(query={"text": "transformer architecture"}, corpus=[...])
# -> [{"id": "doc_42", "score": 0.91, "confidence_interval": [0.87, 0.94]}]
```

---

## Why not build it yourself?

| What you'd need to build | This API |
|---|---|
| Vector DB (Pinecone, Weaviate, Qdrant) + infra setup | Zero infra. Stateless per call. |
| Manual feature selection before embedding | NMI filter runs automatically on your payload |
| Cosine similarity without statistical grounding | Score includes auditable confidence interval |
| Separate pipelines for text vs. tabular vs. vector inputs | One endpoint, all three input types |
| Days of prototyping before first result | Working in under 5 minutes |

---

## The problem with Cosine alone

Cosine similarity treats every dimension equally. In high-dimensional spaces — embeddings, tabular features, tokenized text — most dimensions are noise. Noise inflates similarity scores between unrelated items and collapses them between related ones.

```
Without NMI filter:  query "revenue forecast" -> matches "weather forecast"  (score: 0.82)
With NMI filter:     query "revenue forecast" -> matches "Q3 financial model" (score: 0.89, CI: [0.85, 0.92])
```

This API runs Normalized Mutual Information over the features present in your request payload — query + corpus — before computing cosine distance. Features with low statistical dependence to the query are down-weighted. The result is a distance metric that reflects structure, not just angle.

The confidence interval is not a heuristic. It is derived from the empirical distribution of NMI scores across your corpus, so you can tell the difference between a strong match and a coincidental one.

---

## Endpoint

```
POST https://api.similarity.nexus/v1/search
Authorization: Bearer sk_test_xxxxxxxxxxxxxxxx
Content-Type: application/json
```

```json
{
  "query": {
    "text": "annual recurring revenue growth"
  },
  "corpus": [
    {"id": "doc_1", "text": "SaaS revenue metrics and ARR expansion"},
    {"id": "doc_2", "text": "rainfall patterns in the Amazon basin"},
    {"id": "doc_3", "features": [0.12, 0.87, 0.34, 0.65]}
  ],
  "top_k": 3
}
```

```json
{
  "results": [
    {
      "id": "doc_1",
      "score": 0.934,
      "confidence_interval": [0.901, 0.958],
      "nmi_retained_features": 18
    }
  ],
  "latency_ms": 34,
  "nmi_computed_features": 42
}
```

---

## Input types

| Type | Field | Example |
|---|---|---|
| Raw text | `"text"` | `"transformer architecture"` |
| Pre-computed vector | `"vector"` | `[0.12, 0.87, ..., 0.34]` |
| Tabular features | `"features"` | `[1200.0, 0.43, 7, 0.91]` |

Mix types within the same corpus. The NMI pipeline normalizes across representations before computing distance.

---

## Confidence interval explained

After computing NMI scores for all features in your corpus, the API fits an empirical distribution over those scores. The confidence interval on the final similarity result reflects where that match lands relative to the full distribution — not a fixed threshold, not a model output. If the interval is narrow and high, the match is structurally unambiguous. If it is wide, you have genuine uncertainty that should inform downstream decisions.

This is the property that makes the score **auditable**: given the same payload, any implementation of NMI-weighted cosine will produce the same interval.

---

## Pricing

| Calls / month | Price per call |
|---|---|
| First 10,000 | Free |
| 10,001 - 500,000 | $0.0008 |
| 500,001+ | $0.0004 |

No base fee. No seat license. Pay per search operation.

---

## When to use this instead of a vector database

Use this API when:
- Your dataset is under 100k items and standing up a vector DB is more infrastructure than problem
- You need a confidence-grounded similarity score, not just a ranking
- You are combining text, numeric, and vector inputs in the same search space
- You are prototyping and need a result today, not after a data pipeline is built

Use a dedicated vector database when:
- You are running ANN search over tens of millions of vectors with sub-10ms SLA requirements
- Your access pattern is purely retrieval with no need for statistical grounding on the score

---

## Auth

All requests require a Bearer token in the `Authorization` header.

```bash
curl -X POST https://api.similarity.nexus/v1/search \
  -H "Authorization: Bearer sk_test_xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"query": {"text": "example"}, "corpus": [{"id": "1", "text": "sample document"}], "top_k": 1}'
```

Get your API key at [similarity.nexus/dashboard](https://similarity.nexus/dashboard).

---

## SDK

```bash
pip install similarity-search-sdk   # Python
npm install @nexus/similarity-search  # Node
```

Full SDK reference at [similarity.nexus/docs](https://similarity.nexus/docs).