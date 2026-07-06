# Similarity Search API

Find what's actually similar — not just what's geometrically close.

---

## Install

```bash
pip install similarity-search-sdk
```

## Start searching in 3 lines

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")
results = client.rank(query=[0.12, 0.87, ...], candidates=[[...], [...]], alpha=0.7)
```

`alpha` controls the weight between cosine geometry and mutual information. No index. No infrastructure. One call.

---

## Why not build this yourself

Every team that needs similarity search faces the same fork: use a vector database (Pinecone, Weaviate, Qdrant) or roll a cosine function. Both choices share a silent assumption — geometric closeness is a good enough proxy for real similarity.

It isn't, in noisy domains.

Two vectors can be cosine-similar because they share a directional bias introduced by dataset artifacts, domain shift, or feature correlation that has nothing to do with semantic relatedness. Normalized Mutual Information (NMI) catches this. It penalizes pairs that are geometrically close but statistically independent — the fingerprint of a spurious correlation.

The problem is that NMI and cosine live in separate tools. Combining them manually means:

- Estimating NMI requires discretizing continuous vectors — binning strategy matters, and most implementations get it wrong for high-dimensional data
- Fusing two scores without a principled weighting produces rankings that vary with implementation details, not signal
- None of this runs stateless — most teams end up with a pipeline with intermediate state, a separate scoring step, and drift between the two

This API solves the fusion problem at the request level. The composite score `S = alpha * cosine(u, v) + (1 - alpha) * NMI(u, v)` is computed in a single call using Freedman-Diaconis adaptive binning for the NMI estimation — a method that adjusts bin width to the empirical distribution of each input vector, not a fixed discretization chosen at index-build time.

The result is a score with verifiable statistical properties: if `alpha = 1.0` you get pure cosine; if `alpha = 0.0` you get pure mutual information; anywhere in between you get a controlled blend where both components are on the same `[0, 1]` scale before fusion.

---

## When to use this

- **Duplicate detection** in document corpora where near-identical embeddings hide content that's statistically unrelated
- **Recommendation ranking** where cosine-only similarity produces filter bubbles from correlated features
- **Semantic clustering** validation — use composite score drift across alpha values as a signal for cluster quality
- **Prototype validation** on datasets under 500k items where standing up a vector database adds days of operational overhead before you've confirmed the signal exists

## When not to use this

This API is stateless by design — it does not maintain an index. If your use case requires sub-10ms approximate nearest-neighbor search across tens of millions of pre-indexed vectors with persistent storage, use a dedicated vector database. This primitive is for **scoring and ranking on demand**, not for ANN retrieval at scale.

---

## The `alpha` parameter

`alpha` is the single knob between geometry and information.

| alpha | Behavior |
|-------|----------|
| `1.0` | Pure cosine similarity — fast, standard, no MI penalty |
| `0.7` | Default — cosine-dominant with MI correction for spurious pairs |
| `0.5` | Equal weight — maximum discrimination between geometric and statistical signal |
| `0.0` | Pure NMI — statistical dependence only, geometry ignored |

The right `alpha` depends on your domain. High-noise domains (user behavior, sparse item embeddings) benefit from lower alpha. Dense semantic embeddings from transformer models typically work well at `0.6`-`0.8`.

---

## Full request example

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")

query_vector = [0.12, 0.45, 0.87, 0.33, 0.91]
candidates = [
    [0.11, 0.44, 0.85, 0.35, 0.90],  # geometrically close, statistically dependent
    [0.10, 0.46, 0.88, 0.10, 0.20],  # geometrically close, statistically independent
    [0.80, 0.10, 0.05, 0.60, 0.44],  # geometrically distant
]

results = client.rank(
    query=query_vector,
    candidates=candidates,
    alpha=0.7,
    domain_tag="product_embeddings",  # optional — improves alpha suggestions over time
)

for r in results.ranked:
    print(r.index, r.score_composite, r.score_cosine, r.score_nmi)
```

```
0   0.891   0.943   0.788
1   0.612   0.901   0.142   # <- cosine would have ranked this #1
2   0.201   0.187   0.231
```

Candidate 1 is the case your current stack gets wrong. Cosine ranks it first. The composite score surfaces the statistical independence and drops it to second.

---

## REST API (direct HTTP)

```bash
curl https://api.similaritysearch.io/v1/rank \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.12, 0.45, 0.87, 0.33, 0.91],
    "candidates": [[0.11, 0.44, 0.85, 0.35, 0.90], [0.10, 0.46, 0.88, 0.10, 0.20]],
    "alpha": 0.7,
    "domain_tag": "product_embeddings"
  }'
```

Response:

```json
{
  "ranked": [
    {"index": 0, "score_composite": 0.891, "score_cosine": 0.943, "score_nmi": 0.788},
    {"index": 1, "score_composite": 0.612, "score_cosine": 0.901, "score_nmi": 0.142}
  ],
  "meta": {
    "alpha_used": 0.7,
    "binning_method": "freedman_diaconis",
    "latency_ms": 18
  }
}
```

---

## No infrastructure required

No Docker. No Kubernetes. No persistent index to manage, back up, or keep warm. No embedding model to host. Send vectors, get a ranked list. The entire operational surface is an HTTP request and an API key.

For teams validating a similarity signal before committing to a vector database architecture, this removes the infrastructure decision from the validation loop entirely.

---

## Security

- All requests are authenticated with a bearer token
- Vectors are not stored — they are discarded after scoring
- Aggregate metadata (alpha, dimensionality, domain tag, composite score, latency) is logged for alpha optimization features; no vector data is retained

---

## Support

[docs.similaritysearch.io](https://docs.similaritysearch.io) — API reference, error codes, SDK changelog

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