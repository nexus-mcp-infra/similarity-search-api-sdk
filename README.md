# Similarity Search API

Find the most similar items to a query — ranked by a hybrid NMI+Cosine score that no vector database exposes natively.

---

## Install

```bash
pip install similarity-search-sdk
```

## Quickstart

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")
results = client.rank(query=[0.12, 0.87, 0.34, ...], corpus=my_vectors, top_k=10)
```

## Why not build it yourself?

You could. Here is what that actually costs:

**The NMI trap.** Normalized Mutual Information sounds like a one-liner with `sklearn.metrics`. It is not — not if you want correct rankings. For small corpora (n < 200, exactly the case in analysis scripts and batch jobs), NMI estimators are biased toward larger cardinalities. The fix — Strehl-Ghosh correction over the joint distribution — requires estimating that joint distribution first, which means adaptive binning for dense vectors, and a completely separate code path for discrete probability distributions. Most engineers discover this bug three weeks after shipping, when rankings look plausible but wrong.

**The hybrid scoring gap.** No existing vector database or similarity library exposes `alpha * NMI + (1 - alpha) * Cosine` as a tunable scoring function. Pinecone, Weaviate, and FAISS all assume a pure vector space and optimize for cosine or dot product at index time. Wiring NMI into those pipelines means pulling vectors back out of the index and re-ranking client-side — which is exactly what this API does for you, correctly, per call.

**The infrastructure trap.** Spinning up a vector DB to compare 500 items in a one-off analysis script is not engineering — it is overhead. This API requires no persistent index, no infrastructure provisioning, and no schema. Send your corpus in the request body, get ranked results back. Done.

---

## How scoring works

Every result is ranked by:

```
score = alpha * NMI_corrected(query, candidate) + (1 - alpha) * cosine(query, candidate)
```

`alpha` defaults to `0.5`. Set it to `0.0` for pure cosine (equivalent to any standard vector search). Set it to `1.0` for pure NMI — useful when your vectors represent probability distributions or histograms where directional similarity is less meaningful than shared information structure.

Two input modes, zero configuration required:

| Input type | What the API does |
|---|---|
| Dense float vectors | Adaptive binning -> joint distribution estimation -> Strehl-Ghosh bias correction -> NMI |
| Discrete probability distributions | NMI directly, no binning step |

The API detects which path to take from the shape and values of your input.

---

## Core endpoints

```
POST /v1/rank          — rank a corpus against a single query vector
POST /v1/rank/batch    — rank multiple queries against the same corpus in one call
GET  /v1/health        — confirm API reachability and version
```

Full parameter reference and response schema: **[docs.similarity-search.dev](https://docs.similarity-search.dev)**

---

## Example: discrete distributions

```python
import numpy as np
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="YOUR_API_KEY")

# Topic distributions over a 50-term vocabulary
query_topic  = np.array([0.30, 0.05, 0.20, ...])   # sums to 1.0
corpus_topics = [np.array([...]) for _ in range(200)]

results = client.rank(
    query=query_topic,
    corpus=corpus_topics,
    input_type="distribution",   # skip binning entirely
    alpha=0.8,                   # weight NMI heavily
    top_k=5,
)

for r in results:
    print(r.rank, r.score, r.index)
```

---

## Complexity

- Per-call, in-memory: **O(n log n)** over the corpus you send — dominated by the adaptive binning step for dense vectors.
- No index build time. No cold start. No state between calls.
- Corpus size limit per request: **10,000 vectors**. Above that, use `/v1/rank/batch` with chunked corpora.

---

## Requirements

- Python 3.8+
- An API key — get one at **[similarity-search.dev](https://similarity-search.dev)**

---

## What this is not

- Not a persistent vector database. If you need to store and incrementally update millions of vectors, use Pinecone or Weaviate. Come back to this API when you need NMI-aware re-ranking on top of their recall results.
- Not a training or fine-tuning service. This API scores vectors; it does not produce them.

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