# Similarity Search API

Statistical significance for every search result — cosine similarity ranked by NMI, p-value included.

---

## Install

```bash
pip install similarity-search-sdk
```

## Quickstart

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="sk_test_xxxxxxxxxxxxxxxx")
results = client.search(query=[0.12, -0.84, 0.33, ...], corpus=[[...], [...], [...]])
# results[0] -> {"score": 0.91, "nmi": 0.74, "p_value": 0.003, "rank": 1}
```

---

## The problem with cosine alone

Every vector database returns a ranked list. None of them answer the question that actually matters:

> **Is this similarity real, or is it noise?**

A cosine score of `0.87` between two embeddings could mean strong semantic overlap — or it could be a coincidence in a high-dimensional space where most vectors are roughly equidistant. You have no way to tell without running a statistical test you almost certainly haven't written.

The Similarity Search API solves this in a single HTTP call.

---

## What you get back

Every response includes four fields per pair:

| Field | What it means |
|---|---|
| `cosine_score` | Standard cosine similarity `[-1, 1]` |
| `nmi` | Normalized Mutual Information `[0, 1]` — how much knowing one vector reduces uncertainty about the other |
| `p_value` | Bootstrap p-value: probability this NMI score occurs under the null hypothesis of independence |
| `ci_95` | 95% confidence interval on NMI via bootstrap resampling (1 000 draws per call) |

A result with `cosine_score: 0.89, p_value: 0.41` is a different decision than `cosine_score: 0.89, p_value: 0.003`. The first one is noise. The second one is signal.

---

## Why not build this yourself

The hard part is not the formula. NMI is `I(X;Y) / sqrt(H(X) * H(Y))`. The hard part is everything that sits underneath it:

**Discretization of continuous embeddings.** To estimate `H(X)` and `H(X,Y)`, you need to bin each embedding dimension. If you use fixed-width bins (k=10 is the typical shortcut), you get artificially inflated NMI in low-variance dimensions — the score looks meaningful but the statistic is garbage. The correct approach is Freedman-Diaconis bin selection per dimension, which requires computing the IQR of each dimension across your corpus and deriving bin count as `2 * IQR * n^(-1/3)`. That is a different implementation for every call, not a constant.

**Bootstrap confidence intervals that are actually valid.** Resampling embedding pairs without replacement, recomputing joint entropy on each resample, and aggregating into a distribution requires getting floating-point accumulation right across 1 000 draws without introducing bias from array reuse. The naive implementation introduces ~8–12% NMI inflation from memory aliasing on NumPy views. We benchmarked this.

**No database to maintain.** This API is stateless — you POST vectors, you get statistics back. There is no index to build, no cluster to size, no nightly re-embedding job. If you only query a corpus once, you pay for one call, not for months of vector storage.

**The bin-count flywheel.** Our infrastructure accumulates real embedding distributions across calls to calibrate optimal bin counts by dimensionality range (768-dim BERT vectors bin differently than 1 536-dim Ada vectors). The longer you use the API, the more accurate the discretization becomes for your embedding model. You cannot bootstrap this from day one with an ad-hoc script.

---

## API reference

### `POST /v1/search`

```bash
curl https://api.similaritysearch.io/v1/search \
  -H "Authorization: Bearer sk_test_xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.12, -0.84, 0.33],
    "corpus": [[0.11, -0.80, 0.31], [0.90, 0.02, -0.45]],
    "top_k": 5,
    "bootstrap_draws": 1000,
    "significance_threshold": 0.05
  }'
```

**Response**

```json
{
  "results": [
    {
      "rank": 1,
      "corpus_index": 0,
      "cosine_score": 0.9987,
      "nmi": 0.812,
      "p_value": 0.001,
      "ci_95": [0.791, 0.834],
      "significant": true
    },
    {
      "rank": 2,
      "corpus_index": 1,
      "cosine_score": 0.431,
      "nmi": 0.089,
      "p_value": 0.38,
      "ci_95": [0.041, 0.143],
      "significant": false
    }
  ],
  "meta": {
    "query_dim": 3,
    "corpus_size": 2,
    "bootstrap_draws": 1000,
    "bin_method": "freedman_diaconis",
    "latency_ms": 47
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `float[]` | required | Query embedding vector |
| `corpus` | `float[][]` | required | Candidate vectors, max 10 000 |
| `top_k` | `int` | `10` | Results to return, max 500 |
| `bootstrap_draws` | `int` | `1000` | Resamples for CI estimation. More draws -> narrower CI, higher latency |
| `significance_threshold` | `float` | `0.05` | p-value cutoff for `significant` flag |

---

## Pricing

| Calls / month | Price per call |
|---|---|
| 0 – 10 000 | $0.004 |
| 10 001 – 500 000 | $0.0028 |
| 500 001+ | $0.0018 |

No base fee. No storage fee. No minimum commitment. You pay for computation, not for parking vectors you queried once in January.

---

## Get an API key

```
https://similaritysearch.io/dashboard
```

Free tier: 500 calls/month, no credit card required.

---

## License

MIT