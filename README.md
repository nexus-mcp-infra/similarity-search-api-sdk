# SimilarityAPI

Stateless semantic similarity. No index. No upsert. Just a score.

---

## Install

```bash
pip install similarity-api-client
```

## Three lines to a similarity score

```python
from similarity_api import SimilarityClient

client = SimilarityClient(api_key="sk_test_xxxxxxxxxxxxxxxx")
score = client.compare(embeddings_a=my_vectors, embeddings_b=candidate_vectors, domain="text")
# -> {"score": 0.847, "cosine": 0.791, "nmi": 0.923, "domain": "text", "latency_ms": 12}
```

---

## Why not Pinecone, Weaviate, or raw cosine?

Every vector database in the market charges you for **storing** embeddings, not for **computing** similarity. That means before your first query you must:

1. Provision an index
2. Upsert every vector you want to search
3. Pay for storage you don't need for ad-hoc or low-frequency use cases

SimilarityAPI does none of that. Send your embeddings in the request body, get a calibrated score back. Nothing persisted. No index. No setup time measured in hours.

| | SimilarityAPI | Pinecone / Weaviate | Raw cosine (DIY) |
|---|---|---|---|
| Setup to first result | < 60 seconds | 30-120 min (index + upsert) | Minutes, but no NMI |
| Persistent index required | No | Yes | No |
| Captures non-linear dependence | Yes (NMI) | No | No |
| Domain-calibrated weights | Yes | No | No |
| Pricing model | Per call | Per index / per seat | Infra cost |

---

## The score that actually ranks better

Cosine similarity captures **directional alignment** between vectors. It misses **non-linear statistical dependence** — the signal that matters when your embedding space has cluster structure, distributional skew, or domain-specific correlations.

SimilarityAPI computes:

```
S = alpha * cosine(A, B) + (1 - alpha) * NMI(A, B)
```

Where:
- `cosine(A, B)` — standard cosine similarity over the full embedding vectors
- `NMI(A, B)` — Normalized Mutual Information over the joint activation distribution, computed via NEXUS's information-theoretic module
- `alpha` — a domain-specific weight learned offline via gradient descent over NDCG rankings on MTEB (text), BEIR (retrieval), and internal tabular benchmarks

Typical calibrated values: `alpha_text = 0.61`, `alpha_image = 0.54`, `alpha_tabular = 0.38`. Lower alpha means NMI carries more weight — tabular data has stronger non-linear structure that cosine systematically underweights.

**Benchmark result (BEIR / MTEB subset):** The NMI+Cosine composite ranks correctly ~8.3 percentage points more often than cosine alone on queries where the top-1 cosine result is wrong. That gap is not replicable by copying the interface — it requires the mutual information computation and the domain-calibrated alpha.

---

## API reference

### `POST /v1/similarity`

```json
{
  "embeddings_a": [[0.12, 0.95, ...]],
  "embeddings_b": [[0.08, 0.91, ...]],
  "domain": "text"
}
```

**domain** — one of `"text"` | `"image"` | `"tabular"`. Controls which alpha/beta weights are applied. Required.

**Response**

```json
{
  "score": 0.847,
  "cosine": 0.791,
  "nmi": 0.923,
  "alpha": 0.61,
  "domain": "text",
  "latency_ms": 12
}
```

**What gets logged (and only this):** SHA-256 hash of the input pair, composite score, domain, latency. Raw embeddings are never stored. This log feeds the weight recalibration flywheel — more production calls -> better domain weights -> better ranking for every user.

### `POST /v1/similarity/batch`

Same schema, `embeddings_b` accepts up to 512 vectors. Returns a ranked list with one score object per candidate, sorted by `score` descending.

### `GET /v1/health`

Returns `{"status": "ok", "version": "..."}`. No auth required.

---

## Authentication

Pass your API key in the `Authorization` header:

```bash
curl https://api.similarityapi.io/v1/similarity \
  -H "Authorization: Bearer sk_test_xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"embeddings_a": [[...]], "embeddings_b": [[...]], "domain": "text"}'
```

---

## Pricing

**$0.0008 per call** (single pair comparison).  
**$0.0004 per vector** in a batch call.  
No monthly minimum. No index storage fee. No seat license.

A team running 50,000 ad-hoc comparisons per month pays **$40**. The same workload on a vector database with index storage and per-seat pricing runs **$200-$600** before you count engineering time to maintain the index.

---

## What you don't have to build

- An embedding storage layer
- Index lifecycle management (create, update, delete, re-index on schema change)
- A custom NMI implementation (getting the discretization and normalization right for high-dimensional vectors is non-trivial — see Kraskov et al. 2004)
- Domain-specific weight calibration against public retrieval benchmarks
- A stateless compute service that handles burst traffic without cold-start index load

---

## Get a key

[similarityapi.io/signup](https://similarityapi.io/signup) — free tier includes 1,000 calls/month.