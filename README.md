# Similarity Search API

Stateless similarity search over pre-computed vectors — NMI (normalized mutual information) + cosine fusion, with an entropy-calibrated blending weight computed per request. No vector database, no index to maintain, no infrastructure to run.

Available both as a plain HTTP API and as an MCP server (5 tools) for AI agents.

---

## Important: this operates on vectors, not raw text

This API does **not** embed text for you. `query` and `corpus` entries are pre-computed numeric vectors (e.g. from your own embedding model). If you need text-to-vector embedding first, run that upstream and pass the resulting vectors here.

---

## Base URL
https://similarity-search-api-production.up.railway.app

## Authentication

All business endpoints require an `X-API-Key` header:
X-API-Key: <your key>

`/health` requires no authentication.

## Pricing

Two ways to pay, same endpoints:

- **x402 (pay-per-call, USDC on Base)** — currently on **Base Sepolia testnet**, $0.01/call, no account or API key required beyond the x402 payment flow itself. A request without payment gets `402 Payment Required` with the payment details in the `payment-required` response header.
- **Stripe (metered billing)** — for callers provisioned with an API key and a Stripe customer on the account.

---

## Endpoints

### `POST /similarity/search`
Rank a corpus against a query vector using the composite score.

```json
{
  "query": { "id": "q1", "vector": [0.12, -0.4, 0.91, "..."] },
  "corpus": [
    { "id": "doc1", "vector": [0.10, -0.35, 0.88, "..."] },
    { "id": "doc2", "vector": [0.55, 0.02, -0.14, "..."] }
  ],
  "top_k": 10,
  "nmi_bins": 10,
  "alpha_override": null
}
```

All vectors in `query` and `corpus` must share the same dimensionality (2-4096 dims). `top_k` up to 1000. `alpha_override` (optional) pins the cosine/NMI blend weight instead of calibrating it from corpus entropy.

Response:
```json
{
  "results": [
    { "id": "doc1", "composite_score": 0.91, "cosine_similarity": 0.89, "nmi_score": 0.94, "rank": 1 }
  ],
  "calibrated_alpha": 0.73,
  "corpus_entropy": 3.85,
  "query_id": "q1",
  "corpus_size": 2,
  "latency_ms": 43,
  "request_fingerprint": "..."
}
```

### `POST /similarity/calibrate-alpha/v1`
Compute the entropy-calibrated alpha for a corpus without running a full search - useful for inspecting/debugging calibration behavior before committing to a search call.

### `POST /similarity/batch-score`
Score up to 10,000 `(vector_a, vector_b)` pairs with a fixed `alpha` - no corpus/entropy overhead.

```json
{
  "pairs": [[[0.1, 0.2], [0.15, 0.19]]],
  "alpha": 0.5,
  "nmi_bins": 10
}
```

### `GET /health`
Liveness probe. No auth required. Not billed (excluded from both Stripe and x402).

> Note: `POST /similarity/calibrate-alpha` (without `/v1`) is a deprecated alias kept for backward compatibility - use `/similarity/calibrate-alpha/v1`.

---

## MCP tools

Connect an MCP-compatible client (Claude, Cursor, etc.) to the streamable HTTP endpoint at:
https://similarity-search-api-production.up.railway.app/mcp

Exposes 5 tools: `rank_items_by_nmi_cosine_fusion`, `estimate_corpus_entropy_profile`, `score_pair_nmi_cosine`, `find_outlier_vectors_by_nmi_deficit`, `calibrate_alpha_from_query_entropy`.

---

## The scoring method
composite_score = alpha * cosine(query, doc) + (1 - alpha) * NMI_normalized(query, doc)

`alpha` is calibrated per-request from the Shannon entropy of the submitted corpus (unless you pass `alpha_override`) - high-entropy (dispersed) corpora lean toward cosine; low-entropy (dense/narrow) corpora lean toward NMI, which captures statistical dependence that cosine's geometric angle misses.

---

## Limits

- Corpus size: up to 500,000 items per request
- Vector dimensionality: 2-4,096
- `batch-score` pairs: up to 10,000 per request
