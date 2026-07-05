# nexus-similarity-search-api

Computes stateless NMI-weighted cosine similarity between two collections of heterogeneous records (text + numeric + categorical) and returns ranked results in a single HTTP call, no index or prior setup required.

Auto-generated MCP server by NEXUS FORGE. This package is a thin,
protocol-compliant adapter over the core service (`../core/`) — it
contains no business logic, only tool registration, validation, and
transport.

## Tools

- `rank_by_nmi_weighted_cosine` — Computes NMI-weighted cosine similarity between a query vector and a candidate collection, returning ranked results in a single stateless HTTP call. Use when the feature space is heterogeneous (mixed numeric, categorical, or multimodal) and standard cosine underperforms due to non-linear dependencies. Do NOT use for pure text embeddings where cosine is sufficient, or when you need approximate nearest-neighbor at >1M candidates per call (latency will degrade beyond practical limits).
- `compute_pairwise_nmi_matrix` — Computes the full NMI matrix across feature dimensions for a given collection, exposing per-feature mutual information weights without performing a search. Use this to inspect which feature dimensions carry the most discriminative mutual information before invoking rank_by_nmi_weighted_cosine, or to debug why certain features dominate weighting. Do NOT use as a replacement for the ranked search — this returns a DxD weight matrix, not similarity scores between items.
- `score_heterogeneous_pair` — Computes the NMI-weighted cosine similarity score for exactly one query-candidate pair with full per-dimension weight breakdown. Use for explainability: when you need to understand why two specific items score high or low, or to validate that NMI weighting is behaving correctly on a known pair. Do NOT use in batch loops to simulate collection ranking — rank_by_nmi_weighted_cosine is vectorized and orders of magnitude faster for that purpose.
- `filter_candidates_by_nmi_threshold` — Returns only candidates whose NMI-weighted cosine score meets or exceeds a minimum threshold, without a fixed top_k cutoff. Use when downstream logic requires a quality floor rather than a fixed result count — e.g., deduplication pipelines, semantic clustering seeding, or anomaly filtering where you want all sufficiently similar items. Do NOT use when you need exactly K results; use rank_by_nmi_weighted_cosine with top_k instead.
- `benchmark_nmi_vs_cosine_delta` — Runs both pure cosine and NMI-weighted cosine ranking on the same query and candidates, returning side-by-side rank positions and score deltas per candidate. Use to quantify how much NMI weighting changes ranking order for a specific dataset — critical for justifying adoption to stakeholders or detecting feature spaces where cosine already suffices. Do NOT use in production ranking pipelines — this endpoint is designed for evaluation and costs roughly 2x the compute of a single ranking call.

## Local development

```bash
npm install
cp .env.example .env   # set NEXUS_CORE_BASE_URL
npm run dev
```

## Testing with MCP Inspector

```bash
npm run build
npm run inspector
```

## Production (Streamable HTTP)

```bash
npm run build
npm run start:http
# POST http://localhost:3000/mcp
# GET  http://localhost:3000/health
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `NEXUS_CORE_BASE_URL` | yes | Base URL of the underlying core API |
| `NEXUS_CORE_API_KEY` | no | Bearer token forwarded to the core API |
| `TRANSPORT` | no | `stdio` (default) or `http` |
| `PORT` | no | HTTP port (default 3000) |
