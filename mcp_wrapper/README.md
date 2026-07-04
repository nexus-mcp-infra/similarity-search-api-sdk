# nexus-similarity-search-api

Computes hybrid NMI+Cosine similarity over raw text, categorical, or discrete time-series data on-the-fly, without requiring pre-built vector indexes or embedding pipelines, using entropy-weighted scoring to surface statistically dependent items that cosine similarity misses.

Auto-generated MCP server by NEXUS FORGE. This package is a thin,
protocol-compliant adapter over the core service (`../core/`) — it
contains no business logic, only tool registration, validation, and
transport.

## Tools

- `rank_by_nmi_cosine_hybrid` — Ranks a corpus of raw items (text, discrete categories, or integer time series) against a query using a weighted hybrid of Normalized Mutual Information and cosine similarity, where weights are determined adaptively by the marginal entropy of the corpus. Use when your data has non-linear statistical dependencies that cosine alone would miss, or when working with categorical/discrete distributions without a prebuilt embedding index. Do NOT use for continuous high-dimensional float vectors already embedded — cosine-only is faster and sufficient in that case; do NOT use if corpus exceeds 10,000 items per call (use batch endpoint instead).
- `compute_pairwise_nmi_matrix` — Computes the full N x N Normalized Mutual Information matrix for a set of raw items, returning both the matrix and per-item marginal entropy values. Use for clustering preparation, feature selection diagnostics, or any downstream task that needs the full dependency structure of a corpus rather than a ranked list against a single query. Do NOT use as a ranking primitive — this is O(N^2) and intended for offline analysis, not per-request retrieval. Do NOT call with more than 500 items; use the batch endpoint for larger corpora.
- `estimate_corpus_entropy_profile` — Computes the marginal entropy of each item in a corpus and the joint entropy distribution across the full set, returning the entropy-based NMI weight that rank_by_nmi_cosine_hybrid would apply automatically. Use to audit or preview the adaptive weighting before committing to a ranking call, or to detect degenerate corpora where all items have near-zero entropy (making NMI unreliable). Do NOT use as a substitute for ranking — this endpoint only returns entropy diagnostics, not similarity scores.
- `score_candidate_pair_nmi_cosine` — Computes the NMI score, cosine score, and hybrid weighted score for exactly one (query, candidate) pair without needing a full corpus. Entropy-based weight is estimated from the pair's joint distribution alone (no corpus marginal). Use for spot-checks, unit-level debugging of scoring logic, or integration tests where you need a deterministic score for a known pair. Do NOT use in production ranking loops — call rank_by_nmi_cosine_hybrid instead, which benefits from corpus-level entropy calibration that this endpoint cannot provide for isolated pairs.
- `batch_rank_multiquery_nmi_cosine` — Executes NMI-Cosine hybrid ranking for multiple queries against a shared corpus in a single request, sharing corpus entropy computation across all queries to reduce total cost. Use when you need to rank the same corpus against 2 or more queries simultaneously (e.g., multi-faceted retrieval, ensemble query expansion). Do NOT use for a single query — rank_by_nmi_cosine_hybrid is cheaper. Do NOT use when each query has a different corpus — this endpoint assumes one shared corpus across all queries.

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
