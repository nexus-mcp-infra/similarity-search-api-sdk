# nexus-similarity-search-api

Ejecuta búsqueda de similitud coseno con filtro NMI e intervalo de confianza bootstrap por llamada, devolviendo p-value de significancia estadística por par para distinguir patrones reales de correlaciones aleatorias en embeddings.

Auto-generated MCP server by NEXUS FORGE. This package is a thin,
protocol-compliant adapter over the core service (`../core/`) — it
contains no business logic, only tool registration, validation, and
transport.

## Tools

- `rank_embeddings_by_nmi_cosine` — Ranks a corpus of embeddings against a query vector using a hybrid score that combines cosine similarity with Normalized Mutual Information (NMI) computed via joint-entropy estimation over discretized embedding dimensions. Returns per-pair p-values from bootstrap confidence intervals. Use when you need statistically validated similarity rankings where you must distinguish real dependency patterns from random correlation. Do NOT use for real-time latency-sensitive paths (>500 vectors adds bootstrap overhead), for pure nearest-neighbor ANN tasks where p-values are irrelevant, or when corpus vectors are fewer than 10 (bootstrap intervals become unreliable).
- `estimate_pairwise_nmi_matrix` — Computes the full N×N Normalized Mutual Information matrix for a set of embeddings, returning each cell's NMI score along with a bootstrap-derived p-value under the null hypothesis H0: NMI=0 (independence). Use for clustering pre-analysis, redundancy detection across a document set, or graph-of-similarity construction where edge weights must be statistically grounded. Do NOT use when N > 80 — O(N^2 * bootstrap_iterations) cost makes it prohibitive; use rank_embeddings_by_nmi_cosine in batches instead. Not suitable as a real-time retrieval path.
- `score_candidate_pair_significance` — Computes the hybrid NMI-cosine score and bootstrap p-value for exactly one (query, candidate) embedding pair. Use when you already have a candidate from an external ANN index and need to validate whether the cosine similarity reflects a real statistical dependency — i.e., post-retrieval significance gating. Do NOT use to rank a corpus (use rank_embeddings_by_nmi_cosine instead); calling this in a loop over hundreds of candidates is wasteful because it cannot amortize discretization costs across the corpus.
- `detect_embedding_dimension_redundancy` — Identifies redundant dimensions within a single embedding space by computing pairwise NMI across all D*(D-1)/2 dimension pairs of the provided sample vectors, returning clusters of highly dependent dimensions (NMI above threshold) and a suggested reduced dimensionality. Use before building a similarity pipeline to prune embedding dimensions that carry no additional information — reduces downstream NMI estimation variance and cosine noise. Do NOT use for embeddings with D > 256 (quadratic in D); not intended for runtime retrieval calls, only for offline embedding space analysis.
- `calibrate_nmi_cosine_weight_for_corpus` — Given a labeled calibration set of (query, positive_candidate, negative_candidate) triplets and their embeddings, finds the optimal nmi_cosine_weight w that maximizes separation between positive and negative pairs under the hybrid scoring function, reporting the optimal w with its bootstrap confidence interval and the resulting AUC-ROC. Use once before deploying rank_embeddings_by_nmi_cosine on a specific embedding model and domain to select the best w rather than using the default 0.5. Do NOT use as a runtime call per request — this is a one-time offline calibration step. Requires labeled triplets; if no labels are available, skip and use the default weight.

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
