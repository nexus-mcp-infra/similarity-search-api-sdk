# nexus-similarity-search-api

Computes on-the-fly semantic similarity between raw embedding vectors using a calibrated NMI+Cosine composite score without requiring vector storage, indexing, or upsert operations.

Auto-generated MCP server by NEXUS FORGE. This package is a thin,
protocol-compliant adapter over the core service (`../core/`) — it
contains no business logic, only tool registration, validation, and
transport.

## Tools

- `rank_embeddings_by_nmi_cosine` — Ranks a set of candidate embeddings against a query embedding using a weighted NMI+Cosine composite score calibrated per domain. Use when you need stateless semantic similarity ranking without a vector index or upsert step, especially when candidate correlations are non-linear. Do NOT use if you have a persistent vector store already indexed — latency will be higher than ANN retrieval for corpora above 50k vectors.
- `compute_pairwise_nmi_cosine_matrix` — Computes the full N×N composite similarity matrix for a set of embeddings. Use for clustering preprocessing, graph construction, or reranking pipelines where every pair needs a score. Do NOT use for query-vs-corpus ranking (use rank_embeddings_by_nmi_cosine instead) — cost is O(N²) and grows quadratically.
- `score_embedding_pair_nmi_cosine` — Returns the decomposed composite similarity score (NMI component, cosine component, weighted composite) for exactly one pair of embeddings. Use for debugging, threshold calibration, or audit trails where you need interpretable component breakdown. Do NOT use in batch loops — use rank_embeddings_by_nmi_cosine or compute_pairwise_nmi_cosine_matrix for multiple pairs; per-call overhead makes looping expensive.
- `calibrate_domain_nmi_cosine_weights` — Derives optimal alpha_nmi and alpha_cosine weights for a custom embedding domain by fitting the composite scorer to a labeled relevance dataset you supply. Use when 'text', 'image', and 'tabular' presets underperform on your specific embedding model or corpus distribution. Do NOT use at inference time — run once offline and cache the returned weight profile; recalibrate only when the embedding model changes.
- `explain_nmi_cosine_rank_divergence` — Given a query and candidates, returns a divergence report showing where NMI-informed ranking differs from pure-cosine ranking and why — quantifying non-linear dependency contribution per candidate. Use when auditing model behavior, justifying ranking decisions to stakeholders, or diagnosing unexpected rank positions. Do NOT use in latency-sensitive inference paths — this runs both rankers plus divergence attribution and is 2-3x slower than rank_embeddings_by_nmi_cosine alone.

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
