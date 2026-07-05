# nexus-similarity-search-api

Computes NMI-weighted cosine similarity between a query vector and a corpus in a single stateless HTTP call, returning ranked matches with empirical confidence intervals derived from the NMI distribution over query features.

Auto-generated MCP server by NEXUS FORGE. This package is a thin,
protocol-compliant adapter over the core service (`../core/`) — it
contains no business logic, only tool registration, validation, and
transport.

## Tools

- `rank_vectors_by_nmi_cosine` — Ranks a corpus of vectors against a query vector using NMI-filtered cosine similarity. NMI is computed per-feature across the corpus to suppress noisy dimensions before distance calculation. Use this when you have raw numerical feature vectors and want ranked similarity with confidence intervals. Do NOT use for text inputs (use tokenized_corpus_similarity instead), and do NOT use when corpus size exceeds 50,000 vectors per call — batch instead.
- `compute_tokenized_corpus_similarity` — Accepts pre-tokenized text sequences (lists of token IDs or term-frequency feature arrays) and ranks them against a query sequence using NMI-weighted cosine. NMI is computed over the token co-occurrence feature space to prune uninformative vocabulary dimensions. Use this for sparse text feature vectors or BoW/TF-IDF representations. Do NOT use with dense embedding vectors (use rank_vectors_by_nmi_cosine) and do NOT pass raw strings — tokenize first.
- `extract_nmi_feature_weights` — Computes per-dimension NMI scores between a query vector and a corpus, returning the weight assigned to each feature dimension without performing similarity ranking. Use this to audit which features drive the NMI-cosine score, to tune nmi_threshold before a full ranking call, or to pre-validate corpus quality. Do NOT use as a substitute for ranking — this endpoint returns weights only, not similarity scores.
- `compare_tabular_row_similarity` — Ranks rows in a tabular dataset (mixed numerical features, pre-encoded) against a query row using NMI-weighted cosine. Designed for structured tabular data where feature columns have heterogeneous scales. Handles per-column z-score normalization internally before NMI and cosine steps. Use for structured ML datasets, relational feature tables, or any input that originates as a dataframe row. Do NOT use for raw text or image embeddings — those have different distribution assumptions.
- `estimate_similarity_confidence_band` — Given a set of pre-computed NMI-cosine similarity scores and the underlying NMI weight distribution, returns bootstrap-derived confidence bands for each score. Use this when you already have scores from a prior ranking call and want to widen or narrow confidence intervals at a different confidence level, or to validate stability of results under resampling. Do NOT use as a primary similarity computation — this endpoint takes scores as input, not raw vectors.

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
