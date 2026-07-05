# nexus-similarity-search-api

Ejecuta búsqueda híbrida stateless sobre colecciones arbitrarias fusionando NMI para features categóricas y coseno para features continuas en un score único con desglose por componente, sin índices ni infraestructura persistente.

Auto-generated MCP server by NEXUS FORGE. This package is a thin,
protocol-compliant adapter over the core service (`../core/`) — it
contains no business logic, only tool registration, validation, and
transport.

## Tools

- `rank_hybrid_similarity` — Ranks a corpus of records by hybrid similarity to a query record, fusing NMI for categorical/discrete features and cosine similarity for continuous features into a single adaptive-weighted score. Use when your payload contains a mix of categorical and numeric fields and you need ranked results with per-component score explanation. Do NOT use for pure text embedding search, for corpora larger than 50,000 records per call, or when you need persistent index storage between calls.
- `compute_pairwise_hybrid_matrix` — Computes the full N×N hybrid similarity matrix for a set of records using the same NMI+cosine adaptive fusion. Use for clustering preprocessing, graph construction, or any workflow that needs all pairwise distances in one call. Do NOT use when N > 2000 (O(N^2) complexity becomes prohibitive) or when you only need a single query ranked against a corpus — use rank_hybrid_similarity instead.
- `explain_feature_weight_calibration` — Given a sample of records, returns the adaptive weight assigned to each feature under the NMI+cosine fusion model: marginal entropy for categoricals, L2 norm variance for numerics, and the resulting nmi_weight/cosine_weight split. Use for auditing or understanding why the model weights features as it does before running a large ranking job. Do NOT use as a substitute for actual similarity computation — weights here are descriptive, not prescriptive overrides.
- `filter_by_hybrid_threshold` — Returns all corpus records whose hybrid similarity to the query exceeds a minimum threshold, without ranking. Use when you need a membership decision (similar/not-similar) rather than a ranked list, e.g., deduplication, near-duplicate detection, or candidate set construction. Do NOT use when you need a ranked ordering — rank_hybrid_similarity is more appropriate. Do NOT use with thresholds below 0.05 on high-cardinality categorical corpora, as recall will be near-total and the result set will be unmanageably large.
- `detect_feature_type_schema` — Analyzes a sample of records and returns the inferred type (categorical or continuous) and entropy/variance statistics for each feature key, exactly as the similarity engine would classify them internally. Use before a ranking or filtering call to validate that the engine will treat your features as intended, especially for ambiguous fields (e.g., integer codes that should be categorical). Do NOT use as a general-purpose schema inference tool — it only classifies features into the two types the similarity model supports.

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
