# nexus-similarity-search-api

Ejecuta búsqueda de similitud coseno con filtro NMI e intervalo de confianza bootstrap por llamada, devolviendo p-value de significancia estadística por par para distinguir patrones reales de correlaciones aleatorias en embeddings.

Auto-generated MCP server by NEXUS FORGE. This package is a thin,
protocol-compliant adapter over the core service (`../core/`) — it
contains no business logic, only tool registration, validation, and
transport.

## Tools

<!-- NEXUS PATCH mcp_tool_grounding_similarity_search: originally listed 5
     tools, 4 of which called endpoints with no real implementation in the
     core service. Only the 1 tool with a real backend is listed below. -->

- `rank_embeddings_by_nmi_cosine` — Ranks a corpus of embeddings against a query vector using a composite score that blends cosine similarity with entropy-calibrated Normalized Mutual Information (NMI). Stateless per-call scoring, no vector database required. Use when you need a similarity ranking over an inline corpus. Do NOT use for corpora larger than 500,000 items in a single call, or when you need exact nearest-neighbor search rather than a hybrid statistical score.

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
