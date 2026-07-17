# Calibrated Similarity Search API — AGENTS.md

> Generado a partir del código fuente real deployado (`nexus-mcp-infra/similarity-search-api-sdk`,
> `core/similarity_search_api_api.py`, fetcheado en vivo desde GitHub `main` el 2026-07-17). Cada
> endpoint, tipo, constraint y tool MCP de este documento existe literal en ese archivo — nada fue
> inventado. **Sección MCP reescrita el 2026-07-17** tras confirmar que el commit de merge
> `82292ebc1` (auth + pago x402 en los 3 tools MCP in-process) está efectivamente en producción:
> Railway (`similarity-search-api`, deployment `3a9befdc`, `SUCCESS`) se redeployó a las
> `2026-07-16T20:13:09Z`, 22 segundos después del push a `main` — es el deployment vigente, no hubo
> ninguno posterior.

## Qué hace

Servicio HTTP stateless que calcula similitud entre vectores densos combinando cosine similarity y
Normalized Mutual Information (NMI), con el peso de fusión (`alpha`) auto-calibrado por la entropía
del corpus si no se fija manualmente. No requiere base de datos vectorial ni indexado previo — todo
el corpus se manda inline en cada request.

## Base URL

```
https://similarity-search-api-production.up.railway.app
```

(Railway, sin dominio propio configurado todavía.)

## Autenticación

Header `X-API-Key: <key>`. Si el servidor no tiene `SIMILARITY_API_KEY` seteada, cualquier request
devuelve `503 Service Unavailable` ("API authentication not configured on server") — a diferencia del
otro asset de NEXUS, acá la ausencia de key **no** deja pasar requests sin auth.

## Cobro

Las 3 rutas core (`/similarity/search`, `/similarity/calibrate-alpha/v1`, `/similarity/batch-score`)
requieren pago vía **x402** (protocolo de pago por request en USDC, red **Base Sepolia — testnet, no
mainnet**), `$0.01` por llamada, wallet `0x70e9f8057bb50e31b6ee06958bcbbe7de9daa98f`. Un request sin
pago x402 válido responde `402 Payment Required` con el desafío de pago en el header `payment-required`
(base64, ver `logs/similarity_search_api_prod_validation_2026-07-17.log` para un ejemplo real
decodificado). Confirmado en vivo contra producción el 2026-07-17: sin pago → `402`; con un pago
firmado offline pero de wallet sin fondos → también `402` (`invalid_exact_evm_insufficient_balance` —
el facilitator valida balance on-chain durante el `verify`, no solo la firma). Estas mismas 3 rutas
están excluidas del billing de Stripe (para no cobrar dos veces el mismo request).

**Mismo gate aplica ahora a la superficie MCP** — ver más abajo.

## Endpoints

### `POST /similarity/search`
Búsqueda de similitud sobre un corpus inline, usando el score de fusión NMI+cosine calibrado por
entropía.

Request (`SimilaritySearchRequest`):
| campo | tipo | constraints |
|---|---|---|
| `query` | `CorpusVector` | `{id: string(1-256), vector: number[](2-4096)}` |
| `corpus` | `CorpusVector[]` | 1 a 500,000 items |
| `top_k` | integer | default 10, 1–1000 |
| `nmi_bins` | integer | default 10, 3–50 |
| `alpha_override` | number\|null | 0.0–1.0. Si se omite, alpha se auto-calibra por entropía del corpus |

Response (`SimilaritySearchResponse`): `results[]` (`id`, `composite_score`, `cosine_similarity`,
`nmi_score`, `rank`), `calibrated_alpha`, `corpus_entropy`, `query_id`, `corpus_size`, `latency_ms`,
`request_fingerprint`.

```bash
curl -X POST https://similarity-search-api-production.up.railway.app/similarity/search \
  -H "X-API-Key: $SIMILARITY_API_KEY" -H "Content-Type: application/json" \
  -d '{"query": {"id": "q1", "vector": [0.1, 0.9, 0.3]},
       "corpus": [{"id": "c1", "vector": [0.2, 0.8, 0.25]}],
       "top_k": 5}'
```

(más el header/flujo de pago x402 — ver sección Cobro; sin él, `402` antes de llegar a la validación
de la key.)

### `POST /similarity/calibrate-alpha/v1`
Calcula el `alpha` calibrado por entropía para un corpus sin correr la búsqueda completa — útil para
inspeccionar antes de comprometerse a un `/similarity/search` grande.

Request (`AlphaCalibrateRequest`): `corpus: CorpusVector[]` (1–500,000), `nmi_bins: integer` (default
10, 3–50).
Response (`AlphaCalibrationResponse`): `calibrated_alpha`, `corpus_entropy`, `corpus_size`,
`vector_dim`, `latency_ms`.

### `POST /similarity/batch-score`
Scorea hasta 10,000 pares de vectores con un `alpha` fijo (sin overhead de corpus).

Request (`BatchScoreRequest`): `pairs: [number[], number[]][]` (1–10,000 pares, ambos vectores del
par deben tener igual dimensión), `alpha: number` (default 0.5, 0–1), `nmi_bins: integer` (default 10,
3–50).
Response (`BatchScoreResponse`): `scores[]`, `alpha_used`, `pair_count`, `latency_ms`.

### `GET /health`
Liveness probe, sin autenticación. Devuelve `{"status": "ok", "version": "1.0.0"}`.

## Endpoint deprecado — no usar

`POST /similarity/calibrate-alpha` (sin `/v1`) existe en el código pero devuelve siempre
`501 Not Implemented` con el mensaje "Use the /similarity/calibrate-alpha/v1 POST endpoint with a
JSON body." Queda documentado acá solo para que no se confunda con la ruta real.

## MCP — 2 superficies distintas, no confundir

### Superficie 1: `/mcp` in-process — la real, la que corre en producción

Montada en el mismo proceso FastAPI (`app.mount("/", _nexus_mcp_asgi_app)`), mismo dominio Railway
que el REST de arriba — no hay un segundo servicio. Expone 3 tools reales:

- **`nexus_similarity_search_api_rank_items_by_nmi_cosine_fusion`** — equivalente a
  `POST /similarity/search`. Params: `query_vector`, `corpus_vectors`, `top_k` (default 10, 1–1000),
  `alpha_override` (opcional, 0–1), `n_bins` (default 16, 3–50), `api_key` (string, requerido).
- **`nexus_similarity_search_api_estimate_corpus_entropy_profile`** — equivalente a
  `POST /similarity/calibrate-alpha/v1`. Params: `corpus_vectors`, `n_bins` (default 16, 3–50),
  `api_key` (requerido). Devuelve un `corpus_entropy` agregado — **no** un desglose por dimensión.
- **`nexus_similarity_search_api_score_pair_nmi_cosine`** — equivalente a `POST /similarity/batch-score`
  para exactamente 1 par. Params: `vector_a`, `vector_b`, `n_bins` (default 16, 3–50), `alpha`
  (default 0.5, no auto-calibrado), `api_key` (requerido).

Cada tool llama **directo** a la función de lógica de negocio (`search_corpus_by_calibrated_similarity`,
etc.), sin pasar por HTTP/ASGI/ni el middleware x402 de las rutas REST — necesario porque una llamada
in-process contra esas rutas siempre daría `402` (están gateadas por `PaymentMiddlewareASGI`, sin
distinguir caller interno de externo). Para no dejar las 3 operaciones gratis y sin auth por ese
atajo, cada handler aplica **los mismos 2 gates que la REST, pero a mano**:

1. **`api_key`**: parámetro explícito del tool (mismo secreto que `X-API-Key`/`SIMILARITY_API_KEY`),
   validado con `_require_api_key(key=api_key)` como primera línea del handler. Tiene que venir del
   *caller* — pasarse la key del propio servidor a sí mismo no autentica nada.
2. **Pago x402**: decorador `@_nexus_mcp_x402_wrapper` (`x402.mcp.create_payment_wrapper`, integración
   MCP oficial del SDK x402, no reimplementada a mano) que envuelve el handler con el mismo
   `_nexus_x402_server`/`PaymentRequirements`/precio que usan las rutas REST. Verifica el pago (via
   `_meta`) *antes* de correr el handler y liquida (`settle`) *solo* si el handler retorna sin
   excepción — si `_require_api_key()` rechaza, nunca se llega a cobrar.

**Gotcha histórico, ya arreglado, queda documentado**: la versión anterior de estos 3 tools llamaba
directo a la lógica de negocio para esquivar el `402` de la llamada ASGI interna, pero ese mismo
atajo también saltaba `_require_api_key()` (solo se evalúa vía `Security()`/DI de FastAPI, nunca en
una llamada Python directa) *y* el middleware x402 (el mount de FastMCP en `/` nunca estuvo en
`_NEXUS_X402_ROUTES`) — las 3 operaciones quedaron gratis y sin auth por MCP mientras seguían
protegidas por REST. Corregido por `patch_mcp_x402_auth_gate_similarity_search.py`
(commit `6ae57e63d` + merge `82292ebc1`), confirmado en el deployment vigente de Railway (ver nota al
inicio de este documento).

**No validado end-to-end con un pago real liquidado.** Requiere un cliente MCP capaz de streamable
HTTP + firmar un pago x402 "exact" scheme sobre Base Sepolia con una wallet de testnet fondeada —
pendiente compartido con la REST, no específico de esta superficie (ver memoria de proyecto
`x402-funded-wallet-pending`; el código para construir el payment payload offline con el SDK x402
local ya existe, solo falta el fondeo).

### Superficie 2: `mcp_wrapper/` (TS) — local, nunca deployada, no descubrible

Directorio separado en el mismo repo (`mcp_wrapper/`), pensado originalmente como un microservicio
Node.js aparte del core Python. **Nunca corrió como servicio propio**: el `Procfile` de Railway
lanza únicamente `uvicorn core.similarity_search_api_api:app` (la superficie 1 de arriba) — no hay
ningún proceso Node en producción para este asset. Confirmado en vivo (2026-07-16):

- `GET https://registry.npmjs.org/@nexus-mcp/similarity-search-api` → `404 Not Found`. El paquete
  npm (`@nexus-mcp/similarity-search-api`, bin `similarity-search-api-mcp-server`) nunca se publicó
  de verdad — `NPM_TOKEN` no está seteado, la publicación cayó a modo simulado.
- `registry.modelcontextprotocol.io/v0/servers?search=...` → `{"servers":[],"metadata":{"count":0}}`.
  Tampoco está en el Registro Oficial MCP (requiere un `remote_url` de un deploy real que nunca
  existió).

Única forma de usarlo hoy: clonar `nexus-mcp-infra/similarity-search-api-sdk`, `cd mcp_wrapper`,
`npm install`, `cp .env.example .env` y **corregir** `NEXUS_CORE_BASE_URL` a mano — el `.env.example`
trae `https://similarity-search-api.railway.app`, que no es el dominio real
(`similarity-search-api-production.up.railway.app`). Luego `npm run dev` (stdio) o
`TRANSPORT=http npm run start:http`.

1 tool: **`nexus_similarity_search_api_rank_embeddings_by_nmi_cosine`** — llama por HTTP real
(`coreClient.ts`) a `POST /similarity/search`. Params: `query_vector`, `corpus_vectors`, `corpus_ids`
(array de IDs, separado en vez de venir embebido en cada `CorpusVector` como en la REST/superficie 1),
`top_k` (default 10, 1–1000), `nmi_bins` (default 10, 3–50), `alpha_override` (opcional, 0–1).

**Gotcha real, no arreglado**: `coreClient.ts` solo reenvía el header `x-api-key` (usa
`NEXUS_CORE_API_KEY`) — no tiene ninguna noción de x402. Desde que `/similarity/search` quedó
protegida por `PaymentMiddlewareASGI` (que corre a nivel de middleware ASGI, antes de que la request
llegue a la ruta/dependencia de FastAPI), **cualquier llamada de este wrapper devuelve `402 Payment
Required`, sin importar si la API key es válida**. Este wrapper no se tocó desde el commit `e7427b357`
(el que agregó x402) salvo por el fix de grounding de tools (`10ee34375`, que solo corrigió cuál tool
sobrevive y sus parámetros, no auth/pago) — está roto contra producción hoy tal cual está, y necesita
el mismo flujo de pago x402 que ya tiene la superficie 1 para volver a funcionar.

## Errores

`401` — API key inválida o ausente (REST y MCP superficie 1, vía `_require_api_key`). `422` —
validación de request (ej. dimensión de `query` no coincide con `corpus`, vector con NaN/Inf, o norma
cero). `402` — falta pago x402 válido en una ruta protegida (REST, y ahora también MCP superficie 1 y
2 — ver arriba). `503` — servidor sin `SIMILARITY_API_KEY` configurada.
