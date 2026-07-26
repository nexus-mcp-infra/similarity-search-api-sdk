# Calibrated Similarity Search API — AGENTS.md

> Generado el 2026-07-25 verificando en vivo contra el deployment real
> (`https://similarity-search-api-production.up.railway.app`) con `curl` — no a partir de
> `output/similarity_search_api/core/similarity_search_api_api.py`, que es un snapshot desactualizado
> (generado 2026-07-18, **sin** `_require_api_key`, sin x402, sin rate limiting: expone
> `/v1/similarity/pairwise-score` y otras rutas que no existen en producción). El código real deployado
> se reconstruyó cruzando `output/similarity_search_api/openapi.json` (coincide byte a byte con el
> `/openapi.json` en vivo, confirmado abajo), `github_live_content.py` (snapshot del código fuente real
> tomado el 2026-07-17, ver `CLAUDE.md §4`) y los 6 patches de la raíz del repo que tocan este asset
> (`patch_x402_similarity_search.py`, `patch_mcp_tool_grounding_similarity_search.py`,
> `patch_mcp_tool_grounding_similarity_search_inprocess.py`,
> `patch_mcp_x402_auth_gate_similarity_search.py`,
> `patch_stripe_mcp_billing_exclusion_similarity_search.py`, `patch_rate_limit_similarity_search.py`).
> **Nota de trazabilidad de commits**: solo 2 de esos 6 patches están comiteados en este repo —
> `patch_x402_similarity_search.py` (commit `47c19b3`, 2026-07-14) y `patch_rate_limit_similarity_search.py`
> (commit `6d65a56`). Los otros 4 (`mcp_x402_auth_gate`, ambos `mcp_tool_grounding`,
> `stripe_mcp_billing_exclusion`) aparecen como `??` (untracked) en `git status` — se corrieron a mano
> contra el repo propio del asset (no contra este orquestador) y nunca se comitearon acá. No se inventa
> hash para esos cuatro; su efecto se verificó igual, en vivo, contra producción (ver cada sección).

## Qué hace

Scorer de similitud híbrido stateless: fusiona **NMI normalizada** (Normalized Mutual Information, para
features/vectores categóricos vía discretización + `normalized_mutual_info_score`) con **cosine
similarity** (para vectores continuos), en un score compuesto ponderado por `alpha`. `alpha` se
auto-calibra a partir de la entropía marginal del corpus (`alpha = H(corpus) / (H(corpus) + log2(n_bins))`)
salvo que se pase `alpha_override`/`alpha` explícito. Sin índice vectorial, sin estado entre llamadas —
cada request trae su propio corpus inline. Pensado para ranking/scoring de corpus chicos-a-medianos
(hasta 500k ítems) sin levantar una vector DB.

## Base URL

```
https://similarity-search-api-production.up.railway.app
```

(Railway, sin dominio propio. `GET /` → `404 Not Found` — el mount de FastMCP en `"/"` no responde nada
en la raíz literal; no confundir con "servicio caído".)

## Autenticación

Header `X-API-Key: <key>`. Env var real: `SIMILARITY_API_KEY` — **una sola key** (no una lista;
`BUYWHERE_API_KEYS` del otro asset sí es lista — ver docstring de `patch_rate_limit_similarity_search.py`).

**A diferencia del otro asset de NEXUS** (`useful-data-source-for-agents`, que hace bypass silencioso si
`BUYWHERE_API_KEYS` está vacía — ver su `AGENTS.md`), acá `_require_api_key()` responde `503 Service
Unavailable` ("API authentication not configured on server.") si `SIMILARITY_API_KEY` no está seteada en
el servidor, **antes** de siquiera mirar la key que mandó el caller (confirmado en el código real,
`github_live_content.py:31-42`). Es el comportamiento "seguro" que `CLAUDE.md §Autenticación` del otro
asset usa como contraste. Confirmado indirectamente en vivo: `POST /similarity/calibrate-alpha` con una
key inválida devolvió `401 {"detail":"Invalid or missing API key."}`, no `503` — es decir,
`SIMILARITY_API_KEY` **está** configurada en Railway hoy.

**Gotcha real sobre orden de gates**: en las 3 rutas protegidas también por x402 (ver Cobro), el
middleware de pago corre **antes** que la dependencia de FastAPI que valida la API key — confirmado en
vivo: `POST /similarity/search` con una key claramente inválida (`X-API-Key: totally-wrong-key`), sin
pago, devuelve `402`, no `401`. Incluso con body directamente inválido (no-JSON) sigue devolviendo `402`,
no `422` — el middleware ASGI de x402 se ejecuta antes de que FastAPI llegue a parsear el body o resolver
dependencias. Un caller no puede ni siquiera saber si su API key es válida en esas 3 rutas sin pagar
primero.

## Cobro

Las 3 rutas core (`POST /similarity/search`, `POST /similarity/calibrate-alpha/v1`,
`POST /similarity/batch-score`) requieren pago **x402** (USDC, red **Base Sepolia — testnet**,
`eip155:84532`), `$0.01`/llamada, misma wallet que `useful-data-source-for-agents`
(`0x70e9f8057bb50e31b6ee06958bcbbe7de9daa98f`), facilitator `https://x402.org/facilitator`. Aplicado por
`patch_x402_similarity_search.py` (commit `47c19b3`, 2026-07-14).

Confirmado en vivo — `curl -D-` a `POST /similarity/search` sin pago devuelve:

```
HTTP/1.1 402 Payment Required
payment-required: <base64 de {"x402Version":2,"error":"Payment Required","resource":{...},
  "accepts":[{"scheme":"exact","network":"eip155:84532",
  "asset":"0x036CbD53842c5426634e7929541eC2318f3dCF7e","amount":"10000",
  "payTo":"0x70e9f8057bb50e31b6ee06958bcbbe7de9daa98f","maxTimeoutSeconds":300,
  "extra":{"name":"USDC","version":"2"}}]}>
Content-Length: 2   (body: "{}")
```

`amount":"10000"` = 0.01 USDC en unidades de 6 decimales — coincide exacto con el precio configurado en
el patch.

Excluidas del billing de Stripe (`_NEXUS_BILLING_EXCLUDED_PATHS`) para no cobrar dos veces sobre el mismo
request. **`/mcp` también está en esa exclusión**, agregado por
`patch_stripe_mcp_billing_exclusion_similarity_search.py` (untracked, sin commit) — mismo bug de fondo que
documenta `CLAUDE.md §3` para este asset: `_nexus_usage_middleware` compara contra `request.url.path`, y
el sub-app FastMCP montado en `"/"` es Starlette puro (nunca setea `scope["route"]`); una request real a
`/mcp` tiene `request.url.path == "/mcp"`, que **no** matcheaba el `"/"` ya presente en el set (ese `"/"`
solo cubre la URL raíz literal). Antes del fix, cualquier tráfico de protocolo MCP con status `<400`
(`initialize`, `tools/list` — ninguno de los dos pasa por gate de auth/pago) disparaba un
`stripe.billing.MeterEvent.create()` real. Confirmado indirectamente vía Railway
(`STRIPE_CUSTOMER_ID`/`STRIPE_EVENT_NAME`/`STRIPE_SECRET_KEY` reales, modo test) — no hay forma de
confirmarlo 100% en vivo desde afuera sin acceso al dashboard de Stripe, mismo límite que documentó el
patch original.

**Ruta legacy sin cobertura de pago, pero tampoco explotable**: `POST /similarity/calibrate-alpha` (sin
`/v1`) sigue presente tanto en el OpenAPI en vivo como en el código real, y **no** está en
`_NEXUS_X402_ROUTES` (esa tabla solo tiene la entrada literal `"POST /similarity/calibrate-alpha/v1"`, que
no matchea el path sin sufijo). Requiere `X-API-Key` igual que las demás (`401` confirmado en vivo con key
inválida), pero el handler (`github_live_content.py:281-289`) no calcula nada — devuelve incondicionalmente
`501 Not Implemented`, `"Use the /similarity/calibrate-alpha/v1 POST endpoint with a JSON body."` No pude
confirmar el `501` en vivo con una key válida (no hay `SIMILARITY_API_KEY` real disponible en este
entorno), pero el código no deja ambigüedad: es un stub deprecado, no un bypass real de pago hoy. Si
alguna vez se "completa" en vez de eliminarse, sí sería un bypass real — vale la pena no reimplementar
esa ruta sin agregarla también a `_NEXUS_X402_ROUTES`.

## Rate limiting (por caller, confirmado en vivo)

`patch_rate_limit_similarity_search.py` (commit `6d65a56`, 2026-07-17) agrega un limitador en memoria,
por proceso, sliding-window: **60 requests / 60s por caller** (`NEXUS_RATE_LIMIT_PER_MINUTE` /
`NEXUS_RATE_LIMIT_WINDOW_SECONDS`, configurable en Railway sin redeploy). Identidad resuelta en orden:
wallet pagadora x402 (`X-PAYMENT`) > hash de `X-API-Key` > IP. Corre como middleware ASGI, por lo tanto
cubre también `/mcp` (Starlette puro, no pasaría por un check dentro de cada ruta FastAPI).

**Confirmado en vivo**: 70 requests consecutivas a `POST /similarity/calibrate-alpha` con una API key
inválida (ruta gratis, sin x402, así que cada llamada solo cuesta un `401` — no hay costo real en probar
el límite):

| Requests 1–60 | Requests 61–70 |
|---|---|
| `401 Invalid or missing API key.` (60 de 60) | `429 Too Many Requests` (10 de 10) |

Respuesta del `429`:
```json
{"error":"rate_limited","detail":"Too many requests. Retry after 33s."}
```
con header `Retry-After: 33`. Umbral exacto en 60, coincide con el default documentado.

## Endpoints

### `POST /similarity/search`
Búsqueda/ranking real: score compuesto por ítem del corpus contra un vector query, top-k descendente.

**Usar cuando**: necesitás rankear un corpus (hasta 500k ítems) contra un vector de query con el score
híbrido NMI+cosine, sin mantener índice.
**No usar para**: un solo par aislado sin corpus (usar `/similarity/batch-score`), o solo necesitar el
`alpha` calibrado sin correr el ranking completo (usar `/similarity/calibrate-alpha/v1`).

Body (`SimilaritySearchRequest`): `query` (`CorpusVector{id, vector[2–4096]}`), `corpus`
(1–500000 `CorpusVector`, todas las dimensiones iguales), `top_k` (1–1000, default 10), `nmi_bins`
(3–50, default 10), `alpha_override` (0.0–1.0, opcional).

Response (`SimilaritySearchResponse`): `results[]` (`id`, `composite_score`, `cosine_similarity`,
`nmi_score`, `rank`), `calibrated_alpha`, `corpus_entropy`, `query_id`, `corpus_size`, `latency_ms`,
`request_fingerprint`.

Auth: `X-API-Key` + pago x402. Confirmado en vivo: `402` sin pago (con o sin key, con o sin body válido).

### `POST /similarity/calibrate-alpha/v1`
Calcula el `alpha` auto-calibrado para un corpus sin correr la búsqueda completa.

**Usar cuando**: querés inspeccionar el costo/composición del corpus (entropía, dimensión) antes de
comprometerte a un `/similarity/search` grande.
**No usar para**: obtener resultados rankeados (acá no hay ranking, solo el escalar `alpha`).

Body (`AlphaCalibrateRequest`): `corpus` (1–500000 `CorpusVector`), `nmi_bins` (3–50, default 10).

Response (`AlphaCalibrationResponse`): `calibrated_alpha`, `corpus_entropy`, `corpus_size`, `vector_dim`,
`latency_ms`.

Auth: `X-API-Key` + pago x402. Confirmado en vivo: `402` sin pago.

### `POST /similarity/batch-score`
Scorea hasta 10.000 pares `(vector_a, vector_b)` independientes con un `alpha` **fijo** (no calibrado por
par ni por corpus).

**Usar cuando**: tenés muchos pares ya definidos y solo necesitás el score de cada uno, sin armar un
corpus ni pagar el overhead de ranking.
**No usar para**: ranking contra un corpus completo con `alpha` auto-calibrado (usar `/similarity/search`);
obtener el `alpha` calibrado en sí (usar `/similarity/calibrate-alpha/v1` — acá `alpha` default es 0.5,
aplicado tal cual, **no** se deriva de entropía).

Body (`BatchScoreRequest`): `pairs` (1–10000 tuplas `[vector_a, vector_b]`, misma dimensión por par),
`alpha` (0.0–1.0, default 0.5), `nmi_bins` (3–50, default 10).

Response (`BatchScoreResponse`): `scores[]`, `alpha_used`, `pair_count`, `latency_ms`.

Auth: `X-API-Key` + pago x402. Confirmado en vivo: `402` sin pago.

### `POST /similarity/calibrate-alpha` (legacy, sin `/v1` — deprecado, no usar)
**No usar para nada** — stub deprecado. Requiere `X-API-Key` válida (`401` si falta/es inválida,
confirmado en vivo) pero el handler siempre devuelve `501 Not Implemented` señalando usar
`/similarity/calibrate-alpha/v1`. Ver gotcha en "Cobro" — no está cubierta por x402 (fuera de
`_NEXUS_X402_ROUTES`), pero tampoco ejecuta lógica real, así que hoy no es una forma gratis de obtener
cómputo pago.

### `GET /health`
**Usar para**: liveness probes, monitoreo de uptime.
**No usar para**: nada relacionado a similitud/scoring.

Response: `{"status": "ok", "version": "1.0.0"}`. Sin autenticación, sin pago. Confirmado en vivo (`200`).

## MCP — servidor in-process en `/mcp`, 3 tools reales, auth+pago confirmados en vivo

Servidor MCP embebido (`FastMCP`, `stateless_http=True`) montado en el mismo proceso Railway vía
`app.mount("/", ...)` — no hay segundo servicio. `stateless_http=True` confirmado en vivo: la respuesta a
`initialize` no trae header `mcp-session-id`, y `tools/list` funciona igual en una request HTTP nueva sin
mandar ninguna sesión previa. `serverInfo.version` reportado por `initialize` es `"1.28.1"` — es la
versión del SDK `mcp`/`FastMCP` instalado, **no** la versión del asset (que es `"1.0.0"` según
`info.version` del propio OpenAPI); no confundir ambos números.

`tools/list` en vivo devuelve exactamente 3 tools:

| Tool MCP | Lógica real que llama (directo, sin HTTP) | Auth/pago |
|---|---|---|
| `nexus_similarity_search_api_rank_items_by_nmi_cosine_fusion` | `search_corpus_by_calibrated_similarity()` | `api_key` (param explícito) + x402 |
| `nexus_similarity_search_api_estimate_corpus_entropy_profile` | `inspect_corpus_entropy_and_alpha()` | `api_key` + x402 |
| `nexus_similarity_search_api_score_pair_nmi_cosine` | `score_vector_pairs_with_fixed_alpha()` | `api_key` + x402 |

> **Historia de grounding — 2 rounds de fixes, distintos del `mcp_wrapper/` TS**:
>
> **Round 1** (`patch_mcp_tool_grounding_similarity_search_inprocess.py`, untracked, sin commit): el
> servidor MCP in-process originalmente tenía 5 tools generadas contra rutas ficticias
> (`/v1/similarity/rank-nmi-cosine`, `/v1/similarity/corpus-entropy-profile`, `/v1/similarity/score-pair`,
> `/v1/similarity/outliers-nmi-deficit`, `/v1/similarity/calibrate-alpha`) que nunca existieron en el
> core — mismo root cause que `useful-data-source-for-agents` (fases `_phase_tool_spec()` y la síntesis
> del `api.py` corren como llamadas a Claude independientes, sin memoria compartida). 2 tools
> (`find_outlier_vectors_by_nmi_deficit`, `calibrate_alpha_from_query_entropy`) no tenían equivalente real
> y se eliminaron. Los 3 sobrevivientes originalmente llamaban `_nexus_mcp_call_core()` (ASGI in-process
> contra las rutas reales) — pero para cuando se corrió este patch, esas mismas 3 rutas ya estaban
> envueltas por el middleware x402 (agregado después de que el servidor MCP se generó), así que la
> llamada interna devolvía `402` incluso con ruta y API key correctas. Fix: se eliminó
> `_nexus_mcp_call_core()` y los 3 tools pasaron a llamar **directo** a las funciones de negocio en
> Python (`search_corpus_by_calibrated_similarity`, etc.), saltándose ASGI/HTTP/x402 por completo — mismo
> criterio que ya usa la exclusión de billing de Stripe para tratar esas 3 rutas como internas.
>
> **Bug de fondo introducido por el Round 1, cerrado por el Round 2**
> (`patch_mcp_x402_auth_gate_similarity_search.py`, untracked, sin commit): llamar directo a la lógica de
> negocio se saltea **tanto** `_require_api_key()` (solo se evalúa cuando FastAPI resuelve la dependencia
> vía su router — una llamada Python directa nunca pasa por ahí) **como** el middleware x402 (filtrado
> por path contra 3 rutas REST literales; el mount de FastMCP en `"/"` nunca estuvo en esa lista).
> Resultado, confirmado como el estado real durante una ventana: las 3 operaciones que en REST exigen key
> + pago quedaron gratis y sin auth vía `/mcp`. Fix: se agregó un parámetro `api_key` explícito a cada
> tool + `_require_api_key(key=api_key)` como primera línea del handler, y se decoró cada tool con
> `x402.mcp.create_payment_wrapper()` (integración MCP de primera clase del propio paquete `x402`),
> reusando la misma instancia `_nexus_x402_server`/precio/red que las rutas REST.

**Confirmado en vivo, 2026-07-25** (a diferencia de `useful-data-source-for-agents`, donde este mismo
tipo de fix **no** está aplicado — ver su `AGENTS.md`, sección MCP):

```
tools/call score_pair_nmi_cosine, api_key="fake-key-test", sin X-PAYMENT
→ isError: true, content: {"x402Version":2, "error":"Payment Required",
   "accepts":[{"scheme":"exact","network":"eip155:84532", "amount":"10000",
   "payTo":"0x70e9f...aa98f", ...}], "resource":{"url":"mcp://tool/score_pair_nmi_cosine", ...}}

tools/call score_pair_nmi_cosine, SIN api_key
→ isError: true, "Field required [type=missing] ... api_key"  (falla en validación de schema MCP,
   antes de llegar al wrapper de pago)
```

**Orden de gates confirmado**: con un `api_key` presente pero falso, la respuesta es el challenge de pago
x402 (`Payment Required`), **no** un rechazo de auth — el decorator `@_nexus_mcp_x402_wrapper` envuelve la
función más cerca de la definición que `@_nexus_mcp.tool(...)`, así que su chequeo corre antes de que el
cuerpo del handler llegue a ejecutar `_require_api_key(key=api_key)` (que es la primera línea *dentro* del
handler). Mismo orden que en REST: pago antes que key. Los 3 tools están correctamente protegidos hoy, con
ambos gates activos y verificables en vivo.

Distinto de todo lo anterior: **`mcp_wrapper/`** (paquete TypeScript separado, no corre en Railway, se
instala local apuntando a `NEXUS_CORE_BASE_URL` — en `output/similarity_search_api/mcp_wrapper/` solo
queda `manifest.json`, sin `src/`, igual que en `useful-data-source-for-agents`). Ese wrapper tuvo su
propio bug de grounding, corregido por `patch_mcp_tool_grounding_similarity_search.py` (untracked, sin
commit): de 5 tools originales (`rank_embeddings_by_nmi_cosine`, `estimate_pairwise_nmi_matrix`,
`score_candidate_pair_significance`, `detect_embedding_dimension_redundancy`,
`calibrate_nmi_cosine_weight_for_corpus`) solo `rank_embeddings_by_nmi_cosine` tenía equivalente real
(remapeado a `POST /similarity/search`, parámetros corregidos); los otros 4 describían features nunca
implementadas (matriz NxN, detección de redundancia de dimensiones, calibración por triplets con
AUC-ROC) y se eliminaron. Además, `coreClient.ts` mandaba `Authorization: Bearer <key>` — la API real
espera `X-API-Key`; corregido en el mismo patch (sin esto, incluso con la ruta correcta la llamada hubiera
fallado con `401`, no `404`).

## Errores

`401` — `X-API-Key` inválida o ausente (confirmado en vivo en la ruta legacy
`/similarity/calibrate-alpha`; en las 3 rutas pagas nunca se llega a ver porque x402 corta antes con
`402`, ver gotcha de orden de gates). `402` — falta pago x402 válido, en las 3 rutas REST protegidas y en
los 3 tools MCP protegidos; corre **antes** que la validación de auth/body (confirmado con key inválida,
body inválido, y `api_key` MCP falso, siempre `402`/`Payment Required`). `404` — ruta inexistente,
incluida la raíz literal `GET /` (texto plano `"Not Found"`, confirmado en vivo). `422` — validación de
Pydantic (dimensión de vectores inconsistente, corpus vacío, NaN/Inf en un vector, etc. — ver
`field_validator`s en `github_live_content.py`); no se pudo disparar en vivo sin pasar primero el gate de
pago, confirmado solo por lectura de código real. `429` — rate limit por caller excedido (60 req/60s
default, confirmado en vivo con 70 requests consecutivas: 60× `401`, luego 10× `429` con
`Retry-After`). `501` — ruta legacy `/similarity/calibrate-alpha` (sin `/v1`), siempre, una vez pasado el
gate de auth — confirmado por código real, no se pudo probar en vivo (sin `SIMILARITY_API_KEY` real
disponible en este entorno). `503` — `SIMILARITY_API_KEY` no configurada en el servidor (no es el estado
actual: confirmado que la key SÍ está seteada, porque una key inválida da `401`, no `503`).
