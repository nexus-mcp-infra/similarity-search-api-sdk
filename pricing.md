# Pricing Model — Similarity Search API (NMI-Cosine Hybrid)

---

## Principio de diseño

El activo es **stateless por diseño** (sin persistencia de vectores), lo que elimina el pricing por almacenamiento y hace que **la unidad de valor sea la operación de cómputo estadístico**, no el asiento ni el mes. El costo marginal real está en la discretización Freedman-Diaconis + estimación H(X,Y) + bootstrap de p-values — no en infraestructura de base de datos. El modelo de pricing debe reflejar eso directamente.

---

## Estructura de tiers

### Free Tier — `similarity:free`

| Parámetro | Valor |
|---|---|
| Llamadas / mes | 500 |
| Dimensionalidad máxima por vector | 512 dims |
| Vectores por llamada (`query` + `candidates`) | máx. 20 vectores totales |
| Bootstrap iterations (p-value CI) | 199 iteraciones |
| Intervalo de confianza devuelto | 90% |
| Rate limit | 10 req / min |
| SLA | Best-effort, sin garantía de latencia |
| Soporte | Documentación pública |

**Restricción técnica clave:** con 199 iteraciones bootstrap el error estándar del p-value estimado es ~0.032 (SE = sqrt(p(1-p)/B) para p=0.05, B=199). Suficiente para exploración, insuficiente para decisiones de producción. Esta degradación es técnicamente honesta, no artificial — está documentada en el response body como `bootstrap_precision: "exploratory"`.

**Propósito:** eliminar la fricción de evaluación. Un developer que resuelve "necesito similitud en 50 líneas sin levantar una DB" evalúa el activo en una tarde. El límite de 20 vectores fuerza el upgrade antes de cualquier uso real en pipelines.

---

### Pro Tier — `similarity:pro` (per-call)

**Sin suscripción fija. Sin seats. Sin storage charges.**

| Parámetro | Valor |
|---|---|
| Precio base por llamada | $0.004 |
| Dimensionalidad máxima | 3072 dims (OpenAI text-embedding-3-large) |
| Vectores por llamada | máx. 500 vectores totales |
| Bootstrap iterations | 999 iteraciones |
| Intervalo de confianza devuelto | 95% |
| Rate limit | 300 req / min |
| SLA latencia (p95) | < 800ms para 100 vectores × 1536 dims |
| Soporte | Email, respuesta en 48h |

**Escala de volumen (créditos prepago):**

| Créditos prepago | Precio por llamada | Descuento efectivo |
|---|---|---|
| 0 – 9,999 llamadas | $0.0040 | — |
| 10,000 – 49,999 | $0.0034 | 15% |
| 50,000 – 199,999 | $0.0028 | 30% |
| 200,000+ | $0.0022 | 45% |

**Lógica del precio base $0.0040:**

El costo marginal de cómputo por llamada en 100 vectores × 1536 dims incluye:

- Discretización Freedman-Diaconis por dimensión: O(n · d · log n) donde n = vectores, d = dimensiones
- Estimación H(X,Y) discreta: O(n² · B_avg) donde B_avg es el bin count promedio resultante del ajuste adaptativo (~18 bins en 1536 dims según distribución empírica de embeddings OpenAI)
- Bootstrap con 999 iteraciones: el costo dominante — O(999 · n²) comparaciones de rangos permutados
- En instancia compute-optimized (c6i.xlarge, $0.204/hr), 100 vectores × 1536 dims procesa en ~320ms -> ~630 llamadas/hr -> costo infra = $0.000323/llamada
- Margen 12x sobre costo infra refleja: (a) el diferencial del bin-count flywheel, (b) el moat de la implementación correcta de Freedman-Diaconis vs bins fijos, (c) el valor de la primitiva de decisión (p-value) vs solo ranking

**Modelo mental para el developer:** reemplaza "¿es este score de 0.87 bueno?" por "p-value = 0.031, la similitud es estadísticamente significativa al 5%". Ese salto epistémico vale más que $0.004.

---

### Enterprise Tier — `similarity:enterprise`

**Contrato anual. Precio negociado según volumen comprometido.**

| Parámetro | Valor |
|---|---|
| Precio de entrada | Desde $2,400/año (~600k llamadas a tarifa base) |
| Dimensionalidad máxima | Sin límite (hasta 8192 dims, extensible) |
| Vectores por llamada | máx. 2,000 vectores totales |
| Bootstrap iterations | Configurable: 999 – 9,999 |
| Intervalo de confianza | Configurable: 90%, 95%, 99% |
| Rate limit | Dedicado, negociado (default: 2,000 req/min) |
| SLA latencia (p99) | < 1,200ms para 500 vectores × 3072 dims, contractual |
| Uptime SLA | 99.9% mensual |
| Soporte | Slack dedicado + Technical Account Manager |
| Facturación | Anual prepago con reconciliación trimestral de overage |

**Capacidades exclusivas de Enterprise:**

| Feature | Descripción técnica |
|---|---|
| `bin_calibration_profile` | Acceso al perfil de bin counts óptimos acumulado por el flywheel PostgreSQL para el rango de dimensionalidad del cliente — reduce la varianza del estimador H(X,Y) en embeddings de dominio específico |
| `custom_discretization_method` | Opción de sustituir Freedman-Diaconis por Sturges o Scott rule para compatibilidad con pipelines legacy del cliente |
| `batch_async_endpoint` | POST /v1/similarity/batch — hasta 10,000 pares en un job asíncrono con webhook de completion; Pro solo tiene el endpoint síncrono |
| `audit_log_export` | JSONL firmado de cada llamada con inputs hash, outputs, y metadatos estadísticos — requerido por equipos de compliance en fintech/legaltech |
| `private_deployment_option` | Imagen Docker certificada para VPC del cliente; el flywheel de bins no se comparte con el pool global pero sí se beneficia del seed inicial de distribuciones reales |

---

## Comparativa de tiers

| | Free | Pro | Enterprise |
|---|---|---|---|
| Precio | $0 | $0.004/call | Desde $2,400/año |
| Llamadas/mes | 500 | Ilimitadas (prepago) | Ilimitadas (contractual) |
| Dims máximas | 512 | 3,072 | 8,192+ |
| Vectores/call | 20 | 500 | 2,000 |
| Bootstrap iterations | 199 | 999 | 999–9,999 |
| CI devuelto | 90% | 95% | Configurable |
| p-value en response | Sí | Sí | Sí |
| Bin calibration profile | Global seed | Global seed | Domain-specific |
| Rate limit | 10 req/min | 300 req/min | Dedicado |
| SLA | Best-effort | p95 < 800ms | p99 contractual |
| Batch async | No | No | Sí |
| Audit log export | No | No | Sí |
| Soporte | Docs | Email 48h | Slack + TAM |

---

## Mecanismo de overage (Pro)

Créditos prepago nunca expiran. Si el saldo llega a cero, las llamadas continúan a la tarifa más alta ($0.0040/call) y se facturan a fin de mes con tarjeta registrada. Sin interrupciones de servicio por saldo agotado — el developer que tiene un script en producción no puede permitirse un corte.

Umbral de alerta configurable vía webhook: cuando el saldo cae por debajo del valor definido por el usuario, se envía una notificación antes del corte de facturación.

---

## Rationale anti-commoditization

El pricing no compite con Pinecone, Weaviate, ni Qdrant — esos productos cobran por almacenamiento y queries sobre índices persistidos. Este activo cobra por **el cómputo estadístico de significancia**, que no es substituible por ningún vector DB existente sin implementar el estimador de entropía conjunta con Freedman-Diaconis adaptativo.

El riesgo de commoditización más real es que un developer implemente NMI con bins fijos (k=10) usando `sklearn.metrics.normalized_mutual_info_score` sobre histogramas hardcoded. Ese estimador produce NMI artificialmente alto en dimensiones de baja varianza (fenómeno documentado en Paninski 2003: el estimador plug-in sobreestima MI cuando el número de bins es fijo y la distribución marginal es casi determinista). El flywheel de distribuciones reales que ajusta B por rango de dimensionalidad es el foso técnico que hace que el p-value de este activo sea válido y el del competidor casero sea ruido estadístico — eso es lo que el pricing debe comunicar, y lo que justifica $0.004 sobre $0.