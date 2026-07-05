# Modelo de Pricing — Similarity Search API (NMI-weighted Cosine)

---

## Lógica de segmentación

El costo real de la primitiva escala con dos variables: **n** (items en la colección) y **d** (dimensiones por item). El cómputo es O(n·d·log d) por llamada — el pricing debe reflejar esa curva, no un precio plano que subsidia queries grandes y penaliza queries pequeñas.

La segmentación no es por "features" arbitrarias: es por el umbral donde el cómputo NMI deja de ser trivial y empieza a consumir CPU real.

---

## Tres niveles

### Free Tier — Exploración sin fricción

**Precio:** $0

| Límite | Valor |
|---|---|
| Llamadas por mes | 500 |
| Items por colección (n) | max 100 |
| Dimensiones por item (d) | max 50 |
| Rate limit | 5 req/min |
| SLA | best-effort, sin garantía de latencia |
| Autenticación | API key pública (sin tarjeta) |

**Restricción técnica intencionada:** n=100, d=50 cubre el caso de uso de prototipado (comparar catálogos pequeños, validar que NMI mejora cosine puro en tu dataset) pero no producción. El techo es el argumento de venta hacia Pro, no un castigo.

**Lo que NO incluye:** soporte, SLA, colecciones mixtas de alta cardinalidad categórica (donde NMI tiene mayor ventaja — eso es deliberado).

---

### Pro — Por operación, sin suscripción fija

**Precio base:** $0.004 por llamada estándar

La unidad de cobro es la **operación de búsqueda completa** (un request HTTP con su colección query y su colección target). No hay cobro por almacenamiento porque no hay almacenamiento.

#### Tabla de precios por volumen de compute

El multiplicador refleja el costo real del binning Freedman-Diaconis y la estimación de entropía adaptativa sobre colecciones grandes:

| Tamaño de operación | Precio por llamada |
|---|---|
| n ≤ 500, d ≤ 100 | $0.004 |
| n ≤ 2,000, d ≤ 200 | $0.012 |
| n ≤ 10,000, d ≤ 500 | $0.045 |
| n > 10,000 o d > 500 | cotización automática inline (ver Enterprise) |

**Multiplicador de dimensionalidad:** para d > 200, el precio base se ajusta por factor 1 + 0.15·log2(d/200) — refleja el costo de binning en alta dimensión sin ser punitivo.

#### Condiciones Pro

- Rate limit: 120 req/min, burst hasta 300 req por ventana de 10s
- Latencia p95 garantizada: 800ms para n=500, d=100 (bajo SLA contractual ligero)
- Acceso a parámetros avanzados: `nmi_bins_override`, `cosine_weight_floor`, `feature_types` explícito
- Facturación mensual con crédito prepago disponible ($50 mínimo)
- Sin compromiso de volumen mínimo — el primer mes puede ser $0.004

**Modelo de crédito prepago (opcional):**

| Crédito cargado | Descuento efectivo |
|---|---|
| $50 | 0% (precio base) |
| $200 | 8% |
| $500 | 15% |
| $2,000 | 22% |

---

### Enterprise — Compute garantizado + acuerdo de volumen

**Precio:** contrato anual con precio por operación negociado, mínimo $1,500/mes garantizado.

Enterprise no es "más features" — es un acuerdo sobre **throughput garantizado y SLA duro** para colecciones de alta cardinalidad donde el cómputo NMI es intensivo.

#### Lo que cambia respecto a Pro

| Dimensión | Pro | Enterprise |
|---|---|---|
| n máximo por llamada | 10,000 | ilimitado (acordado en contrato) |
| d máximo por llamada | 500 | ilimitado |
| Latencia p95 garantizada | 800ms (n=500) | acordada por tier de compute |
| Rate limit | 120 req/min | throughput dedicado acordado |
| SLA uptime | 99.5% best-effort | 99.9% con penalización contractual |
| Soporte | async (48h) | canal dedicado, respuesta 4h |
| Auditoría de pesos NMI | no | sí — endpoint `/explain` incluido |
| Datos de entrenamiento / fine-tuning del binning | no | posible bajo NDA técnico |

#### Modelo de precios Enterprise

El precio por operación baja con el volumen comprometido:

| Volumen mensual comprometido | Precio por operación |
|---|---|
| 50,000 ops/mes | $0.0032 |
| 200,000 ops/mes | $0.0024 |
| 1,000,000 ops/mes | $0.0016 |
| > 1,000,000 ops/mes | negociación directa |

El precio nunca sube por dimensionalidad alta en Enterprise — ese costo está absorbido en el mínimo mensual. Es el argumento para migrar desde Pro cuando d > 300 se vuelve frecuente.

---

## Justificación matemática del pricing

El costo computacional por llamada es O(n·d·log d). Para los tres puntos de precio de referencia:

- **n=500, d=100:** ~500·100·6.6 = 330,000 operaciones elementales. Precio: $0.004. Margen bruto estimado >80% en hardware commodity.
- **n=2,000, d=200:** ~2,000·200·7.6 = 3,040,000 ops. Precio: $0.012. Factor de escala ~9.2x compute, ~3x precio — el margen comprime intencionalmente para competir con post-processing manual.
- **n=10,000, d=500:** ~10,000·500·8.9 = 44,500,000 ops. Precio: $0.045. Factor ~134x compute, ~11x precio — aquí el valor de no mantener vector DB propio absorbe la diferencia.

El ratio **precio / compute** decae deliberadamente con el tamaño: el valor entregado (evitar un vector DB + schema design + NMI manual) crece superlinealmente con n y d, por lo que el precio puede crecer sublinealmente y seguir siendo percibido como barato.

---

## Comparativa de costo vs. alternativas

| Escenario | Esta API (Pro) | Pinecone serverless + post-NMI manual | Weaviate + custom module |
|---|---|---|---|
| 500 productos, búsqueda única | $0.004 | ~$0.08 (upsert + query + compute NMI manual) | $0 compute + ~2h setup inicial |
| 10,000 productos, 1,000 búsquedas/día | $45/día | ~$180/día (storage + queries + CPU NMI) | infra fija ~$200/mes + mantenimiento |
| Setup time to first result | < 5 min | 2-4 horas | 1-2 días |

El argumento de precio no es "somos baratos" — es **costo total de operación cero para infraestructura** más la métrica NMI que ninguno de esos proveedores expone nativamente.

---

## Condiciones transversales

- **No hay cobro por llamadas que retornan error 4xx** (input inválido, autenticación fallida) — solo se cobra compute real ejecutado.
- **Idempotencia:** el modelo stateless hace que cada llamada sea idempotente por diseño; no hay riesgo de cobro doble por retry.
- **Grandfathering:** los usuarios Pro en precio base $0.004 mantienen ese precio 12 meses ante cualquier revisión de tarifa.
- **Overage en Free:** las llamadas 501+ en Free son rechazadas con HTTP 429, nunca cobradas silenciosamente.