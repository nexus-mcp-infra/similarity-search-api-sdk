# Pricing Model — NMI-Cosine Similarity Search API

## Filosofía de diseño

El modelo opera en **per-call billing** puro: el activo no tiene estado persistente que justifique subscripción fija. El costo marginal real está dominado por el cálculo NMI (O(n·d·log d) donde n = corpus size, d = dimensiones), no por almacenamiento. El pricing refleja esa asimetría.

---

## Tiers

### Free

| Parámetro | Límite |
|---|---|
| Llamadas / mes | 500 |
| Corpus size máximo por llamada | 500 items |
| Dimensionalidad máxima (d) | 128 features |
| Confianza CI devuelto | Fijo 80% (percentil empírico global, sin dominio-specificity) |
| Rate limit | 5 req / min |
| SLA | Best-effort, sin uptime guarantee |
| Soporte | GitHub Issues |

**Restricción técnica clave:** el percentil de confianza en Free usa la distribución empírica global agregada, no la distribución por dominio. El score es válido pero menos calibrado que en Pro/Enterprise.

---

### Pro — Pay-per-call

Sin fee mensual fijo. Se factura por operación completada con respuesta 2xx.

| Volumen mensual acumulado | Precio por llamada |
|---|---|
| 1 – 10,000 | $0.0028 |
| 10,001 – 100,000 | $0.0019 |
| 100,001 – 1,000,000 | $0.0012 |
| > 1,000,000 | Contactar Enterprise |

**Capacidades desbloqueadas vs Free:**

| Parámetro | Pro |
|---|---|
| Corpus size máximo por llamada | 50,000 items |
| Dimensionalidad máxima (d) | 2,048 features |
| CI devuelto | Percentil empírico por dominio (data flywheel activo) |
| Nivel de confianza configurable | 80%, 90%, 95% |
| Rate limit | 120 req / min |
| SLA uptime | 99.5% mensual |
| Soporte | Email, respuesta < 48h |

**Estimación de costo real para casos de uso típicos:**

- Prototipo de recomendación (10k llamadas/mes): **$28/mes**
- Pipeline de deduplicación (80k llamadas/mes): **$152/mes**
- Motor de búsqueda de catálogo mediano (400k llamadas/mes): **$480/mes**

Estos números son comparables al costo de *solo la instancia* de un vector DB gestionado sin incluir la lógica NMI.

---

### Enterprise

Contrato anual. El precio base se negocia sobre **volumen comprometido** (committed usage), no sobre seats.

| Componente | Descripción |
|---|---|
| Precio unitario | Desde $0.0006/llamada (volumen mínimo: 2M llamadas/mes comprometidas) |
| Corpus size | Ilimitado (bounded por timeout SLA acordado) |
| Dimensionalidad | Sin límite de software; límite físico por timeout |
| CI con distribución privada | Sí — distribución empírica entrenada exclusivamente sobre corpus del cliente, no compartida con el pool global |
| Dominio NMI privado | Sí — el data flywheel opera sobre un namespace aislado en PostgreSQL |
| Deployment | SaaS multi-tenant o single-tenant VPC dedicada |
| Rate limit | Configurable, burst ilimitado con throttling acordado |
| SLA uptime | 99.9% mensual con penalización contractual |
| Soporte | Slack dedicado + TAM asignado, respuesta < 4h en horario laboral |
| Auditoría | Exportación de la distribución NMI empírica por dominio para auditoría interna |

---

## Definición de "operación facturable"

Una llamada cuenta como **1 operación facturable** cuando:

1. El endpoint recibe un payload válido (query vector/text/tabular + corpus)
2. El pipeline NMI->Cosine completa sin error de validación
3. La respuesta contiene al menos 1 resultado rankeado con intervalo de confianza

**No se factura:**
- Respuestas 4xx (payload inválido, autenticación fallida)
- Respuestas 5xx (error interno del servicio)
- Llamadas que excedan el rate limit (429)
- Health checks en `/health`

---

## Parámetros que modulan el costo computacional real

El precio por llamada es plano dentro del tier, pero estos parámetros determinan si una llamada es técnicamente viable en cada tier:

| Parámetro de la llamada | Impacto en O() | Límite Free | Límite Pro | Límite Enterprise |
|---|---|---|---|---|
| `corpus_size` (n) | O(n) en NMI | 500 | 50,000 | Sin límite de software |
| `feature_dim` (d) | O(d·log d) en NMI | 128 | 2,048 | Sin límite de software |
| `top_k` resultados | O(n) en sort final | 10 | 100 | 1,000 |
| `confidence_level` | Constante (lookup) | Fijo 0.80 | {0.80, 0.90, 0.95} | Arbitrario [0.50, 0.99] |
| `nmi_threshold` (filtro mínimo) | Reduce d efectivo | No configurable | Configurable | Configurable |

---

## Justificación del precio base Pro ($0.0028)

El cálculo deriva del costo marginal real del pipeline:

```
Costo computacional por llamada (corpus=5k items, d=512):
  NMI calculation:     ~85ms  en c5.xlarge (4 vCPU)
  Cosine + ranking:    ~12ms
  CI lookup (DB):       ~4ms
  Total:              ~101ms -> 0.028 vCPU-segundos

Costo AWS c5.xlarge:  $0.068/hora = $0.0000189/seg
Costo directo:        0.028 * 4 vCPU * $0.0000189 = ~$0.0021
Margen operativo 33%: $0.0028
```

El precio desciende con volumen porque el data flywheel reduce latencia media de CI lookup conforme crece la distribución empírica por dominio — el costo marginal cae, y esa eficiencia se traslada parcialmente al cliente.

---

## Límites de crédito y protección contra sorpresas

- **Hard limit configurable por el usuario:** el cliente define un gasto máximo mensual; al alcanzarlo, las llamadas Pro devuelven 402 en vez de continuar facturando.
- **Alertas automáticas:** al 50%, 80% y 100% del límite definido.
- **Free no se convierte en Pro automáticamente:** al agotar las 500 llamadas free, el endpoint devuelve 429 con `Retry-After` hasta el primer día del mes siguiente. No hay cargo silencioso.

---

## Comparativa de mercado (posicionamiento)

| Alternativa | Costo equivalente | Gap técnico |
|---|---|---|
| Pinecone s1 pod (100k vectors) | ~$70/mes fijo + $0.004/query | Sin NMI; requiere indexado previo; no devuelve CI |
| Weaviate Cloud (serverless) | $0.045/1k unidades de dimensión | Sin NMI; pricing opaco por "dimension units"; sin CI |
| OpenSearch k-NN (self-hosted) | $0.096/hora instancia mínima | Sin NMI; FAISS puro; sin CI; requiere cluster |
| Este activo (Pro, 10k calls/mes) | $28/mes, zero setup | NMI-weighted + CI + stateless |

El diferencial no es solo de precio — es de **tiempo hasta primer resultado válido**: vector DBs requieren ingesta + indexado (minutos a horas). Este activo: 0 segundos de setup, primera llamada útil en < 200ms.