# Pricing Model — Similarity Search API (NMI-Hybrid)

## Filosofía de diseño

El score híbrido H(q,d) es una operación estadística con costo computacional variable según N (tamaño del corpus) y V (vocabulario). El pricing refleja ese costo real: **se cobra por operación de búsqueda**, no por asiento ni por mes de acceso. El corpus es efímero por default; la persistencia es el único estado que cuesta por tiempo.

---

## Tiers

### Free

| Parámetro | Límite |
|---|---|
| Búsquedas / día | 100 |
| Corpus size / query | max 1 000 items |
| Dimensiones por ítem | max 512 tokens / serie de 256 puntos |
| Persistencia de corpus | ninguna — cada request es stateless |
| Métricas expuestas | H(q,d) únicamente |
| Rate limit | 10 req / min |
| SLA | best-effort, sin garantía de latencia |
| Soporte | documentación pública + GitHub Issues |

**Restricción de diseño:** el tier free no expone `alpha(C)` ni el desglose `cosine_component` / `nmi_component` del score — solo el scalar final. Eso no es capricho de negocio: sin ver la entropía marginal del corpus no se puede replicar la ponderación, lo que protege el moat técnico incluso en uso gratuito.

---

### Pro — por operación

Precio base por búsqueda (una query contra un corpus, retorna top-k resultados rankeados por H):

| Corpus size (N) | Precio por búsqueda |
|---|---|
| N <= 10 000 | $0.0008 |
| 10 001 <= N <= 100 000 | $0.0020 |
| N > 100 000 | $0.0045 |

**Justificación de la escala de precios:**

El costo computacional del cálculo NMI sobre corpus crudo crece como O(N * \|V\|) donde \|V\| es el vocabulario efectivo. El componente cosine es O(N * d) con d fijo. La ponderación adaptativa alpha(C) = H\_marginal(C) / log2(\|V\|) requiere un pase completo sobre el corpus para calcular la distribución marginal antes de cada búsqueda stateless. A N=100k con \|V\|~50k tokens, ese pase representa ~5×10^9 operaciones elementales — de ahí el salto de precio en el tramo superior.

**Extras facturables en Pro:**

| Add-on | Precio |
|---|---|
| Corpus persistido (hot, TTL 24h) | $0.0002 / item / día |
| Desglose de score (`alpha`, `cosine_component`, `nmi_component`) | +$0.0003 / búsqueda |
| Batch de queries (hasta 50 queries contra mismo corpus) | precio del tramo × 0.7 (30 % descuento) |
| Webhook de resultado asíncrono (corpus > 50k, latencia > 2s) | sin costo adicional — incluido para N > 50k |

**Compromiso de latencia Pro (P95):**

| N | Latencia P95 |
|---|---|
| <= 10 000 | 180 ms |
| <= 100 000 | 1 400 ms |
| > 100 000 | respuesta asíncrona, webhook <= 8s |

**Rate limit Pro:** 120 req / min por API key. Burst de hasta 300 req durante 10s, luego throttle a 120.

**Facturación:** prepago en créditos. Mínimo de compra: $10. Sin vencimiento de créditos en los primeros 12 meses. Sin costo de setup ni suscripción mensual fija.

---

### Enterprise

Dirigido a: corpus persistidos > 500k items, SLA contractual, necesidad de despliegue en VPC propia o on-premise, o integración de dominio propietario (tokenización custom, series temporales con semántica de negocio).

| Parámetro | Enterprise |
|---|---|
| Precio por búsqueda | negociado por volumen comprometido (escala desde $0.0030 a N > 100k) |
| Corpus size | ilimitado — se dimensiona infraestructura dedicada |
| Corpus persistido | caliente sin TTL, índice de entropía marginal precalculado y actualizado en streaming |
| Desglose de score | incluido por default |
| Tokenización custom | sí — pipeline de discretización propietario integrable vía webhook |
| SLA uptime | 99.9 % mensual, contractual |
| Latencia P95 garantizada | definida por contrato según N y QPS comprometido |
| Soporte | canal dedicado, ingeniero de soluciones asignado, runbooks privados |
| Despliegue | SaaS managed / VPC del cliente (AWS, GCP, Azure) / on-premise (Docker + Helm) |
| Auditoría matemática | acceso al informe de calibración de alpha(C) por corpus, útil para equipos de ciencia de datos que necesitan justificar el score ante stakeholders internos |

**Condición de entrada a Enterprise:** volumen comprometido >= $2 000 / mes o requerimiento técnico que el tier Pro no puede satisfacer (SLA contractual, despliegue privado, corpus > 500k).

---

## Comparativa de valor vs. alternativas

| Capacidad | Esta API | Pinecone / Weaviate | sklearn / faiss (self-hosted) |
|---|---|---|---|
| NMI nativo como métrica | si | no | no |
| Score híbrido NMI+Cosine ponderado por entropía | si | no | no — requiere reimplementar alpha(C) |
| Sin embedding pipeline previo | si | no — requiere embeddings | si, pero requiere infraestructura |
| Stateless, sin índice persistente | si (Free y Pro base) | no | depende del setup |
| One-off sobre corpus < 100k sin servidor | si | no — overhead de servidor vectorial | si, pero no como servicio |
| Costo de una búsqueda puntual sobre 10k items | $0.0008 | ~$0.10 (ingest + query en Pinecone serverless) | $0 pero horas de setup + infra |

El precio de referencia de $0.0008 para N<=10k es deliberadamente agresivo frente al costo real de Pinecone serverless (~$0.08 por 1M dimensiones de ingest + $0.10 por 1M queries leídas), porque el caso de uso principal del tier Pro es el developer con colecciones pequeñas que no puede justificar el overhead de una base de datos vectorial. El moat no está en el precio — está en que el score NMI-híbrido ponderado por entropía marginal del corpus específico no es replicable one-to-one con sklearn sin conocer la distribución real del corpus del cliente.

---

## Modelo de créditos — mecánica operativa

```
1 búsqueda (N <= 10k)                    = 1 crédito     ($0.0008 / crédito al comprar el pack de entrada)
1 búsqueda (10k < N <= 100k)             = 2.5 créditos
1 búsqueda (N > 100k)                    = 5.6 créditos
Add-on score breakdown                   = +0.375 créditos / búsqueda
Corpus persistido                        = 0.25 créditos / item / día
Batch 50 queries (N <= 10k)              = 35 créditos (vs 50 sin descuento)
```

**Packs de créditos:**

| Pack | Créditos | Precio | Precio por crédito |
|---|---|---|---|
| Starter | 12 500 | $10 | $0.0008 |
| Growth | 75 000 | $55 | $0.00073 |
| Scale | 250 000 | $165 | $0.00066 |
| Pro Max | 1 000 000 | $600 | $0.00060 |

El descuento por volumen en créditos (hasta 25 % en Pro Max vs Starter) es la palanca de retención para developers que ya validaron el caso de uso en Free y están escalando — sin bloquearlos en un contrato mensual.