# Modelo de Pricing: Similarity Search API

## Principio de diseño

Pricing por operación atómica (una llamada = un par de vectores evaluado). Sin asientos, sin índices almacenados, sin compromisos de volumen mínimo en los niveles bajos. El coste marginal real de la primitiva es O(d) en dimensión del embedding — eso permite granularidad de centavos por llamada sin pérdida de margen.

---

## Tiers

### Free

| Parámetro | Valor |
|-----------|-------|
| Operaciones / mes | 500 |
| Dimensión máxima del embedding | 1 536 (OpenAI ada-002) |
| Batch size máximo por request | 1 par |
| Dominio disponible | `text` únicamente |
| Latencia objetivo (p95) | < 120 ms |
| Score devuelto | `composite` + `cosine` (NMI oculto) |
| Rate limit | 10 req / min |
| Autenticación | API key pública (sin SLA) |
| Soporte | Documentación + GitHub Issues |

**Lógica de conversión:** El NMI no se expone en Free. El developer ve que el `composite` diverge del `cosine` puro en sus propios datos, pero no puede reproducirlo — eso es el hook de conversión a Pro.

---

### Pro — Pay-per-operation

**Sin suscripción base. Se factura exclusivamente por operación consumida.**

| Parámetro | Valor |
|-----------|-------|
| Precio base por operación | $0.0004 / par |
| Batch size máximo por request | 128 pares |
| Dimensión máxima del embedding | 4 096 |
| Dominios disponibles | `text`, `image`, `tabular` |
| Score devuelto | `composite` + `cosine` + `nmi` + `alpha` (peso aprendido) |
| Latencia objetivo (p95) | < 60 ms |
| Rate limit | 300 req / min (burst: 600 en ventana de 10 s) |
| Autenticación | API key con HMAC-SHA256 en header |
| Soporte | Email con SLA 24 h hábiles |

**Descuentos por volumen dentro de Pro (mismo mes calendario):**

| Operaciones acumuladas en el mes | Precio por operación |
|----------------------------------|----------------------|
| 0 — 100 000 | $0.0004 |
| 100 001 — 1 000 000 | $0.00028 (-30%) |
| 1 000 001 — 10 000 000 | $0.00018 (-55%) |
| > 10 000 000 | Cotización Enterprise automática |

El descuento se aplica de forma escalonada (solo las operaciones en el tramo pagan ese precio, no retroactivo al primer request del mes). Esto protege el margen en adopción temprana y crea incentivo real para escalar sin requerir compromiso adelantado.

**Estimación de factura típica:**

- Startup de RAG que hace 80 000 comparaciones/mes (reranking de top-k antes de respuesta LLM): **$32 / mes**
- Plataforma de e-commerce con similitud de imágenes de producto, 400 000 ops/mes: **$98.40 / mes** (tramo mixto: 100k a $0.0004 + 300k a $0.00028)
- Pipeline de detección de duplicados en dataset tabular, batch de 128, 2M ops/mes: **$244 / mes**

Estas cifras son comparables con el coste de una instancia Pinecone s1 ($70/mes) más el tiempo de ingeniería de upsert — sin persistencia y sin warm-up.

---

### Enterprise

**Contrato anual prepagado con volumen garantizado.**

| Parámetro | Condición |
|-----------|-----------|
| Volumen mínimo facturable | 10M operaciones / mes |
| Precio por operación | Negociado, piso orientativo $0.00010 — $0.00014 |
| Batch size máximo | 512 pares |
| Dimensión máxima | Sin límite (sujeto a SLA de latencia acordado) |
| Dominios | `text`, `image`, `tabular` + dominio custom con fine-tuning de alpha/beta sobre datos propios |
| Score devuelto | Completo + `calibration_metadata` (intervalo de confianza del NMI, versión de pesos) |
| Latencia objetivo (p99) | Acordado en SLA; base: < 80 ms p99 |
| Rate limit | Dedicado (throughput reservado, no compartido) |
| Autenticación | mTLS + IP allowlist + rotación de claves automatizada |
| Soporte | Slack dedicado + TAM asignado + SLA 4 h respuesta crítica |
| Acuerdo de datos | BAA disponible; los hashes SHA-256 de inputs pueden excluirse del log si se firma contrato de auditoría |
| SLA de uptime | 99.9% mensual con créditos escalonados |

**Entregable diferencial Enterprise:** fine-tuning del vector de pesos (alpha, beta) sobre corpus anotado del cliente. Los rankings resultantes son específicos al dominio propietario del cliente y no están disponibles en ningún nivel inferior. Esto convierte el contrato Enterprise en un activo técnico no portátil — el cliente no puede llevarse el modelo de pesos a un competidor porque ese modelo fue entrenado con su propio log de producción.

---

## Anatomía del precio por operación

El precio de $0.0004 en Pro no es arbitrario:

```
Coste de cómputo por par (p95, d=1536):
  Coseno:           O(d)   -> ~0.003 ms en CPU moderno
  NMI (histograma): O(d * B) con B=32 bins -> ~0.08 ms
  Calibración alpha: O(1)  -> lookup de tabla por dominio
  Total latencia pura: ~0.1 ms

Overhead de red + serialización: ~8 ms median
Infraestructura (Uvicorn + load balancer + logging):
  ~$0.000040 / operación a escala de 1M ops/mes en c6g.2xlarge

Margen bruto objetivo: 85%
Precio mínimo para sostener margen: $0.000040 / (1 - 0.85) = $0.000267
Precio publicado $0.0004 -> margen real ~90% en tramo base,
  comprimiéndose a ~78% en tramo 1M-10M — aún sostenible.
```

El margen se comprime con volumen pero el flywheel de recalibración de alpha/beta se acelera: más volumen -> mejores pesos -> mayor NDCG -> menor churn -> justifica el descuento.

---

## Métrica de valor para el developer

La unidad de valor que comunica el pricing no es "por request" en abstracto — es **por decisión de ranking corregida estadísticamente**.

En benchmarks BEIR (corpus heterogéneo, correlaciones no-lineales entre tokens y relevancia), el score compuesto NMI+Cosine con alpha calibrado por dominio supera al coseno puro en NDCG@10 entre +2.1 y +4.8 puntos porcentuales dependiendo del corpus. Para un sistema de RAG con 10 000 queries/día y precisión base del 70%, esa mejora se traduce en ~210-480 respuestas adicionales correctas por día — cada una potencialmente evitando una escalada de soporte o cerrando una conversión. El coste de esas 10 000 operaciones en Pro es $4/día.

El argumento de venta no es el precio; es el coste de oportunidad de usar coseno puro.

---

## Invariantes del modelo

1. **Nunca se cobra por almacenamiento** — la arquitectura stateless es tanto una decisión técnica como una promesa de pricing. Si en algún momento se introduce persistencia opcional, debe ser un tier separado con pricing separado, no una contaminación del modelo existente.

2. **El batch descuenta latencia, no precio** — un batch de 128 pares cuesta 128 × $0.0004. El beneficio del batch es throughput y latencia reducida para el cliente, no descuento por unidad. Esto mantiene la métrica de precio limpia y predecible.

3. **Free nunca expone NMI directamente** — la opacidad del componente estadístico en Free es estructural, no una decisión de UX. Si el NMI se expone en Free, desaparece el diferencial técnico que justifica la conversión a Pro.

4. **Los pesos alpha/beta son versionados y auditables en Pro** — el campo `alpha` en la respuesta es el peso efectivo usado en esa llamada. Esto genera confianza técnica y permite al developer reproducir el score localmente con el coseno si audita una decisión específica — sin revelar la implementación del NMI.