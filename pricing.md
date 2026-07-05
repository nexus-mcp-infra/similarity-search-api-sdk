# Modelo de Pricing — Similarity Search API

## Lógica de Unidad Facturable

La unidad de cobro es la **comparación híbrida individual**: un par (query, candidate) para el que se ejecuta el pipeline NMI+coseno completo con calibración de pesos por entropía. Una llamada con `top_k=20` sobre un corpus de 500 items factura **500 comparaciones**, no 1 request. Esto alinea el precio con el costo computacional real: la calibración de entropía marginal corre por feature por par, no por request.

---

## Tiers

### Free

| Parámetro | Límite |
|---|---|
| Comparaciones / mes | 50 000 |
| Corpus por request | máx. 500 items |
| top\_k | máx. 10 |
| Explicabilidad por componente | incluida (w\_nmi, w\_coseno, score dominante) |
| Rate limit | 10 req / min |
| Retención de logs de pesos | ninguna — DuckDB flywheel no alimentado |
| SLA | sin garantía |

**Objetivo**: desarrollador que evalúa la API con un corpus de prueba. El límite de 500 items por corpus no es arbitrario: por debajo de ese umbral la latencia p99 es < 80 ms en hardware de referencia (4 vCPU, 8 GB RAM), lo que garantiza una experiencia de evaluación representativa sin subsidiar workloads de producción.

---

### Pro — Por Operación

**Precio: $0.0000 04 por comparación híbrida** (USD, facturación mensual por consumo real)

Ejemplos de escala real:

| Caso de uso | Corpus | top\_k | Comparaciones / llamada | Precio / llamada |
|---|---|---|---|---|
| Matching de producto e-commerce | 10 000 items | 20 | 10 000 | $0.004 |
| Deduplicación de registros | 50 000 items | 5 | 50 000 | $0.020 |
| Recomendación de contenido | 100 000 items | 50 | 100 000 | $0.040 |

| Parámetro | Límite |
|---|---|
| Comparaciones / mes | ilimitadas (mínimo facturable: $5 / mes) |
| Corpus por request | máx. 100 000 items |
| top\_k | máx. 100 |
| Explicabilidad por componente | incluida |
| Rate limit | 120 req / min |
| Retención de logs de pesos | 30 días — alimenta flywheel propio del cliente |
| SLA | 99.5 % uptime mensual |
| Soporte | async, < 48 h |

**Commitment discount**: prepago de 500 M comparaciones -> $0.000000 32 por comparación (20 % de descuento). Precio final: $160 por bloque prepagado vs. $200 en pay-as-you-go.

---

### Enterprise

**Precio: contrato anual, base $1 200 / mes + overage a $0.000000 28 / comparación sobre cuota base de 5 000 M comparaciones / mes**

| Parámetro | Límite |
|---|---|
| Comparaciones / mes incluidas | 5 000 M (base) |
| Corpus por request | sin límite documentado — negociado por SLA de latencia |
| top\_k | sin límite |
| Explicabilidad por componente | incluida + exportación de distribución de pesos por dominio |
| Rate limit | negociado — burst hasta 2 000 req / min |
| Retención de logs de pesos | 12 meses + acceso SQL directo a DuckDB snapshot propio |
| Flywheel de calibración | el cliente obtiene los umbrales de entropía calibrados sobre su dominio específico |
| Despliegue | opción VPC dedicada (precio adicional: $800 / mes por región) |
| SLA | 99.9 % uptime mensual + RTO < 15 min |
| Soporte | canal dedicado, < 4 h, escalación a ingeniería |

**Diferenciador Enterprise vs. Pro**: el acceso a la distribución de pesos histórica (w\_nmi / w\_coseno por dominio) es el activo de datos que Enterprise compra además del cómputo. Un cliente de e-commerce que acumula 6 meses de distribuciones puede ajustar sus propios modelos downstream con señales de dependencia estadística que ningún otro proveedor le da desagregadas.

---

## Estructura de Costos de Referencia (justificación del precio)

El costo unitario de $0.0000004 / comparación se deriva de:

- **Costo de cómputo**: calibración de entropía marginal por feature -> O(n · f) donde n = número de items en corpus, f = número de features. En hardware de referencia (c6i.xlarge, $0.17 / h), 1 M comparaciones con f = 10 features corren en ~2.3 s -> costo de cómputo puro: ~$0.000000109 / comparación.
- **Margen de operación + infra + DuckDB storage**: ~2.7x sobre costo de cómputo.
- **Resultado**: $0.0000004 / comparación -> margen bruto estimado ~63 % en Pro, ~71 % en Enterprise (por economías de escala en hardware dedicado).

Estos márgenes son conservadores para una API stateless: no hay costo de índice persistente, replicación de vectores ni infraestructura de escritura — la ventaja de diseño stateless se traslada directamente a margen.

---

## Señales de Upgrade Automático

El sistema debe emitir estos eventos para activar flujos de upgrade:

| Evento | Acción sugerida |
|---|---|
| Free: > 80 % de 50 000 comparaciones consumidas en los primeros 15 días del mes | Email con estimación de costo Pro para el patrón de uso observado |
| Pro: corpus > 80 000 items en > 30 % de las llamadas del mes | Alerta: latencia p99 puede superar SLA; sugerir Enterprise con burst |
| Pro: gasto mensual > $800 por 3 meses consecutivos | Propuesta Enterprise automática — break-even es a ~$1 000 / mes de gasto Pro |
| Enterprise: overage > 20 % de cuota base en 2 meses seguidos | Renegociación de cuota base |

---

## Lo que el Pricing NO incluye (límites explícitos)

- **Almacenamiento de corpus**: la API es stateless por diseño. El cliente envía el corpus en cada request. No hay tier de "corpus almacenado" — eso es exactamente lo que este producto evita ser.
- **Fine-tuning de umbrales de entropía**: el umbral H < 1.5 bits es el default calibrado. Enterprise obtiene umbrales calibrados por su dominio via flywheel; Pro no. Esta diferencia es el argumento de upgrade más fuerte para clientes con dominio especializado (geoespacial, texto médico, catálogos industriales).
- **Embeddings propios del cliente**: si el cliente quiere inyectar vectores pre-computados en lugar de features raw, eso es una feature de roadmap — no está en v1 y no se cobra ni promete.