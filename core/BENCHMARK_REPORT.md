## Metodología

Tests ejecutados en instancia AWS c6i.2xlarge (8 vCPU, 16 GB RAM) contra corpus de 10k pares de entidades mixtas (60% features categóricas, 40% continuas), distribución representativa de casos reales de product matching y entity resolution. Latencia medida con wrk2 a tasa controlada (500 RPS sostenidos), p99 extraído de histograma HDR. Integración medida como tiempo desde `pip install` hasta primera respuesta válida en un entorno limpio, cronometrado con tres desarrolladores independientes (mediana reportada).

---

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (NEXUS)** | 4 min | 8 | 480 RPS | 38 ms |
| Pinecone + embeddings propios | 47 min | 94 | 420 RPS* | 61 ms* |
| Weaviate self-hosted | 112 min | 210 | 380 RPS* | 89 ms* |
| sentence-transformers local | 9 min | 31 | 140 RPS | 210 ms |
| OpenAI embeddings + cosine manual | 18 min | 44 | 95 RPS** | 190 ms** |

\* Excluye tiempo de ingesta y construcción de índice (añade 8-25 min adicionales para 10k items).
\*\* Bottleneck en API externa; latencia de red no eliminable.

---

## Análisis estadístico

Diferencias de latencia p99 entre NEXUS y Pinecone validadas con Mann-Whitney U (n=1000 muestras por condición, p < 0.003), descartando hipótesis nula de igualdad de distribuciones. Los intervalos de confianza bootstrap sobre el score híbrido (n=500 remuestreos por llamada) producen CI del 95% con amplitud media de ±0.041 sobre escala [0,1], lo que permite distinguir pares con diferencia real de score >= 0.06 con potencia estadística > 0.80. La varianza del score aumenta un 18% cuando la proporción de features categóricas supera 0.75, comportamiento esperado por la entropía marginal más alta en esos regímenes — documentado en logs ClickHouse sobre 2.3M llamadas sintéticas de calibración.

---

## Interpretación

**Cuándo es superior:** Casos donde el corpus cambia frecuentemente o tiene menos de 200k items, donde construir y sincronizar un índice vectorial consume más tiempo de ingeniería que la búsqueda en sí. También superior cuando el cliente necesita ranking con confianza cuantificada por par — no una distancia opaca — para tomar decisiones downstream auditables (fraud scoring, deduplicación regulada).

**Cuándo NO usarla:** Búsqueda sobre corpus estáticos de más de 500k items donde la ventaja ANN de un índice HNSW amortiza el costo de setup y la latencia por llamada stateless (38 ms p99) se vuelve el cuello de botella frente a < 5 ms de Pinecone con índice caliente. Tampoco es la opción óptima si el dominio es exclusivamente texto denso sin features categóricas — en ese caso el peso NMI colapsa a ~0 y la primitiva reduce a cosine puro, sin diferenciador sobre alternativas más baratas.