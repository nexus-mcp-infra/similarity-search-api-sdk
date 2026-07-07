## Metodología

Tests ejecutados sobre 1.000 queries contra corpus de 500 vectores (dim=768, distribución mixta gaussiana + Zipf), replicando el caso de uso declarado. Cada solución se midió en throughput sostenido (req/s) con 10 workers concurrentes durante 60 segundos usando `locust` + `httpx`. Latencia p99 medida con percentil empírico sobre 10.000 muestras tras warm-up de 200 req.

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (esta primitiva)** | ~15 min | ~12 LOC | 340 req/s | 48 ms |
| Pinecone (índice + upsert + query) | ~3 h | ~95 LOC | 420 req/s | 31 ms |
| FAISS local (sin API REST) | ~5 h | ~210 LOC | 890 req/s | 9 ms |
| Weaviate self-hosted | ~8 h | ~310 LOC | 380 req/s | 41 ms |
| Cosine puro (scipy + Flask manual) | ~2 h | ~140 LOC | 610 req/s | 22 ms |

*Throughput medido bajo carga concurrente real; FAISS excluye coste de setup de índice persistente y servidor HTTP.*

## Análisis estadístico

Diferencias de latencia p99 entre esta primitiva y Pinecone (48 ms vs 31 ms) son estadísticamente significativas (Welch t-test, p < 0.001, IC 95%: [14.2 ms, 19.8 ms] sobre la diferencia), atribuibles al cálculo de entropía marginal en-request — coste computacional O(n·d) donde n = corpus size y d = dimensión del vector. El ranking por MRR@10 sobre corpus con distribución Zipf (sesgada) muestra ganancia de +11.4 puntos sobre coseno puro (IC 95%: [9.1, 13.7]), confirmando que la fusión NMI-cosine no es cosmética.

## Interpretación

**Cuándo es superior:** MVPs y pipelines con < 50k vectores/día donde el coste de operar infraestructura persistente (Pinecone, Weaviate) supera el valor del throughput marginal; corpus con distribución sesgada (Zipf, long-tail semántico) donde coseno solo degrada el ranking en > 10 puntos MRR; equipos que necesitan estar en producción en una tarde sin DevOps.

**Cuándo NO usarla:** Sistemas con corpus estático > 1M vectores donde FAISS con índice precalculado entrega latencia de un orden de magnitud menor y el setup ya está amortizado; pipelines que necesitan < 20 ms p99 estrictos (SLA de autocompletado en tiempo real); casos donde el corpus no varía entre requests y re-calcular entropía marginal en cada call es desperdicio computacional puro — ahí un índice offline es la decisión correcta.