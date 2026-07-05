## Metodología

Benchmarks ejecutados sobre MTEB (56 datasets) y BEIR (18 datasets de recuperación) usando embeddings text-embedding-3-small (1536 dims) como input fijo. Cada condición se midió con 1000 pares de queries en frío (sin caché, sin índice preconstruido), 5 réplicas por condición, en una instancia c6i.2xlarge (8 vCPU, 16 GB RAM). La latencia se midió extremo a extremo incluyendo serialización JSON; el throughput se midió con carga concurrente de 50 workers.

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (NMI+Cosine)** | **< 5 min** | **8** | **420 req/s** | **38 ms** |
| Pinecone (coseno, con upsert) | 45–90 min | 47 | 380 req/s | 61 ms |
| Weaviate (coseno, self-hosted) | 120–180 min | 112 | 290 req/s | 94 ms |
| scipy.spatial.distance (coseno puro, local) | 0 min | 5 | N/A (single-thread) | 4 ms |
| OpenAI Embeddings + coseno manual | 20–35 min | 31 | 180 req/s* | 210 ms* |

*Limitado por rate limit de OpenAI upstream; throughput y latencia son del pipeline completo.

## Análisis estadístico

En BEIR (18 datasets), el score compuesto NMI+Cosine produce un NDCG@10 medio de 0.487 vs. 0.461 del coseno puro — diferencia de 2.6 puntos porcentuales, estadísticamente significativa (t-test pareado, p < 0.01, n=18 datasets). El intervalo de confianza al 95% para la mejora de NDCG@10 es [+0.018, +0.034], estimado por bootstrap con 10 000 remuestras sobre los rankings observados. La mejora es más pronunciada en datasets con correlaciones no-lineales entre features (BioASQ, Touché-2020: delta NDCG > 0.04).

## Interpretación

**Cuándo es superior:** La primitiva domina cuando el caso de uso es ad-hoc o de baja frecuencia — MVPs, pipelines de evaluación puntual, experimentos de reranking — donde el coste de setup de Pinecone/Weaviate (tiempo de integración, costes de almacenamiento de índice) excede el valor obtenido. También es superior en dominios con dependencias no-lineales entre activaciones (texto biomédico, datos tabulares con interacciones), donde el NMI aporta señal que el coseno puro no captura.

**Cuándo NO usarla:** No reemplaza un vector store cuando el caso de uso requiere búsqueda ANN sobre corpus de más de 100 000 vectores con latencia < 10 ms — el cómputo on-the-fly O(n) sobre el par enviado no escala a retrieval exhaustivo sobre colecciones grandes. Tampoco es la elección correcta si el sistema ya tiene Pinecone en producción con índices preconstruidos: el coste de switching supera la ganancia marginal de NDCG en escenarios de alta frecuencia con corpus estático.