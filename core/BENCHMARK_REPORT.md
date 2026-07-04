## Metodología

Tests ejecutados sobre corpus sintéticos de 1k, 10k y 50k documentos (texto corto, categorías discretas, series temporales de 24 puntos), en instancia c6i.2xlarge (8 vCPU, 16 GB RAM). Cada configuración se ejecutó 500 iteraciones en frío (sin índice precalentado); latencias medidas con `perf_counter_ns`, throughput con wrk2 a tasa sostenida. Comparadores: Pinecone Serverless (cosine nativo), Weaviate OSS 1.24 (cosine+BM25), sklearn `NearestNeighbors` con NMI manual.

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput (QPS) | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (NMI+Cosine híbrido)** | 4 min | 8 | 1.840 | 38 ms |
| Pinecone Serverless | 22 min | 47 | 3.200 | 22 ms |
| Weaviate OSS 1.24 | 68 min | 130 | 2.100 | 31 ms |
| sklearn NMI manual | 11 min | 94 | 210 | 480 ms |

Pinecone requiere pipeline previo de embeddings (OpenAI/Cohere); throughput medido ya con embeddings cacheados, ventaja artificial. Sin caché, su latencia efectiva sube a 290 ms p99. El score híbrido opera sobre datos crudos — latencia incluye cálculo de H_marginal(corpus) completo.

## Análisis estadístico

Diferencias de latencia p99 entre la API y Weaviate (38 ms vs 31 ms) no son estadísticamente significativas al 95% (IC solapado: [33–44] ms vs [27–36] ms, Mann-Whitney U, p = 0.09). La diferencia en throughput frente a sklearn NMI manual (1.840 vs 210 QPS) sí es significativa (p < 0.001, d de Cohen = 2.3), explicada por la vectorización NumPy del cálculo de entropía marginal versus el loop Python puro. El intervalo de confianza del score híbrido H(q,d) sobre recall@10 es [0.71, 0.78] vs [0.61, 0.68] de cosine puro en corpus con dependencias no lineales (correlación categórica simulada, 10k docs).

## Interpretación

**Cuándo es superior:** Colecciones entre 5k y 100k items donde el overhead de un servidor vectorial es desproporcionado; datos con dependencia estadística no lineal (categorías, distribuciones multimodales) donde cosine pierde hasta 14 puntos de recall@10 frente al score híbrido; scripts one-off o pipelines batch sin infraestructura persistente. El alpha(C) adaptativo captura la entropía real del corpus específico — ventaja no replicable sin conocer esa distribución.

**Cuándo NO usarla:** Colecciones superiores a 500k items donde FAISS con índice IVF amortiza el costo de construcción y supera 8.000 QPS sostenidos; casos donde la latencia p99 < 20 ms es requisito duro (Pinecone con embeddings precalculados gana en ese escenario); datos continuos de alta dimensión (>768 dims, embeddings densos) donde NMI pierde poder estadístico por escasez de bins y cosine es suficiente.