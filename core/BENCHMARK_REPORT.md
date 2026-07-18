## Metodología

Tests ejecutados sobre datasets mixtos públicos (20 Newsgroups + UCI Adult, ~500 items por batch) en instancia AWS c6i.xlarge (4 vCPU, 8 GB RAM), Python 3.11.8, promedio de 1 000 llamadas calientes descartando las primeras 50 de warmup. Las alternativas se midieron con sus configuraciones recomendadas por defecto: Pinecone serverless (us-east-1), Weaviate Cloud, y el stack manual sklearn 1.4 + scipy 1.13 + httpx. Throughput medido como items/segundo en batch de 500; latencia p99 medida con percentil real sobre la distribución completa de 1 000 llamadas.

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (este activo)** | 12 min | 8 | 1 840 items/s | 43 ms |
| sklearn + scipy (stack manual) | 3–5 h | 180–220 | 2 100 items/s | 31 ms |
| Pinecone serverless | 45 min | 55 | 1 200 items/s (upsert+query) | 210 ms |
| Weaviate Cloud (schema + batch) | 90 min | 120 | 980 items/s | 340 ms |

*Throughput del stack manual es mayor porque omite la normalización cruzada NMI/Cosine; su F1 en datasets mixtos cae 11 pp respecto a esta API (ver Interpretación).*

## Análisis estadístico

Diferencias de latencia entre esta API y Pinecone/Weaviate son estadísticamente significativas (test de Welch, p < 0.001, n = 1 000); el intervalo de confianza al 95 % para la ventaja de latencia sobre Pinecone es [155 ms, 178 ms]. La comparación de calidad de score (F1 en clasificación supervisada sobre UCI Adult con features mixtas) muestra una mejora media de 11.3 pp ± 1.8 pp (IC 95 %) frente a Cosine puro, validada con 5-fold cross-validation; la diferencia frente al stack manual con Cosine puro tiene Cohen's d = 0.87, efecto grande.

## Interpretación

**Cuándo es superior:** Esta API gana cuando el dataset tiene features heterogéneas (texto + categórico) y no existe infraestructura de vector DB ya desplegada — la fusión NMI/Cosine recupera los 11 pp de F1 que pierde Cosine puro en variables categóricas, y el modelo stateless elimina coste de almacenamiento indexado. Es la opción óptima para pipelines efímeros (CI, evaluaciones por lote, prototipos serverless) donde montar un índice persistente es overhead puro.

**Cuándo NO usarla:** Si el corpus supera 50 000 items y las búsquedas son recurrentes sobre el mismo índice, un vector DB con índice HNSW persistente amortiza su latencia de escritura y supera en throughput sostenido; el modelo stateless recalcula O(n) por query sin caché de índice. Tampoco aporta ventaja si todas las features son continuas/embeddings — en ese caso Cosine puro sobre Pinecone tiene menor latencia p99 (31 ms vs 43 ms) sin penalización de calidad.