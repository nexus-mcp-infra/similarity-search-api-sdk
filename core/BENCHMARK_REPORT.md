## Metodología

Tests ejecutados sobre 50 datasets sintéticos de tamaño N={100, 500, 2000} items con proporciones variables de features categóricas (20%-80%) y continuas. Cada condición se repitió 200 veces; se midió tiempo wall-clock end-to-end (cliente HTTP → respuesta JSON) con wrk2 a carga sostenida de 50 RPS. La comparación incluyó Pinecone (pod s1.x1), Weaviate OSS (Docker local, HNSW default) y FAISS-CPU (flat index, no-GPU) como referencia de búsqueda vectorial pura.

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (esta)** | ~4 min | 8-12 | 340 RPS | 87 ms |
| Pinecone + embeddings | 45-75 min | 80-130 | 420 RPS | 52 ms |
| Weaviate OSS | 60-90 min | 110-160 | 380 RPS | 61 ms |
| FAISS-CPU (flat) | 20-35 min | 55-90 | 510 RPS | 38 ms |

Throughput medido con payload de 200 items, top-k=10, features 40% categóricas / 60% continuas. Los incumbentes requieren índice pre-construido; esta API opera stateless sobre payload raw.

## Análisis estadístico

Diferencias de latencia entre esta API y FAISS-CPU son estadísticamente significativas (Mann-Whitney U, p < 0.001); la penalización de 49 ms p99 frente a FAISS refleja el cómputo NMI en tiempo real, no overhead de red. La ventaja en calidad de ranking sobre datasets mixtos se midió con NDCG@10: esta API obtiene 0.84 ± 0.03 (IC 95%) vs 0.71 ± 0.04 para cosine-only (FAISS/Pinecone) cuando el porcentaje de features categóricas supera el 35%, diferencia de 0.13 puntos con p < 0.01.

## Interpretación

**Cuándo es superior:** catálogos de productos, datasets de CRM o inventarios donde coexisten campos de texto libre, categorías (tipo, región, SKU) y numerics — escenarios donde cosine-only pierde señal en variables ordinales y categóricas. También es la opción correcta cuando el equipo necesita validar product-market fit de búsqueda semántica antes de comprometer presupuesto en infraestructura persistente: el coste de oportunidad de 45-75 minutos de setup y $70+/mes de pod se elimina completamente.

**Cuándo NO usarla:** cargas superiores a ~300 RPS sostenidas con corpus estático grande (N > 10.000 items por llamada), donde FAISS con índice HNSW amortiza el costo de indexación y entrega latencias p99 < 40 ms imposibles de igualar en modo stateless. Tampoco es adecuada cuando el dominio es texto puro sin features categóricas (búsqueda semántica en documentos largos), caso en que embeddings densos pre-computados dominan en calidad de ranking y la contribución del término NMI colapsa a ruido.