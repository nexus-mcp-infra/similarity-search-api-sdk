## Metodología

Tests ejecutados sobre corpus sintéticos de 1K, 10K y 100K vectores (dim=384, modelo `all-MiniLM-L6-v2`) usando pares con similitud controlada: 30% semánticamente relacionados, 70% correlacionados por sesgo del embedding space (subespacio denso de términos financieros). Cada condición se repitió 200 veces; tiempos medidos con `perf_counter_ns` excluyendo serialización HTTP. Baseline de comparación: Pinecone Serverless, Qdrant Cloud (free tier) y SciPy `cosine_similarity` directa.

---

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput (req/s) | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API** | < 5 min (HTTP directo) | 8-12 | 340 | 38 ms |
| Pinecone Serverless | 45-90 min (index + upsert) | 60-120 | 410 | 22 ms |
| Qdrant Cloud | 30-60 min (colección + schema) | 80-140 | 380 | 19 ms |
| SciPy cosine directo | 0 min | 5 | 1,200 | 4 ms |

*Throughput medido a corpus=10K, batch=50 pares, concurrencia=8. Latencia incluye cálculo NMI + p-value Bonferroni.*

---

## Análisis estadístico

La diferencia en tasa de falsos positivos semánticos entre coseno puro y el score compuesto S=0.6·coseno+0.4·NMI es de 31 puntos porcentuales (IC 95%: [27.4%, 34.6%], n=6,000 pares, chi-cuadrado p<0.001). El p-value calibrado por Bonferroni con m=corpus_size mantiene FWER<0.05 en todos los tamaños testados; a corpus=100K la corrección eleva el umbral de significancia efectivo a α=5×10⁻⁷, lo que filtra correctamente el 94% de correlaciones espaciales espurias identificadas en el conjunto de control.

---

## Interpretación

**Cuándo es superior:** datasets menores a 500K ítems con uso esporádico o por lotes donde montar una vector DB implica coste fijo no recuperable; pipelines de deduplicación, recomendación o clustering donde distinguir similitud semántica real de artefacto del embedding space es crítico para la calidad del output downstream; equipos sin infraestructura ML que necesitan un p-value interpretable sin implementar corrección estadística propia.

**Cuándo NO usarla:** búsqueda de nearest-neighbor a escala >1M vectores con SLA de latencia p99<10 ms — Pinecone/Qdrant con HNSW son estructuralmente más rápidos en ese régimen; casos donde el embedding space está bien calibrado y el usuario solo necesita ranking ordinal sin significancia estadística, donde SciPy directo a 4 ms p99 y 0 LOC de integración domina en coste-beneficio.