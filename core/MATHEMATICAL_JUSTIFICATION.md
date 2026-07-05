# Justificación Matemática: Similarity Search API

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión de un desarrollador integrando una API sigue $T = b \cdot \log_2(n+1)$. Con $n=5$ endpoints, el overhead cognitivo es $b \cdot \log_2(6) \approx 2.58b$; duplicarlo a $n=10$ lo eleva a $3.46b$ — un 34% más de fricción sin ganancia funcional para una primitiva stateless. Para esta API, la superficie mínima es exactamente la necesaria: `score`, `rank`, `batch_score`, `calibrate_domain_weights` y `health`. Ningún endpoint adicional añade capacidad que no sea composición de estos.

## 2. Pricing Per-Call (Elasticidad Precio-Demanda)

Para casos de uso ad-hoc la demanda es altamente elástica: $\varepsilon = \frac{\Delta Q / Q}{\Delta P / P} \ll -1$. Un modelo por asiento impone coste fijo $C_{seat}$ independiente del volumen, lo que hace que el coste marginal por query $C_{seat}/q$ sea prohibitivo en $q$ bajos (MVP, prototipos, pipelines batch esporádicos). El modelo per-call fija $P_{call}$ constante, alineando el coste con el valor recibido y capturando el segmento de alta elasticidad que los modelos de índice almacenado (Pinecone tier mínimo: $\$70$/mes antes de la primera query) excluyen estructuralmente.

## 3. Estructura de Datos: Embeddings en Request Body

Persistir vectores introduce $O(n)$ overhead de upsert antes de cualquier búsqueda, más latencia de red en ida-vuelta adicional. La arquitectura stateless reduce la complejidad operacional a $O(d)$ por par de embeddings de dimensión $d$: el cómputo de coseno es $O(d)$ y la estimación de NMI por histograma adaptativo es $O(d \log d)$ (ordenación del espacio de activaciones). El log de SHA-256 es $O(1)$ por llamada y no recupera el vector original, preservando privacidad con costo computacional despreciable frente al cómputo del score.

## 4. Invariante Matemático: Superioridad del Score Compuesto

El invariante es que el score $S = \alpha \cdot \cos(\mathbf{u}, \mathbf{v}) + (1-\alpha) \cdot \widehat{NMI}(\mathbf{u}, \mathbf{v})$ domina estocásticamente al coseno puro en NDCG cuando la distribución conjunta de activaciones no es gaussiana. Formalmente, si $I(U;V) > H(U) \cdot H(V) \cdot \cos^2\theta$ — es decir, cuando la dependencia mutua excede lo explicable por alineación lineal — entonces $\mathbb{E}[\text{NDCG}(S)] > \mathbb{E}[\text{NDCG}(\cos)]$. Los pesos $\alpha^*$ que maximizan NDCG sobre el log de producción se obtienen por descenso de gradiente sobre pares etiquetados reales, haciéndolos estrictamente específicos al dominio y al corpus de llamadas observado: no son transferibles sin los datos.

## 5. Límites Teóricos del Sistema

El sistema no puede garantizar coherencia transitiva de rankings: si $S(a,b) > S(a,c)$ y $S(a,c) > S(a,d)$, no se garantiza $S(b,c) > S(b,d)$ porque NMI no define una métrica en sentido estricto (no cumple desigualdad triangular). Tampoco escala a búsqueda sobre colecciones grandes sin índice: la complejidad de fuerza bruta es $O(n \cdot d \log d)$ sobre $n$ candidatos, lo que hace que comparaciones contra corpus de $n > 10^4$ vectores requieran HNSW o IVF externo — exactamente el caso de uso que esta primitiva no pretende cubrir. El límite de validez es similitud puntual o batch pequeño ($n \leq 512$), donde el diferenciador estadístico opera sin degradación de latencia.