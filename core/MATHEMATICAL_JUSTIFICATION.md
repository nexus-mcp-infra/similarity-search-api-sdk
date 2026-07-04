# Justificación Matemática: Similarity Search API con Score Híbrido NMI+Cosine

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión cognitiva sigue $T = b \cdot \log_2(n+1)$, donde cada endpoint adicional incrementa la latencia de integración de forma logarítmica. Con 5 endpoints, $T \approx b \cdot 2.58$; con 10, $T \approx b \cdot 3.46$ — un 34% más de fricción de onboarding sin añadir valor computacional. Para una primitiva per-call donde el desarrollador decide en el momento de integración si la adopta, reducir $T$ es directamente reducir churn pre-conversión.

## 2. Pricing Per-Call vs. Por Asiento (Elasticidad Precio-Demanda)

La elasticidad precio-demanda $\varepsilon = \frac{\partial Q / Q}{\partial P / P}$ es más elástica en uso episódico (scripts one-off, colecciones $< 100$k items) que en uso continuo. Un modelo por asiento fija $P$ independientemente de $Q$, lo que hace que para $Q$ bajo el coste medio por query sea prohibitivo y el usuario abandone. Per-call alinea $P \propto Q$, manteniendo el coste marginal constante y capturando usuarios de cola larga que serían precio-excluidos en un modelo fijo — exactamente el segmento de dolor declarado.

## 3. Estructura de Datos: Por Qué Matrices de Frecuencia Conjunta On-the-Fly

NMI requiere $I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$, computable en $O(|V|^2)$ sobre la matriz de co-ocurrencia. Mantener un índice persistente añadiría $O(N \cdot |V|)$ de escritura en ingest para un beneficio de $O(1)$ en lectura — trade-off desfavorable para $N < 100$k. La representación como histogramas de frecuencia dispersos (dict de conteos) da complejidad de construcción $O(N \cdot L)$ donde $L$ es longitud media de documento, con footprint de memoria $O(|V|)$ en lugar de $O(N \cdot |V|)$ para una matriz densa.

## 4. Invariante Matemático: Ponderación Adaptativa por Entropía Marginal

El invariante central es que $\alpha(C) = H_{\text{marginal}}(C) / \log_2|V|$ es la entropía normalizada del corpus, acotada en $[0,1]$ por definición. Cuando el corpus es uniforme ($H_{\text{marginal}} \to \log_2|V|$), $\alpha \to 1$ y el score colapsa a cosine puro — correcto, porque NMI aporta poco cuando no hay concentración estadística. Cuando el corpus es concentrado ($H_{\text{marginal}} \ll \log_2|V|$), $\alpha \to 0$ y NMI domina, capturando dependencias no lineales que cosine ignora. El score $H(q,d) = \alpha(C)\cdot\text{cosine}(q,d) + (1-\alpha(C))\cdot\text{NMI}(q,d)$ hereda acotamiento en $[0,1]$ de ambos sumandos bajo sus normalizaciones estándar.

## 5. Límites Teóricos del Sistema

NMI requiere estimación de distribuciones conjuntas: para vocabularios con $|V| > 10^5$ tokens únicos, la matriz de co-ocurrencia dispersa supera los 2 GB de memoria de trabajo y la estimación de $p(x,y)$ se vuelve estadísticamente ruidosa con menos de $O(|V|^2 / \epsilon^2)$ muestras para garantizar error $\epsilon$ en las probabilidades. El sistema no puede reemplazar búsqueda ANN (Approximate Nearest Neighbor) sobre $N > 500$k documentos — la complejidad de NMI pairwise es $O(N^2 \cdot |V|)$, no reducible por árboles KD ni HNSW porque NMI no satisface la desigualdad triangular y por tanto no admite indexación métrica. Ese es el límite duro: esta primitiva es óptima para el régimen $N < 100$k, no un reemplazo de FAISS.