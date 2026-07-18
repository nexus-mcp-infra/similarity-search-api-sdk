# Justificación Matemática — Similarity Search API (Hybrid NMI+Cosine Scorer)

---

## 1. Máximo 5 Endpoints — Hick's Law

El tiempo de decisión de un developer al integrar una API sigue $T = b \cdot \log_2(n+1)$, donde $n$ es el número de opciones cognitivas disponibles. Con 5 endpoints, $T \approx b \cdot 2.58$; con 10, $T \approx b \cdot 3.46$ — un 34% más de fricción de integración sin ganancia funcional. Para esta primitiva stateless, la superficie mínima correcta es: `score_hybrid_batch`, `score_hybrid_pair`, `detect_feature_schema`, `calibrate_weights`, `health` — cada uno con responsabilidad no solapada y sin estado compartido entre llamadas.

---

## 2. Pricing Per-Call vs Por Asiento — Elasticidad Precio-Demanda

La elasticidad precio-demanda para infraestructura de búsqueda efímera es alta en la dimensión de volumen y baja en la dimensión de asiento: un equipo de 3 devs puede generar $10^5$ llamadas/día o 0, dependiendo del pipeline. El modelo por asiento cobra la media de una distribución bimodal, extrayendo valor subóptimo. Per-call alinea el coste marginal del proveedor ($\approx O(n \cdot d)$ por batch de $n$ items en $d$ dimensiones) con el ingreso marginal, maximizando el excedente del productor sin distorsionar el volumen de uso del comprador.

---

## 3. Estructura de Datos — Complejidad Algorítmica

El scorer stateless recibe el corpus como matriz densa $X \in \mathbb{R}^{n \times d_c}$ para features continuas y tabla de frecuencias conjuntas $C \in \mathbb{Z}^{n \times d_k}$ para categóricas. Cosine sobre $X$ tiene complejidad $O(n^2 d_c)$ con producto matricial BLAS; NMI sobre $C$ requiere $O(n^2 k)$ donde $k$ es la cardinalidad media de categorías. Separar los dos espacios de representación antes de fusionar evita el coste de $O(n^2(d_c + k))$ sobre una matriz unificada mal tipada, y permite paralelizar los dos cómputos independientemente antes del merge ponderado.

---

## 4. Invariante Matemático de Corrección

El score híbrido $S = w_{cat} \cdot \widetilde{NMI} + w_{cont} \cdot \cos\theta$ es correcto si y solo si ambos términos están en el mismo rango $[0,1]$ y los pesos satisfacen $w_{cat} + w_{cont} = 1$, con $w_{cat} = d_k / (d_k + d_c)$ derivado de la proporción de features detectadas. $\widetilde{NMI} = \frac{I(X;Y)}{\sqrt{H(X)H(Y)}} \in [0,1]$ por definición de información mutua normalizada; $\cos\theta \in [-1,1]$ se proyecta a $[0,1]$ via $\frac{1+\cos\theta}{2}$. Este invariante garantiza que $S$ es una media convexa de dos métricas de dependencia estadísticamente compatibles — propiedad que un wrapper naive de sklearn + scipy no preserva sin la normalización cruzada explícita.

---

## 5. Límites Teóricos del Sistema

La ausencia de índice persistente implica complejidad de búsqueda $O(n)$ exacta, no $O(\log n)$ aproximada de HNSW/IVF. Para $n > 10^4$ items con $d_c > 768$ (embeddings densos), el tiempo de respuesta supera los 200ms en una sola instancia — este diseño es correcto para búsqueda efímera en batches moderados, no para recuperación online sobre corpus de millones de vectores. Adicionalmente, NMI asume que las distribuciones marginales $P(X)$ y $P(Y)$ son estimables con suficiente soporte en el corpus recibido por llamada; con $n < 30$ por categoría, la estimación de entropía tiene sesgo $O(k/n)$ que degrada la fiabilidad del score categórico.