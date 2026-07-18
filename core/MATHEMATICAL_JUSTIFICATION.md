# Justificación Matemática: Similarity Search API

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión de integración sigue $T = b \cdot \log_2(n+1)$. Con $n=5$ endpoints, $T \approx 2.58b$; con $n=10$, $T \approx 3.46b$ — un 34% más de fricción cognitiva sin añadir cobertura funcional. Para esta primitiva, las operaciones atómicas son exactamente cinco: compute similarity, batch similarity, calibrate alpha, introspect entropy, health. Cualquier endpoint adicional colapsa una de estas responsabilidades en ambigüedad, no en potencia.

## 2. Pricing Per-Call vs Por Asiento (Elasticidad Precio-Demanda)

La elasticidad $\varepsilon = \frac{\partial Q / Q}{\partial P / P}$ para infraestructura de búsqueda stateless es alta en el margen: el developer con corpus de 10k documentos no activa una suscripción fija mensual, pero sí paga por 500 llamadas si el costo marginal es menor que el costo de setup alternativo (Pinecone: ~$70/mes mínimo). El modelo per-call captura exactamente el excedente del consumidor en corpora pequeños y medianos — el segmento donde esta primitiva tiene ventaja estructural — sin subsidiar casos de uso de alto volumen con infraestructura propia.

## 3. Estructura de Datos y Complejidad Algorítmica

El score compuesto $S = \alpha \cdot \cos(\mathbf{u}, \mathbf{v}) + (1-\alpha) \cdot \text{NMI}(X, Y)$ requiere dos estructuras: vectores densos para cosine ($O(d)$ por par, donde $d$ es dimensión de embedding) y tablas de frecuencia conjunta para NMI ($O(n \log n)$ para construir $H(X,Y)$ via sort-and-count sobre $n$ ítems del corpus entrante). El diseño stateless obliga a que ambas estructuras vivan en memoria de request — lo que fuerza el límite de 500k ítems: más allá, $O(n \log n)$ en RAM de un solo worker supera 2GB con vectores float32 de dimensión 768.

## 4. Invariante Matemático de Corrección

El invariante es que $\alpha \in (0,1)$ está determinado únicamente por la entropía marginal del corpus: $\alpha = H(X) / (H(X) + H_{\max})$, donde $H_{\max} = \log_2(n)$ es la entropía máxima para $n$ ítems equiprobables. Esto garantiza que $\alpha \to 1$ cuando el corpus es semánticamente disperso (alta $H$, cosine discrimina mejor entre vectores alejados) y $\alpha \to 0$ cuando el corpus está en clusters densos (baja $H$, NMI captura dependencia no-lineal que cosine no puede distinguir por proximidad geométrica). El score es correcto — en el sentido de Bayes-optimal para la distribución observada — porque $\alpha$ es una estadística suficiente de la distribución empírica del corpus.

## 5. Límites Teóricos del Sistema

**Lo que no puede hacer y por qué:**

- **Corpus > 500k ítems**: la complejidad $O(n \log n)$ de construcción de $H(X,Y)$ en RAM stateless excede el presupuesto de latencia de una API síncrona por encima de ese umbral — no es una limitación de implementación sino de la definición de stateless.
- **Embeddings de dimensión variable entre llamadas**: cosine similarity no es comparable entre espacios métricos distintos; $\cos(\mathbf{u}, \mathbf{v})$ requiere $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ con $d$ fijo por corpus — la API rechaza requests mixtos por consistencia algebraica, no por convención.
- **Dependencias causales**: NMI mide $I(X;Y) = H(X) + H(Y) - H(X,Y)$, que es simétrica. Esta primitiva no puede detectar $X \to Y$ vs $Y \to X$ — para causalidad se requiere el módulo `src/math/causal`, fuera del scope de esta API.