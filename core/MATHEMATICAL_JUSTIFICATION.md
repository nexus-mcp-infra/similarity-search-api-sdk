# Justificación Matemática: Similarity Search API (NMI+Cosine Fusion)

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión del desarrollador sobre cómo integrar una API sigue $T = b \cdot \log_2(n+1)$. Con $n=5$ endpoints, $T \approx 2.58b$; con $n=12$, $T \approx 3.70b$ — un 43% más de fricción cognitiva antes de la primera llamada exitosa. Para una primitiva cuyo diferenciador es eliminar setup, aumentar $n$ contradice directamente la propuesta de valor: cada endpoint adicional implica una decisión de routing que el cliente debe tomar antes de recibir valor.

## 2. Pricing Per-Call vs Por Asiento (Elasticidad Precio-Demanda)

La elasticidad precio-demanda para herramientas de infraestructura de evaluación es altamente elástica ($|\varepsilon| > 1$) en la fase de prueba: un costo fijo de \$70/mes genera una barrera de adopción independiente del volumen, colapsando la demanda a cero para el segmento de 500 queries/día. El modelo per-call alinea $\text{Costo} = p \cdot q$ con la curva de valor real del cliente — quien procesa 500 queries/día paga proporcional a 500, no a la capacidad de un pod ocioso. Esto maximiza el área bajo la curva de adopción en la cola larga del mercado.

## 3. Estructura de Datos: Payload Raw sin Índice (Complejidad Algorítmica)

Un índice HNSW persistente tiene costo de construcción $O(n \log n)$ y costo de query $O(\log n)$, pero requiere $O(n \cdot d)$ almacenamiento y estado mutable. Para catálogos pequeños ($n \leq 10^4$ items), la búsqueda exhaustiva sobre payload raw tiene costo $O(n \cdot d)$ por query — igual orden que HNSW en la práctica para $n$ moderado, pero con costo de setup $O(1)$ y cero estado. La estructura elegida es una matriz densa efímera $\mathbf{X} \in \mathbb{R}^{n \times d}$ construida en memoria por request y descartada al responder, eliminando el overhead de serialización/deserialización de índice.

## 4. Invariante Matemático del Scoring Unificado

El invariante central es que el score fusionado $S = \alpha \cdot \cos(\mathbf{u}, \mathbf{v}) + (1-\alpha) \cdot \widehat{\text{NMI}}(f_i, f_j)$ permanece en $[0, 1]$ para todo $\alpha \in [0,1]$, dado que $\cos \in [-1,1]$ se normaliza a $\frac{1+\cos}{2} \in [0,1]$ y $\text{NMI} \in [0,1]$ por construcción: $\text{NMI}(X,Y) = \frac{2 \cdot I(X;Y)}{H(X)+H(Y)}$ con $H$ estimada via conteos suavizados con corrección de Laplace ($+1$ por bin), garantizando $H > 0$ incluso en distribuciones degeneradas. Este invariante hace el ranking total-ordenado y comparable entre llamadas con distintos $\alpha$, sin normalización post-hoc.

## 5. Límites Teóricos del Sistema

El diseño stateless impone $n \leq O(10^4)$ items por llamada antes de que la latencia por request supere umbrales aceptables (~200ms): la complejidad $O(n^2)$ del cálculo NMI entre pares de features crece cuadráticamente con el catálogo, haciendo este sistema **incorrecto por diseño** para repositorios de millones de items — exactamente el caso de uso donde Pinecone tiene ventaja asintótica. Adicionalmente, NMI requiere variables con soporte discreto finito; features continuas de alta cardinalidad ($|\mathcal{V}| \to \infty$) colapsan la estimación de entropía hacia ruido, por lo que el pipeline no puede sustituir embeddings semánticos densos en dominios puramente textuales de vocabulario abierto.