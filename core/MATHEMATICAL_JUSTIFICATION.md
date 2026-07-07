# Justificación Matemática — Similarity Search API

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión de integración sigue $T = b \cdot \log_2(n+1)$, donde $n$ es el número de operaciones expuestas. Con $n=5$, el desarrollador alcanza decisión óptima en $T \approx 2.58b$ unidades cognitivas; duplicar a $n=10$ incrementa ese coste un 38% sin añadir cobertura funcional. Para una primitiva stateless cuyo valor es la ausencia de setup, reducir fricción de integración *es* el producto — la API debe ser operable sin leer documentación completa.

## 2. Pricing Per-Call (Elasticidad Precio-Demanda)

El caso de uso central (similitud ad-hoc sobre colecciones <100k) tiene consumo altamente variable: un cliente dispara 3 llamadas/día en desarrollo y 10k en producción. Con precio por asiento, la elasticidad $\varepsilon = \partial Q / \partial P \cdot P/Q$ castiga precisamente los clientes de alto crecimiento, invirtiendo el flywheel. Per-call alinea coste marginal del cliente con valor marginal entregado: cada llamada produce un score $s \in [0,1]$ con intervalo de confianza bootstrap, y el cliente paga exactamente por esa unidad de certeza, no por capacidad instalada.

## 3. Estructura de Datos: Matriz de Features Heterogénea

El input se representa como $\mathbf{x} = (\mathbf{x}_c, \mathbf{x}_d)$ donde $\mathbf{x}_c \in \mathbb{R}^{p}$ son features continuas y $\mathbf{x}_d \in \{1,\ldots,K_j\}^{q}$ son categoriales. Esta separación permite computar $\text{NMI}(X_j, Y_j) = I(X_j;Y_j)/H(X_j,Y_j)$ en $O(K_j^2)$ por feature categorial y similitud coseno $\cos(\mathbf{u},\mathbf{v}) = \mathbf{u}^\top\mathbf{v}/(\|\mathbf{u}\|\|\mathbf{v}\|)$ en $O(p)$ sobre el bloque continuo — sin serializar a un formato unificado que destruiría la semántica de escala. La detección de tipo en tiempo de llamada corre en $O(p+q)$, dominada por el cómputo NMI cuando $q > 0$.

## 4. Invariante Matemático del Score Híbrido

El peso dinámico $\lambda = q/(p+q)$ garantiza que el score híbrido $s = \lambda \cdot \text{NMI}_{\text{agg}} + (1-\lambda) \cdot \cos(\mathbf{u},\mathbf{v})$ permanece en $[0,1]$ para todo input válido, dado que $\text{NMI} \in [0,1]$ por construcción (normalización por entropía conjunta) y $\cos \in [-1,1]$ se remapea a $[0,1]$. El invariante crítico es que $\lambda$ se recalcula por llamada a partir de los datos observados, no de un schema declarado — la corrección del score no depende de que el cliente clasifique sus propias features, eliminando una fuente de error humano sistemática.

## 5. Límites Teóricos del Sistema

La primitiva es asintóticamente ineficiente para colecciones $|C| > 10^5$: sin índice, la complejidad de búsqueda del top-$k$ es $O(|C| \cdot (p + K_{\max}^2 \cdot q))$, donde el término NMI domina cuando $q \gg p$. El bootstrap CI con $n=500$ remuestreos introduce un factor constante de $500\times$ sobre el cómputo del score, no paralelizable por encima de los cores disponibles por worker — a $p+q > 200$ features por par, la latencia excede SLA razonables (< 200ms) sin GPU. El sistema tampoco puede garantizar estabilidad del score cuando las distribuciones marginales de features categoriales tienen soporte $K_j < 5$: con entropía conjunta $H(X_j,Y_j) \approx 0$, el NMI normalizado es numéricamente inestable y el CI bootstrap diverge.