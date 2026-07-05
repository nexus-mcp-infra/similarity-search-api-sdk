# Justificación Matemática: NMI-Weighted Similarity Search API

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión del desarrollador sigue $T = b \cdot \log_2(n+1)$. Con $n=5$ endpoints, $T \approx 2.58b$; duplicar a $n=10$ eleva $T$ a $3.46b$ — un 34% más de fricción cognitiva sin añadir valor diferencial. Para una primitiva stateless cuya propuesta central es *zero-setup*, cada endpoint adicional contradice el invariante de mínima resistencia de adopción.

## 2. Pricing Per-Call vs. Por Asiento (Elasticidad Precio-Demanda)

La elasticidad precio-demanda $\varepsilon = \frac{\partial Q / Q}{\partial P / P}$ es más inelástica en el margen cuando el costo es proporcional al valor generado. Un desarrollador que compara 500 productos en un carrito paga exactamente por esa operación; con precio por asiento pagaría en periodos de cero uso, desplazando la curva de demanda hacia sustitutos. El modelo per-call alinea $P$ con $\Delta \text{valor}$ por llamada, maximizando el área bajo la curva de disposición a pagar en workloads intermitentes.

## 3. Estructura de Datos y Complejidad Algorítmica

El núcleo opera sobre matrices densas $A \in \mathbb{R}^{n \times d}$ y $B \in \mathbb{R}^{m \times d}$. El cómputo NMI por feature usa binning Freedman-Diaconis: ancho de bin $h = 2 \cdot \text{IQR}(x) \cdot k^{-1/3}$, con costo $O(k \log k)$ por feature para el sort previo. El vector de pesos $\mathbf{w} \in \mathbb{R}^d$ se aplica como $\tilde{A} = A \cdot \text{diag}(\mathbf{w})$ en $O(nd)$, y el producto de similitud $\tilde{A}\tilde{B}^\top$ en $O(nmd)$ via BLAS. El costo total $O(n \cdot d \log d + nmd)$ es óptimo para el caso stateless: no existe estructura de índice pre-computable sin violar la constraint de zero-persistencia.

## 4. Invariante Matemático de Corrección

El score híbrido $\text{sim}(a,b) = \cos(\mathbf{w} \odot a,\, \mathbf{w} \odot b)$ es correcto si y solo si $w_i = \text{NMI}(X_i, Y) \in [0,1]$ constituye una ponderación válida, lo que se garantiza por la propiedad $\text{NMI} = \frac{I(X;Y)}{\sqrt{H(X)H(Y)}}$ donde $I(X;Y) \geq 0$ y $\text{NMI} \leq 1$ por desigualdad de información mutua. El invariante es: **features con dependencia mutua nula con el target reciben $w_i \approx 0$, colapsando su contribución al score sin requerir selección explícita de features**, lo que hace el resultado agnóstico al tipo de feature (continua, categórica, mixta) sin lógica de branching condicional.

## 5. Límites Teóricos del Sistema

El estimador de entropía discreta converge con error $O(|\mathcal{X}|/k)$ donde $|\mathcal{X}|$ es el cardinal del alfabeto: para features categóricas de alta cardinalidad ($|\mathcal{X}| \gg \sqrt{k}$), el NMI estará subestimado y los pesos serán imprecisos. El sistema no puede operar correctamente sobre colecciones con $k < 30$ items por feature continua — el estimador Freedman-Diaconis colapsa a un solo bin, degenerando $\text{NMI} \to 0$ artificialmente. Asimismo, la complejidad $O(nmd)$ hace inviable el regime $n, m > 10^4$ en una sola llamada HTTP dentro de presupuestos de latencia P99 razonables ($< 2s$): este no es un motor de búsqueda a escala de corpus; es una primitiva de comparación de colecciones medianas sin índice.