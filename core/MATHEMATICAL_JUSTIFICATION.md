# Justificación Matemática: Similarity Search API (NMI-Cosine Fusion)

## 1. Máximo 5 Endpoints — Hick's Law

$T = b \cdot \log_2(n + 1)$ predice que el tiempo de decisión de integración crece logarítmicamente con el número de opciones. Con $n = 5$ endpoints, $T \approx 2.58b$; con $n = 10$, $T \approx 3.46b$ — un 34% más de fricción cognitiva sin ganancia funcional proporcional. Para una primitiva stateless donde el valor está en la operación atómica (enviar corpus, recibir scores), la superficie mínima maximiza la tasa de adopción sin degradar expresividad.

## 2. Pricing Per-Call — Elasticidad Precio-Demanda

La elasticidad $\varepsilon = \frac{\partial Q / Q}{\partial P / P}$ para herramientas de infraestructura de bajo volumen es altamente elástica: un developer con 500 búsquedas/día rechaza una suscripción fija de \$49/mes pero acepta \$0.002/call (coste total: \$1/día). El modelo per-call alinea el coste marginal del consumidor con el coste marginal real de cómputo ($O(n \cdot d)$ por request), eliminando la barrera de entrada que hace que el developer elija FAISS local sobre una API REST con valor real.

## 3. Estructura de Datos — Complejidad Algorítmica

El corpus se representa como matriz densa $\mathbf{V} \in \mathbb{R}^{n \times d}$ en memoria durante el request. El coseno se calcula como $\mathbf{V}_{\text{norm}} \cdot \mathbf{q}_{\text{norm}}^\top$ en $O(n \cdot d)$ con una sola operación BLAS. La NMI requiere histogramas de densidad conjunta sobre el corpus completo: $O(n \log n)$ para el sort más $O(n \cdot B)$ para binning con $B$ bins. La fusión resultante en $O(n)$ es estrictamente más barata que construir un índice HNSW ($O(n \log n)$ construcción + almacenamiento persistente), lo que justifica el modelo stateless para $n \lesssim 10^4$ vectores por call.

## 4. Invariante Matemático — Corrección de la Fusión Adaptativa

El score fusionado $S = (1 - w_{\text{nmi}}) \cdot \cos(\mathbf{v}, \mathbf{q}) + w_{\text{nmi}} \cdot \text{NMI}(\mathbf{v}, \mathbf{q})$ es correcto si y solo si $w_{\text{nmi}} \in [0, 1]$ y se deriva de la entropía marginal real del corpus: $w_{\text{nmi}} = H(\mathbf{V}) / (H(\mathbf{V}) + H_0)$, donde $H_0$ es la entropía de referencia de una distribución uniforme con el mismo soporte. El invariante es que $w_{\text{nmi}} \to 0$ cuando $H(\mathbf{V}) \to 0$ (corpus degenerado, coseno domina) y $w_{\text{nmi}} \to 1$ cuando $H(\mathbf{V}) \to H_0$ (corpus uniforme, NMI compensa la pérdida de discriminabilidad coseno). El ranking es monótonamente consistente con este peso bajo cualquier distribución de entrada.

## 5. Límites Teóricos del Sistema

El modelo stateless impone dos límites duros derivados de la arquitectura. Primero, la complejidad $O(n \cdot d)$ por request hace que $n > 10^4$ vectores con $d = 1536$ (ada-002) consuma $\sim$2.4 GB de RAM por worker, lo que convierte el sistema en inviable sin sharding — este régimen requiere un índice persistente (Pinecone, Weaviate) y está fuera del scope de diseño. Segundo, la NMI requiere el corpus completo para calcular distribuciones marginales conjuntas; la API no puede aproximar $H(\mathbf{V})$ con un corpus parcial sin introducir sesgo de estimación que rompe el invariante de corrección — no existe versión online o incremental de esta primitiva que mantenga la garantía matemática sin almacenamiento de estado.