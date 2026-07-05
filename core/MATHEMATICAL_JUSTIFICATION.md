# Justificación Matemática: Similarity Search API con Scoring NMI-Cosine

## 1. Máximo 5 endpoints (Hick's Law)

El tiempo de decisión de integración sigue $T = b \cdot \log_2(n+1)$, donde $n$ es el número de opciones disponibles. Con 5 endpoints, $T \approx b \cdot 2.58$; con 10, $T \approx b \cdot 3.46$ — un 34% de fricción cognitiva adicional sin ganancia funcional. Para una primitiva stateless cuyo valor está en el cálculo, no en la superficie, cada endpoint adicional más allá de 5 reduce la tasa de adopción sin aumentar el moat técnico.

## 2. Pricing per-call vs. por asiento (elasticidad precio-demanda)

La elasticidad cruzada del uso real es $E = \frac{\Delta Q / Q}{\Delta P / P}$, y para workloads de script one-shot (un dolor explícito del developer: "consulto vectores una vez") la demanda es altamente elástica al coste fijo: un fee mensual convierte el caso de uso en irracional económicamente. El modelo per-call alinea el precio con $C_{marginal}$ real del cómputo NMI+bootstrap, que es $O(n \cdot B \cdot d)$ donde $B$ es el número de réplicas bootstrap y $d$ la dimensionalidad — coste variable que no debe amortizarse sobre asientos inactivos.

## 3. Estructura de datos y complejidad algorítmica

El estimador de entropía conjunta discreta $H(X,Y) = -\sum_{x,y} p(x,y) \log p(x,y)$ requiere una tabla de contingencia sparse de dimensiones $k_X \times k_Y$, donde $k_i$ es el número de bins adaptativos por dimensión según Freedman-Diaconis: $k_i = \lceil (x_{max} - x_{min}) / (2 \cdot IQR_i \cdot n^{-1/3}) \rceil$. Almacenarla como `scipy.sparse.csr_matrix` reduce la complejidad espacial de $O(k^2)$ a $O(nnz)$ donde $nnz \ll k^2$ en embeddings de alta dimensión con distribuciones concentradas — crítico porque un embedding de 1536 dimensiones con $k=15$ bins generaría $15^2 = 225$ celdas por par dimensional, inmanejable en denso para comparaciones batch.

## 4. Invariante matemático de corrección

El invariante que garantiza que el NMI reportado es una métrica probabilística válida — y no un número arbitrario — es la normalización estricta $NMI(X,Y) = \frac{2 \cdot I(X;Y)}{H(X) + H(Y)} \in [0,1]$, donde $I(X;Y) = H(X) + H(Y) - H(X,Y) \geq 0$ por la desigualdad de Jensen aplicada a la divergencia KL. El p-value bootstrap es válido si y solo si los bins adaptativos Freedman-Diaconis eliminan el sesgo de estimación por dimensiones de baja varianza — bins fijos rompen este invariante porque inflan $I(X;Y)$ artificialmente en dimensiones constantes, haciendo que $NMI \to 1$ por artefacto de discretización, no por dependencia real.

## 5. Límites teóricos del sistema

El estimador NMI converge al valor poblacional con error $O(k^2 / n)$ según la expansión de Miller-Madow, lo que impone un límite inferior práctico de $n \geq 30$ vectores por llamada para que el intervalo de confianza bootstrap sea informativo — con $n < 10$ el p-value no tiene potencia estadística suficiente para rechazar $H_0: I(X;Y)=0$. El sistema tampoco puede detectar dependencias no lineales que no sobrevivan a la discretización en bins marginales: relaciones del tipo $Y = f(X)$ con $f$ altamente no monótona y localizada requieren estimadores de entropía de kernel (KDE), fuera del alcance computacional de una llamada stateless con SLA de latencia acotada.