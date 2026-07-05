# Justificación Matemática: Similarity Search API Híbrida NMI+Coseno

## 1. Por qué máximo 5 endpoints (Hick's Law)

Hick's Law establece $T = b \cdot \log_2(n+1)$: el tiempo de decisión crece logarítmicamente con el número de opciones. Para una API stateless donde el integrador debe elegir el endpoint correcto sin documentación contextual, cada endpoint adicional aumenta la fricción cognitiva de adopción. Con 5 endpoints el término $\log_2(6) \approx 2.58$ mantiene la decisión en menos de 3 bits de información — el umbral empírico donde la elección sigue siendo automática y no deliberativa para un desarrollador con contexto de búsqueda.

## 2. Por qué pricing per-call vs por asiento (elasticidad precio-demanda)

La elasticidad-precio de una API de búsqueda stateless es alta en volumen bajo y baja en volumen alto: $\varepsilon = \partial Q / \partial P \cdot P/Q$. El pricing por asiento introduce un costo fijo que destruye el excedente del consumidor precisamente en el segmento de corpus pequeño ($< 100\text{k}$ items) donde el dolor existe — esos usuarios abandonan antes de validar el valor. El modelo per-call convierte el precio en variable proporcional al valor recibido, maximizando el área bajo la curva de demanda y capturando el segmento que Pinecone/Weaviate pierde por sobredimensionamiento.

## 3. Por qué esta estructura de datos (complejidad algorítmica)

El scoring híbrido opera sobre payloads arbitrarios en memoria sin índice persistente: la complejidad por consulta es $O(n \cdot d)$ donde $n$ es el tamaño del corpus y $d$ es la dimensionalidad efectiva. Para $n < 100\text{k}$ y $d < 512$, el producto $n \cdot d < 5 \times 10^7$ operaciones — ejecutable en $< 200\text{ms}$ con NumPy vectorizado sin estructuras de índice auxiliares. Mantener los vectores continuos en `float32` y las distribuciones categóricas como histogramas de frecuencia normalizados en `dict[str, float]` minimiza la huella de memoria y elimina el overhead de serialización que impondría un índice persistente como HNSW.

## 4. El invariante matemático que hace esta solución correcta

El score híbrido $S = w_{nmi} \cdot \text{NMI}(X,Y) + (1 - w_{nmi}) \cdot \cos(\mathbf{u}, \mathbf{v})$ es correcto porque ambos componentes están acotados en $[0, 1]$: NMI por definición ($\text{NMI} = 2I(X;Y)/(H(X)+H(Y))$, donde $I \leq \min(H(X),H(Y))$) y coseno normalizado por construcción. El peso $w_{nmi} = \sum H_{cat} / (\sum H_{cat} + \sum H_{cont})$ también vive en $[0,1]$, garantizando que $S \in [0,1]$ para cualquier composición de payload sin requerir renormalización post-hoc. Este invariante hace el score directamente comparable entre queries con distinta composición de features.

## 5. Límites teóricos del sistema (qué no puede hacer y por qué)

El sistema es estadísticamente ciego a dependencias de orden superior entre features: NMI captura dependencia mutua par a par ($I(X;Y)$) pero no interacciones condicionales del tipo $I(X;Y|Z)$, lo que lo hace incorrecto para dominios donde la similitud emerge de correlaciones entre tres o más variables simultáneamente. El umbral de entropía $H < 1.5$ bits como clasificador categórico/continuo es una heurística calibrada empíricamente, no un clasificador Bayes-óptimo — features ordinales con cardinalidad media (ej. ratings 1–10) caerán en la región de ambigüedad $H \approx 3.3$ bits y serán enrutadas como continuas, perdiendo la invariancia a cardinalidad que NMI garantiza. Finalmente, la complejidad $O(n \cdot d)$ establece un techo duro alrededor de $n = 500\text{k}$ items antes de que la latencia supere los SLAs de uso interactivo ($< 500\text{ms}$) — escalar más allá requiere índices aproximados (HNSW, IVF) que rompen la promesa stateless.