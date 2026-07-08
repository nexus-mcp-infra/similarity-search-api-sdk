# Pricing

El modelo de suscripción fija presupone que el desarrollador puede predecir con antelación cuántas búsquedas va a ejecutar en un período dado. Pero el caso de uso central de esta API —análisis ad-hoc sobre corpus variables, scripts de exploración, pipelines de enriquecimiento que se disparan por eventos— es estructuralmente impredecible: un investigador puede no hacer ninguna llamada durante semanas y luego lanzar miles en una tarde. Forzar ese patrón dentro de un tier mensual significa que el usuario paga por capacidad que no consume, o que se frena antes de completar un análisis porque se aproxima al límite. Ninguno de esos dos comportamientos es aceptable para una primitiva de infraestructura cuyo valor es precisamente eliminar la fricción de montar un sistema de similitud propio.

La tarifa decreciente por volumen existe porque el costo marginal real de cada llamada no es lineal desde la perspectiva operativa: la infraestructura de estimación de distribuciones conjuntas, la corrección de bias Strehl-Ghosh y la fusión NMI+Cosine tienen un costo fijo de inicialización por request que se amortiza mejor en lotes grandes. Trasladar esa economía al precio no es un descuento de marketing, es una señal honesta de la estructura de costos subyacente. Un usuario que envía un corpus de varios miles de items en una sola llamada está aprovechando la complejidad O(n log n) de forma más eficiente que quien hace llamadas unitarias sucesivas, y el precio debe reflejar esa diferencia sin crear categorías artificiales que obliguen a reorganizar el código para "entrar en el tier correcto".

La ausencia de compromiso mínimo es consecuencia directa del perfil del adoptante temprano de esta primitiva: equipos de ML que están evaluando si NMI como métrica de ranking mejora sus resultados frente a coseno puro, o desarrolladores que necesitan similitud semántica en un microservicio sin justificar la infraestructura de un vector store. Pedirle a ese perfil que firme un volumen mínimo antes de haber validado la utilidad técnica es el mecanismo más rápido para perder esa adopción. El crecimiento de uso debe ser consecuencia de que la primitiva resuelve el problema —rankings correctos donde NMI sin corrección de bias los distorsiona— no de un contrato que lo garantice artificialmente.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |