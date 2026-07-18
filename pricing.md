# Pricing

El modelo de suscripción fija asume que el valor que extraes de una primitiva de similitud es constante mes a mes — pero el patrón de uso real es radicalmente distinto: un pipeline de recomendación lanza ráfagas de comparaciones durante ingesta de nuevo contenido y luego queda silencioso, una búsqueda semántica sobre un corpus editorial pico los lunes y es irrelevante los fines de semana, un sistema de deduplicación corre una vez contra el catálogo completo y no vuelve a ejecutarse hasta el próximo ciclo de actualización. Cobrar un tier fijo en esos patrones significa que el desarrollador paga por capacidad que no consume durante el valle, o se queda corto en el pico y tiene que negociar un upgrade de plan en el peor momento. El pricing por llamada elimina esa fricción estructural: el costo sigue exactamente la curva de valor generado, sin que el desarrollador tenga que predecir su volumen con semanas de antelación.

La tarifa decreciente por volumen no es un descuento de fidelidad — es el reflejo directo de la economía de la primitiva. El costo marginal de ejecutar la calibración automática de alpha vía entropía del corpus, calcular H(X), H(Y) y H(X,Y) sobre el corpus entrante y producir el score compuesto NMI + cosine tiene componentes fijos de arranque por request y componentes variables que escalan con O(n log n) sobre el tamaño del corpus, pero la infraestructura subyacente amortiza mejor a mayor volumen sostenido. Trasladar esa economía al precio es lo técnicamente honesto: quien hace millones de comparaciones al mes está usando la primitiva de forma que permite optimizar rutas de cómputo, cachear distribuciones de entropía intermedias y reducir latencia media — y merece un precio marginal más bajo que refleje esa eficiencia compartida.

Finalmente, la ausencia de compromiso mínimo es parte del diferenciador técnico, no solo comercial. El problema que esta API resuelve existe precisamente porque las alternativas — Pinecone, Weaviate, soluciones vectoriales con estado persistente — requieren onboarding, configuración de índices y contratos antes de que el desarrollador haya validado si la señal NMI + cosine es útil para su corpus específico. Imponer un mínimo mensual reproduce exactamente la barrera que hace que esas alternativas sean inadecuadas para corpora pequeños y medianos. El modelo sin piso permite que un desarrollador evalúe la primitiva contra su corpus real con una sola llamada, confirme que la calibración automática de alpha mejora su métrica de negocio, y escale desde ahí — sin haber asumido ningún riesgo financiero antes de ver el valor.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |