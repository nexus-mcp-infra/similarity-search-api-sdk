# Pricing

El modelo de suscripción fija por tiers asume que el desarrollador sabe de antemano cuántas llamadas va a necesitar, en qué frecuencia y para qué volumen de datos — exactamente lo que no ocurre cuando el caso de uso es búsqueda de similitud ad-hoc sobre payloads sin estado. La Similarity Search API está diseñada precisamente para el escenario donde no existe una base de datos que gestionar ni un índice que mantener: cada llamada es autónoma, el costo computacional está determinado por el tamaño del payload y la entropía marginal de sus dimensiones, y ese costo varía con cada request. Cobrar una tarifa plana mensual sobre ese patrón de uso significaría subsidiar a quien sobreestima su consumo y penalizar a quien lo usa de forma puntual e intensa — ninguno de los dos escenarios refleja el valor real entregado por llamada.

El cálculo NMI+Cosine ponderado por entropía no tiene costo fijo: el trabajo computacional crece con la complejidad distribucional del input, no con el tiempo transcurrido ni con asientos activos. Un tier de suscripción desconecta ese costo real del precio cobrado, creando un modelo donde el proveedor asume el riesgo de los patrones de uso más pesados sin capturar el valor correspondiente. Usage-based con tarifa decreciente por volumen resuelve esto con precisión quirúrgica: cada llamada se paga por lo que consume, y el precio marginal cae a medida que el volumen acumula señal estadística de que el uso es sostenido y predecible — lo que reduce el costo operativo de servir ese cliente y justifica transferirle parte de ese ahorro.

El descuento por volumen sin compromiso mínimo no es generosidad comercial: es el mecanismo que elimina la fricción de adopción inicial. Un desarrollador que evalúa si la fusión NMI+Cosine resuelve su caso de uso mixto no va a comprometerse a un tier mensual antes de validar el score en sus propios datos. Con tarifa decreciente y sin mínimo, el primer experimento cuesta exactamente lo que vale una llamada, y si el patrón escala, el precio por unidad baja solo. Eso alinea el incentivo del desarrollador con el del proveedor: más uso produce mejor economía para ambos, sin negociación de contratos, sin cambio de plan y sin techo artificial que fuerce una conversación de ventas cuando el volumen crece.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |