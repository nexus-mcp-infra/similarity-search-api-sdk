# Pricing

El modelo de tarifa decreciente por volumen existe porque el patrón de consumo de una primitiva stateless como esta es fundamentalmente diferente al de un servicio con estado. Quien adopta esta API no tiene infraestructura que mantener ni índice que sincronizar: cada llamada es independiente, lo que significa que el volumen de uso puede crecer de forma no lineal y no planificada. Un equipo puede hacer diez llamadas en desarrollo, diez mil durante un experimento de relevancia, y volver a cero la semana siguiente. Obligarles a comprometerse con un tier mensual fijo castiga exactamente el comportamiento que hace valiosa una primitiva stateless: la capacidad de escalar y desescalar sin fricción operativa. El compromiso mínimo convierte una ventaja arquitectónica en una penalización financiera.

La tarifa decreciente refleja además la estructura real de costes computacionales de esta primitiva. El cálculo del score híbrido NMI+Cosine con calibración dinámica de pesos y bootstrap de intervalos de confianza sobre quinientos remuestreos tiene un coste fijo de inicialización por llamada que se amortiza a medida que el cliente genera volumen. A bajo volumen ese coste fijo pesa más por unidad; a alto volumen, menos. El pricing que no refleja esa curva de amortización o bien cobra de más al cliente en volumen alto, o bien subsidia el volumen bajo con márgenes insostenibles. La tarifa decreciente alinea el precio con la economía real del cálculo, no con una ficción administrativa de tiers.

Finalmente, el modelo sin techo ni compromiso mínimo es coherente con el flywheel técnico que alimenta la primitiva. Cada llamada en producción contribuye a las distribuciones de scores acumuladas en ClickHouse, que permiten recalibrar los pesos default del score híbrido con datos reales por dominio. Limitar artificialmente el volumen con un techo o desincentivar el uso bajo con un mínimo obligatorio frenaría precisamente el flujo de datos que mejora la calibración para todos los clientes. El pricing abierto y proporcional al uso es, en este caso, también la política que maximiza la velocidad de mejora del producto: más llamadas reales significan mejores pesos default, que producen scores más precisos, que generan más adopción.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |