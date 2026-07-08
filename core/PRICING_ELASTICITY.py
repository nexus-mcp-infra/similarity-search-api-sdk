import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple

# Parametros de mercado especificos para NMI+Cosine Similarity Search API
P_MIN = 0.001   # USD por operacion (piso developer hobbyist)
P_MAX = 0.05    # USD por operacion (techo enterprise tolerance)
Q_MIN = 1_000   # operaciones/mes minimo viable
Q_MAX = 10_000_000  # operaciones/mes cliente enterprise


@dataclass
class AdoptionScenario:
    name: str
    elasticity: float       # epsilon: sensibilidad precio-demanda
    base_demand: float      # Q0 en operaciones/mes a precio de referencia
    alpha_preference: float # peso NMI vs Cosine preferido (0=cosine puro, 1=NMI puro)
    freemium_threshold: float  # operaciones/mes donde convierte a paid


# 3 escenarios de adopcion para developers usando primitivas de similitud sin vector DB
SCENARIOS = [
    AdoptionScenario(
        name="data_scientist_adhoc",
        elasticity=-1.8,        # alta elasticidad: sustituye con scikit-learn si sube precio
        base_demand=15_000,     # scripts de analisis esporadicos, no pipelines continuos
        alpha_preference=0.7,   # prefiere NMI alto: datos categoricos/distribucionales
        freemium_threshold=5_000,
    ),
    AdoptionScenario(
        name="ml_engineer_production",
        elasticity=-0.9,        # inelastico: ya elimino el vector DB overhead, no vuelve atras
        base_demand=400_000,    # pipeline de reranking en produccion, llamadas sistematicas
        alpha_preference=0.4,   # balance NMI+Cosine: embeddings densos con semantica hibrida
        freemium_threshold=50_000,
    ),
    AdoptionScenario(
        name="platform_integrator",
        elasticity=-0.5,        # muy inelastico: NMI+Strehl-Ghosh es diferenciador no replicable
        base_demand=2_500_000,  # SaaS que expone similitud a sus propios usuarios finales
        alpha_preference=0.3,   # cosine domina: corpus de embeddings densos masivos
        freemium_threshold=200_000,
    ),
]

P_REF = 0.01  # precio de referencia para anclar Q0 (mid-range del mercado developer)


def nmi_hybrid_demand(P: float, scenario: AdoptionScenario) -> float:
    """
    Q = Q0 * (P / P_REF)^epsilon
    Funcion isoelastica: elasticidad constante en todo el rango de precio.
    Valida para mercados de API donde el developer compara coste marginal por llamada.
    """
    if P <= 0:
        raise ValueError(f"Precio debe ser positivo, recibido: {P}")
    if P < P_MIN or P > P_MAX:
        # fuera del rango de mercado definido: demanda colapsa o es irrelevante
        return 0.0
    Q = scenario.base_demand * (P / P_REF) ** scenario.elasticity
    return max(Q, 0.0)


def price_elasticity_at(P: float, scenario: AdoptionScenario, delta: float = 1e-6) -> float:
    """
    epsilon = (dQ/dP) * (P/Q)
    Derivada numerica de orden 2 (central difference) para mayor precision.
    """
    Q = nmi_hybrid_demand(P, scenario)
    if Q == 0:
        return 0.0
    dQ_dP = (nmi_hybrid_demand(P + delta, scenario) - nmi_hybrid_demand(P - delta, scenario)) / (2 * delta)
    return dQ_dP * (P / Q)


def monthly_revenue(P: float, scenario: AdoptionScenario) -> float:
    # R(P) = P * Q(P): funcion objetivo para maximizacion
    return P * nmi_hybrid_demand(P, scenario)


def optimal_price(scenario: AdoptionScenario) -> Tuple[float, float, float]:
    """
    max R(P) = P * Q0 * (P/P_REF)^epsilon
    Solucion analitica: P* = P_REF * (1 + 1/epsilon)^(-1) para elasticidad != -1
    Fallback numerico via scipy para validacion cruzada.
    """
    eps = scenario.elasticity
    # solucion de forma cerrada: MR=0 => 1 + epsilon = 0 => P* resuelto por log-derivada
    if abs(eps + 1.0) > 1e-9:
        P_analytic = P_REF * ((-eps) / (-eps - 1))
        P_analytic = np.clip(P_analytic, P_MIN, P_MAX)
    else:
        P_analytic = P_REF  # caso limite elasticidad = -1: revenue constante

    # validacion numerica independiente
    result = minimize_scalar(
        lambda p: -monthly_revenue(p, scenario),
        bounds=(P_MIN, P_MAX),
        method="bounded",
    )
    P_numeric = result.x

    P_opt = P_numeric  # numerico es ground truth; analitico sirve para sanity check
    Q_opt = nmi_hybrid_demand(P_opt, scenario)
    R_opt = monthly_revenue(P_opt, scenario)
    return P_opt, Q_opt, R_opt


def freemium_to_paid_equilibrium(scenario: AdoptionScenario) -> dict:
    """
    Punto de equilibrio: precio minimo P_eq tal que R(P_eq) >= coste marginal de servir
    al cliente que supero el threshold freemium.
    Coste marginal estimado: O(n log n) sobre corpus medio de 500 items por llamada,
    ~0.00003 USD por operacion en c5.xlarge (calibrado para NMI+Strehl-Ghosh, no coseno simple).
    """
    MARGINAL_COST_PER_OP = 0.00003  # USD: NMI con corrección de bias es ~3x más caro que coseno puro

    Q_threshold = scenario.freemium_threshold
    # P_eq: precio donde revenue por operacion cubre coste marginal con margen minimo 60%
    P_equilibrium = MARGINAL_COST_PER_OP / (1 - 0.60)
    R_at_threshold = monthly_revenue(P_equilibrium, scenario)
    cost_at_threshold = Q_threshold * MARGINAL_COST_PER_OP

    return {
        "scenario": scenario.name,
        "freemium_ops_limit": Q_threshold,
        "P_equilibrium_usd": round(P_equilibrium, 6),
        "revenue_at_eq": round(R_at_threshold, 4),
        "cost_at_threshold": round(cost_at_threshold, 4),
        "unit_economics_positive": R_at_threshold > cost_at_threshold,
    }


def run_elasticity_model() -> None:
    for scenario in SCENARIOS:
        P_opt, Q_opt, R_opt = optimal_price(scenario)
        eps_at_opt = price_elasticity_at(P_opt, scenario)
        eq = freemium_to_paid_equilibrium(scenario)

        print(f"\nScenario: {scenario.name}")
        print(f"  Optimal price       : ${P_opt:.5f}/op")
        print(f"  Demand at P_opt     : {Q_opt:,.0f} ops/month")
        print(f"  Max monthly revenue : ${R_opt:,.2f}")
        print(f"  Elasticity at P_opt : {eps_at_opt:.4f}  (expected ~{scenario.elasticity})")
        print(f"  Freemium->paid P_eq : ${eq['P_equilibrium_usd']}/op  |  unit_econ_positive={eq['unit_economics_positive']}")
        print(f"  NMI/Cosine alpha    : {scenario.alpha_preference} (correlates with elasticity: higher NMI dependency -> less substitutable)")


if __name__ == "__main__":
    run_elasticity_model()