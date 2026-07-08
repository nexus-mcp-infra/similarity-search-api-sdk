import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import NamedTuple

# --- Parámetros de mercado calibrados al segmento developer tools ---
P_MIN = 0.001   # USD por operación, floor observado en vector DB competitors
P_MAX = 0.05    # USD por operación, ceiling willingness-to-pay developers
Q_BASELINE = 500_000  # operaciones/mes punto de referencia (mid-market team)

# Elasticidad empírica estimada para dev tools SaaS per-call: rango -1.2 a -2.5
# NMI+coseno es diferenciador con sustitutos imperfectos -> elasticidad moderada
EPSILON_BASE = -1.6

# --- Función de demanda con pendiente log-lineal, estándar en SaaS per-call ---
# Q(P) = Q_baseline * (P / P_ref)^epsilon
# Forma power law: elasticidad constante, apropiada cuando no hay coste de switching fijo

P_REF = 0.008  # precio de referencia: midpoint del rango ajustado por competencia

def stateless_similarity_demand(price: float, epsilon: float = EPSILON_BASE, q_ref: float = Q_BASELINE) -> float:
    # Demanda en operaciones/mes dado un precio por operación
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")
    return q_ref * (price / P_REF) ** epsilon

def point_elasticity(price: float, epsilon: float = EPSILON_BASE) -> float:
    # En modelo power law, elasticidad puntual == epsilon por construcción
    # Verificación analítica: dQ/dP * P/Q = epsilon * Q/P * P/Q = epsilon
    return epsilon  # constante en este modelo; varía si se usa forma logística

def monthly_revenue(price: float, epsilon: float = EPSILON_BASE, q_ref: float = Q_BASELINE) -> float:
    # R(P) = P * Q(P); maximizar sobre P en [P_MIN, P_MAX]
    return price * stateless_similarity_demand(price, epsilon, q_ref)

def optimal_price_nmi_cosine_api(
    epsilon: float = EPSILON_BASE,
    q_ref: float = Q_BASELINE,
    p_bounds: tuple = (P_MIN, P_MAX)
) -> dict:
    # Maximización numérica porque el óptimo analítico P* = P_ref*(epsilon/(1+epsilon))
    # puede quedar fuera del rango de mercado -> clip necesario
    analytical_optimum = P_REF * (epsilon / (1 + epsilon))  # negativo/infinito si epsilon=-1

    result = minimize_scalar(
        lambda p: -monthly_revenue(p, epsilon, q_ref),
        bounds=p_bounds,
        method='bounded'
    )
    p_opt = result.x
    q_opt = stateless_similarity_demand(p_opt, epsilon, q_ref)
    r_opt = p_opt * q_opt

    return {
        "optimal_price_usd": round(p_opt, 5),
        "optimal_demand_ops_month": round(q_opt),
        "optimal_revenue_usd_month": round(r_opt, 2),
        "analytical_unconstrained_optimum": round(analytical_optimum, 5),
        "elasticity_at_optimum": point_elasticity(p_opt, epsilon),
        "revenue_maximizing_condition": "MR=0 -> epsilon*(1+epsilon)^-1 constrained to market bounds"
    }

# --- 3 escenarios de adopción: segmentos reales del mercado developer ---

@dataclass
class AdoptionScenario:
    name: str
    q_ref: float      # volumen base operaciones/mes
    epsilon: float    # elasticidad propia del segmento
    freemium_ops: int # operaciones gratuitas/mes en tier free
    conversion_rate: float  # tasa freemium->paid empírica dev tools: 2-8%

SCENARIOS = [
    AdoptionScenario(
        name="indie_developer",
        q_ref=3_000,           # 1K-10K ops/mes: side project o prototipo
        epsilon=-2.1,          # alta sensibilidad precio: sustituye con coseno puro si caro
        freemium_ops=1_000,
        conversion_rate=0.03   # 3% conversión, referencia: Pinecone free tier
    ),
    AdoptionScenario(
        name="growth_startup",
        q_ref=150_000,         # 50K-500K ops/mes: producto en producción temprana
        epsilon=-1.6,          # elasticidad base: tiene presupuesto pero compara alternativas
        freemium_ops=10_000,
        conversion_rate=0.055  # 5.5%: ya integró NMI+coseno en pipeline, switching cost real
    ),
    AdoptionScenario(
        name="mid_market_platform",
        q_ref=4_000_000,       # 1M-10M ops/mes: plataforma con corpus efímero recurrente
        epsilon=-1.1,          # baja elasticidad: NMI+coseno sin estado es diferenciador sin sustituto
        freemium_ops=50_000,
        conversion_rate=0.07   # 7%: evaluación técnica formal -> conversión más alta
    ),
]

class ScenarioResult(NamedTuple):
    name: str
    optimal_price: float
    peak_revenue_month: float
    freemium_breakeven_price: float  # precio donde pagar < coste de superar free tier
    paid_volume_at_optimal: float

def simulate_adoption_scenario(scenario: AdoptionScenario) -> ScenarioResult:
    opt = optimal_price_nmi_cosine_api(
        epsilon=scenario.epsilon,
        q_ref=scenario.q_ref
    )
    p_opt = opt["optimal_price_usd"]
    r_peak = opt["optimal_revenue_usd_month"]

    # Punto freemium->paid: precio donde el coste mensual iguala valor marginal del volumen incremental
    # Q_free es gratis; developer paga solo si Q(P)*P < perceived_value
    # Simplificado: breakeven donde revenue de operaciones sobre free_tier = 0 -> P tal que
    # Q(P) == freemium_ops (demanda cae al nivel del tier gratuito)
    # P_breakeven = P_REF * (freemium_ops / scenario.q_ref)^(1/epsilon)
    if scenario.epsilon != 0:
        p_breakeven = P_REF * (scenario.freemium_ops / scenario.q_ref) ** (1.0 / scenario.epsilon)
    else:
        p_breakeven = float('inf')

    # Clip al rango de mercado real
    p_breakeven = float(np.clip(p_breakeven, P_MIN, P_MAX))

    q_paid = stateless_similarity_demand(p_opt, scenario.epsilon, scenario.q_ref)

    return ScenarioResult(
        name=scenario.name,
        optimal_price=round(p_opt, 5),
        peak_revenue_month=round(r_peak, 2),
        freemium_breakeven_price=round(p_breakeven, 5),
        paid_volume_at_optimal=round(q_paid)
    )

# --- Equilibrio freemium -> paid usando condición de indiferencia ---

def freemium_paid_equilibrium(
    scenario: AdoptionScenario,
    cost_per_op_usd: float = 0.00015  # coste operativo estimado: KDE+NMI O(n*d), n=1K, d=768
) -> dict:
    # Developer convierte cuando: valor_NMI_coseno > precio AND volumen > free_tier
    # Condición de equilibrio: P* tal que (P - cost) * converted_volume = 0 en margen
    # converted_volume = scenario.q_ref * scenario.conversion_rate
    converted_ops = scenario.q_ref * scenario.conversion_rate
    p_opt_result = optimal_price_nmi_cosine_api(scenario.epsilon, scenario.q_ref)
    p_opt = p_opt_result["optimal_price_usd"]

    margin_per_op = p_opt - cost_per_op_usd
    monthly_margin = margin_per_op * converted_ops

    # Precio mínimo viable: donde margen cubre coste fijo de infraestructura (estimado $200/mes)
    infra_fixed_cost = 200.0
    p_floor_viable = cost_per_op_usd + infra_fixed_cost / max(converted_ops, 1)

    return {
        "scenario": scenario.name,
        "converted_ops_month": round(converted_ops),
        "optimal_price": p_opt,
        "cost_per_op": cost_per_op_usd,
        "margin_per_op": round(margin_per_op, 5),
        "monthly_margin_usd": round(monthly_margin, 2),
        "minimum_viable_price": round(float(np.clip(p_floor_viable, P_MIN, P_MAX)), 5),
        "freemium_equilibrium": "convert when p_opt > cost_per_op AND ops > freemium_ops"
    }

# --- Ejecución del modelo completo ---

if __name__ == "__main__":
    import json

    market_optimum = optimal_price_nmi_cosine_api()
    print("MARKET OPTIMUM:", json.dumps(market_optimum, indent=2))

    for scenario in SCENARIOS:
        result = simulate_adoption_scenario(scenario)
        equilibrium = freemium_paid_equilibrium(scenario)
        print(f"\nSCENARIO [{result.name}]")
        print("  adoption:", result._asdict())
        print("  equilibrium:", equilibrium)