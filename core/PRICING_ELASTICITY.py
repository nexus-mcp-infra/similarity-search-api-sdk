import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import NamedTuple

# Parametros de mercado derivados de developer tools con pricing por operacion
P_MIN = 0.001   # USD por operacion (floor: competencia con FAISS local ~0)
P_MAX = 0.05    # USD por operacion (ceiling: dolor de adopcion en devs indie)
Q_BASE = 50_000 # operaciones/mes baseline para un dev mid-size

# Elasticidad empirica para developer tools: inelastica en pain points, elastica en nice-to-have
# NMI-cosine fusion resuelve pain point real -> elasticidad moderada (-1.8 a -2.5)
EPSILON_REF = -2.1  # punto de referencia: elasticidad precio-demanda del segmento

@dataclass
class DemandPoint:
    price: float
    quantity: float
    revenue: float
    elasticity: float

class AdoptionScenario(NamedTuple):
    name: str
    q_base: float       # operaciones/mes baseline
    price_sensitivity: float  # escala el exponente de elasticidad
    freemium_threshold: int   # ops/mes donde el usuario considera pagar

def nmi_cosine_demand_curve(price: float, q_base: float, epsilon: float, p_ref: float) -> float:
    """
    Q(P) = Q_base * (P / P_ref) ^ epsilon
    Modelo potencia clasico: elasticidad constante en log-log space.
    Apropiado para developer tools donde la decision de adopcion es binaria
    a escala de equipo pero continua a escala de mercado agregado.
    """
    if price <= 0:
        raise ValueError(f"Price must be positive, got {price}")
    return q_base * (price / p_ref) ** epsilon

def point_elasticity(price: float, q_base: float, epsilon: float, p_ref: float) -> float:
    """
    Para demanda potencia Q = Q_base * (P/P_ref)^e, la elasticidad puntual
    es exactamente epsilon en todo el rango — propiedad util para pricing por calls.
    Verificacion analitica: dQ/dP = epsilon * Q_base * (P/P_ref)^(epsilon-1) * (1/P_ref)
    => (dQ/dP)*(P/Q) = epsilon. Consistencia garantizada.
    """
    q = nmi_cosine_demand_curve(price, q_base, epsilon, p_ref)
    dq_dp = epsilon * q_base * (price / p_ref) ** (epsilon - 1) / p_ref
    return dq_dp * (price / q)

def revenue(price: float, q_base: float, epsilon: float, p_ref: float) -> float:
    """R(P) = P * Q(P) — objetivo de maximizacion directo."""
    return price * nmi_cosine_demand_curve(price, q_base, epsilon, p_ref)

def optimal_price(q_base: float, epsilon: float, p_ref: float) -> DemandPoint:
    """
    Maximiza R(P) = P * Q_base * (P/P_ref)^epsilon en [P_MIN, P_MAX].
    Analitico: dR/dP = 0 => P* = P_ref * (1 / (1 + epsilon)) si epsilon < -1.
    Resolvemos numericamente para respetar los bounds del mercado real.
    """
    if epsilon >= -1:
        raise ValueError(f"Elasticity {epsilon} implies no finite optimum; must be < -1")

    result = minimize_scalar(
        lambda p: -revenue(p, q_base, epsilon, p_ref),
        bounds=(P_MIN, P_MAX),
        method="bounded"
    )
    p_star = result.x
    q_star = nmi_cosine_demand_curve(p_star, q_base, epsilon, p_ref)
    r_star = p_star * q_star
    e_star = point_elasticity(p_star, q_base, epsilon, p_ref)
    return DemandPoint(price=p_star, quantity=q_star, revenue=r_star, elasticity=e_star)

def freemium_to_paid_equilibrium(scenario: AdoptionScenario, p_paid: float, epsilon: float, p_ref: float) -> dict:
    """
    Punto de equilibrio: el usuario migra a paid cuando el costo de oportunidad
    de la friccion freemium (rate limiting, latencia degradada) supera p_paid * Q.
    Modelamos friction_cost = alpha * Q^beta donde alpha calibra el pain point NMI-cosine
    (usuarios con distribuciones sesgadas sienten el fallo de coseno puro antes).
    Equilibrio: friction_cost(Q*) = p_paid * Q* => Q* resuelto iterativamente.
    """
    alpha = 0.0008  # costo de friccion por op en freemium (estimado: reintento + ingenieria)
    beta = 1.15     # superlinear: a mayor volumen, mas doloroso el workaround manual

    q_paid = nmi_cosine_demand_curve(p_paid, scenario.q_base, epsilon, p_ref)
    friction_at_threshold = alpha * (scenario.freemium_threshold ** beta)
    paid_cost_at_threshold = p_paid * scenario.freemium_threshold

    # Q_eq donde friction_cost == paid_cost
    q_eq = (alpha / p_paid) ** (1 / (1 - beta))  # solucion analitica de alpha*Q^beta = p*Q

    return {
        "scenario": scenario.name,
        "q_paid_demand": round(q_paid),
        "freemium_threshold_ops": scenario.freemium_threshold,
        "friction_cost_at_threshold_usd": round(friction_at_threshold, 4),
        "paid_cost_at_threshold_usd": round(paid_cost_at_threshold, 4),
        "equilibrium_ops_per_month": round(q_eq),
        "converts_at_threshold": friction_at_threshold >= paid_cost_at_threshold,
    }

# EXACTAMENTE 3 escenarios de adopcion para NMI-cosine Similarity Search API
ADOPTION_SCENARIOS = [
    AdoptionScenario(
        name="indie_mvp",           # dev solo, MVP con ~500-5K ops/dia
        q_base=3_000,
        price_sensitivity=2.4,      # muy elastico: presupuesto limitado
        freemium_threshold=500,
    ),
    AdoptionScenario(
        name="startup_semantic_search",  # equipo 2-10, feature de busqueda en producto
        q_base=120_000,
        price_sensitivity=1.9,           # moderado: valora stateless sobre setup VectorDB
        freemium_threshold=10_000,
    ),
    AdoptionScenario(
        name="enterprise_reranking_pipeline",  # ML team, NMI-cosine como reranker en RAG
        q_base=2_500_000,
        price_sensitivity=1.4,                 # inelastico: precision > costo marginal
        freemium_threshold=100_000,
    ),
]

def simulate_scenario(scenario: AdoptionScenario, p_ref: float = 0.01) -> dict:
    epsilon = -scenario.price_sensitivity  # negativo: ley de demanda
    opt = optimal_price(scenario.q_base, epsilon, p_ref)
    eq = freemium_to_paid_equilibrium(scenario, opt.price, epsilon, p_ref)

    price_sweep = np.linspace(P_MIN, P_MAX, 200)
    revenues = [revenue(p, scenario.q_base, epsilon, p_ref) for p in price_sweep]
    max_rev_idx = int(np.argmax(revenues))

    return {
        "scenario": scenario.name,
        "optimal_price_usd_per_op": round(opt.price, 5),
        "demand_at_optimal_ops_per_month": round(opt.quantity),
        "max_monthly_revenue_usd": round(opt.revenue, 2),
        "elasticity_at_optimum": round(opt.elasticity, 3),
        "price_at_sweep_max": round(price_sweep[max_rev_idx], 5),  # validacion cruzada
        "freemium_equilibrium": eq,
    }

if __name__ == "__main__":
    import json
    results = [simulate_scenario(s) for s in ADOPTION_SCENARIOS]
    print(json.dumps(results, indent=2))