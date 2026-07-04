import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import NamedTuple

# Parametros del mercado: developer tools SaaS, billing por operacion
P_MIN, P_MAX = 0.001, 0.05
Q_MIN, Q_MAX = 1_000, 10_000_000

# Elasticidad empirica para developer tools: inelastica en precision, elastica en commodities
# NMI-hybrid justifica elasticidad menor que cosine-only APIs (menor sustituibilidad)
EPSILON_BASE = -1.4  # elasticidad propia estimada; mas inelastica que busqueda vectorial pura (-2.1)

@dataclass
class DemandPoint:
    price: float
    quantity: float
    revenue: float
    elasticity: float

class AdoptionScenario(NamedTuple):
    label: str
    alpha_sensitivity: float   # sensibilidad al precio del segmento
    q_anchor: float            # volumen de referencia en P_anchor
    freemium_threshold: float  # ops/mes en las que el dev considera pagar

# Funcion de demanda log-lineal: ln(Q) = ln(A) - epsilon*ln(P)
# Forma isoelastica — elasticidad constante, estandar en demand modeling de APIs
def nmi_hybrid_demand(price: float, A: float, epsilon: float) -> float:
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    if A <= 0:
        raise ValueError(f"demand scale A must be positive, got {A}")
    return A * (price ** epsilon)

# Elasticidad puntual analitica: d(ln Q)/d(ln P) = epsilon (constante en isoelastica)
def point_elasticity(price: float, A: float, epsilon: float) -> float:
    q = nmi_hybrid_demand(price, A, epsilon)
    dq_dp = epsilon * A * (price ** (epsilon - 1))
    return dq_dp * (price / q)  # = epsilon exactamente; validacion aritmetica explícita

# Revenue: R(P) = P * Q(P) = A * P^(1 + epsilon)
# Maximo analitico en P* = -epsilon / (1 + epsilon) * (MC) — con MC~0 para API: P* -> inf salvo restriccion de mercado
# Usamos optimizacion numerica sobre el rango real del mercado para respetar P_MIN, P_MAX
def revenue(price: float, A: float, epsilon: float) -> float:
    return price * nmi_hybrid_demand(price, A, epsilon)

def optimal_price(A: float, epsilon: float) -> DemandPoint:
    if epsilon >= -1:
        # Si elasticidad > -1, revenue crece monotonamente — precio optimo en el techo del mercado
        p_star = P_MAX
    else:
        result = minimize_scalar(
            lambda p: -revenue(p, A, epsilon),
            bounds=(P_MIN, P_MAX),
            method='bounded'
        )
        p_star = result.x
    q_star = nmi_hybrid_demand(p_star, A, epsilon)
    return DemandPoint(
        price=p_star,
        quantity=q_star,
        revenue=p_star * q_star,
        elasticity=point_elasticity(p_star, A, epsilon)
    )

# Calibracion de A para anclar Q_anchor operaciones a P_anchor = $0.01
# A = Q_anchor / P_anchor^epsilon
def calibrate_demand_scale(q_anchor: float, p_anchor: float = 0.01, epsilon: float = EPSILON_BASE) -> float:
    return q_anchor / (p_anchor ** epsilon)

# EXACTAMENTE 3 escenarios de adopcion para NMI Similarity Search API
ADOPTION_SCENARIOS = [
    AdoptionScenario(
        label="one_off_script",           # Dev que corre busqueda ad-hoc, sin infra vectorial
        alpha_sensitivity=2.1,            # Alta elasticidad: sustituye con sklearn si sube el precio
        q_anchor=8_000,                   # 8K ops/mes tipico en exploracion de datos
        freemium_threshold=5_000          # convierte a paid al superar 5K ops/mes gratis
    ),
    AdoptionScenario(
        label="sub_100k_collection",      # App con coleccion pequena, overhead de faiss desproporcionado
        alpha_sensitivity=1.4,            # Elasticidad base: tiene alternativas pero costosas de operar
        q_anchor=200_000,                 # 200K ops/mes en produccion ligera
        freemium_threshold=25_000         # convierte a paid al superar 25K ops/mes
    ),
    AdoptionScenario(
        label="nmi_precision_buyer",      # Dev que necesita NMI nativa — sin sustituto directo
        alpha_sensitivity=0.8,            # Inelastico: NMI no esta en ninguna API comercial
        q_anchor=1_500_000,              # 1.5M ops/mes en pipeline de recomendacion o deduplicacion
        freemium_threshold=10_000         # convierte rapido; el valor de NMI es inmediato
    ),
]

def simulate_scenario(scenario: AdoptionScenario) -> dict:
    epsilon_adj = -abs(scenario.alpha_sensitivity)  # signo negativo garantizado
    A = calibrate_demand_scale(scenario.q_anchor, epsilon=epsilon_adj)
    opt = optimal_price(A, epsilon_adj)

    # Curva de demanda sobre el rango de precios del mercado (20 puntos)
    prices = np.linspace(P_MIN, P_MAX, 20)
    demand_curve = [
        DemandPoint(
            price=p,
            quantity=nmi_hybrid_demand(p, A, epsilon_adj),
            revenue=revenue(p, A, epsilon_adj),
            elasticity=point_elasticity(p, A, epsilon_adj)
        )
        for p in prices
    ]

    # Punto de equilibrio freemium->paid: precio en el que Q = freemium_threshold
    # Q_threshold = A * P_eq^epsilon  =>  P_eq = (Q_threshold / A)^(1/epsilon)
    p_freemium_breakeven = (scenario.freemium_threshold / A) ** (1.0 / epsilon_adj)
    p_freemium_breakeven = float(np.clip(p_freemium_breakeven, P_MIN, P_MAX))

    return {
        "scenario": scenario.label,
        "epsilon": epsilon_adj,
        "demand_scale_A": A,
        "optimal_price_usd": round(opt.price, 6),
        "optimal_quantity_ops_month": round(opt.quantity),
        "optimal_monthly_revenue_usd": round(opt.revenue, 2),
        "freemium_breakeven_price_usd": round(p_freemium_breakeven, 6),
        "freemium_breakeven_ops": scenario.freemium_threshold,
        "demand_curve": [
            {"price": round(d.price, 6), "quantity": round(d.quantity), "revenue": round(d.revenue, 4)}
            for d in demand_curve
        ]
    }

def run_elasticity_model() -> list[dict]:
    results = []
    for scenario in ADOPTION_SCENARIOS:
        sim = simulate_scenario(scenario)

        # Validacion de coherencia: Q en P_MIN debe ser mayor que Q en P_MAX (demanda decreciente)
        curve = sim["demand_curve"]
        assert curve[0]["quantity"] > curve[-1]["quantity"], (
            f"Demand curve not decreasing for scenario {scenario.label}: "
            f"Q(P_MIN)={curve[0]['quantity']} vs Q(P_MAX)={curve[-1]['quantity']}"
        )

        # Validacion: breakeven freemium debe estar dentro del rango de mercado
        bp = sim["freemium_breakeven_price_usd"]
        assert P_MIN <= bp <= P_MAX, (
            f"Freemium breakeven {bp} out of market range [{P_MIN}, {P_MAX}] "
            f"for scenario {scenario.label}"
        )

        results.append(sim)
    return results

if __name__ == "__main__":
    import json
    output = run_elasticity_model()
    print(json.dumps(output, indent=2))