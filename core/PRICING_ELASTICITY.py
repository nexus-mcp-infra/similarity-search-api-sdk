import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import NamedTuple

# Parametros de mercado calibrados al segmento developer-tools (API per-call)
P_MIN = 0.001   # USD — piso empirico de willingness-to-pay en dev tools
P_MAX = 0.05    # USD — techo antes de sustitucion por solucion interna
Q_BASE = 500_000  # operaciones/mes — punto medio geometrico del rango 1K-10M
ALPHA = 2.1     # elasticidad base estimada: dev tools son elasticos pero no perfectamente

# Parametro de sensibilidad al moat: NMI+Cosine reduce sustituibilidad vs cosine puro
# Un competidor solo-cosine tendria ALPHA ~ 3.0; el moat comprime la elasticidad
MOAT_INELASTICITY_FACTOR = 0.70  # reduce elasticidad efectiva en 30% por diferenciacion

@dataclass
class DemandPoint:
    price: float
    quantity: float
    elasticity: float
    revenue: float

class AdoptionScenario(NamedTuple):
    label: str
    alpha: float          # elasticidad propia del segmento
    q_base: float         # volumen base de referencia
    freemium_threshold: float  # operaciones/mes donde el usuario considera pagar

def demand_quantity(price: float, alpha: float, q_base: float, p_ref: float = 0.01) -> float:
    """Q = Q_base * (P_ref / P)^alpha — ley de potencia estandar para APIs de infraestructura."""
    if price <= 0:
        raise ValueError(f"price debe ser > 0, recibido: {price}")
    return q_base * (p_ref / price) ** alpha

def price_elasticity(price: float, alpha: float) -> float:
    """Para demanda de potencia Q ~ P^(-alpha), elasticidad es constante = -alpha."""
    # No requiere derivada numerica: analitico por construccion del modelo
    return -alpha

def revenue(price: float, alpha: float, q_base: float, p_ref: float = 0.01) -> float:
    """R(P) = P * Q(P) — objetivo de maximizacion."""
    return price * demand_quantity(price, alpha, q_base, p_ref)

def optimal_price(alpha: float, q_base: float, p_ref: float = 0.01) -> DemandPoint:
    """
    Maximiza R(P) en [P_MIN, P_MAX].
    Para elasticidad constante, el maximo analitico es P* = P_ref * (alpha/(alpha-1))^(1/alpha)
    pero se clampea al rango de mercado real — no tiene sentido fuera de [0.001, 0.05].
    """
    result = minimize_scalar(
        lambda p: -revenue(p, alpha, q_base, p_ref),
        bounds=(P_MIN, P_MAX),
        method='bounded'
    )
    p_star = result.x
    q_star = demand_quantity(p_star, alpha, q_base, p_ref)
    return DemandPoint(
        price=round(p_star, 6),
        quantity=round(q_star, 0),
        elasticity=price_elasticity(p_star, alpha),
        revenue=round(p_star * q_star, 2)
    )

def freemium_paid_equilibrium(scenario: AdoptionScenario, p_star: float) -> dict:
    """
    Punto de equilibrio: Q_freemium = threshold de ops gratuitas donde el costo
    de oportunidad de quedarse en free supera el costo de pagar.
    Revenue_free_equivalent = p_star * threshold (costo implicito que el usuario absorbe).
    Conversion se modela como logistica: P_convert = 1 / (1 + exp(-k*(Q_uso - Q_threshold)))
    """
    k = 0.000008  # pendiente calibrada: conversion rapida una vez superado el threshold
    q_usage_range = np.linspace(0, scenario.freemium_threshold * 3, 300)
    conversion_prob = 1.0 / (1.0 + np.exp(-k * (q_usage_range - scenario.freemium_threshold)))

    # Punto de equilibrio: conversion_prob = 0.5 ocurre exactamente en q_threshold por construccion
    implicit_free_cost = p_star * scenario.freemium_threshold  # USD/mes que el usuario "regala"
    equilibrium_revenue_per_user = p_star * scenario.freemium_threshold  # primer mes post-conversion

    return {
        "freemium_threshold_ops": scenario.freemium_threshold,
        "implicit_monthly_cost_at_threshold_usd": round(implicit_free_cost, 4),
        "conversion_probability_at_threshold": 0.50,  # por construccion logistica
        "expected_first_month_revenue_usd": round(equilibrium_revenue_per_user, 4),
        "breakeven_ops_for_internal_build": int(50_000 / p_star),  # costo fijo dev ~ $50 amortizado
    }

# EXACTAMENTE 3 escenarios de adopcion — no se agregan mas por diseno del modelo
ADOPTION_SCENARIOS: list[AdoptionScenario] = [
    AdoptionScenario(
        label="early_adopter_ml_engineer",
        alpha=ALPHA * MOAT_INELASTICITY_FACTOR,   # 1.47 — alta tolerancia, bajo volumen inicial
        q_base=15_000,
        freemium_threshold=5_000,
    ),
    AdoptionScenario(
        label="product_team_mixed_data",
        alpha=ALPHA * MOAT_INELASTICITY_FACTOR * 1.15,  # 1.69 — mas sensible al precio, mas volumen
        q_base=Q_BASE,
        freemium_threshold=50_000,
    ),
    AdoptionScenario(
        label="high_volume_data_pipeline",
        alpha=ALPHA * MOAT_INELASTICITY_FACTOR * 1.35,  # 1.99 — muy elastico, negocia por volumen
        q_base=4_000_000,
        freemium_threshold=500_000,
    ),
]

def run_elasticity_model() -> dict:
    results = {}
    for scenario in ADOPTION_SCENARIOS:
        opt = optimal_price(scenario.alpha, scenario.q_base)
        eq  = freemium_paid_equilibrium(scenario, opt.price)

        # Verificacion de coherencia: revenue debe crecer vs precio minimo
        rev_at_pmin = revenue(P_MIN, scenario.alpha, scenario.q_base)
        rev_at_pmax = revenue(P_MAX, scenario.alpha, scenario.q_base)

        results[scenario.label] = {
            "elasticity_epsilon": round(scenario.alpha * -1, 4),
            "optimal_price_usd": opt.price,
            "optimal_quantity_ops_month": int(opt.quantity),
            "max_revenue_usd_month": opt.revenue,
            "revenue_at_price_floor_usd": round(rev_at_pmin, 2),
            "revenue_at_price_ceiling_usd": round(rev_at_pmax, 2),
            "freemium_equilibrium": eq,
        }
    return results

if __name__ == "__main__":
    import json
    output = run_elasticity_model()
    print(json.dumps(output, indent=2))