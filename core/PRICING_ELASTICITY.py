import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple

# src/math/information: entropia marginal calibra alpha en cada request
# H(X) = -sum(p * log2(p)), base natural para NMI normalizado
def corpus_entropy(token_freq: np.ndarray) -> float:
    p = token_freq / token_freq.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

# alpha = H(corpus) / (H(corpus) + H_max): mayor entropia -> mayor peso cosine
def calibrate_alpha(h_corpus: float, h_max: float) -> float:
    if h_max <= 0:
        raise ValueError(f"h_max debe ser > 0, recibido: {h_max}")
    return h_corpus / (h_corpus + h_max)

@dataclass
class SimilaritySearchDemandParams:
    # Willingness-to-pay range: $0.001-$0.05 por operacion
    wtp_min: float = 0.001
    wtp_max: float = 0.050
    # Volumen mensual por cliente: 1K-10M operaciones
    vol_min: float = 1_000
    vol_max: float = 10_000_000
    # Elasticidad empirica developer tools: tipicamente -1.2 a -2.5
    base_elasticity: float = -1.8
    # Tamano del mercado: desarrolladores con corpus 10k-500k items
    market_size_clients: int = 8_500

@dataclass
class AdoptionScenario:
    name: str
    corpus_size: int          # items en el corpus
    ops_per_month: float      # operaciones/mes por cliente
    price_sensitivity: float  # multiplicador sobre elasticidad base
    freemium_ops_free: int    # operaciones gratis en tier freemium
    conversion_rate: float    # fraccion que convierte a paid

def demand_curve(price: float, params: SimilaritySearchDemandParams, sensitivity: float) -> float:
    # Q(P) = Q_ref * (P / P_ref)^epsilon: ley de potencia sobre precio
    if price <= 0:
        raise ValueError(f"price debe ser > 0, recibido: {price}")
    p_ref = (params.wtp_min + params.wtp_max) / 2.0   # $0.0255 precio de referencia
    q_ref = np.sqrt(params.vol_min * params.vol_max)   # media geometrica: ~100k ops
    epsilon = params.base_elasticity * sensitivity
    return float(q_ref * (price / p_ref) ** epsilon)

def price_elasticity(price: float, params: SimilaritySearchDemandParams, sensitivity: float) -> float:
    # epsilon = (dQ/dP) * (P/Q): analitico desde demanda potencia -> igual al exponente
    epsilon = params.base_elasticity * sensitivity
    return epsilon  # para curva potencia, elasticidad es constante = exponente

def revenue(price: float, params: SimilaritySearchDemandParams, sensitivity: float) -> float:
    # R(P) = P * Q(P) * N_clientes: revenue total del mercado
    q = demand_curve(price, params, sensitivity)
    return price * q * params.market_size_clients

def optimal_price(params: SimilaritySearchDemandParams, sensitivity: float) -> Tuple[float, float, float]:
    # max R(P) en [wtp_min, wtp_max] via Brent: sin supuesto de forma analitica
    result = minimize_scalar(
        lambda p: -revenue(p, params, sensitivity),
        bounds=(params.wtp_min, params.wtp_max),
        method="bounded"
    )
    p_star = float(result.x)
    q_star = demand_curve(p_star, params, sensitivity)
    r_star = revenue(p_star, params, sensitivity)
    return p_star, q_star, r_star

def freemium_conversion_threshold(scenario: AdoptionScenario, params: SimilaritySearchDemandParams) -> float:
    # Punto de equilibrio: revenue_paid >= costo_oportunidad de ops gratuitas
    # P_breakeven tal que P * (Q - Q_free) = 0 -> P_breakeven donde conversion se sostiene
    # Derivado de: conversion_rate * ops_per_month * P = freemium_ops_free * P_marginal_cost
    # P_marginal_cost estimado: O(n log n) compute ~ $0.0003 por op en 500k items
    marginal_cost_per_op = 3e-4
    numerator = scenario.freemium_ops_free * marginal_cost_per_op
    denominator = scenario.conversion_rate * (scenario.ops_per_month - scenario.freemium_ops_free)
    if denominator <= 0:
        raise ValueError("ops_per_month debe superar freemium_ops_free para calcular breakeven")
    return float(numerator / denominator)

def simulate_scenario(scenario: AdoptionScenario, params: SimilaritySearchDemandParams) -> dict:
    p_star, q_star, r_star = optimal_price(params, scenario.price_sensitivity)
    eps = price_elasticity(p_star, params, scenario.price_sensitivity)
    breakeven_price = freemium_conversion_threshold(scenario, params)
    monthly_revenue_per_client = p_star * scenario.ops_per_month * scenario.conversion_rate
    return {
        "scenario": scenario.name,
        "corpus_size": scenario.corpus_size,
        "optimal_price_per_op": round(p_star, 5),
        "elasticity_at_optimum": round(eps, 3),
        "market_revenue_monthly": round(r_star, 2),
        "revenue_per_client_monthly": round(monthly_revenue_per_client, 2),
        "freemium_breakeven_price": round(breakeven_price, 6),
        "freemium_viable": breakeven_price < p_star,
    }

# EXACTAMENTE 3 escenarios de adopcion: early-adopter, growth, scale
SCENARIOS = [
    AdoptionScenario(
        name="early_adopter_small_corpus",
        corpus_size=10_000,
        ops_per_month=5_000,
        price_sensitivity=0.85,    # menos sensible: dolor alto, alternativas caras
        freemium_ops_free=1_000,
        conversion_rate=0.12,
    ),
    AdoptionScenario(
        name="growth_mid_corpus",
        corpus_size=150_000,
        ops_per_month=200_000,
        price_sensitivity=1.00,    # elasticidad base: mercado competitivo
        freemium_ops_free=10_000,
        conversion_rate=0.07,
    ),
    AdoptionScenario(
        name="scale_large_corpus",
        corpus_size=480_000,
        ops_per_month=3_000_000,
        price_sensitivity=1.30,    # mas sensible: volumen alto, negocian precio
        freemium_ops_free=50_000,
        conversion_rate=0.04,
    ),
]

if __name__ == "__main__":
    params = SimilaritySearchDemandParams()

    # Validacion de calibracion alpha con entropia sintetica del corpus
    synthetic_freqs = np.random.dirichlet(np.ones(1000)) * 480_000
    h_corpus = corpus_entropy(synthetic_freqs)
    h_max = np.log2(len(synthetic_freqs))
    alpha = calibrate_alpha(h_corpus, h_max)
    print(f"alpha calibrado (corpus 480k items): {alpha:.4f}")
    print(f"H(corpus)={h_corpus:.3f} bits, H_max={h_max:.3f} bits")
    print()

    for scenario in SCENARIOS:
        result = simulate_scenario(scenario, params)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print()