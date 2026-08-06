from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import normalized_mutual_info_score
import hashlib
import os
import time

app = FastAPI(
    title="Calibrated Similarity Search API",
    description="Stateless NMI + cosine fusion with entropy-driven alpha calibration",
    version="1.0.0",
    contact={"email": "dasaanrod@gmail.com"},
)

# --- NEXUS: x402 (pago por llamada en USDC, Base Sepolia testnet) ---
from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.schemas import Network
from x402.server import x402ResourceServer

_NEXUS_X402_EVM_ADDRESS = "0x70e9f8057bb50e31b6ee06958bcbbe7de9daa98f"
_NEXUS_X402_NETWORK: Network = "eip155:84532"  # Base Sepolia (testnet) -- cambiar a eip155:8453 + facilitator mainnet para produccion
_NEXUS_X402_PRICE = "$0.01"

_nexus_x402_facilitator = HTTPFacilitatorClient(
    FacilitatorConfig(url="https://x402.org/facilitator")
)
_nexus_x402_server = x402ResourceServer(_nexus_x402_facilitator)
_nexus_x402_server.register(_NEXUS_X402_NETWORK, ExactEvmServerScheme())

# --- PATCH x402_revenue_events_hook ---
# Cierra CLAUDE.md SS9.57 pendiente #1 / SS9.58: settle_payment() real
# (REST o MCP, ambos llaman sobre esta MISMA instancia de
# x402ResourceServer) no dejaba ninguna fila en revenue_events. Portado
# a mano del mismo fix ya aplicado a mcp_wrapper_generator.py -- ver
# patch_x402_revenue_events_hook.py y CLAUDE.md SS9.58 para el diseno
# completo (por que on_after_settle() y no un punto proxy pre-settlement).
# Registrado aca, ANTES del branching de NEXUS_X402_FREE_MODE mas abajo
# en este archivo -- queda listo para cuando el freemium termine, sin
# necesitar un segundo patch.
_NEXUS_SUPABASE_URL = os.getenv("SUPABASE_URL")
_NEXUS_SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
_NEXUS_ASSET_NAME = os.getenv("NEXUS_ASSET")


async def _nexus_supabase_insert(table, payload):
    if not _NEXUS_SUPABASE_URL or not _NEXUS_SUPABASE_ANON_KEY:
        return
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                f"{_NEXUS_SUPABASE_URL}/rest/v1/{table}",
                json=payload,
                headers={
                    "apikey": _NEXUS_SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {_NEXUS_SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
    except Exception:
        pass  # nunca romper el flujo real por un fallo de telemetria


async def _nexus_log_x402_revenue_event(ctx) -> None:
    """
    AfterSettleHook real -- dispara SOLO cuando x402ResourceServer
    confirma settle_result.success=True (ver _settle_payment_core en
    x402/server_base.py). Duck-typed, nunca levanta: un bug aca no debe
    poder romper la respuesta de pago real que el caller ya esta por
    recibir.
    """
    try:
        result = getattr(ctx, "result", None)
        requirements = getattr(ctx, "requirements", None)
        if result is None or requirements is None or not getattr(result, "success", False):
            return
        # amount en unidades atomicas de USDC (6 decimales) -- amount_eur
        # no es EUR, es el valor numerico crudo, sin conversion FX (mismo
        # mismatch de schema ya documentado en el generador).
        raw_amount = getattr(requirements, "amount", None)
        amount_eur = int(raw_amount) / 1_000_000 if raw_amount is not None else None
        await _nexus_supabase_insert("revenue_events", {
            "asset_name": _NEXUS_ASSET_NAME,
            "amount_eur": amount_eur,
            "pricing_model": "x402",
            "stripe_event_id": None,
            "customer_id": getattr(result, "payer", None),
        })
    except Exception:
        pass


def _nexus_register_x402_revenue_logging(x402_server) -> None:
    """Registrado UNA vez, cubre settle_payment() real de REST y MCP
    por igual -- ambos llaman sobre la misma instancia compartida."""
    x402_server.on_after_settle(_nexus_log_x402_revenue_event)


_nexus_register_x402_revenue_logging(_nexus_x402_server)

_NEXUS_X402_ROUTES: dict[str, RouteConfig] = {
    "POST /similarity/search": RouteConfig(
        accepts=[PaymentOption(scheme="exact", pay_to=_NEXUS_X402_EVM_ADDRESS, price=_NEXUS_X402_PRICE, network=_NEXUS_X402_NETWORK)],
        mime_type="application/json",
        description="Stateless NMI+cosine fusion similarity search over an inline corpus",
    ),
    "POST /similarity/calibrate-alpha/v1": RouteConfig(
        accepts=[PaymentOption(scheme="exact", pay_to=_NEXUS_X402_EVM_ADDRESS, price=_NEXUS_X402_PRICE, network=_NEXUS_X402_NETWORK)],
        mime_type="application/json",
        description="Entropy-calibrated alpha for a corpus, sin correr la busqueda completa",
    ),
    "POST /similarity/batch-score": RouteConfig(
        accepts=[PaymentOption(scheme="exact", pay_to=_NEXUS_X402_EVM_ADDRESS, price=_NEXUS_X402_PRICE, network=_NEXUS_X402_NETWORK)],
        mime_type="application/json",
        description="Scoring de hasta 10000 pares de vectores con alpha fijo",
    ),
}

# --- NEXUS PATCH x402_free_mode_similarity_search ---
# Experimento freemium (validar demanda antes de decidir mainnet/MPP,
# ver research 2026-07-26). NEXUS_X402_FREE_MODE=true desactiva el gate
# de pago x402 SIN tocar el resto (auth por API key, rate limiting,
# exclusion de billing de Stripe -- ninguno de esos se toca aca). Se
# lee UNA sola vez al arrancar el proceso -- Railway reinicia el
# servicio en cada cambio de env var (mismo comportamiento ya
# documentado para set_variables() en CLAUDE.md SS3.3), asi que
# alternar el flag en Railway sin pushear codigo nuevo alcanza.
_NEXUS_X402_FREE_MODE = os.environ.get("NEXUS_X402_FREE_MODE", "").strip().lower() == "true"
if not _NEXUS_X402_FREE_MODE:
    app.add_middleware(PaymentMiddlewareASGI, routes=_NEXUS_X402_ROUTES, server=_nexus_x402_server)

# --- NEXUS: x402scan discovery -- x-payment-info en openapi.json ---
# x402scan (Merit-Systems) no tiene un .well-known/x402 ratificado --
# confirmado 2026-07-26 cruzando x402-foundation/x402 (sin mencion),
# x402scan-skills (deprecado, sin schema) y la spec vigente en
# agentcash.dev/discovery (mismo equipo detras de x402scan): el
# mecanismo real es leer /openapi.json buscando la extension
# "x-payment-info" por operacion. Sin esto, x402scan reporta "no
# discovery document found" pese a que /openapi.json ya se sirve.
# Las 3 operaciones y el precio salen de _NEXUS_X402_ROUTES /
# _NEXUS_X402_PRICE ya definidos arriba -- nada nuevo se inventa aca.
_NEXUS_X402_OPENAPI_OPERATIONS = [
    ("post", "/similarity/search"),
    ("post", "/similarity/calibrate-alpha/v1"),
    ("post", "/similarity/batch-score"),
]


def _nexus_x402_openapi_with_payment_info():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi as _nexus_get_openapi
    schema = _nexus_get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        contact=app.contact,
    )
    for _nexus_method, _nexus_path in _NEXUS_X402_OPENAPI_OPERATIONS:
        _nexus_operation = schema.get("paths", {}).get(_nexus_path, {}).get(_nexus_method)
        if _nexus_operation is None:
            continue
        _nexus_operation["x-payment-info"] = {
            "price": {
                "mode": "fixed",
                "currency": "USD",
                "amount": _NEXUS_X402_PRICE.lstrip("$"),
            },
            "protocols": [{"x402": {}}],
        }
        _nexus_operation.setdefault("responses", {})["402"] = {"description": "Payment Required"}
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _nexus_x402_openapi_with_payment_info

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
_VALID_API_KEY = os.environ.get("SIMILARITY_API_KEY", "")

MAX_CORPUS_ITEMS = 500_000
MAX_VECTOR_DIM = 4_096
MIN_VECTOR_DIM = 2
NMI_BINS_DEFAULT = 10
NMI_BINS_MIN = 3
NMI_BINS_MAX = 50


def _require_api_key(key: str = Security(API_KEY_HEADER)) -> str:
    if not _VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication not configured on server.",
        )
    if not key or key != _VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return key


class CorpusVector(BaseModel):
    id: str = Field(..., min_length=1, max_length=256)
    vector: list[float] = Field(..., min_length=MIN_VECTOR_DIM, max_length=MAX_VECTOR_DIM)

    @field_validator("vector")
    @classmethod
    def _vector_must_be_finite(cls, v: list[float]) -> list[float]:
        arr = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Vector contains NaN or Inf values.")
        if np.linalg.norm(arr) == 0.0:
            raise ValueError("Zero-norm vector has no direction and cannot be used.")
        return v


class SimilaritySearchRequest(BaseModel):
    query: CorpusVector
    corpus: list[CorpusVector] = Field(..., min_length=1, max_length=MAX_CORPUS_ITEMS)
    top_k: int = Field(default=10, ge=1, le=1_000)
    nmi_bins: int = Field(default=NMI_BINS_DEFAULT, ge=NMI_BINS_MIN, le=NMI_BINS_MAX)
    alpha_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Pin alpha manually [0,1]. If None, entropy-calibrated alpha is used.",
    )

    @field_validator("corpus")
    @classmethod
    def _corpus_vectors_uniform_dim(cls, v: list[CorpusVector]) -> list[CorpusVector]:
        if not v:
            return v
        dim = len(v[0].vector)
        for item in v[1:]:
            if len(item.vector) != dim:
                raise ValueError(
                    f"All corpus vectors must share the same dimension. "
                    f"Expected {dim}, got {len(item.vector)} for id='{item.id}'."
                )
        return v


class SimilarityResult(BaseModel):
    id: str
    composite_score: float
    cosine_similarity: float
    nmi_score: float
    rank: int


class SimilaritySearchResponse(BaseModel):
    results: list[SimilarityResult]
    calibrated_alpha: float
    corpus_entropy: float
    query_id: str
    corpus_size: int
    latency_ms: float
    request_fingerprint: str


class AlphaCalibrationResponse(BaseModel):
    calibrated_alpha: float
    corpus_entropy: float
    corpus_size: int
    vector_dim: int
    latency_ms: float


class BatchScoreRequest(BaseModel):
    pairs: list[tuple[list[float], list[float]]] = Field(..., min_length=1, max_length=10_000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    nmi_bins: int = Field(default=NMI_BINS_DEFAULT, ge=NMI_BINS_MIN, le=NMI_BINS_MAX)

    @field_validator("pairs")
    @classmethod
    def _pairs_must_be_valid(cls, pairs):
        for i, (a, b) in enumerate(pairs):
            if len(a) != len(b):
                raise ValueError(f"Pair {i}: vectors must have equal dimension ({len(a)} != {len(b)}).")
            if len(a) < MIN_VECTOR_DIM:
                raise ValueError(f"Pair {i}: vectors must have at least {MIN_VECTOR_DIM} dimensions.")
            if len(a) > MAX_VECTOR_DIM:
                raise ValueError(f"Pair {i}: vectors exceed max dimension {MAX_VECTOR_DIM}.")
        return pairs


class BatchScoreResponse(BaseModel):
    scores: list[float]
    alpha_used: float
    pair_count: int
    latency_ms: float


def _normalize_l2(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-norm vector.")
    return vec / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _normalize_l2(a)
    b_n = _normalize_l2(b)
    return float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))


def _discretize_vector(vec: np.ndarray, n_bins: int) -> np.ndarray:
    reshaped = vec.reshape(-1, 1)
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
    try:
        binned = kbd.fit_transform(reshaped).astype(np.int32).flatten()
    except ValueError:
        kbd2 = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform", subsample=None)
        binned = kbd2.fit_transform(reshaped).astype(np.int32).flatten()
    return binned


def _nmi_between_vectors(a: np.ndarray, b: np.ndarray, n_bins: int) -> float:
    a_disc = _discretize_vector(a, n_bins)
    b_disc = _discretize_vector(b, n_bins)
    score = normalized_mutual_info_score(a_disc, b_disc, average_method="arithmetic")
    return float(np.clip(score, 0.0, 1.0))


def _marginal_entropy_of_matrix(matrix: np.ndarray, n_bins: int) -> float:
    n_items, dim = matrix.shape
    entropies = np.zeros(dim)
    for d in range(dim):
        col = matrix[:, d]
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        try:
            binned = kbd.fit_transform(col.reshape(-1, 1)).flatten().astype(np.int32)
        except ValueError:
            kbd2 = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform", subsample=None)
            binned = kbd2.fit_transform(col.reshape(-1, 1)).flatten().astype(np.int32)
        counts = np.bincount(binned, minlength=n_bins).astype(np.float64)
        probs = counts / counts.sum()
        entropies[d] = scipy_entropy(probs, base=2)
    return float(np.mean(entropies))


def _calibrate_alpha(corpus_entropy: float, n_bins: int) -> float:
    h_max = np.log2(n_bins)
    if h_max <= 0.0:
        return 0.5
    alpha = corpus_entropy / (corpus_entropy + h_max)
    return float(np.clip(alpha, 0.0, 1.0))


def _composite_score(cosine_sim: float, nmi: float, alpha: float) -> float:
    cosine_normalized = (cosine_sim + 1.0) / 2.0
    return float(alpha * cosine_normalized + (1.0 - alpha) * nmi)


def _request_fingerprint(query_id: str, corpus_size: int, timestamp_ms: float) -> str:
    raw = f"{query_id}:{corpus_size}:{timestamp_ms}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@app.post(
    "/similarity/search",
    response_model=SimilaritySearchResponse,
    summary="Stateless NMI+cosine fusion search over an inline corpus",
)
def search_corpus_by_calibrated_similarity(
    request: SimilaritySearchRequest,
    _key: str = Security(_require_api_key),
) -> SimilaritySearchResponse:
    t0 = time.perf_counter()

    query_vec = np.asarray(request.query.vector, dtype=np.float64)
    corpus_ids = [item.id for item in request.corpus]
    corpus_matrix = np.array([item.vector for item in request.corpus], dtype=np.float64)

    query_dim = query_vec.shape[0]
    corpus_dim = corpus_matrix.shape[1]
    if query_dim != corpus_dim:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Query vector dimension ({query_dim}) must match corpus dimension ({corpus_dim}).",
        )

    corpus_entropy = _marginal_entropy_of_matrix(corpus_matrix, request.nmi_bins)

    if request.alpha_override is not None:
        alpha = request.alpha_override
    else:
        alpha = _calibrate_alpha(corpus_entropy, request.nmi_bins)

    cosine_sims = np.array(
        [_cosine_similarity(query_vec, corpus_matrix[i]) for i in range(len(corpus_ids))],
        dtype=np.float64,
    )
    nmi_scores = np.array(
        [_nmi_between_vectors(query_vec, corpus_matrix[i], request.nmi_bins) for i in range(len(corpus_ids))],
        dtype=np.float64,
    )
    composite_scores = np.array(
        [_composite_score(float(cosine_sims[i]), float(nmi_scores[i]), alpha) for i in range(len(corpus_ids))],
        dtype=np.float64,
    )

    top_k = min(request.top_k, len(corpus_ids))
    top_indices = np.argpartition(composite_scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(composite_scores[top_indices])[::-1]]

    results = [
        SimilarityResult(
            id=corpus_ids[idx],
            composite_score=round(float(composite_scores[idx]), 6),
            cosine_similarity=round(float(cosine_sims[idx]), 6),
            nmi_score=round(float(nmi_scores[idx]), 6),
            rank=rank + 1,
        )
        for rank, idx in enumerate(top_indices)
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    ts = time.time() * 1000.0

    return SimilaritySearchResponse(
        results=results,
        calibrated_alpha=round(alpha, 6),
        corpus_entropy=round(corpus_entropy, 6),
        query_id=request.query.id,
        corpus_size=len(corpus_ids),
        latency_ms=round(latency_ms, 3),
        request_fingerprint=_request_fingerprint(request.query.id, len(corpus_ids), ts),
    )


@app.post(
    "/similarity/calibrate-alpha",
    response_model=AlphaCalibrationResponse,
    summary="Compute entropy-calibrated alpha for a corpus without running a query",
)
def compute_corpus_entropy_calibrated_alpha(
    corpus: list[CorpusVector] = ...,
    nmi_bins: int = NMI_BINS_DEFAULT,
    _key: str = Security(_require_api_key),
) -> AlphaCalibrationResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Use the /similarity/calibrate-alpha/v1 POST endpoint with a JSON body.",
    )


class AlphaCalibrateRequest(BaseModel):
    corpus: list[CorpusVector] = Field(..., min_length=1, max_length=MAX_CORPUS_ITEMS)
    nmi_bins: int = Field(default=NMI_BINS_DEFAULT, ge=NMI_BINS_MIN, le=NMI_BINS_MAX)


@app.post(
    "/similarity/calibrate-alpha/v1",
    response_model=AlphaCalibrationResponse,
    summary="Inspect entropy-calibrated alpha for a corpus before committing to a search",
)
def inspect_corpus_entropy_and_alpha(
    request: AlphaCalibrateRequest,
    _key: str = Security(_require_api_key),
) -> AlphaCalibrationResponse:
    t0 = time.perf_counter()

    corpus_matrix = np.array([item.vector for item in request.corpus], dtype=np.float64)
    dims = [len(item.vector) for item in request.corpus]
    if len(set(dims)) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="All corpus vectors must share the same dimension.",
        )

    corpus_entropy = _marginal_entropy_of_matrix(corpus_matrix, request.nmi_bins)
    alpha = _calibrate_alpha(corpus_entropy, request.nmi_bins)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return AlphaCalibrationResponse(
        calibrated_alpha=round(alpha, 6),
        corpus_entropy=round(corpus_entropy, 6),
        corpus_size=len(request.corpus),
        vector_dim=corpus_matrix.shape[1],
        latency_ms=round(latency_ms, 3),
    )


@app.post(
    "/similarity/batch-score",
    response_model=BatchScoreResponse,
    summary="Score up to 10,000 vector pairs with a fixed alpha — no corpus overhead",
)
def score_vector_pairs_with_fixed_alpha(
    request: BatchScoreRequest,
    _key: str = Security(_require_api_key),
) -> BatchScoreResponse:
    t0 = time.perf_counter()

    scores = []
    for a_raw, b_raw in request.pairs:
        a = np.asarray(a_raw, dtype=np.float64)
        b = np.asarray(b_raw, dtype=np.float64)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="One or more vectors contain NaN or Inf values.",
            )
        cos_sim = _cosine_similarity(a, b)
        nmi = _nmi_between_vectors(a, b, request.nmi_bins)
        score = _composite_score(cos_sim, nmi, request.alpha)
        scores.append(round(score, 6))

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return BatchScoreResponse(
        scores=scores,
        alpha_used=request.alpha,
        pair_count=len(scores),
        latency_ms=round(latency_ms, 3),
    )


# --- NEXUS PATCH health_real_core_computation ---
# GET /health devolvia {"status": "ok"} hardcodeado siempre (bug real,
# ver CLAUDE.md SS9.8/9.12/9.13) -- este asset no tiene upstream externo
# (confirmado), asi que la unica senal honesta disponible es correr de
# verdad el computo central (cosine + NMI + composite score) contra un
# par de vectores dummy fijos y deterministas, y confirmar que sklearn/
# numpy siguen funcionando -- no solo que el proceso responde. El
# contador del rate limiter se reporta aparte, explicitamente NO ligado
# a status: llegar a _NEXUS_RATE_LIMIT_MAX_TRACKED solo evict al caller
# mas viejo (LRU, ver PATCH rate_limit_similarity_search mas abajo en
# este mismo archivo), no degrada el servicio -- inflar status con eso
# seria el mismo tipo de bug ya corregido en BuyWhere.
_NEXUS_HEALTH_CHECK_VECTOR_A = [0.10, 0.85, 0.32, 0.67, 0.05, 0.94]
_NEXUS_HEALTH_CHECK_VECTOR_B = [0.92, 0.15, 0.60, 0.28, 0.77, 0.03]


@app.get(
    "/health",
    summary="Liveness probe — no auth required",
    openapi_extra={"security": []},
)
def liveness_probe() -> dict:
    try:
        _a = np.asarray(_NEXUS_HEALTH_CHECK_VECTOR_A, dtype=np.float64)
        _b = np.asarray(_NEXUS_HEALTH_CHECK_VECTOR_B, dtype=np.float64)
        _cos = _cosine_similarity(_a, _b)
        _nmi = _nmi_between_vectors(_a, _b, NMI_BINS_MIN)
        _composite = _composite_score(_cos, _nmi, 0.5)
        _core_ok = bool(
            np.isfinite(_cos) and -1.0 <= _cos <= 1.0
            and np.isfinite(_nmi) and 0.0 <= _nmi <= 1.0
            and np.isfinite(_composite) and 0.0 <= _composite <= 1.0
        )
    except Exception:
        _cos = _nmi = _composite = None
        _core_ok = False

    return {
        "status": "ok" if _core_ok else "degraded",
        "core_computation": {
            "ok": _core_ok,
            "sample_cosine_similarity": _cos,
            "sample_nmi": _nmi,
            "sample_composite_score": _composite,
        },
        "rate_limiter": {
            "note": (
                "Informational only -- NOT part of status. Hitting "
                "max_tracked_callers evicts the oldest tracked caller "
                "(LRU); it does not degrade the service."
            ),
            "tracked_callers": len(_nexus_rate_limit_state),
            "max_tracked_callers": _NEXUS_RATE_LIMIT_MAX_TRACKED,
        },
        "version": "1.0.0",
    }

# --- NEXUS: servidor MCP real montado en el mismo proceso (inyectado por forge_agent) ---
# Reemplaza el wrapper Node/TypeScript separado -- un solo deploy, sin
# segundo servicio, sin salto de red interno. Ver mcp_wrapper_generator.py
# (v2.0) para el razonamiento completo, incluido el gotcha de
# session_manager que explica el patron startup/shutdown de abajo.

from typing import Annotated, Any, Literal
from contextlib import AsyncExitStack as _NexusMcpExitStack
import os
import httpx
from pydantic import Field
from mcp.server.fastmcp import FastMCP as _NexusFastMCP
from mcp.server.transport_security import TransportSecuritySettings

# --- NEXUS: PATCH fix_mcp_dns_rebinding_host_deployed_outputs ---
# FastMCP() sin host/transport_security explicito activa proteccion
# anti DNS-rebinding con allowlist localhost-only por default del SDK,
# rechazando con 421 "Invalid Host header" cualquier request real
# contra el dominio publico de Railway (bug real confirmado en
# produccion 2026-07-09 probando /mcp contra el deploy real). Se pasa
# transport_security explicito leyendo RAILWAY_PUBLIC_DOMAIN en
# runtime -- Railway lo inyecta automaticamente en cada servicio.
_nexus_railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "*")

_nexus_mcp = _NexusFastMCP(
    'nexus-similarity-search-api',
    stateless_http=True,
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        # --- PATCH fix_mcp_dns_rebinding_bare_host ---
        # Railway (como cualquier proxy HTTPS estandar) manda el Host
        # header SIN puerto explicito -- "dominio:*" nunca matchea eso,
        # solo matchea "dominio:443". Se agrega tambien el dominio
        # pelado para cubrir ambos casos (bug real confirmado en
        # produccion 2026-07-09: primer fix desplegado, /mcp seguia
        # devolviendo 421 tras el redeploy).
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            _nexus_railway_domain,
            _nexus_railway_domain + ":*",
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://127.0.0.1:*",
            "https://" + _nexus_railway_domain,
        ],
    ),
)


# _nexus_mcp_call_core() eliminada: llamaba a las rutas HTTP reales via
# ASGI in-process, pero esas mismas rutas estan protegidas por el
# middleware x402 (ver "NEXUS: x402" mas arriba) -- cualquier request,
# incluido este interno, exige un pago valido y devuelve 402 Payment
# Required en vez de la respuesta real. Los 3 tools MCP que sobreviven
# ahora llaman DIRECTO a las funciones de logica de negocio
# (search_corpus_by_calibrated_similarity, etc.), sin pasar por
# ASGI/HTTP/x402 -- mismo criterio que ya usa la exclusion de billing
# de Stripe para tratar estas 3 rutas como internas.


# --- NEXUS PATCH mcp_x402_auth_gate_similarity_search ---
# patch_mcp_tool_grounding_similarity_search_inprocess.py made these 3 tools
# call the business-logic functions directly to dodge the 402 from the old
# ASGI-transport internal call -- but that direct call also skipped
# _require_api_key() (only evaluated by FastAPI's own DI, never by a plain
# Python call) AND the x402 PaymentMiddlewareASGI (registered via
# app.add_middleware(), filtered by path in _NEXUS_X402_ROUTES -- the
# FastMCP mount at "/" was never in that list). Net effect: these 3 paid,
# API-key-gated operations became free and unauthenticated over MCP while
# staying gated over REST. Fixed here with the same two checks the REST
# routes already enforce, applied explicitly instead of via FastAPI DI:
#   - auth: _require_api_key() is a plain function, not FastAPI-DI-only --
#     called directly with an explicit `api_key` tool parameter (must come
#     from the caller, matching X-API-Key; passing the server's own
#     _VALID_API_KEY back to itself would authenticate nothing).
#   - payment: x402's own MCP integration (x402.mcp.create_payment_wrapper,
#     installed as part of this asset's x402 dependency) wraps the FastMCP
#     tool handler with the SAME _nexus_x402_server/PaymentOption/price the
#     REST routes use -- not a hand-rolled reimplementation. Verification
#     runs before the handler; settlement only runs if the handler returns
#     without raising, so a bad api_key aborts before any charge lands.

from x402.mcp import create_payment_wrapper as _nexus_mcp_x402_wrapper_factory
from x402.schemas.config import ResourceConfig as _NexusX402ResourceConfig

# create_payment_wrapper() necesita PaymentRequirements (asset + amount ya
# resueltos), no PaymentOption (price string sin resolver) -- son tipos
# distintos en el SDK. En vez de derivar a mano el asset USDC/decimales, se
# reusa el mismo camino que ya usa la libreria puertas adentro para las
# rutas REST (ver x402.http.x402_http_server._build_payment_requirements_from_options):
# ResourceConfig(misma price/network/pay_to que el PaymentOption de REST) ->
# server.build_payment_requirements(). Requiere que el server este
# inicializado (fetch de "supported" contra el facilitator); se garantiza
# una sola vez sin duplicar la llamada si algo mas ya lo inicializo antes.
# --- NEXUS PATCH x402_free_mode_similarity_search (MCP surface) ---
# Mismo flag que el gate REST (_NEXUS_X402_FREE_MODE, definido mas
# arriba junto a PaymentMiddlewareASGI). En modo libre, el wrapper es
# un decorator identity -- no se inicializa el server contra el
# facilitator ni se construyen PaymentRequirements, evitando un
# roundtrip de red innecesario mientras el experimento esta activo.
if _NEXUS_X402_FREE_MODE:
    _nexus_mcp_x402_wrapper = lambda fn: fn
else:
    if not getattr(_nexus_x402_server, "_initialized", False):
        _nexus_x402_server.initialize()

    _NEXUS_MCP_X402_RESOURCE_CONFIG = _NexusX402ResourceConfig(
        scheme="exact",
        pay_to=_NEXUS_X402_EVM_ADDRESS,
        price=_NEXUS_X402_PRICE,
        network=_NEXUS_X402_NETWORK,
    )
    _NEXUS_MCP_X402_ACCEPTS = _nexus_x402_server.build_payment_requirements(_NEXUS_MCP_X402_RESOURCE_CONFIG)
    _nexus_mcp_x402_wrapper = _nexus_mcp_x402_wrapper_factory(_nexus_x402_server, accepts=_NEXUS_MCP_X402_ACCEPTS)

@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_items_by_nmi_cosine_fusion', description="Ranks a corpus of items against a query vector using a calibrated fusion score (alpha * cosine + (1-alpha) * NMI_normalizado), where alpha is auto-derived from the corpus's marginal entropy unless overridden. Results are identified by their 0-indexed position in corpus_vectors (this tool does not accept explicit item IDs). Use this when you need semantically-calibrated similarity over a stateless corpus of up to 500k items without a vector database. Do NOT use for purely geometric nearest-neighbor search where NMI overhead is unnecessary, nor for corpora larger than 500k items per call. Requires a valid api_key (same as X-API-Key) and an x402 payment.")
@_nexus_mcp_x402_wrapper
async def rank_items_by_nmi_cosine_fusion(query_vector: Annotated[list[float], Field(..., description='Dense numeric vector representing the query item. Must have the same dimensionality as all corpus_vectors entries.', min_length=2, max_length=4096)], corpus_vectors: Annotated[list[list[float]], Field(..., description='List of dense numeric vectors forming the corpus to rank against. Each inner array must match query_vector dimensionality. Maximum 500000 entries.', min_length=1, max_length=500000)], top_k: Annotated[float, Field(10, description='Number of top-ranked results to return, ordered by descending fusion score. Capped at 1000 by the core service regardless of corpus size.', ge=1, le=1000)], alpha_override: Annotated[float, Field(None, description='Fixed alpha weight for cosine component in [0.0, 1.0]. If omitted, alpha is auto-calibrated from corpus entropy. Set to 1.0 to use pure cosine; 0.0 for pure NMI.', ge=0.0, le=1.0)], n_bins: Annotated[float, Field(16, description='Number of histogram bins used to discretize continuous dimensions when estimating NMI. Must be between 3 and 50.', ge=3, le=50)], api_key: Annotated[str, Field(..., description='API key required for this paid operation -- same secret configured as X-API-Key on the REST endpoints (SIMILARITY_API_KEY). Payment (x402) alone is not sufficient; both gates must pass.')]) -> dict[str, Any]:
    """NMI-Cosine Fused Similarity Ranking"""
    _require_api_key(key=api_key)
    corpus_ids = [str(i) for i in range(len(corpus_vectors))]
    request_obj = SimilaritySearchRequest(
        query=CorpusVector(id="query", vector=query_vector),
        corpus=[CorpusVector(id=cid, vector=vec) for cid, vec in zip(corpus_ids, corpus_vectors)],
        top_k=int(top_k),
        nmi_bins=int(n_bins),
        alpha_override=alpha_override,
    )
    response = search_corpus_by_calibrated_similarity(request_obj, _key=_VALID_API_KEY)
    return response.model_dump()

@_nexus_mcp.tool(name='nexus_similarity_search_api_estimate_corpus_entropy_profile', description="Computes the aggregate entropy-calibrated alpha for a corpus without running a full search -- useful to inspect before committing to a large rank_items_by_nmi_cosine_fusion call. Returns a single aggregate corpus_entropy value, NOT a per-dimension breakdown -- the real logic only exposes the mean marginal entropy across dimensions, not H(X_d) per individual dimension. Do NOT use expecting per-dimension granularity. Requires a valid api_key (same as X-API-Key) and an x402 payment.")
@_nexus_mcp_x402_wrapper
async def estimate_corpus_entropy_profile(corpus_vectors: Annotated[list[list[float]], Field(..., description='List of dense numeric vectors for which to compute the aggregate entropy and calibrated alpha. Each inner array must be the same length. Maximum 500000 entries.', min_length=1, max_length=500000)], n_bins: Annotated[float, Field(16, description='Number of histogram bins for entropy discretization. Must be between 3 and 50; should match the n_bins used in rank_items_by_nmi_cosine_fusion for the profile to be consistent.', ge=3, le=50)], api_key: Annotated[str, Field(..., description='API key required for this paid operation -- same secret configured as X-API-Key on the REST endpoints (SIMILARITY_API_KEY). Payment (x402) alone is not sufficient; both gates must pass.')]) -> dict[str, Any]:
    """Corpus Entropy and Calibrated Alpha Estimator"""
    _require_api_key(key=api_key)
    corpus_ids = [str(i) for i in range(len(corpus_vectors))]
    request_obj = AlphaCalibrateRequest(
        corpus=[CorpusVector(id=cid, vector=vec) for cid, vec in zip(corpus_ids, corpus_vectors)],
        nmi_bins=int(n_bins),
    )
    response = inspect_corpus_entropy_and_alpha(request_obj, _key=_VALID_API_KEY)
    return response.model_dump()

@_nexus_mcp.tool(name='nexus_similarity_search_api_score_pair_nmi_cosine', description="Computes the NMI-cosine fusion score for exactly one (query, target) vector pair at a fixed alpha. Use for explainability, debugging, or unit-level validation of fusion scores before running full corpus ranking. Unlike corpus-level ranking, alpha is NOT auto-calibrated for a single pair -- the real logic requires a fixed alpha (default 0.5); pass alpha explicitly for a specific blend. Do NOT use in a loop to score many pairs; batch them into rank_items_by_nmi_cosine_fusion instead. Requires a valid api_key (same as X-API-Key) and an x402 payment.")
@_nexus_mcp_x402_wrapper
async def score_pair_nmi_cosine(vector_a: Annotated[list[float], Field(..., description='First dense numeric vector of the pair. Must have the same dimensionality as vector_b.', min_length=2, max_length=4096)], vector_b: Annotated[list[float], Field(..., description='Second dense numeric vector of the pair. Must have the same dimensionality as vector_a.', min_length=2, max_length=4096)], n_bins: Annotated[float, Field(16, description='Histogram bins for NMI discretization. Must be between 3 and 50.', ge=3, le=50)], alpha: Annotated[float, Field(0.5, description='Fixed alpha weight for the cosine component in [0.0, 1.0], applied as-is -- not auto-calibrated. Default 0.5 matches the core service default.', ge=0.0, le=1.0)], api_key: Annotated[str, Field(..., description='API key required for this paid operation -- same secret configured as X-API-Key on the REST endpoints (SIMILARITY_API_KEY). Payment (x402) alone is not sufficient; both gates must pass.')]) -> dict[str, Any]:
    """Single-Pair NMI-Cosine Scorer"""
    _require_api_key(key=api_key)
    request_obj = BatchScoreRequest(pairs=[(vector_a, vector_b)], alpha=alpha, nmi_bins=int(n_bins))
    response = score_vector_pairs_with_fixed_alpha(request_obj, _key=_VALID_API_KEY)
    return response.model_dump()


# Crea el sub-app ASGI de streamable HTTP -- DEBE llamarse antes de
# poder acceder a _nexus_mcp.session_manager (se crea de forma
# perezosa, ver docstring del modulo).
# Se monta en "/" (no en "/mcp"): streamable_http_app() YA expone su
# propia ruta interna en "/mcp" -- montarlo de nuevo en "/mcp" duplica
# el path a "/mcp/mcp" y da 404 (bug real encontrado probando esto en
# runtime con un cliente MCP de verdad, no algo teorico).
_nexus_mcp_asgi_app = _nexus_mcp.streamable_http_app()
_nexus_mcp_stack = _NexusMcpExitStack()


@app.on_event("startup")
async def _nexus_mcp_startup():
    await _nexus_mcp_stack.enter_async_context(_nexus_mcp.session_manager.run())


@app.on_event("shutdown")
async def _nexus_mcp_shutdown():
    await _nexus_mcp_stack.aclose()




# PATCH remove_nexus_debug_host_endpoint: /_nexus_debug_host removido.
# Exponia todas las variables RAILWAY_* sin auth en un endpoint publico
# -- fuga de informacion real, no solo cruft de diagnostico temporal.

# --- NEXUS: A2A Agent Card (spec v0.3.0/v1.0) para discovery ---
# Ruta correcta del spec vigente -- NO /agent.json (deprecado en
# v0.1.0), NO /v1/agent-card.json ni /v2/agent-card.json (no son parte
# del spec real, un crawler las prueba igual sin encontrar nada). Debe
# registrarse ANTES del app.mount("/", ...) de mas abajo: Starlette
# matchea rutas en el orden en que se agregan a app.routes, y un Mount
# en "/" intercepta cualquier path si se agrega primero.
@app.get("/.well-known/agent-card.json", include_in_schema=False)
async def _nexus_a2a_agent_card() -> dict:
    return {
        "name": "Calibrated Similarity Search API",
        "description": "Stateless NMI + cosine fusion with entropy-driven alpha calibration",
        "url": "https://similarity-search-api-production.up.railway.app",
        "version": "1.0.0",
        "documentationUrl": "https://similarity-search-api-production.up.railway.app/docs",
        "provider": {
            "organization": "nexus-mcp-infra",
            "url": "https://github.com/nexus-mcp-infra/similarity-search-api-sdk",
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "additionalInterfaces": [
            {"url": "https://similarity-search-api-production.up.railway.app/mcp", "transport": "MCP"},
        ],
        "securitySchemes": {
            "apiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            "x402Payment": {
                "type": "apiKey",
                "in": "header",
                "name": "X-PAYMENT",
                "description": "x402 payment proof (USDC, Base Sepolia testnet) required on paid operations -- not a classic auth scheme.",
            },
        },
        "security": [{"apiKeyHeader": [], "x402Payment": []}],
        "skills": [
            {
                "id": "nexus_similarity_search_api_rank_items_by_nmi_cosine_fusion",
                "name": "NMI-Cosine Fused Similarity Ranking",
                "description": "Ranks a corpus of items against a query vector using a calibrated fusion score (alpha * cosine + (1-alpha) * NMI_normalizado), where alpha is auto-derived from the corpus's marginal entropy unless overridden. Results are identified by their 0-indexed position in corpus_vectors (this tool does not accept explicit item IDs). Requires a valid api_key (same as X-API-Key) and an x402 payment.",
                "tags": ["similarity-search", "embeddings", "nmi", "cosine-similarity"],
            },
            {
                "id": "nexus_similarity_search_api_estimate_corpus_entropy_profile",
                "name": "Corpus Entropy and Calibrated Alpha Estimator",
                "description": "Computes the aggregate entropy-calibrated alpha for a corpus without running a full search -- useful to inspect before committing to a large rank_items_by_nmi_cosine_fusion call. Returns a single aggregate corpus_entropy value, not a per-dimension breakdown. Requires a valid api_key (same as X-API-Key) and an x402 payment.",
                "tags": ["entropy", "calibration"],
            },
            {
                "id": "nexus_similarity_search_api_score_pair_nmi_cosine",
                "name": "Single-Pair NMI-Cosine Scorer",
                "description": "Computes the NMI-cosine fusion score for exactly one (query, target) vector pair at a fixed alpha. Use for explainability, debugging, or unit-level validation of fusion scores before running full corpus ranking. Requires a valid api_key (same as X-API-Key) and an x402 payment.",
                "tags": ["similarity-search", "debugging"],
            },
        ],
        "metadata": {
            "protocol_note": (
                "This service implements the Model Context Protocol (MCP) at /mcp, "
                "not A2A's own JSONRPC/gRPC/HTTP+JSON task methods (message/send, "
                "tasks/get, etc.). This Agent Card is provided for discovery/indexing "
                "purposes; A2A-conformant task orchestration is not implemented."
            ),
        },
    }


# --- NEXUS: favicon.ico real para checklist de x402scan ---
# x402scan.com/resources/register senala /favicon.ico ausente al leer
# el listado (2026-07-26). Icono generico solido, sin decision de
# branding -- generado con stdlib puro (struct+zlib), sin Pillow.
# Debe registrarse ANTES del app.mount("/", ...) de mas abajo, mismo
# motivo que el Agent Card: Starlette matchea rutas en el orden en que
# se agregan a app.routes.
import base64 as _nexus_favicon_base64

_NEXUS_FAVICON_ICO = _nexus_favicon_base64.b64decode(
    "AAABAAEAICAAAAEAIABpAAAAFgAAAIlQTkcNChoKAAAADUlIRFIAAAAgAAAAIAgGAAAAc3p69AAAADBJREFUeNrtziEBAAAIAzCS4MlA/1wQ42ZiftWzl1QCAgICAgICAgICAgICAgLpwAO9cwRbSHXMRQAAAABJRU5ErkJggg=="
)


@app.get("/favicon.ico", include_in_schema=False)
async def _nexus_favicon():
    from fastapi import Response
    return Response(content=_NEXUS_FAVICON_ICO, media_type="image/x-icon")


app.mount("/", _nexus_mcp_asgi_app)

# --- NEXUS: reporte de uso real a Stripe (inyectado por forge_output_saver_v6) ---
# --- HOTFIX: excluir paths de monitoreo/sistema del billing (ver Fase 0.5) ---
# --- NEXUS PATCH stripe_mcp_billing_exclusion ---
# /mcp agregado como entrada explicita: el sub-app FastMCP montado en "/"
# es Starlette puro (no FastAPI), nunca setea scope["route"] -- este
# middleware compara contra request.url.path, que para cualquier request a
# /mcp vale literalmente "/mcp" (el "/" ya presente en el set solo cubre la
# URL raiz, no subrutas del mount). Sin esto, trafico de protocolo MCP
# (initialize, tools/list -- ninguno pasa por gate de auth/pago) se
# facturaba igual que una operacion de negocio real. Confirmado en Railway:
# STRIPE_CUSTOMER_ID/STRIPE_EVENT_NAME/STRIPE_SECRET_KEY reales, modo test.
_NEXUS_BILLING_EXCLUDED_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc", "/favicon.ico", "/mcp", "/similarity/search", "/similarity/calibrate-alpha/v1", "/similarity/batch-score", "/.well-known/agent-card.json"}  # x402 cubre estas 3 -- Stripe no debe cobrarlas de nuevo; agent-card.json es discovery, no negocio
@app.middleware("http")
async def _nexus_usage_middleware(request, call_next):
    response = await call_next(request)
    try:
        if (
            request.url.path not in _NEXUS_BILLING_EXCLUDED_PATHS
            and response.status_code < 400
        ):
            import os as _nexus_os
            import stripe as _nexus_stripe
            _customer_id = _nexus_os.environ.get("STRIPE_CUSTOMER_ID")
            _event_name = _nexus_os.environ.get("STRIPE_EVENT_NAME")
            _secret_key = _nexus_os.environ.get("STRIPE_SECRET_KEY")
            if _customer_id and _event_name and _secret_key:
                _nexus_stripe.api_key = _secret_key
                _nexus_stripe.billing.MeterEvent.create(
                    event_name=_event_name,
                    payload={
                        "stripe_customer_id": _customer_id,
                        "value": "1",
                    },
                )
    except Exception:
        pass  # nunca romper la response real por un fallo de billing
    return response


# --- NEXUS PATCH rate_limit_similarity_search ---
# Rate limiting por caller (Fase 1). Identidad resuelta con la mejor senal
# disponible: wallet pagadora x402 (X-PAYMENT) > API key (X-API-Key,
# hasheada) > IP del cliente. Corre como middleware ASGI, por lo tanto
# cubre tanto las rutas REST como el sub-app FastMCP montado en "/" (ver
# docstring de patch_rate_limit_similarity_search.py para el detalle
# completo de estas decisiones). Sin dependencias nuevas (solo stdlib).
import base64 as _nexus_rl_base64
import hashlib as _nexus_rl_hashlib
import json as _nexus_rl_json
import os as _nexus_rl_os
import threading as _nexus_rl_threading
import time as _nexus_rl_time
from collections import OrderedDict as _NexusRLOrderedDict, deque as _nexus_rl_deque

from fastapi import Request as _NexusRLRequest
from fastapi.responses import JSONResponse as _NexusRLJSONResponse

_NEXUS_RATE_LIMIT_MAX_REQUESTS = int(_nexus_rl_os.environ.get("NEXUS_RATE_LIMIT_PER_MINUTE", "60"))
_NEXUS_RATE_LIMIT_WINDOW_SECONDS = float(_nexus_rl_os.environ.get("NEXUS_RATE_LIMIT_WINDOW_SECONDS", "60"))
_NEXUS_RATE_LIMIT_MAX_TRACKED = int(_nexus_rl_os.environ.get("NEXUS_RATE_LIMIT_MAX_TRACKED", "10000"))
_NEXUS_RATE_LIMIT_EXEMPT_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc", "/favicon.ico", "/.well-known/agent-card.json"}

_nexus_rate_limit_lock = _nexus_rl_threading.Lock()
_nexus_rate_limit_state = _NexusRLOrderedDict()


def _nexus_rate_limit_extract_wallet(payment_header):
    try:
        padded = payment_header + "=" * (-len(payment_header) % 4)
        payload = _nexus_rl_json.loads(_nexus_rl_base64.b64decode(padded))
        payer = payload.get("payload", {}).get("authorization", {}).get("from")
        return payer.lower() if isinstance(payer, str) and payer else None
    except Exception:
        return None


def _nexus_rate_limit_caller_id(request):
    payment_header = request.headers.get("x-payment")
    if payment_header:
        wallet = _nexus_rate_limit_extract_wallet(payment_header)
        if wallet:
            return f"wallet:{wallet}"
    api_key = request.headers.get("x-api-key")
    if api_key:
        digest = _nexus_rl_hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"apikey:{digest}"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


def _nexus_rate_limit_check(caller_id):
    now = _nexus_rl_time.monotonic()
    window = _NEXUS_RATE_LIMIT_WINDOW_SECONDS
    with _nexus_rate_limit_lock:
        bucket = _nexus_rate_limit_state.get(caller_id)
        if bucket is None:
            if len(_nexus_rate_limit_state) >= _NEXUS_RATE_LIMIT_MAX_TRACKED:
                _nexus_rate_limit_state.popitem(last=False)
            bucket = _nexus_rl_deque()
            _nexus_rate_limit_state[caller_id] = bucket
        else:
            _nexus_rate_limit_state.move_to_end(caller_id)
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= _NEXUS_RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(window - (now - bucket[0])) + 1)
            return False, retry_after
        bucket.append(now)
        return True, 0


@app.middleware("http")
async def _nexus_rate_limit_middleware(request: _NexusRLRequest, call_next):
    if request.url.path in _NEXUS_RATE_LIMIT_EXEMPT_PATHS:
        return await call_next(request)
    caller_id = _nexus_rate_limit_caller_id(request)
    allowed, retry_after = _nexus_rate_limit_check(caller_id)
    if not allowed:
        return _NexusRLJSONResponse(
            status_code=429,
            content={"error": "rate_limited", "detail": f"Too many requests. Retry after {retry_after}s."},
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)
# --- END NEXUS PATCH rate_limit_similarity_search ---


# --- NEXUS PATCH traffic_log_similarity_search ---
# Instrumentacion minima para el experimento freemium (ver
# patch_x402_free_mode_similarity_search.py): loguea IP truncada (nunca
# completa) + method/path/status por request a stdout, capturado por
# Railway. Cubre REST y el sub-app FastMCP montado en "/" (ambos pasan
# por el mismo stack ASGI). No escribe a Supabase ni a ningun store
# externo -- ver docstring del patch para el alcance deliberado.
def _nexus_traffic_log_truncate_ip(raw_ip):
    if not raw_ip:
        return "unknown"
    if ":" in raw_ip and "." not in raw_ip:
        segments = [s for s in raw_ip.split(":") if s]
        head = segments[:4] if len(segments) >= 4 else segments
        return (":".join(head) + "::/64") if head else "unknown"
    octets = raw_ip.split(".")
    if len(octets) == 4 and all(o.isdigit() for o in octets):
        return f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
    return "unknown"


@app.middleware("http")
async def _nexus_traffic_log_middleware(request, call_next):
    response = await call_next(request)
    try:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            raw_ip = forwarded.split(",")[0].strip()
        else:
            raw_ip = request.client.host if request.client else None
        ip_range = _nexus_traffic_log_truncate_ip(raw_ip)
        print(
            f"[NEXUS_TRAFFIC] ip_range={ip_range} method={request.method} "
            f"path={request.url.path} status={response.status_code}",
            flush=True,
        )
    except Exception:
        pass  # nunca romper la response real por un fallo de logging
    return response
# --- END NEXUS PATCH traffic_log_similarity_search ---
