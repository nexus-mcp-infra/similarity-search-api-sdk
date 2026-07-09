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
)

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


@app.get(
    "/health",
    summary="Liveness probe — no auth required",
)
def liveness_probe() -> dict:
    return {"status": "ok", "version": "1.0.0"}

# --- NEXUS: servidor MCP real montado en el mismo proceso (inyectado por forge_agent) ---
# Reemplaza el wrapper Node/TypeScript separado -- un solo deploy, sin
# segundo servicio, sin salto de red interno. Ver mcp_wrapper_generator.py
# (v2.0) para el razonamiento completo, incluido el gotcha de
# session_manager que explica el patron startup/shutdown de abajo.

from typing import Annotated, Any, Literal
from contextlib import AsyncExitStack as _NexusMcpExitStack

import httpx
from pydantic import Field
from mcp.server.fastmcp import FastMCP as _NexusFastMCP

_nexus_mcp = _NexusFastMCP('nexus-similarity-search-api', stateless_http=True)


async def _nexus_mcp_call_core(method: str, path: str, params: dict) -> Any:
    """
    Llama al endpoint real del core -- via ASGI in-process (sin red
    real, sin segundo proceso), no un HTTP call externo. `app` ya
    existe en este mismo modulo (es el FastAPI que FORGE genero).
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nexus-internal") as client:
        if method == "GET":
            resp = await client.get(path, params=params)
        else:
            resp = await client.post(path, json=params)
        resp.raise_for_status()
        return resp.json()


@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_items_by_nmi_cosine_fusion', description="Ranks a corpus of items against a query vector using a calibrated fusion score (alpha * cosine + (1-alpha) * NMI_normalizado), where alpha is auto-derived from the query vector's marginal entropy relative to the corpus distribution. Use this when you need semantically-calibrated similarity over a stateless corpus of up to 500k items without a vector database. Do NOT use for purely geometric nearest-neighbor search where NMI overhead is unnecessary, nor for corpora larger than 500k items per call.")
async def rank_items_by_nmi_cosine_fusion(query_vector: Annotated[list[float], Field(..., description='Dense numeric vector representing the query item. Must have the same dimensionality as all corpus_vectors entries.', min_length=2, max_length=4096)], corpus_vectors: Annotated[list[list[float]], Field(..., description='List of dense numeric vectors forming the corpus to rank against. Each inner array must match query_vector dimensionality. Maximum 500000 entries.', min_length=1, max_length=500000)], top_k: Annotated[float, Field(10, description='Number of top-ranked results to return, ordered by descending fusion score. Must be between 1 and the corpus size.', ge=1, le=500000)], alpha_override: Annotated[float, Field(None, description='Fixed alpha weight for cosine component in [0.0, 1.0]. If omitted, alpha is auto-calibrated from corpus entropy via src/math/information. Set to 1.0 to use pure cosine; 0.0 for pure NMI. Use override only when you have a domain-specific prior on the geometry-vs-dependence tradeoff.', ge=0.0, le=1.0)], n_bins: Annotated[float, Field(16, description='Number of histogram bins used to discretize continuous dimensions when estimating H(X), H(Y), H(X,Y) for NMI computation. Higher values increase resolution but raise O(n log n) cost. Recommended range: 8-64.', ge=4, le=128)]) -> dict[str, Any]:
    """NMI-Cosine Fused Similarity Ranking"""
    params = {"query_vector": query_vector, "corpus_vectors": corpus_vectors, "top_k": top_k, "alpha_override": alpha_override, "n_bins": n_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/rank-nmi-cosine', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_estimate_corpus_entropy_profile', description="Computes per-dimension marginal entropy H(X_d) and the aggregate joint entropy estimate H(X) for a corpus using src/math/information, returning the entropy profile and the auto-calibrated alpha that rank_items_by_nmi_cosine_fusion would apply. Use this before batch ranking jobs to inspect the corpus's information geometry and decide whether to override alpha or adjust n_bins. Do NOT use as a general statistics endpoint — it is scoped exclusively to the entropy quantities needed for NMI-cosine calibration.")
async def estimate_corpus_entropy_profile(corpus_vectors: Annotated[list[list[float]], Field(..., description='List of dense numeric vectors for which to compute the entropy profile. Each inner array must be the same length. Maximum 500000 entries.', min_length=1, max_length=500000)], n_bins: Annotated[float, Field(16, description='Number of histogram bins for entropy discretization. Must match the n_bins value you intend to use in rank_items_by_nmi_cosine_fusion for the profile to be consistent.', ge=4, le=128)]) -> dict[str, Any]:
    """Corpus Entropy Profile Estimator"""
    params = {"corpus_vectors": corpus_vectors, "n_bins": n_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/corpus-entropy-profile', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_score_pair_nmi_cosine', description='Computes the NMI-cosine fusion score and its decomposed components (cosine, NMI, auto-calibrated alpha) for exactly one (query, target) vector pair. Use for explainability, debugging, or unit-level validation of fusion scores before running full corpus ranking. Do NOT use in batch loops to score many pairs — each call recomputes entropy from scratch; use rank_items_by_nmi_cosine_fusion for multi-item ranking instead.')
async def score_pair_nmi_cosine(vector_a: Annotated[list[float], Field(..., description='First dense numeric vector of the pair. Must have the same dimensionality as vector_b.', min_length=2, max_length=4096)], vector_b: Annotated[list[float], Field(..., description='Second dense numeric vector of the pair. Must have the same dimensionality as vector_a.', min_length=2, max_length=4096)], n_bins: Annotated[float, Field(16, description='Histogram bins for NMI discretization. Use the same value as in corpus-level calls to ensure score comparability.', ge=4, le=128)], alpha_override: Annotated[float, Field(None, description='Fixed alpha in [0.0, 1.0] for the fusion formula. If omitted, alpha is derived from the marginal entropies of the two vectors alone — note this differs from corpus-level alpha calibration.', ge=0.0, le=1.0)]) -> dict[str, Any]:
    """Single-Pair NMI-Cosine Scorer"""
    params = {"vector_a": vector_a, "vector_b": vector_b, "n_bins": n_bins, "alpha_override": alpha_override}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/score-pair', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_find_outlier_vectors_by_nmi_deficit', description='Identifies vectors in a corpus whose NMI with the corpus centroid falls below a threshold, signaling low statistical dependence with the dominant corpus distribution. Useful for corpus quality auditing, deduplication preprocessing, and detecting off-distribution items before ranking. Do NOT use as a general anomaly detection tool — the NMI deficit metric is meaningful only relative to a shared embedding space where cosine proximity is also meaningful.')
async def find_outlier_vectors_by_nmi_deficit(corpus_vectors: Annotated[list[list[float]], Field(..., description='Corpus of dense numeric vectors to audit for NMI-deficit outliers. All vectors must share the same dimensionality.', min_length=2, max_length=500000)], nmi_deficit_threshold: Annotated[float, Field(0.05, description='Minimum acceptable NMI score (in [0.0, 1.0]) between a vector and the corpus centroid. Vectors below this threshold are returned as outliers. Default 0.05 flags only strongly off-distribution items.', ge=0.0, le=1.0)], n_bins: Annotated[float, Field(16, description='Histogram bins for NMI estimation. Consistent with the value used in rank_items_by_nmi_cosine_fusion for comparable NMI scores.', ge=4, le=128)], return_scores: Annotated[bool, Field(True, description='If true, include the NMI deficit score for each outlier index in the response. Set false for minimal payloads when only the index list is needed.')]) -> dict[str, Any]:
    """NMI-Deficit Outlier Detector"""
    params = {"corpus_vectors": corpus_vectors, "nmi_deficit_threshold": nmi_deficit_threshold, "n_bins": n_bins, "return_scores": return_scores}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/outliers-nmi-deficit', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_calibrate_alpha_from_query_entropy', description='Given a single query vector and a representative sample of the corpus, computes the entropy-derived alpha that rank_items_by_nmi_cosine_fusion would auto-assign for that query. Returns alpha, H(query), and the corpus entropy summary used in calibration. Use when you need deterministic, pre-computed alpha values for reproducible ranking pipelines or when logging calibration decisions for auditability. Do NOT use to calibrate alpha for a different corpus than the one that will be used in the subsequent ranking call — the alpha is corpus-distribution-specific.')
async def calibrate_alpha_from_query_entropy(query_vector: Annotated[list[float], Field(..., description='The query vector for which alpha will be calibrated. Must match the dimensionality of corpus_sample vectors.', min_length=2, max_length=4096)], corpus_sample: Annotated[list[list[float]], Field(..., description='Representative sample of the corpus used to estimate the corpus entropy baseline. Does not need to be the full corpus — a sample of 1000-10000 vectors is sufficient for stable alpha calibration.', min_length=10, max_length=10000)], n_bins: Annotated[float, Field(16, description='Histogram bins for entropy estimation. Must match the n_bins used in the subsequent rank_items_by_nmi_cosine_fusion call for the calibrated alpha to be valid.', ge=4, le=128)]) -> dict[str, Any]:
    """Per-Query Alpha Entropy Calibrator"""
    params = {"query_vector": query_vector, "corpus_sample": corpus_sample, "n_bins": n_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/calibrate-alpha', params)


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


app.mount("/", _nexus_mcp_asgi_app)

# --- NEXUS: reporte de uso real a Stripe (inyectado por forge_output_saver_v6) ---
@app.middleware("http")
async def _nexus_usage_middleware(request, call_next):
    response = await call_next(request)
    try:
        if response.status_code < 400:
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
