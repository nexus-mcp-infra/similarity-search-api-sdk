from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cosine as cosine_distance
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("similarity_search_api")

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
VALID_API_KEYS = set(filter(None, os.environ.get("API_KEYS", "").split(",")))

app = FastAPI(
    title="Stateless Similarity Search API",
    description="Per-call NMI+Cosine fused similarity ranking over ephemeral corpora. No index setup required.",
    version="1.0.0",
)


def _require_valid_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if VALID_API_KEYS and api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return api_key


class SimilarityRequest(BaseModel):
    corpus: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of embedding vectors forming the ephemeral corpus. All vectors must share the same dimensionality.",
    )
    query: list[float] = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Query embedding vector. Must match corpus vector dimensionality.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Number of top results to return.",
    )
    nmi_bandwidth: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="KDE bandwidth for NMI estimation. If None, Scott's rule is applied automatically.",
    )

    @field_validator("corpus")
    @classmethod
    def corpus_vectors_uniform_dim(cls, v):
        if not v:
            raise ValueError("corpus must contain at least one vector.")
        dim = len(v[0])
        if dim == 0:
            raise ValueError("corpus vectors must have at least one dimension.")
        for i, vec in enumerate(v):
            if len(vec) != dim:
                raise ValueError(
                    f"All corpus vectors must share the same dimensionality. "
                    f"Vector 0 has dim={dim}, vector {i} has dim={len(vec)}."
                )
        return v

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v):
        if not v:
            raise ValueError("query vector must not be empty.")
        return v


class RankedItem(BaseModel):
    corpus_index: int
    composite_score: float
    cosine_similarity: float
    nmi_score: float
    nmi_weight: float


class SimilarityResponse(BaseModel):
    results: list[RankedItem]
    corpus_size: int
    query_dim: int
    adaptive_nmi_weight: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


def _cosine_similarities(corpus_matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    corpus_norms = np.linalg.norm(corpus_matrix, axis=1)

    zero_query = query_norm < 1e-12
    zero_corpus = corpus_norms < 1e-12

    if zero_query:
        return np.zeros(len(corpus_matrix))

    dot_products = corpus_matrix @ query_vec
    denom = corpus_norms * query_norm
    similarities = np.where(zero_corpus, 0.0, dot_products / denom)
    return np.clip(similarities, -1.0, 1.0)


def _estimate_nmi_kde(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: Optional[float],
) -> float:
    n = len(x)
    if n < 4:
        return 0.0

    if bandwidth is None:
        bw_x = float(np.std(x)) * (n ** (-1.0 / 5.0))
        bw_y = float(np.std(y)) * (n ** (-1.0 / 5.0))
    else:
        bw_x = bandwidth
        bw_y = bandwidth

    bw_x = max(bw_x, 1e-10)
    bw_y = max(bw_y, 1e-10)

    grid_size = min(32, n)
    x_grid = np.linspace(x.min() - 2 * bw_x, x.max() + 2 * bw_x, grid_size)
    y_grid = np.linspace(y.min() - 2 * bw_y, y.max() + 2 * bw_y, grid_size)

    kde_x = gaussian_kde(x, bw_method=bw_x / (np.std(x) + 1e-12) if np.std(x) > 1e-12 else "scott")
    kde_y = gaussian_kde(y, bw_method=bw_y / (np.std(y) + 1e-12) if np.std(y) > 1e-12 else "scott")
    kde_joint = gaussian_kde(np.vstack([x, y]), bw_method=bw_x / (np.std(x) + 1e-12) if np.std(x) > 1e-12 else "scott")

    px = kde_x(x_grid)
    py = kde_y(y_grid)

    px = np.clip(px, 1e-300, None)
    py = np.clip(py, 1e-300, None)
    px /= px.sum()
    py /= py.sum()

    XX, YY = np.meshgrid(x_grid, y_grid)
    eval_points = np.vstack([XX.ravel(), YY.ravel()])
    pxy = kde_joint(eval_points).reshape(grid_size, grid_size)
    pxy = np.clip(pxy, 1e-300, None)
    pxy /= pxy.sum()

    outer = np.outer(py, px)
    outer = np.clip(outer, 1e-300, None)

    mi = float(np.sum(pxy * np.log(pxy / outer)))
    mi = max(mi, 0.0)

    hx = -float(np.sum(px * np.log(px)))
    hy = -float(np.sum(py * np.log(py)))

    denom = min(hx, hy)
    if denom < 1e-12:
        return 0.0

    nmi = min(mi / denom, 1.0)
    return float(nmi)


def _corpus_variance_weight(corpus_matrix: np.ndarray) -> float:
    inter_item_var = float(np.mean(np.var(corpus_matrix, axis=0)))
    total_range = float(np.ptp(corpus_matrix))
    if total_range < 1e-12:
        return 0.0
    normalized_var = min(inter_item_var / (total_range ** 2 + 1e-12), 1.0)
    nmi_weight = float(np.tanh(3.0 * normalized_var))
    return nmi_weight


def _batch_nmi_scores(
    corpus_matrix: np.ndarray,
    query_vec: np.ndarray,
    bandwidth: Optional[float],
) -> np.ndarray:
    n, d = corpus_matrix.shape
    nmi_scores = np.zeros(n)

    for dim_idx in range(d):
        corpus_dim = corpus_matrix[:, dim_idx]
        query_val = float(query_vec[dim_idx])
        x_extended = np.append(corpus_dim, query_val)

        for item_idx in range(n):
            y_pair = np.array([corpus_dim[item_idx], query_val])
            x_pair = np.array([corpus_dim[item_idx], query_val])
            nmi_scores[item_idx] += _estimate_nmi_kde(corpus_dim, np.full(n, corpus_dim[item_idx]), bandwidth)

    dim_nmi = np.zeros(n)
    for item_idx in range(n):
        per_dim = np.zeros(d)
        for dim_idx in range(d):
            corpus_col = corpus_matrix[:, dim_idx]
            item_projection = np.full(len(corpus_col), corpus_matrix[item_idx, dim_idx])
            query_projection = query_vec[dim_idx]
            x_signal = corpus_col
            y_signal = np.full(len(corpus_col), corpus_matrix[item_idx, dim_idx])
            per_dim[dim_idx] = _estimate_nmi_kde(x_signal, y_signal, bandwidth)
        dim_nmi[item_idx] = float(np.mean(per_dim))

    return dim_nmi


def _compute_item_nmi(
    corpus_matrix: np.ndarray,
    item_idx: int,
    query_vec: np.ndarray,
    bandwidth: Optional[float],
) -> float:
    n, d = corpus_matrix.shape
    per_dim_nmi = np.zeros(d)
    for dim_idx in range(d):
        corpus_col = corpus_matrix[:, dim_idx]
        item_val = corpus_matrix[item_idx, dim_idx]
        query_val = query_vec[dim_idx]
        signal_a = corpus_col
        signal_b = np.full(n, item_val)
        per_dim_nmi[dim_idx] = _estimate_nmi_kde(signal_a, signal_b, bandwidth)
    return float(np.mean(per_dim_nmi))


@app.get("/health", response_model=HealthResponse, tags=["operational"])
def health_check():
    return HealthResponse(status="ok", version="1.0.0")


@app.post(
    "/rank",
    response_model=SimilarityResponse,
    tags=["core"],
    summary="Rank corpus items by NMI+Cosine fused similarity against a query — stateless, per-call.",
)
def rank_by_fused_similarity(
    request: SimilarityRequest,
    api_key: str = Security(_require_valid_key),
):
    t0 = time.perf_counter()

    corpus_matrix = np.array(request.corpus, dtype=np.float64)
    query_vec = np.array(request.query, dtype=np.float64)

    n, d = corpus_matrix.shape

    if d != len(query_vec):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Dimensionality mismatch: corpus vectors have dim={d}, "
                f"query has dim={len(query_vec)}."
            ),
        )

    if not np.all(np.isfinite(corpus_matrix)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="corpus contains non-finite values (NaN or Inf). All embeddings must be finite.",
        )
    if not np.all(np.isfinite(query_vec)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="query contains non-finite values (NaN or Inf).",
        )

    cosine_sims = _cosine_similarities(corpus_matrix, query_vec)
    cosine_normalized = (cosine_sims + 1.0) / 2.0

    nmi_weight = _corpus_variance_weight(corpus_matrix)
    cosine_weight = 1.0 - nmi_weight

    nmi_scores = np.zeros(n)
    if nmi_weight > 1e-4:
        for item_idx in range(n):
            nmi_scores[item_idx] = _compute_item_nmi(
                corpus_matrix, item_idx, query_vec, request.nmi_bandwidth
            )

    composite_scores = cosine_weight * cosine_normalized + nmi_weight * nmi_scores

    top_k = min(request.top_k, n)
    top_indices = np.argpartition(composite_scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(composite_scores[top_indices])[::-1]]

    results = [
        RankedItem(
            corpus_index=int(idx),
            composite_score=float(composite_scores[idx]),
            cosine_similarity=float(cosine_sims[idx]),
            nmi_score=float(nmi_scores[idx]),
            nmi_weight=float(nmi_weight),
        )
        for idx in top_indices
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "rank completed corpus_size=%d dim=%d top_k=%d nmi_weight=%.4f latency_ms=%.2f",
        n, d, top_k, nmi_weight, latency_ms,
    )

    return SimilarityResponse(
        results=results,
        corpus_size=n,
        query_dim=d,
        adaptive_nmi_weight=float(nmi_weight),
        latency_ms=round(latency_ms, 3),
    )


@app.post(
    "/score",
    tags=["core"],
    summary="Compute the fused NMI+Cosine score for a single corpus/query pair — no ranking, minimal payload.",
)
def score_single_pair(
    request: SimilarityRequest,
    api_key: str = Security(_require_valid_key),
):
    request.top_k = len(request.corpus)
    response = rank_by_fused_similarity(request, api_key=api_key)
    if not response.results:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No results produced. Verify corpus and query inputs.",
        )
    return {
        "composite_score": response.results[0].composite_score,
        "cosine_similarity": response.results[0].cosine_similarity,
        "nmi_score": response.results[0].nmi_score,
        "adaptive_nmi_weight": response.adaptive_nmi_weight,
        "latency_ms": response.latency_ms,
    }


@app.post(
    "/introspect",
    tags=["diagnostics"],
    summary="Return corpus statistics used to compute adaptive NMI weight — useful for debugging blend behavior.",
)
def introspect_corpus_statistics(
    request: SimilarityRequest,
    api_key: str = Security(_require_valid_key),
):
    t0 = time.perf_counter()

    corpus_matrix = np.array(request.corpus, dtype=np.float64)
    query_vec = np.array(request.query, dtype=np.float64)
    n, d = corpus_matrix.shape

    if d != len(query_vec):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dimensionality mismatch: corpus dim={d}, query dim={len(query_vec)}.",
        )

    if not np.all(np.isfinite(corpus_matrix)) or not np.all(np.isfinite(query_vec)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Non-finite values detected in corpus or query.",
        )

    per_dim_var = np.var(corpus_matrix, axis=0)
    inter_item_var = float(np.mean(per_dim_var))
    total_range = float(np.ptp(corpus_matrix))
    normalized_var = min(inter_item_var / (total_range ** 2 + 1e-12), 1.0)
    nmi_weight = float(np.tanh(3.0 * normalized_var))
    cosine_weight = 1.0 - nmi_weight

    cosine_sims = _cosine_similarities(corpus_matrix, query_vec)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "corpus_size": n,
        "embedding_dim": d,
        "inter_item_variance_mean": round(inter_item_var, 8),
        "total_value_range": round(total_range, 8),
        "normalized_variance": round(normalized_var, 8),
        "adaptive_nmi_weight": round(nmi_weight, 6),
        "adaptive_cosine_weight": round(cosine_weight, 6),
        "cosine_similarity_stats": {
            "min": round(float(cosine_sims.min()), 6),
            "max": round(float(cosine_sims.max()), 6),
            "mean": round(float(cosine_sims.mean()), 6),
            "std": round(float(cosine_sims.std()), 6),
        },
        "kde_bandwidth_mode": "scott_rule" if request.nmi_bandwidth is None else "manual",
        "latency_ms": round(latency_ms, 3),
    }

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


@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_corpus_by_nmi_cosine_fusion', description='Ranks all corpus vectors against a query vector using a fused NMI+cosine score. NMI captures nonlinear statistical dependence; cosine captures geometric proximity. The composite score reduces false negatives on noisy or high-dimensional corpora where cosine alone fails. Use when you need a single ranked list from an ephemeral corpus without prior indexing. Do NOT use when corpus exceeds 50,000 vectors per call (latency will exceed SLA), when you need approximate nearest neighbors with sub-linear time, or when corpus is reused across many queries (use a persistent index instead).')
async def rank_corpus_by_nmi_cosine_fusion(corpus_vectors: Annotated[list[list[float]], Field(..., description='Matrix of corpus item embeddings. Each row is one item vector. All rows must have the same dimensionality as query_vector. Shape: [n_items, n_dims]. Accepts float32-compatible values.', min_length=1, max_length=50000)], query_vector: Annotated[list[float], Field(..., description='Single embedding vector representing the query item. Must match dimensionality of corpus_vectors columns. Values should be finite floats; NaN or Inf will cause a 422 error.', min_length=1, max_length=8192)], top_k: Annotated[float, Field(10, description='Number of top-ranked corpus items to return. Must be <= len(corpus_vectors). Returns indices and scores in descending composite score order.', ge=1, le=1000)], nmi_weight: Annotated[float, Field(0.5, description='Weight assigned to the NMI component in the fused score. Composite = nmi_weight * nmi_score + (1 - nmi_weight) * cosine_score. Set closer to 1.0 for semantically noisy corpora; closer to 0.0 for geometrically clean embedding spaces.', ge=0.0, le=1.0)], nmi_bins: Annotated[float, Field(16, description='Number of histogram bins used when estimating the joint distribution for NMI calculation over continuous vector dimensions. Higher values increase NMI precision but raise compute time. Recommended range: 8-32 for d >= 128.', ge=4, le=64)]) -> dict[str, Any]:
    """NMI+Cosine Fusion Ranking"""
    params = {"corpus_vectors": corpus_vectors, "query_vector": query_vector, "top_k": top_k, "nmi_weight": nmi_weight, "nmi_bins": nmi_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/rank', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_compute_pairwise_nmi_cosine_matrix', description='Computes the full n x n composite NMI+cosine similarity matrix for a set of vectors. Useful for clustering, deduplication, or graph construction where all pairwise relationships matter. Do NOT use for query-to-corpus ranking (use rank_corpus_by_nmi_cosine_fusion instead); do NOT use when n > 2,000 per call due to O(n^2 * d) complexity.')
async def compute_pairwise_nmi_cosine_matrix(vectors: Annotated[list[list[float]], Field(..., description='Set of embedding vectors to compare pairwise. Shape: [n, d]. All rows must share the same dimensionality. Output matrix will be symmetric of shape [n, n].', min_length=2, max_length=2000)], nmi_weight: Annotated[float, Field(0.5, description='Weight for NMI component in composite score. Same semantics as in rank_corpus_by_nmi_cosine_fusion.', ge=0.0, le=1.0)], nmi_bins: Annotated[float, Field(16, description='Histogram bins for joint distribution estimation per vector pair. Applies uniformly to all pairs.', ge=4, le=64)]) -> dict[str, Any]:
    """Pairwise Fusion Score Matrix"""
    params = {"vectors": vectors, "nmi_weight": nmi_weight, "nmi_bins": nmi_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/pairwise-matrix', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_decompose_nmi_cosine_scores', description='Returns the individual NMI score, cosine score, and fused composite score for each corpus item against the query, without ranking. Use when you need full score transparency to tune nmi_weight, audit model behavior, or build custom re-ranking logic on top. Do NOT use when you only need the top-k result — rank_corpus_by_nmi_cosine_fusion is more efficient for that case.')
async def decompose_nmi_cosine_scores(corpus_vectors: Annotated[list[list[float]], Field(..., description='Corpus embedding matrix. Shape: [n_items, n_dims]. Same constraints as rank_corpus_by_nmi_cosine_fusion.', min_length=1, max_length=10000)], query_vector: Annotated[list[float], Field(..., description='Query embedding vector. Must match corpus_vectors column dimensionality.', min_length=1, max_length=8192)], nmi_bins: Annotated[float, Field(16, description='Histogram bins for NMI estimation. Affects NMI precision; cosine scores are unaffected.', ge=4, le=64)]) -> dict[str, Any]:
    """Score Component Decomposition"""
    params = {"corpus_vectors": corpus_vectors, "query_vector": query_vector, "nmi_bins": nmi_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/decompose', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_estimate_nmi_score_between_vectors', description='Computes the normalized mutual information score between exactly two continuous embedding vectors by estimating their joint probability distribution via histogram binning. Use when you need to verify the NMI signal in isolation for a single pair, or to calibrate nmi_bins before a bulk ranking call. Do NOT use as a loop replacement for corpus ranking — calling this n times for n corpus items costs O(n) round trips vs. a single rank call.')
async def estimate_nmi_score_between_vectors(vector_a: Annotated[list[float], Field(..., description='First embedding vector. Must be same length as vector_b. Values must be finite floats.', min_length=1, max_length=8192)], vector_b: Annotated[list[float], Field(..., description='Second embedding vector. Must be same length as vector_a.', min_length=1, max_length=8192)], nmi_bins: Annotated[float, Field(16, description='Number of bins for joint histogram estimation. Increasing bins reduces bias but increases variance for short vectors (d < 64). Default of 16 is calibrated for d >= 128.', ge=4, le=64)]) -> dict[str, Any]:
    """NMI Score Between Two Vectors"""
    params = {"vector_a": vector_a, "vector_b": vector_b, "nmi_bins": nmi_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/nmi-pair', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_filter_corpus_by_fusion_threshold', description='Returns all corpus items whose composite NMI+cosine score against the query meets or exceeds a minimum threshold, without imposing a fixed top_k cutoff. Use when the relevant set size is unknown in advance and you want all items above a quality floor (e.g., deduplication, candidate retrieval before re-ranking). Do NOT use when you need exactly k results — use rank_corpus_by_nmi_cosine_fusion instead. Do NOT use with very low thresholds (< 0.1) on large corpora as it may return the full corpus.')
async def filter_corpus_by_fusion_threshold(corpus_vectors: Annotated[list[list[float]], Field(..., description='Corpus embedding matrix. Shape: [n_items, n_dims]. Items are returned in descending composite score order.', min_length=1, max_length=50000)], query_vector: Annotated[list[float], Field(..., description='Query embedding vector. Must match corpus_vectors column dimensionality.', min_length=1, max_length=8192)], min_fusion_score: Annotated[float, Field(..., description='Minimum composite NMI+cosine score (inclusive) for an item to be included in the response. Composite scores are in [0.0, 1.0]. Recommended starting value: 0.5 for dense embedding spaces.', ge=0.0, le=1.0)], nmi_weight: Annotated[float, Field(0.5, description='Weight for NMI component. Must be consistent with how min_fusion_score was calibrated.', ge=0.0, le=1.0)], nmi_bins: Annotated[float, Field(16, description='Histogram bins for NMI estimation per corpus item.', ge=4, le=64)]) -> dict[str, Any]:
    """Threshold-Based Corpus Filter"""
    params = {"corpus_vectors": corpus_vectors, "query_vector": query_vector, "min_fusion_score": min_fusion_score, "nmi_weight": nmi_weight, "nmi_bins": nmi_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/filter', params)


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
