from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import numpy as np
from scipy.stats import entropy as scipy_entropy
import asyncpg
import asyncio
import time
import os
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("similarity_api")

DB_DSN = os.environ.get("DATABASE_URL", "")
API_KEY_SECRET = os.environ.get("API_KEY_SECRET", "")

db_pool: Optional[asyncpg.Pool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    if DB_DSN:
        try:
            db_pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=10)
            await _ensure_flywheel_table()
            logger.info("DB pool ready")
        except Exception as exc:
            logger.warning(f"DB unavailable, flywheel disabled: {exc}")
    yield
    if db_pool:
        await db_pool.close()


app = FastAPI(
    title="NMI-Cosine Similarity API",
    version="1.0.0",
    description="Stateless similarity scoring: S = alpha*cosine + (1-alpha)*NMI",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=512)


def _freedman_diaconis_bins(data: np.ndarray) -> int:
    n = len(data)
    if n < 2:
        return 2
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    if iqr == 0:
        bins = int(np.ceil(np.sqrt(n)))
    else:
        h = 2.0 * iqr * n ** (-1.0 / 3.0)
        data_range = data.max() - data.min()
        bins = int(np.ceil(data_range / h)) if h > 0 else int(np.ceil(np.sqrt(n)))
    return max(2, min(bins, 64))


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    raw = float(np.dot(u, v) / (norm_u * norm_v))
    return float(np.clip(raw, -1.0, 1.0))


def _normalized_mutual_information(u: np.ndarray, v: np.ndarray) -> float:
    n = len(u)
    all_vals = np.concatenate([u, v])
    n_bins = _freedman_diaconis_bins(all_vals)
    global_min = all_vals.min()
    global_max = all_vals.max()
    if global_max == global_min:
        return 1.0
    edges = np.linspace(global_min, global_max, n_bins + 1)
    u_idx = np.searchsorted(edges[1:-1], u, side="right")
    v_idx = np.searchsorted(edges[1:-1], v, side="right")
    joint_counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    for i, j in zip(u_idx, v_idx):
        joint_counts[i, j] += 1.0
    joint_prob = joint_counts / n
    p_u = joint_prob.sum(axis=1)
    p_v = joint_prob.sum(axis=0)
    h_u = scipy_entropy(p_u + 1e-12)
    h_v = scipy_entropy(p_v + 1e-12)
    joint_flat = joint_prob[joint_prob > 0]
    h_joint = -np.sum(joint_flat * np.log(joint_flat + 1e-12))
    mutual_info = h_u + h_v - h_joint
    denom = h_u + h_v
    if denom < 1e-12:
        return 1.0
    nmi = 2.0 * mutual_info / denom
    return float(np.clip(nmi, 0.0, 1.0))


def _cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norm_q = np.linalg.norm(query)
    norms = np.linalg.norm(matrix, axis=1)
    if norm_q == 0.0:
        return np.zeros(len(matrix))
    dots = matrix @ query
    denom = norms * norm_q
    safe_denom = np.where(denom == 0, 1e-12, denom)
    return np.clip(dots / safe_denom, -1.0, 1.0)


def _nmi_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.array([_normalized_mutual_information(query, row) for row in matrix])


async def _ensure_flywheel_table():
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS nmi_cosine_flywheel (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                alpha FLOAT NOT NULL,
                vector_dim INTEGER NOT NULL,
                domain_tag TEXT,
                score_composite FLOAT NOT NULL,
                latency_ms FLOAT NOT NULL
            )
        """)


async def _record_flywheel(
    alpha: float,
    vector_dim: int,
    domain_tag: Optional[str],
    score_composite: float,
    latency_ms: float,
):
    if not db_pool:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO nmi_cosine_flywheel
                   (alpha, vector_dim, domain_tag, score_composite, latency_ms)
                   VALUES ($1, $2, $3, $4, $5)""",
                alpha,
                vector_dim,
                domain_tag,
                score_composite,
                latency_ms,
            )
    except Exception as exc:
        logger.warning(f"Flywheel write failed: {exc}")


def _require_api_key(x_api_key: str = Header(...)):
    if not API_KEY_SECRET:
        return
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class VectorPairRequest(BaseModel):
    vector_a: list[float] = Field(..., min_length=2, max_length=8192)
    vector_b: list[float] = Field(..., min_length=2, max_length=8192)
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight for cosine component; (1-alpha) weights NMI")
    domain_tag: Optional[str] = Field(None, max_length=64)

    @field_validator("vector_a", "vector_b", mode="before")
    @classmethod
    def no_nan_inf(cls, v):
        if v is None:
            raise ValueError("Vector must not be None")
        for val in v:
            if not isinstance(val, (int, float)):
                raise ValueError(f"All vector elements must be numeric, got {type(val)}")
            if not np.isfinite(val):
                raise ValueError("Vector elements must be finite (no NaN or Inf)")
        return v

    @model_validator(mode="after")
    def same_length(self):
        if len(self.vector_a) != len(self.vector_b):
            raise ValueError(
                f"Vectors must have equal dimension: got {len(self.vector_a)} vs {len(self.vector_b)}"
            )
        return self


class ScoreResponse(BaseModel):
    score_composite: float
    cosine_similarity: float
    nmi: float
    alpha: float
    latency_ms: float


class BatchRequest(BaseModel):
    query: list[float] = Field(..., min_length=2, max_length=8192)
    candidates: list[list[float]] = Field(..., min_length=1, max_length=512)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    top_k: int = Field(10, ge=1, le=512)
    domain_tag: Optional[str] = Field(None, max_length=64)

    @field_validator("query", mode="before")
    @classmethod
    def query_no_nan_inf(cls, v):
        if v is None:
            raise ValueError("Query vector must not be None")
        for val in v:
            if not isinstance(val, (int, float)):
                raise ValueError(f"Query elements must be numeric")
            if not np.isfinite(val):
                raise ValueError("Query elements must be finite")
        return v

    @model_validator(mode="after")
    def validate_candidates(self):
        dim = len(self.query)
        for idx, cand in enumerate(self.candidates):
            if len(cand) != dim:
                raise ValueError(
                    f"Candidate {idx} has dim {len(cand)}, expected {dim} to match query"
                )
            for val in cand:
                if not isinstance(val, (int, float)) or not np.isfinite(val):
                    raise ValueError(f"Candidate {idx} contains non-finite or non-numeric value")
        return self


class RankedCandidate(BaseModel):
    index: int
    score_composite: float
    cosine_similarity: float
    nmi: float


class BatchScoreResponse(BaseModel):
    results: list[RankedCandidate]
    alpha: float
    latency_ms: float


class AlphaInsightResponse(BaseModel):
    domain_tag: str
    suggested_alpha: float
    confidence: str
    sample_count: int
    note: str


@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/v1/score", response_model=ScoreResponse, summary="Score a single vector pair")
async def score_pair(
    req: VectorPairRequest,
    _: None = Depends(_require_api_key),
):
    t0 = time.perf_counter()
    u = np.array(req.vector_a, dtype=np.float64)
    v = np.array(req.vector_b, dtype=np.float64)
    cosine = _cosine_similarity(u, v)
    nmi = _normalized_mutual_information(u, v)
    composite = req.alpha * cosine + (1.0 - req.alpha) * nmi
    latency_ms = (time.perf_counter() - t0) * 1000.0
    asyncio.ensure_future(
        _record_flywheel(req.alpha, len(u), req.domain_tag, composite, latency_ms)
    )
    return ScoreResponse(
        score_composite=round(composite, 8),
        cosine_similarity=round(cosine, 8),
        nmi=round(nmi, 8),
        alpha=req.alpha,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/v1/rank", response_model=BatchScoreResponse, summary="Rank candidates against a query vector")
async def rank_candidates(
    req: BatchRequest,
    _: None = Depends(_require_api_key),
):
    t0 = time.perf_counter()
    query = np.array(req.query, dtype=np.float64)
    matrix = np.array(req.candidates, dtype=np.float64)
    cosines = _cosine_similarity_batch(query, matrix)
    nmis = _nmi_batch(query, matrix)
    composites = req.alpha * cosines + (1.0 - req.alpha) * nmis
    top_k = min(req.top_k, len(req.candidates))
    top_indices = np.argpartition(composites, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(composites[top_indices])[::-1]]
    results = [
        RankedCandidate(
            index=int(i),
            score_composite=round(float(composites[i]), 8),
            cosine_similarity=round(float(cosines[i]), 8),
            nmi=round(float(nmis[i]), 8),
        )
        for i in top_indices
    ]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if results:
        best_composite = float(composites[top_indices[0]])
        asyncio.ensure_future(
            _record_flywheel(req.alpha, len(query), req.domain_tag, best_composite, latency_ms)
        )
    return BatchScoreResponse(results=results, alpha=req.alpha, latency_ms=round(latency_ms, 3))


@app.post("/v1/explain", summary="Decompose score components for interpretability")
async def explain_score(
    req: VectorPairRequest,
    _: None = Depends(_require_api_key),
):
    t0 = time.perf_counter()
    u = np.array(req.vector_a, dtype=np.float64)
    v = np.array(req.vector_b, dtype=np.float64)
    cosine = _cosine_similarity(u, v)
    nmi = _normalized_mutual_information(u, v)
    composite = req.alpha * cosine + (1.0 - req.alpha) * nmi
    cosine_contribution = req.alpha * cosine
    nmi_contribution = (1.0 - req.alpha) * nmi
    tension = abs(cosine - nmi)
    interpretation = (
        "high_geometric_low_statistical"
        if cosine > nmi + 0.2
        else "high_statistical_low_geometric"
        if nmi > cosine + 0.2
        else "aligned"
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "score_composite": round(composite, 8),
        "cosine_similarity": round(cosine, 8),
        "nmi": round(nmi, 8),
        "cosine_contribution": round(cosine_contribution, 8),
        "nmi_contribution": round(nmi_contribution, 8),
        "metric_tension": round(tension, 8),
        "interpretation": interpretation,
        "alpha": req.alpha,
        "vector_dim": len(u),
        "latency_ms": round(latency_ms, 3),
    }


@app.get("/v1/alpha-insight/{domain_tag}", response_model=AlphaInsightResponse, summary="Suggest optimal alpha for a domain")
async def alpha_insight(
    domain_tag: str,
    _: None = Depends(_require_api_key),
):
    if not domain_tag or len(domain_tag) > 64:
        raise HTTPException(status_code=422, detail="domain_tag must be between 1 and 64 characters")
    if not db_pool:
        raise HTTPException(
            status_code=503,
            detail="Flywheel DB unavailable; alpha-insight requires historical data"
        )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT alpha, score_composite
               FROM nmi_cosine_flywheel
               WHERE domain_tag = $1
               ORDER BY created_at DESC
               LIMIT 1000""",
            domain_tag,
        )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No flywheel data for domain_tag='{domain_tag}'. Score at least one pair with this tag first."
        )
    alphas = np.array([r["alpha"] for r in rows], dtype=np.float64)
    scores = np.array([r["score_composite"] for r in rows], dtype=np.float64)
    n_bins = min(20, len(alphas))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(alphas, bin_edges[1:-1])
    bin_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    for i, s in zip(bin_indices, scores):
        bin_means[i] += s
        bin_counts[i] += 1
    observed = bin_counts > 0
    bin_means[observed] /= bin_counts[observed]
    best_bin = int(np.argmax(bin_means * observed))
    suggested_alpha = float((bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0)
    confidence = (
        "high" if bin_counts[best_bin] >= 50
        else "medium" if bin_counts[best_bin] >= 10
        else "low"
    )
    return AlphaInsightResponse(
        domain_tag=domain_tag,
        suggested_alpha=round(suggested_alpha, 3),
        confidence=confidence,
        sample_count=int(len(rows)),
        note="Suggested alpha maximizes mean composite score observed for this domain. Use as starting point, not ground truth.",
    )

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


@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_by_nmi_cosine_fusion', description='Ranks a candidate set against a query vector using a weighted fusion of Cosine similarity and Normalized Mutual Information (NMI). Use this when you need rankings that penalize geometrically close but statistically independent pairs — ideal for noisy recommendation, semantic deduplication, or clustering pre-processing. Do NOT use when candidates are pre-indexed in a vector store (use a native ANN query instead) or when NMI computation cost is unacceptable for very high-dimensional discrete histograms (dim > 4096 with many bins).')
async def rank_by_nmi_cosine_fusion(query_vector: Annotated[list[float], Field(..., description='Dense float vector representing the query item. Must have the same dimensionality as every vector in candidate_matrix.', min_length=2, max_length=4096)], candidate_matrix: Annotated[list[list[float]], Field(..., description='List of candidate dense float vectors to rank. Each row must match the dimensionality of query_vector. Maximum 2000 candidates per request.', min_length=1, max_length=2000)], alpha: Annotated[float, Field(0.7, description='Mixing weight in [0.0, 1.0]. Final score = alpha * cosine + (1 - alpha) * nmi. alpha=1.0 reduces to pure cosine; alpha=0.0 reduces to pure NMI. Values around 0.6-0.75 are typical for semantic search with noise.', ge=0.0, le=1.0)], nmi_bins: Annotated[float, Field(32, description='Number of histogram bins used to estimate joint and marginal distributions for NMI computation. Higher values increase resolution but raise compute cost. Recommended range 8-64; values above 128 are rarely beneficial.', ge=4, le=128)], top_k: Annotated[float, Field(10, description='Number of top-ranked results to return. Must be <= len(candidate_matrix). Set to -1 to return all candidates ranked.', ge=-1, le=2000)], candidate_ids: Annotated[list[str], Field(None, description='Optional list of string identifiers aligned with candidate_matrix rows. When provided, each result includes the corresponding ID. If omitted, results use zero-based integer indices as identifiers.', max_length=2000)]) -> dict[str, Any]:
    """NMI-Cosine Fused Ranking"""
    params = {"query_vector": query_vector, "candidate_matrix": candidate_matrix, "alpha": alpha, "nmi_bins": nmi_bins, "top_k": top_k, "candidate_ids": candidate_ids}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/rank-nmi-cosine', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_score_vector_pair_nmi_cosine', description='Computes the full NMI-Cosine fusion score for exactly one pair of vectors, returning the composite score plus each component (cosine, nmi) individually for interpretability. Use this for auditing, threshold calibration, or explaining why two items were ranked together. Do NOT use for bulk ranking — calling this in a loop over thousands of pairs is inefficient; use rank_by_nmi_cosine_fusion instead.')
async def score_vector_pair_nmi_cosine(vector_a: Annotated[list[float], Field(..., description='First dense float vector of the pair.', min_length=2, max_length=4096)], vector_b: Annotated[list[float], Field(..., description='Second dense float vector of the pair. Must have identical dimensionality to vector_a.', min_length=2, max_length=4096)], alpha: Annotated[float, Field(0.7, description='Mixing weight in [0.0, 1.0]. Final score = alpha * cosine + (1 - alpha) * nmi. Must match the alpha used in ranking if this call is used for threshold calibration.', ge=0.0, le=1.0)], nmi_bins: Annotated[float, Field(32, description='Number of histogram bins for NMI distribution estimation. Must match the value used in ranking calls to produce comparable scores.', ge=4, le=128)]) -> dict[str, Any]:
    """Single-Pair NMI-Cosine Score"""
    params = {"vector_a": vector_a, "vector_b": vector_b, "alpha": alpha, "nmi_bins": nmi_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/score-pair', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_compute_pairwise_nmi_cosine_matrix', description='Computes the full N x N NMI-Cosine fusion score matrix for a set of vectors. Returns a symmetric matrix where entry [i][j] is the composite similarity between vector i and vector j. Use this as input to clustering algorithms, graph construction, or duplicate detection pipelines. Do NOT use with N > 500 — the O(N^2) pairs make this expensive beyond that size; prefer rank_by_nmi_cosine_fusion with each vector as a query instead.')
async def compute_pairwise_nmi_cosine_matrix(vector_matrix: Annotated[list[list[float]], Field(..., description='List of dense float vectors. Each pair of rows will be scored. All rows must share the same dimensionality. Maximum 500 vectors.', min_length=2, max_length=500)], alpha: Annotated[float, Field(0.7, description='Mixing weight in [0.0, 1.0] for cosine vs NMI component.', ge=0.0, le=1.0)], nmi_bins: Annotated[float, Field(32, description='Number of histogram bins for NMI estimation across all pairs.', ge=4, le=128)], vector_ids: Annotated[list[str], Field(None, description='Optional identifiers aligned with vector_matrix rows. When provided, the response labels matrix rows and columns with these IDs.', max_length=500)]) -> dict[str, Any]:
    """Pairwise Fusion Score Matrix"""
    params = {"vector_matrix": vector_matrix, "alpha": alpha, "nmi_bins": nmi_bins, "vector_ids": vector_ids}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/pairwise-matrix', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_calibrate_alpha_for_domain', description='Given a labeled set of vector pairs with ground-truth similarity judgments (positive/negative), finds the alpha value that maximizes ranking quality (AUC-ROC) of the NMI-Cosine fusion score over that dataset. Use this once per domain or dataset shift to choose the right alpha before deploying rank_by_nmi_cosine_fusion. Do NOT use on unlabeled data — without ground-truth labels the optimization has no signal and will return a meaningless alpha.')
async def calibrate_alpha_for_domain(pair_vectors_a: Annotated[list[list[float]], Field(..., description='List of first vectors in each labeled pair. Must align row-by-row with pair_vectors_b and pair_labels.', min_length=10, max_length=5000)], pair_vectors_b: Annotated[list[list[float]], Field(..., description='List of second vectors in each labeled pair. Must align row-by-row with pair_vectors_a and pair_labels.', min_length=10, max_length=5000)], pair_labels: Annotated[list[float], Field(..., description='Binary ground-truth similarity labels: 1 = similar, 0 = dissimilar. Must align with pair_vectors_a and pair_vectors_b. Minimum 5 positive and 5 negative examples required.', ge=0, le=1, min_length=10, max_length=5000)], nmi_bins: Annotated[float, Field(32, description='Number of histogram bins for NMI estimation. Use the same value you plan to use in production ranking calls.', ge=4, le=128)], alpha_resolution: Annotated[float, Field(0.05, description='Step size for grid search over alpha in (0.0, 1.0). Smaller values yield finer calibration at higher compute cost. Recommended range: 0.05 to 0.01.', ge=0.01, le=0.25)]) -> dict[str, Any]:
    """Alpha Weight Calibration"""
    params = {"pair_vectors_a": pair_vectors_a, "pair_vectors_b": pair_vectors_b, "pair_labels": pair_labels, "nmi_bins": nmi_bins, "alpha_resolution": alpha_resolution}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/calibrate-alpha', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_detect_statistical_geometric_divergence', description='For each candidate vector, computes the signed divergence between its cosine rank and its NMI rank relative to the query. High divergence flags candidates that are geometrically close but statistically independent (cosine high, NMI low) — these are the false positives that pure cosine search misses. Use this to audit an existing cosine-only ranking or to identify noise sources in a dataset. Do NOT use as a primary ranking tool — it diagnoses divergence, it does not produce a final composite ranking; use rank_by_nmi_cosine_fusion for that.')
async def detect_statistical_geometric_divergence(query_vector: Annotated[list[float], Field(..., description='Dense float query vector used as the reference for both cosine and NMI rank computation.', min_length=2, max_length=4096)], candidate_matrix: Annotated[list[list[float]], Field(..., description='Candidate dense float vectors to analyze for rank divergence. Maximum 2000 candidates.', min_length=2, max_length=2000)], nmi_bins: Annotated[float, Field(32, description='Number of histogram bins for NMI estimation.', ge=4, le=128)], divergence_threshold: Annotated[float, Field(5, description='Minimum absolute rank divergence (in rank positions) to flag a candidate as divergent. Candidates with |cosine_rank - nmi_rank| >= this value are included in the flagged output list.', ge=1, le=1999)], candidate_ids: Annotated[list[str], Field(None, description='Optional identifiers aligned with candidate_matrix rows for labeling the divergence report.', max_length=2000)]) -> dict[str, Any]:
    """NMI-Cosine Divergence Detector"""
    params = {"query_vector": query_vector, "candidate_matrix": candidate_matrix, "nmi_bins": nmi_bins, "divergence_threshold": divergence_threshold, "candidate_ids": candidate_ids}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/detect-divergence', params)


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
