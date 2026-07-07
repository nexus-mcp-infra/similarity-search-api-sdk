from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np
from scipy.stats import chi2_contingency
from scipy.spatial.distance import cosine as cosine_distance
import os
import math

app = FastAPI(
    title="Similarity Search API",
    description="Cosine + NMI composite similarity with calibrated p-values. No vector DB required.",
    version="1.0.0",
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
_VALID_API_KEY = os.environ.get("SIMILARITY_API_KEY", "")


def _authenticate(api_key: str = Security(API_KEY_HEADER)) -> str:
    if not _VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: SIMILARITY_API_KEY not set.",
        )
    if api_key != _VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return api_key


def _freedman_diaconis_bins(data: np.ndarray) -> int:
    n = len(data)
    if n < 2:
        return 1
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    if iqr == 0.0:
        return max(1, int(np.sqrt(n)))
    bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
    data_range = data.max() - data.min()
    if bin_width == 0.0 or data_range == 0.0:
        return 1
    return max(1, int(math.ceil(data_range / bin_width)))


def _segment_magnitudes(embedding: np.ndarray) -> np.ndarray:
    D = len(embedding)
    k = max(1, int(math.isqrt(D)))
    segment_size = D // k
    magnitudes = np.empty(k)
    for i in range(k):
        start = i * segment_size
        end = start + segment_size if i < k - 1 else D
        magnitudes[i] = np.linalg.norm(embedding[start:end])
    return magnitudes


def _activation_histogram(embedding: np.ndarray, bins: Optional[int] = None) -> np.ndarray:
    magnitudes = _segment_magnitudes(embedding)
    if bins is None:
        bins = _freedman_diaconis_bins(magnitudes)
    hist, _ = np.histogram(magnitudes, bins=bins, density=False)
    return hist.astype(np.float64)


def _shannon_entropy(distribution: np.ndarray) -> float:
    total = distribution.sum()
    if total == 0.0:
        return 0.0
    p = distribution / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _normalized_mutual_information(hist_p: np.ndarray, hist_q: np.ndarray) -> tuple[float, float]:
    n_bins = max(len(hist_p), len(hist_q))
    if len(hist_p) < n_bins:
        hist_p = np.pad(hist_p, (0, n_bins - len(hist_p)))
    if len(hist_q) < n_bins:
        hist_q = np.pad(hist_q, (0, n_bins - len(hist_q)))

    joint = np.outer(hist_p, hist_q)
    total = joint.sum()
    if total == 0.0:
        return 0.0, 1.0

    joint_normalized = joint / total
    p_marginal = joint_normalized.sum(axis=1, keepdims=True)
    q_marginal = joint_normalized.sum(axis=0, keepdims=True)

    independence = p_marginal * q_marginal
    nonzero = (joint_normalized > 0) & (independence > 0)
    mi = float(np.sum(
        joint_normalized[nonzero] * np.log(joint_normalized[nonzero] / independence[nonzero])
    ))

    h_p = _shannon_entropy(hist_p)
    h_q = _shannon_entropy(hist_q)
    denom = math.sqrt(h_p * h_q)
    nmi = mi / denom if denom > 0.0 else 0.0
    nmi = float(np.clip(nmi, 0.0, 1.0))

    contingency_table = np.outer(hist_p, hist_q)
    row_sums = contingency_table.sum(axis=1)
    col_sums = contingency_table.sum(axis=0)
    nonzero_rows = row_sums > 0
    nonzero_cols = col_sums > 0
    trimmed = contingency_table[np.ix_(nonzero_rows, nonzero_cols)]

    if trimmed.shape[0] < 2 or trimmed.shape[1] < 2:
        chi2_pvalue = 1.0
    else:
        try:
            _, chi2_pvalue, _, _ = chi2_contingency(trimmed, correction=False)
        except ValueError:
            chi2_pvalue = 1.0

    return nmi, float(chi2_pvalue)


def _bonferroni_correction(raw_pvalue: float, corpus_size: int) -> float:
    m = max(1, corpus_size)
    corrected = min(1.0, raw_pvalue * m)
    return float(corrected)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(1.0 - cosine_distance(a, b))


def _validate_embedding(raw: list, field_name: str) -> np.ndarray:
    if raw is None or len(raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{field_name}' must be a non-empty list of floats.",
        )
    try:
        arr = np.array(raw, dtype=np.float64)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{field_name}' contains non-numeric values.",
        )
    if arr.ndim != 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{field_name}' must be a 1-D array.",
        )
    if not np.all(np.isfinite(arr)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{field_name}' contains NaN or Inf values.",
        )
    return arr


class PairSimilarityRequest(BaseModel):
    embedding_a: list[float] = Field(..., min_length=1, max_length=32768)
    embedding_b: list[float] = Field(..., min_length=1, max_length=32768)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    corpus_size: int = Field(default=1, ge=1, le=500000)

    @field_validator("embedding_a", "embedding_b")
    @classmethod
    def embeddings_must_be_finite(cls, v):
        if any(not math.isfinite(x) for x in v):
            raise ValueError("Embedding contains NaN or Inf.")
        return v


class PairSimilarityResponse(BaseModel):
    cosine_similarity: float
    nmi: float
    composite_score: float
    p_value_raw: float
    p_value_bonferroni: float
    statistically_significant: bool
    alpha_used: float
    corpus_size: int


class BatchSimilarityRequest(BaseModel):
    query_embedding: list[float] = Field(..., min_length=1, max_length=32768)
    corpus_embeddings: list[list[float]] = Field(..., min_length=1, max_length=5000)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=500)
    significance_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    @field_validator("query_embedding")
    @classmethod
    def query_must_be_finite(cls, v):
        if any(not math.isfinite(x) for x in v):
            raise ValueError("Query embedding contains NaN or Inf.")
        return v


class BatchSimilarityResult(BaseModel):
    index: int
    cosine_similarity: float
    nmi: float
    composite_score: float
    p_value_bonferroni: float
    statistically_significant: bool


class BatchSimilarityResponse(BaseModel):
    results: list[BatchSimilarityResult]
    corpus_size: int
    alpha_used: float
    significance_threshold: float
    top_k_requested: int


class SignificanceFilterRequest(BaseModel):
    query_embedding: list[float] = Field(..., min_length=1, max_length=32768)
    corpus_embeddings: list[list[float]] = Field(..., min_length=1, max_length=5000)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    significance_threshold: float = Field(default=0.05, ge=0.001, le=0.5)
    min_composite_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SignificanceFilterResponse(BaseModel):
    significant_indices: list[int]
    significant_count: int
    corpus_size: int
    rejection_rate: float


@app.post("/v1/similarity/pair", response_model=PairSimilarityResponse)
def compute_pair_similarity(
    request: PairSimilarityRequest,
    api_key: str = Security(_authenticate),
) -> PairSimilarityResponse:
    emb_a = _validate_embedding(request.embedding_a, "embedding_a")
    emb_b = _validate_embedding(request.embedding_b, "embedding_b")

    if len(emb_a) != len(emb_b):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Embedding dimensions must match: got {len(emb_a)} vs {len(emb_b)}.",
        )

    cosine_sim = _cosine_similarity(emb_a, emb_b)

    shared_bins = max(
        _freedman_diaconis_bins(_segment_magnitudes(emb_a)),
        _freedman_diaconis_bins(_segment_magnitudes(emb_b)),
    )
    hist_a = _activation_histogram(emb_a, bins=shared_bins)
    hist_b = _activation_histogram(emb_b, bins=shared_bins)

    nmi, p_raw = _normalized_mutual_information(hist_a, hist_b)
    p_bonferroni = _bonferroni_correction(p_raw, request.corpus_size)
    composite = request.alpha * cosine_sim + (1.0 - request.alpha) * nmi

    return PairSimilarityResponse(
        cosine_similarity=round(cosine_sim, 6),
        nmi=round(nmi, 6),
        composite_score=round(composite, 6),
        p_value_raw=round(p_raw, 8),
        p_value_bonferroni=round(p_bonferroni, 8),
        statistically_significant=p_bonferroni < 0.05,
        alpha_used=request.alpha,
        corpus_size=request.corpus_size,
    )


@app.post("/v1/similarity/batch", response_model=BatchSimilarityResponse)
def compute_batch_similarity(
    request: BatchSimilarityRequest,
    api_key: str = Security(_authenticate),
) -> BatchSimilarityResponse:
    query = _validate_embedding(request.query_embedding, "query_embedding")
    corpus_size = len(request.corpus_embeddings)

    if corpus_size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="corpus_embeddings must contain at least one vector.",
        )

    query_seg = _segment_magnitudes(query)
    query_bins = _freedman_diaconis_bins(query_seg)
    query_hist = _activation_histogram(query, bins=query_bins)

    scored: list[tuple[int, float, float, float, float]] = []

    for idx, raw_vec in enumerate(request.corpus_embeddings):
        candidate = _validate_embedding(raw_vec, f"corpus_embeddings[{idx}]")
        if len(candidate) != len(query):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"corpus_embeddings[{idx}] dimension {len(candidate)} != query dimension {len(query)}.",
            )

        cosine_sim = _cosine_similarity(query, candidate)
        cand_hist = _activation_histogram(candidate, bins=query_bins)
        nmi, p_raw = _normalized_mutual_information(query_hist, cand_hist)
        p_bonferroni = _bonferroni_correction(p_raw, corpus_size)
        composite = request.alpha * cosine_sim + (1.0 - request.alpha) * nmi
        scored.append((idx, cosine_sim, nmi, composite, p_bonferroni))

    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[: request.top_k]

    results = [
        BatchSimilarityResult(
            index=item[0],
            cosine_similarity=round(item[1], 6),
            nmi=round(item[2], 6),
            composite_score=round(item[3], 6),
            p_value_bonferroni=round(item[4], 8),
            statistically_significant=item[4] < request.significance_threshold,
        )
        for item in top
    ]

    return BatchSimilarityResponse(
        results=results,
        corpus_size=corpus_size,
        alpha_used=request.alpha,
        significance_threshold=request.significance_threshold,
        top_k_requested=request.top_k,
    )


@app.post("/v1/similarity/filter-significant", response_model=SignificanceFilterResponse)
def filter_statistically_significant(
    request: SignificanceFilterRequest,
    api_key: str = Security(_authenticate),
) -> SignificanceFilterResponse:
    query = _validate_embedding(request.query_embedding, "query_embedding")
    corpus_size = len(request.corpus_embeddings)

    if corpus_size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="corpus_embeddings must contain at least one vector.",
        )

    query_bins = _freedman_diaconis_bins(_segment_magnitudes(query))
    query_hist = _activation_histogram(query, bins=query_bins)

    significant_indices: list[int] = []

    for idx, raw_vec in enumerate(request.corpus_embeddings):
        candidate = _validate_embedding(raw_vec, f"corpus_embeddings[{idx}]")
        if len(candidate) != len(query):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"corpus_embeddings[{idx}] dimension {len(candidate)} != query dimension {len(query)}.",
            )

        cosine_sim = _cosine_similarity(query, candidate)
        cand_hist = _activation_histogram(candidate, bins=query_bins)
        nmi, p_raw = _normalized_mutual_information(query_hist, cand_hist)
        p_bonferroni = _bonferroni_correction(p_raw, corpus_size)
        composite = request.alpha * cosine_sim + (1.0 - request.alpha) * nmi

        if p_bonferroni < request.significance_threshold and composite >= request.min_composite_score:
            significant_indices.append(idx)

    rejection_rate = round(1.0 - len(significant_indices) / corpus_size, 4)

    return SignificanceFilterResponse(
        significant_indices=significant_indices,
        significant_count=len(significant_indices),
        corpus_size=corpus_size,
        rejection_rate=rejection_rate,
    )


@app.get("/v1/health")
def health_check() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/v1/schema/composite-score")
def describe_composite_score_formula() -> dict:
    return {
        "formula": "S = alpha * cosine_similarity + (1 - alpha) * NMI",
        "alpha_default": 0.6,
        "alpha_range": [0.0, 1.0],
        "nmi_definition": "MI(P, Q) / sqrt(H(P) * H(Q))",
        "activation_histogram": {
            "method": "segment embedding into k=floor(sqrt(D)) windows, compute L2 norm per window",
            "binning": "Freedman-Diaconis rule applied to per-segment magnitudes",
        },
        "p_value": {
            "base_test": "chi-squared test of independence on joint activation histogram",
            "correction": "Bonferroni: p_corrected = min(1, p_raw * corpus_size)",
            "interpretation": "p_bonferroni < 0.05 means similarity is statistically significant given corpus size",
        },
        "when_cosine_misleads": (
            "High cosine (> 0.85) with low NMI (< 0.3) indicates vectors are geometrically "
            "close due to embedding space density, not shared informational structure. "
            "The composite score and p-value together disambiguate this case."
        ),
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


@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_by_nmi_cosine_composite', description='Ranks a corpus of candidate embeddings against a query embedding using a composite score that fuses cosine similarity with normalized mutual information (NMI) computed over activation-histogram distributions of each embedding. Returns ranked candidates with composite score, raw cosine, raw NMI, and calibrated p-value indicating whether the similarity is statistically significant given corpus size. Use when you need to distinguish genuine semantic similarity from spatial correlation artifacts in dense embedding subspaces. Do NOT use for exact nearest-neighbor lookup where statistical significance is irrelevant, or when corpus has fewer than 30 vectors (p-value calibration degrades below this threshold).')
async def rank_by_nmi_cosine_composite(query_embedding: Annotated[list[float], Field(..., description='Dense float vector representing the query item. Must match dimensionality of all candidate_embeddings vectors. Typically 128–4096 dims depending on model.', min_length=32, max_length=8192)], candidate_embeddings: Annotated[list[list[float]], Field(..., description='2-D array of candidate float vectors to rank. Each row is one embedding. All rows must share the same dimensionality as query_embedding. Minimum 30 rows for reliable p-value calibration.', min_length=2, max_length=10000)], histogram_bins: Annotated[float, Field(16, description="Number of bins used to discretize each embedding's activation distribution before computing NMI. Higher values increase NMI resolution but raise compute cost. Typical range 8–64; default 16 balances precision vs. speed.", ge=4, le=128)], alpha: Annotated[float, Field(0.05, description="Significance threshold for the calibrated p-value. Results with p > alpha are flagged as statistically non-significant. Does not filter results, only sets the 'significant' boolean field in each result row.", ge=0.001, le=0.2)], top_k: Annotated[float, Field(10, description='Maximum number of top-ranked results to return, ordered by composite score descending. Set to 0 to return all candidates.', ge=0, le=10000)], composite_weight_nmi: Annotated[float, Field(0.4, description='Weight assigned to normalized NMI in the composite score formula: composite = (1 - w) * cosine + w * nmi_normalized. Weight for cosine is implicitly (1 - composite_weight_nmi). Use higher values when embedding space is known to be anisotropic or cluster-dense.', ge=0.0, le=1.0)]) -> dict[str, Any]:
    """NMI-Cosine Composite Ranking"""
    params = {"query_embedding": query_embedding, "candidate_embeddings": candidate_embeddings, "histogram_bins": histogram_bins, "alpha": alpha, "top_k": top_k, "composite_weight_nmi": composite_weight_nmi}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/rank-composite', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_compare_embedding_pair_significance', description="Computes the composite similarity score and calibrated p-value for exactly one query-candidate pair, given a reference corpus used solely for null-distribution calibration. Use when you need to audit a single pair's similarity in isolation — e.g., deduplication checks, plagiarism signals, or validating a specific match from a prior ranking call. Do NOT use as a batch ranking method (use rank_by_nmi_cosine_composite instead); invoking this tool in a loop over N candidates is O(N) API calls and defeats the purpose.")
async def compare_embedding_pair_significance(embedding_a: Annotated[list[float], Field(..., description='First embedding vector of the pair being tested. Dimensionality must match embedding_b and all reference_corpus vectors.', min_length=32, max_length=8192)], embedding_b: Annotated[list[float], Field(..., description='Second embedding vector of the pair being tested.', min_length=32, max_length=8192)], reference_corpus: Annotated[list[list[float]], Field(..., description='Background corpus of embeddings used to build the null distribution for p-value calibration. Must have at least 30 rows. Not ranked or scored — used only for statistical reference.', min_length=30, max_length=10000)], histogram_bins: Annotated[float, Field(16, description='Bin count for activation-histogram discretization, consistent with what was used in any prior ranking call for comparability.', ge=4, le=128)]) -> dict[str, Any]:
    """Pairwise NMI-Cosine Significance Test"""
    params = {"embedding_a": embedding_a, "embedding_b": embedding_b, "reference_corpus": reference_corpus, "histogram_bins": histogram_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/pair-significance', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_estimate_corpus_anisotropy', description='Analyzes a corpus of embeddings to quantify anisotropy — the degree to which embeddings cluster in a low-dimensional subspace, which inflates cosine similarity scores for unrelated items. Returns the explained-variance ratio of the top principal components, average pairwise cosine baseline, and a recommended composite_weight_nmi value to compensate for the measured anisotropy. Use this BEFORE your first ranking call on a new corpus to calibrate composite_weight_nmi appropriately. Do NOT use as a substitute for actual similarity ranking — it produces diagnostics, not similarity scores.')
async def estimate_corpus_anisotropy(corpus_embeddings: Annotated[list[list[float]], Field(..., description='Full corpus of embeddings to analyze. Minimum 30 vectors required for meaningful PCA-based anisotropy estimation. Sampling a representative subset is acceptable for large corpora.', min_length=30, max_length=50000)], pca_components: Annotated[float, Field(20, description='Number of principal components to inspect when computing explained-variance ratio. Set to a value that covers the subspace you suspect is dominant (e.g., 10–50 for typical transformer embeddings).', ge=2, le=512)]) -> dict[str, Any]:
    """Corpus Anisotropy Diagnostics"""
    params = {"corpus_embeddings": corpus_embeddings, "pca_components": pca_components}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/corpus-anisotropy', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_filter_spurious_cosine_matches', description='Given a pre-ranked list of (candidate_index, cosine_score) pairs and the corresponding embeddings, recomputes NMI for each pair and removes matches whose composite score drops below a significance threshold — i.e., matches that were high-cosine but low-NMI, indicating spatial correlation rather than informational dependence. Returns the filtered and re-ranked list with composite scores and p-values. Use as a post-processing step on results from any cosine-only retrieval system (FAISS, Pinecone, etc.) to remove false positives before surfacing results to end users. Do NOT use as a primary ranking method when you have access to raw embeddings upfront — use rank_by_nmi_cosine_composite directly instead.')
async def filter_spurious_cosine_matches(query_embedding: Annotated[list[float], Field(..., description='The original query embedding used to produce the cosine-ranked candidate list.', min_length=32, max_length=8192)], candidate_embeddings: Annotated[list[list[float]], Field(..., description='Embeddings of the pre-ranked candidates, in the same order as cosine_scores. Must have the same number of rows as cosine_scores.', min_length=1, max_length=1000)], cosine_scores: Annotated[list[float], Field(..., description='Pre-computed cosine similarity scores for each candidate, parallel to candidate_embeddings. Values must be in [-1.0, 1.0]. Passed in to avoid redundant recomputation.', ge=-1.0, le=1.0, min_length=1, max_length=1000)], min_composite_score: Annotated[float, Field(0.5, description='Minimum composite score threshold below which a candidate is classified as a spurious match and excluded from the output. Candidates at or above this value are retained and re-ranked.', ge=0.0, le=1.0)], histogram_bins: Annotated[float, Field(16, description='Bin count for NMI histogram discretization. Should match the value used in any prior ranking call for score consistency.', ge=4, le=128)]) -> dict[str, Any]:
    """Spurious Cosine Match Filter"""
    params = {"query_embedding": query_embedding, "candidate_embeddings": candidate_embeddings, "cosine_scores": cosine_scores, "min_composite_score": min_composite_score, "histogram_bins": histogram_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/filter-spurious', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_calibrate_pvalue_null_distribution', description='Precomputes the null distribution of composite similarity scores for a given corpus by sampling random pairs and fitting a parametric distribution (beta or gamma) to their composite scores. Returns the distribution parameters and a lookup table of composite-score-to-p-value mappings. Use this once per stable corpus to cache calibration parameters and pass them to ranking calls, avoiding redundant null-distribution recomputation on every query. Do NOT call this on every query — it is intended for one-time or periodic calibration when corpus membership changes significantly (more than 10% churn).')
async def calibrate_pvalue_null_distribution(corpus_embeddings: Annotated[list[list[float]], Field(..., description='Full corpus of embeddings from which random pairs are sampled to build the null distribution. Minimum 50 vectors for stable distribution fitting.', min_length=50, max_length=50000)], n_null_samples: Annotated[float, Field(1000, description='Number of random pairs to sample from the corpus when constructing the null distribution. More samples produce a more stable p-value calibration at higher compute cost. 500–2000 is typical.', ge=100, le=10000)], histogram_bins: Annotated[float, Field(16, description='Bin count for NMI discretization, must match the value you intend to use in subsequent ranking calls for calibration to be valid.', ge=4, le=128)], composite_weight_nmi: Annotated[float, Field(0.4, description='NMI weight used in composite scoring, must match the value you intend to use in subsequent ranking calls.', ge=0.0, le=1.0)]) -> dict[str, Any]:
    """Null Distribution p-value Calibration"""
    params = {"corpus_embeddings": corpus_embeddings, "n_null_samples": n_null_samples, "histogram_bins": histogram_bins, "composite_weight_nmi": composite_weight_nmi}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/calibrate-null', params)


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
