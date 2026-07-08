from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
import numpy as np
from scipy.special import digamma
from scipy.stats import entropy as scipy_entropy
import os
import time
import hashlib
import hmac

app = FastAPI(
    title="NMI-Cosine Similarity Search API",
    version="1.0.0",
    description="Hybrid NMI+Cosine ranking over in-memory corpora. No persistent index required.",
)

bearer_scheme = HTTPBearer()

VALID_API_KEYS: set[str] = set(
    k.strip() for k in os.environ.get("SIMILARITY_API_KEYS", "").split(",") if k.strip()
)


def _authenticate(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> str:
    token = credentials.credentials
    if not VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No API keys configured on server. Set SIMILARITY_API_KEYS env var.",
        )
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    for key in VALID_API_KEYS:
        if hmac.compare_digest(hashlib.sha256(key.encode()).hexdigest(), token_hash):
            return token
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


class DenseVectorCorpus(BaseModel):
    query: list[float] = Field(..., min_length=2, max_length=16384)
    corpus: list[list[float]] = Field(..., min_length=1, max_length=10000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=500)
    n_bins: Optional[int] = Field(default=None, ge=2, le=256)

    @field_validator("corpus")
    @classmethod
    def corpus_vectors_nonempty(cls, v: list[list[float]]) -> list[list[float]]:
        for i, vec in enumerate(v):
            if len(vec) < 2:
                raise ValueError(f"corpus[{i}] must have at least 2 dimensions")
            if len(vec) > 16384:
                raise ValueError(f"corpus[{i}] exceeds max dimension 16384")
        return v

    @model_validator(mode="after")
    def dimensions_consistent(self) -> "DenseVectorCorpus":
        d = len(self.query)
        for i, vec in enumerate(self.corpus):
            if len(vec) != d:
                raise ValueError(
                    f"corpus[{i}] dimension {len(vec)} != query dimension {d}"
                )
        return self


class DiscreteDistributionCorpus(BaseModel):
    query: list[float] = Field(..., min_length=2, max_length=65536)
    corpus: list[list[float]] = Field(..., min_length=1, max_length=10000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=500)

    @field_validator("query")
    @classmethod
    def query_is_valid_distribution(cls, v: list[float]) -> list[float]:
        arr = np.asarray(v, dtype=np.float64)
        if np.any(arr < 0.0):
            raise ValueError("query distribution must be non-negative")
        total = arr.sum()
        if total <= 0.0:
            raise ValueError("query distribution must sum to a positive value")
        return (arr / total).tolist()

    @field_validator("corpus")
    @classmethod
    def corpus_distributions_valid(cls, v: list[list[float]]) -> list[list[float]]:
        result = []
        for i, dist in enumerate(v):
            if len(dist) < 2:
                raise ValueError(f"corpus[{i}] must have at least 2 bins")
            arr = np.asarray(dist, dtype=np.float64)
            if np.any(arr < 0.0):
                raise ValueError(f"corpus[{i}] has negative probabilities")
            total = arr.sum()
            if total <= 0.0:
                raise ValueError(f"corpus[{i}] sums to zero")
            result.append((arr / total).tolist())
        return result

    @model_validator(mode="after")
    def dimensions_consistent(self) -> "DiscreteDistributionCorpus":
        d = len(self.query)
        for i, dist in enumerate(self.corpus):
            if len(dist) != d:
                raise ValueError(
                    f"corpus[{i}] length {len(dist)} != query length {d}"
                )
        return self


class SimilarityResult(BaseModel):
    index: int
    score: float
    nmi: float
    cosine: float


class SearchResponse(BaseModel):
    results: list[SimilarityResult]
    alpha: float
    input_type: Literal["dense_vector", "discrete_distribution"]
    corpus_size: int
    query_dim: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    numpy_version: str
    scipy_version: str


def _adaptive_bin_count(n_samples: int, user_bins: Optional[int]) -> int:
    if user_bins is not None:
        return user_bins
    sturges = int(np.ceil(np.log2(n_samples) + 1))
    scott_bandwidth = 3.5 * 1.0 / (n_samples ** (1.0 / 3.0))
    fd_bins = max(1, int(np.ceil(1.0 / scott_bandwidth)))
    return max(2, min(sturges, fd_bins, 64))


def _joint_histogram_normalized(
    x: np.ndarray, y: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1)
    y_edges = np.linspace(y.min() - 1e-10, y.max() + 1e-10, n_bins + 1)
    joint_counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    n = joint_counts.sum()
    if n == 0:
        return np.zeros((n_bins, n_bins)), np.zeros(n_bins), np.zeros(n_bins)
    p_xy = joint_counts / n
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    return p_xy, p_x, p_y


def _entropy_from_pmf(p: np.ndarray) -> float:
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def _miller_madow_correction(n_samples: int, n_nonzero_bins: int) -> float:
    return (n_nonzero_bins - 1.0) / (2.0 * max(n_samples, 1))


def _nmi_dense_strehl_ghosh(
    x: np.ndarray, y: np.ndarray, n_bins: int
) -> float:
    n = len(x)
    p_xy, p_x, p_y = _joint_histogram_normalized(x, y, n_bins)

    h_x_raw = _entropy_from_pmf(p_x)
    h_y_raw = _entropy_from_pmf(p_y)
    h_xy_raw = _entropy_from_pmf(p_xy.ravel())

    n_nonzero_x = int(np.sum(p_x > 0))
    n_nonzero_y = int(np.sum(p_y > 0))
    n_nonzero_xy = int(np.sum(p_xy > 0))

    correction_x = _miller_madow_correction(n, n_nonzero_x)
    correction_y = _miller_madow_correction(n, n_nonzero_y)
    correction_xy = _miller_madow_correction(n, n_nonzero_xy)

    h_x = h_x_raw + correction_x
    h_y = h_y_raw + correction_y
    h_xy = h_xy_raw + correction_xy

    mi = max(0.0, h_x + h_y - h_xy)

    denom = h_x + h_y
    if denom < 1e-12:
        return 0.0

    nmi_strehl_ghosh = 2.0 * mi / denom
    return float(np.clip(nmi_strehl_ghosh, 0.0, 1.0))


def _nmi_discrete_native(p: np.ndarray, q: np.ndarray) -> float:
    epsilon = 1e-12
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + epsilon)
    q = q / (q.sum() + epsilon)

    p_xy = np.outer(p, q)
    marginal_product = np.outer(p, q)

    mask = (p_xy > epsilon) & (marginal_product > epsilon)
    mi = float(np.sum(p_xy[mask] * np.log(p_xy[mask] / (marginal_product[mask] + epsilon))))
    mi = max(0.0, mi)

    h_p = float(-np.sum(p[p > epsilon] * np.log(p[p > epsilon])))
    h_q = float(-np.sum(q[q > epsilon] * np.log(q[q > epsilon])))

    denom = h_p + h_q
    if denom < 1e-12:
        return 1.0 if np.allclose(p, q, atol=1e-6) else 0.0

    nmi = 2.0 * mi / denom
    return float(np.clip(nmi, 0.0, 1.0))


def _cosine_similarity_batch(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-12:
        return np.zeros(len(corpus))
    corpus_norms = np.linalg.norm(corpus, axis=1)
    zero_mask = corpus_norms < 1e-12
    dots = corpus @ query
    with np.errstate(invalid="ignore", divide="ignore"):
        cosines = np.where(zero_mask, 0.0, dots / (corpus_norms * query_norm))
    return np.clip(cosines, -1.0, 1.0)


def _hybrid_scores_dense(
    query: np.ndarray,
    corpus: np.ndarray,
    alpha: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cosines = _cosine_similarity_batch(query, corpus)
    n = len(corpus)
    nmis = np.empty(n, dtype=np.float64)
    for i in range(n):
        nmis[i] = _nmi_dense_strehl_ghosh(query, corpus[i], n_bins)
    cosine_norm = (cosines + 1.0) / 2.0
    scores = alpha * nmis + (1.0 - alpha) * cosine_norm
    return scores, nmis, cosines


def _hybrid_scores_discrete(
    query: np.ndarray,
    corpus: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(corpus)
    nmis = np.empty(n, dtype=np.float64)
    for i in range(n):
        nmis[i] = _nmi_discrete_native(query, corpus[i])
    cosines = _cosine_similarity_batch(query, corpus)
    cosine_norm = (cosines + 1.0) / 2.0
    scores = alpha * nmis + (1.0 - alpha) * cosine_norm
    return scores, nmis, cosines


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    import scipy
    return HealthResponse(
        status="ok",
        version="1.0.0",
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
    )


@app.post(
    "/search/dense",
    response_model=SearchResponse,
    tags=["search"],
    summary="Rank dense vectors by NMI+Cosine hybrid score with Strehl-Ghosh bias correction",
)
def search_dense_vectors(
    payload: DenseVectorCorpus,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> SearchResponse:
    _authenticate(credentials)
    t0 = time.perf_counter()

    query_arr = np.asarray(payload.query, dtype=np.float64)
    corpus_arr = np.asarray(payload.corpus, dtype=np.float64)

    n_samples = len(query_arr)
    n_bins = _adaptive_bin_count(n_samples, payload.n_bins)

    if np.all(query_arr == query_arr[0]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query vector is constant — NMI is undefined for zero-variance inputs.",
        )

    scores, nmis, cosines = _hybrid_scores_dense(
        query_arr, corpus_arr, payload.alpha, n_bins
    )

    top_k = min(payload.top_k, len(scores))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = [
        SimilarityResult(
            index=int(i),
            score=float(scores[i]),
            nmi=float(nmis[i]),
            cosine=float(cosines[i]),
        )
        for i in top_indices
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return SearchResponse(
        results=results,
        alpha=payload.alpha,
        input_type="dense_vector",
        corpus_size=len(corpus_arr),
        query_dim=len(query_arr),
        latency_ms=round(latency_ms, 3),
    )


@app.post(
    "/search/distributions",
    response_model=SearchResponse,
    tags=["search"],
    summary="Rank discrete probability distributions by NMI+Cosine hybrid score (no binning required)",
)
def search_discrete_distributions(
    payload: DiscreteDistributionCorpus,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> SearchResponse:
    _authenticate(credentials)
    t0 = time.perf_counter()

    query_arr = np.asarray(payload.query, dtype=np.float64)
    corpus_arr = np.asarray(payload.corpus, dtype=np.float64)

    scores, nmis, cosines = _hybrid_scores_discrete(query_arr, corpus_arr, payload.alpha)

    top_k = min(payload.top_k, len(scores))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = [
        SimilarityResult(
            index=int(i),
            score=float(scores[i]),
            nmi=float(nmis[i]),
            cosine=float(cosines[i]),
        )
        for i in top_indices
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return SearchResponse(
        results=results,
        alpha=payload.alpha,
        input_type="discrete_distribution",
        corpus_size=len(corpus_arr),
        query_dim=len(query_arr),
        latency_ms=round(latency_ms, 3),
    )


@app.post(
    "/alpha/calibrate",
    tags=["math"],
    summary="Estimate optimal alpha from labeled pairs using NMI-vs-Cosine variance decomposition",
)
def calibrate_alpha(
    pairs: list[dict],
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> dict:
    _authenticate(credentials)

    if not pairs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="pairs list must not be empty",
        )
    if len(pairs) > 2000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum 2000 pairs per calibration call",
        )

    nmi_scores = []
    cosine_scores = []

    for idx, pair in enumerate(pairs):
        try:
            a = np.asarray(pair["a"], dtype=np.float64)
            b = np.asarray(pair["b"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"pairs[{idx}]: each pair must have keys 'a' and 'b' with numeric arrays. Error: {exc}",
            )

        if len(a) != len(b):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"pairs[{idx}]: 'a' and 'b' must have equal length",
            )
        if len(a) < 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"pairs[{idx}]: vectors must have at least 2 dimensions",
            )

        n_bins = _adaptive_bin_count(len(a), None)
        nmi_val = _nmi_dense_strehl_ghosh(a, b, n_bins)
        cos_val = float(
            np.clip(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12),
                -1.0,
                1.0,
            )
        )
        nmi_scores.append(nmi_val)
        cosine_scores.append((cos_val + 1.0) / 2.0)

    nmi_arr = np.asarray(nmi_scores)
    cos_arr = np.asarray(cosine_scores)

    var_nmi = float(np.var(nmi_arr))
    var_cos = float(np.var(cos_arr))
    total_var = var_nmi + var_cos

    if total_var < 1e-12:
        optimal_alpha = 0.5
        rationale = "both metrics have near-zero variance across pairs — default alpha=0.5 applied"
    else:
        optimal_alpha = float(np.clip(var_nmi / total_var, 0.0, 1.0))
        rationale = (
            f"variance-weighted: var(NMI)={var_nmi:.4f}, var(Cosine)={var_cos:.4f} — "
            f"higher variance metric contributes more to discrimination"
        )

    return {
        "optimal_alpha": round(optimal_alpha, 4),
        "var_nmi": round(var_nmi, 6),
        "var_cosine": round(var_cos, 6),
        "n_pairs": len(pairs),
        "rationale": rationale,
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


@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_by_nmi_cosine_hybrid', description='Rankea un corpus de vectores contra un vector query usando scoring híbrido NMI+Cosine ponderado por alpha. Úsalo cuando el corpus completo cabe en una sola llamada (<= 10000 vectores) y no se requiere índice persistente. NO usarlo si los vectores son continuos sin interpretación distribucional y alpha=1.0 (en ese caso usa rank_by_cosine_only para evitar overhead de estimación NMI).')
async def rank_by_nmi_cosine_hybrid(query_vector: Annotated[list[float], Field(..., description='Vector de consulta. Debe tener la misma dimensionalidad que cada vector del corpus. Acepta valores reales; si representa una distribución, debe sumar 1.0 (se valida internamente).', min_length=2, max_length=4096)], corpus_vectors: Annotated[list[list[float]], Field(..., description='Matriz de vectores candidatos [n_docs x n_dims]. Cada fila es un documento. Máximo 10000 filas; todas las filas deben tener la misma longitud que query_vector.', min_length=1, max_length=10000)], alpha: Annotated[float, Field(0.5, description='Peso del componente NMI en el score híbrido: score = alpha * NMI + (1 - alpha) * Cosine. 0.0 = solo coseno, 1.0 = solo NMI. Valores recomendados: 0.3-0.7 para corpus distribucionales mixtos.', ge=0.0, le=1.0)], top_k: Annotated[float, Field(10, description='Número de resultados más similares a devolver, ordenados por score descendente.', ge=1, le=1000)], bias_correction: Annotated[bool, Field(True, description='Aplica corrección de bias de Strehl-Ghosh para NMI normalizado en muestras pequeñas. Recomendado true cuando n_dims < 50. Añade ~15% de latencia.')], n_bins: Annotated[float, Field(10, description='Número de bins para discretización al estimar distribuciones conjuntas de vectores continuos. Ignorado si los vectores ya son distribuciones discretas (suman 1.0). Rango útil: 5-20; valores altos aumentan precisión pero requieren más muestras.', ge=2, le=50)]) -> dict[str, Any]:
    """NMI+Cosine Hybrid Ranking"""
    params = {"query_vector": query_vector, "corpus_vectors": corpus_vectors, "alpha": alpha, "top_k": top_k, "bias_correction": bias_correction, "n_bins": n_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/rank-nmi-cosine', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_by_cosine_only', description='Rankea un corpus contra un query usando exclusivamente similitud coseno. Úsalo cuando los vectores son embeddings densos continuos sin componente distribucional y se requiere máxima velocidad. NO usarlo si el diferenciador NMI es necesario para el caso de uso — en ese caso usa rank_by_nmi_cosine_hybrid con alpha apropiado.')
async def rank_by_cosine_only(query_vector: Annotated[list[float], Field(..., description='Vector de consulta de dimensionalidad arbitraria pero consistente con corpus_vectors.', min_length=2, max_length=4096)], corpus_vectors: Annotated[list[list[float]], Field(..., description='Matriz de vectores candidatos [n_docs x n_dims]. Máximo 10000 filas.', min_length=1, max_length=10000)], top_k: Annotated[float, Field(10, description='Número de resultados más similares a devolver.', ge=1, le=1000)]) -> dict[str, Any]:
    """Pure Cosine Similarity Ranking"""
    params = {"query_vector": query_vector, "corpus_vectors": corpus_vectors, "top_k": top_k}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/rank-cosine', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_compute_pairwise_nmi_matrix', description='Calcula la matriz NMI normalizada (con corrección Strehl-Ghosh opcional) para todos los pares de un conjunto de vectores. Úsalo para clustering exploratorio, análisis de redundancia o construcción de grafos de similitud. NO usarlo como paso previo a rank_by_nmi_cosine_hybrid — ese endpoint ya incorpora NMI internamente y calcular la matriz completa sería redundante y más costoso.')
async def compute_pairwise_nmi_matrix(vectors: Annotated[list[list[float]], Field(..., description='Conjunto de vectores [n x d]. La matriz resultante será [n x n]. Límite: 500 vectores (la operación es O(n^2 * d)).', min_length=2, max_length=500)], bias_correction: Annotated[bool, Field(True, description='Aplica corrección de bias de Strehl-Ghosh. Obligatorio para conjuntos con d < 30 para evitar sobreestimación sistemática de NMI.')], n_bins: Annotated[float, Field(10, description='Bins para discretización de vectores continuos en estimación de entropía conjunta.', ge=2, le=50)], return_upper_triangle_only: Annotated[bool, Field(False, description='Si true, devuelve solo el triángulo superior de la matriz (la matriz NMI es simétrica), reduciendo el payload a la mitad.')]) -> dict[str, Any]:
    """Pairwise NMI Matrix"""
    params = {"vectors": vectors, "bias_correction": bias_correction, "n_bins": n_bins, "return_upper_triangle_only": return_upper_triangle_only}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/pairwise-nmi-matrix', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_score_discrete_distribution_pair', description='Calcula el NMI normalizado entre exactamente dos distribuciones de probabilidad discretas, con corrección de bias y reporte de entropías marginales y conjunta. Úsalo para diagnóstico, validación o cuando solo se necesita comparar un par específico sin overhead de ranking. NO usarlo en bucles sobre corpus grandes — rank_by_nmi_cosine_hybrid es O(n log n) y más eficiente para ese caso.')
async def score_discrete_distribution_pair(distribution_p: Annotated[list[float], Field(..., description='Primera distribución de probabilidad discreta. Debe sumar 1.0 (tolerancia 1e-6). Misma longitud que distribution_q.', min_length=2, max_length=4096)], distribution_q: Annotated[list[float], Field(..., description='Segunda distribución de probabilidad discreta. Debe sumar 1.0 (tolerancia 1e-6). Misma longitud que distribution_p.', min_length=2, max_length=4096)], bias_correction: Annotated[bool, Field(True, description='Aplica corrección de Strehl-Ghosh. Especialmente importante cuando el número de categorías (longitud del vector) supera las muestras efectivas implícitas en la distribución.')]) -> dict[str, Any]:
    """NMI Score for Distribution Pair"""
    params = {"distribution_p": distribution_p, "distribution_q": distribution_q, "bias_correction": bias_correction}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/score-distribution-pair', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_calibrate_alpha_for_corpus', description='Sugiere el valor óptimo de alpha para rank_by_nmi_cosine_hybrid dado un corpus de muestra, midiendo la concordancia de ranking entre NMI puro y Cosine puro (correlación de Spearman entre órdenes). Un alpha óptimo minimiza la divergencia de rankings ponderada por la confianza estadística de NMI. Úsalo una vez antes de desplegar el ranking en producción para calibrar el hiperparámetro. NO usarlo en cada petición de ranking — el resultado es estable para corpus con distribución similar.')
async def calibrate_alpha_for_corpus(sample_query_vectors: Annotated[list[list[float]], Field(..., description='Conjunto de vectores query representativos del caso de uso [n_queries x n_dims]. Mínimo 5 queries para que la calibración sea estadísticamente significativa.', min_length=5, max_length=200)], sample_corpus_vectors: Annotated[list[list[float]], Field(..., description='Muestra del corpus sobre el que se realizarán los rankings reales [n_docs x n_dims]. Debe ser representativa de la distribución real del corpus.', min_length=10, max_length=2000)], alpha_candidates: Annotated[list[float], Field(None, description='Valores de alpha a evaluar. Si se omite, se usa una grilla uniforme de 0.0 a 1.0 con paso 0.1. Cada valor debe estar en [0.0, 1.0].', min_length=2, max_length=20)], n_bins: Annotated[float, Field(10, description='Bins para discretización usados durante la calibración. Debe coincidir con el valor que se usará en producción.', ge=2, le=50)]) -> dict[str, Any]:
    """Alpha Calibration via NMI-Cosine Agreement"""
    params = {"sample_query_vectors": sample_query_vectors, "sample_corpus_vectors": sample_corpus_vectors, "alpha_candidates": alpha_candidates, "n_bins": n_bins}
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
