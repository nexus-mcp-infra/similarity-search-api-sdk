from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, model_validator
from typing import Any
import numpy as np
from scipy.stats import entropy as scipy_entropy
import os
import hashlib
import time

from src.math.information import NormalizedMutualInformation, TransferEntropy
from src.math.causal import DoCalculus, CausalDAG, build_nexus_dag
from src.math.game_theory import NashEquilibrium, MarketEntryGame
from src.math.statistics import Statistics

app = FastAPI(
    title="NMI Similarity Search API",
    version="1.0.0",
    description="Stateless NMI-weighted cosine similarity over heterogeneous feature collections. Zero infrastructure, per-call.",
    docs_url="/docs",
    redoc_url=None,
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
_VALID_API_KEYS: set[str] = {k for k in os.environ.get("NMI_API_KEYS", "").split(",") if k}

MAX_COLLECTION_SIZE = 10_000
MAX_QUERY_SIZE = 500
MAX_DIMENSIONS = 2_048
MIN_DIMENSIONS = 1


class FeatureVector(BaseModel):
    id: str = Field(..., min_length=1, max_length=256)
    features: list[float | int | str] = Field(..., min_length=MIN_DIMENSIONS, max_length=MAX_DIMENSIONS)
    metadata: dict[str, Any] | None = None


class SimilaritySearchRequest(BaseModel):
    query: list[FeatureVector] = Field(..., min_length=1, max_length=MAX_QUERY_SIZE)
    collection: list[FeatureVector] = Field(..., min_length=1, max_length=MAX_COLLECTION_SIZE)
    top_k: int = Field(default=10, ge=1, le=500)
    nmi_weight_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Convex blend: final_score = alpha * nmi_cosine + (1 - alpha) * pure_cosine. Alpha=1.0 is full NMI-weighted.",
    )
    feature_types: list[str] | None = Field(
        default=None,
        description="Per-dimension type hint: 'continuous' or 'categorical'. If None, auto-inferred.",
    )

    @model_validator(mode="after")
    def validate_dimension_consistency(self) -> "SimilaritySearchRequest":
        dims = {len(v.features) for v in self.query + self.collection}
        if len(dims) > 1:
            raise ValueError(
                f"All feature vectors must share identical dimensionality. Found: {sorted(dims)}"
            )
        if self.feature_types is not None:
            d = len(self.query[0].features)
            if len(self.feature_types) != d:
                raise ValueError(
                    f"feature_types length ({len(self.feature_types)}) must equal feature dimensionality ({d})."
                )
            invalid = {t for t in self.feature_types if t not in ("continuous", "categorical")}
            if invalid:
                raise ValueError(f"Invalid feature_types values: {invalid}. Allowed: 'continuous', 'categorical'.")
        return self


class SimilarityResult(BaseModel):
    query_id: str
    matches: list[dict[str, Any]]
    nmi_weights: list[float]
    computation_ms: float


class SimilaritySearchResponse(BaseModel):
    results: list[SimilarityResult]
    total_comparisons: int
    api_version: str = "1.0.0"


class NMIWeightedSimilarityEngine:

    def __init__(self, nmi_calculator: NormalizedMutualInformation, stats: Statistics):
        self._nmi = nmi_calculator
        self._stats = stats

    def _infer_feature_types(self, matrix: np.ndarray) -> list[str]:
        types = []
        for dim_idx in range(matrix.shape[1]):
            col = matrix[:, dim_idx]
            unique_ratio = len(np.unique(col)) / len(col)
            types.append("categorical" if unique_ratio < 0.05 else "continuous")
        return types

    def _freedman_diaconis_bins(self, data: np.ndarray) -> int:
        n = len(data)
        if n < 2:
            return 1
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr == 0:
            return max(1, int(np.sqrt(n)))
        h = 2.0 * iqr * (n ** (-1.0 / 3.0))
        data_range = data.max() - data.min()
        if h == 0:
            return max(1, int(np.sqrt(n)))
        return max(1, int(np.ceil(data_range / h)))

    def _encode_features_to_numeric(self, raw_features: list[list[Any]]) -> np.ndarray:
        n = len(raw_features)
        d = len(raw_features[0])
        matrix = np.zeros((n, d), dtype=np.float64)
        for dim_idx in range(d):
            col_vals = [row[dim_idx] for row in raw_features]
            if isinstance(col_vals[0], str):
                unique_vals = list(dict.fromkeys(col_vals))
                mapping = {v: i for i, v in enumerate(unique_vals)}
                matrix[:, dim_idx] = np.array([float(mapping[v]) for v in col_vals])
            else:
                matrix[:, dim_idx] = np.array([float(v) for v in col_vals])
        return matrix

    def _compute_nmi_weights(
        self,
        collection_matrix: np.ndarray,
        query_vector: np.ndarray,
        feature_types: list[str],
    ) -> np.ndarray:
        n, d = collection_matrix.shape
        weights = np.zeros(d, dtype=np.float64)

        combined = np.vstack([collection_matrix, query_vector.reshape(1, -1)])

        for dim_idx in range(d):
            x_col = combined[:, dim_idx]
            y_col = np.linalg.norm(combined, axis=1)

            try:
                if feature_types[dim_idx] == "categorical":
                    nmi_val = self._nmi.compute_discrete(x_col, y_col)
                else:
                    n_bins = self._freedman_diaconis_bins(x_col)
                    nmi_val = self._nmi.compute_continuous(x_col, y_col, bins=n_bins)
            except Exception:
                nmi_val = 0.0

            weights[dim_idx] = max(0.0, float(nmi_val))

        weight_sum = weights.sum()
        if weight_sum < 1e-12:
            weights = np.ones(d, dtype=np.float64) / d
        else:
            weights = weights / weight_sum

        return weights

    def _cosine_similarity_weighted(
        self,
        query_vec: np.ndarray,
        collection_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        w_sqrt = np.sqrt(weights)
        q_weighted = query_vec * w_sqrt
        c_weighted = collection_matrix * w_sqrt[np.newaxis, :]

        q_norm = np.linalg.norm(q_weighted)
        c_norms = np.linalg.norm(c_weighted, axis=1)

        if q_norm < 1e-12:
            return np.zeros(len(collection_matrix), dtype=np.float64)

        dot_products = c_weighted @ q_weighted
        denominator = c_norms * q_norm
        denominator = np.where(denominator < 1e-12, 1e-12, denominator)

        return dot_products / denominator

    def _pure_cosine_similarity(
        self,
        query_vec: np.ndarray,
        collection_matrix: np.ndarray,
    ) -> np.ndarray:
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-12:
            return np.zeros(len(collection_matrix), dtype=np.float64)
        c_norms = np.linalg.norm(collection_matrix, axis=1)
        dot_products = collection_matrix @ query_vec
        denominator = c_norms * q_norm
        denominator = np.where(denominator < 1e-12, 1e-12, denominator)
        return dot_products / denominator

    def search_single_query(
        self,
        query_vec: np.ndarray,
        collection_matrix: np.ndarray,
        feature_types: list[str],
        top_k: int,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self._compute_nmi_weights(collection_matrix, query_vec, feature_types)

        nmi_scores = self._cosine_similarity_weighted(query_vec, collection_matrix, weights)
        pure_scores = self._pure_cosine_similarity(query_vec, collection_matrix)

        blended = alpha * nmi_scores + (1.0 - alpha) * pure_scores

        top_k_clamped = min(top_k, len(blended))
        top_indices = np.argpartition(blended, -top_k_clamped)[-top_k_clamped:]
        top_indices = top_indices[np.argsort(blended[top_indices])[::-1]]

        return top_indices, blended[top_indices], weights


def _require_valid_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if not _VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key store not configured. Set NMI_API_KEYS environment variable.",
        )
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    valid_hashes = {hashlib.sha256(k.encode()).hexdigest() for k in _VALID_API_KEYS}
    if key_hash not in valid_hashes:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return api_key


def _build_engine() -> NMIWeightedSimilarityEngine:
    nmi_calc = NormalizedMutualInformation()
    stats = Statistics()
    return NMIWeightedSimilarityEngine(nmi_calculator=nmi_calc, stats=stats)


_engine = _build_engine()


@app.post(
    "/v1/similarity/nmi-ranked",
    response_model=SimilaritySearchResponse,
    summary="NMI-weighted cosine similarity search over an ad-hoc collection",
    status_code=status.HTTP_200_OK,
)
async def nmi_ranked_similarity_search(
    request: SimilaritySearchRequest,
    _: str = Security(_require_valid_api_key),
) -> SimilaritySearchResponse:
    t_start = time.perf_counter()

    all_vectors = request.query + request.collection
    raw_features = [v.features for v in all_vectors]
    full_matrix = _engine._encode_features_to_numeric(raw_features)

    n_query = len(request.query)
    collection_matrix = full_matrix[n_query:]

    if request.feature_types is not None:
        feature_types = request.feature_types
    else:
        feature_types = _engine._infer_feature_types(full_matrix)

    results: list[SimilarityResult] = []

    for qi, qv in enumerate(request.query):
        query_vec = full_matrix[qi]
        q_start = time.perf_counter()

        top_indices, scores, weights = _engine.search_single_query(
            query_vec=query_vec,
            collection_matrix=collection_matrix,
            feature_types=feature_types,
            top_k=request.top_k,
            alpha=request.nmi_weight_alpha,
        )

        q_elapsed_ms = (time.perf_counter() - q_start) * 1_000

        matches = []
        for rank, (idx, score) in enumerate(zip(top_indices.tolist(), scores.tolist())):
            item = request.collection[idx]
            matches.append(
                {
                    "rank": rank + 1,
                    "id": item.id,
                    "score": round(float(score), 8),
                    "metadata": item.metadata,
                }
            )

        results.append(
            SimilarityResult(
                query_id=qv.id,
                matches=matches,
                nmi_weights=[round(float(w), 8) for w in weights.tolist()],
                computation_ms=round(q_elapsed_ms, 3),
            )
        )

    total_elapsed_ms = (time.perf_counter() - t_start) * 1_000

    return SimilaritySearchResponse(
        results=results,
        total_comparisons=len(request.query) * len(request.collection),
    )


@app.post(
    "/v1/similarity/nmi-weights-only",
    summary="Compute NMI weight vector for a collection without running full search",
    status_code=status.HTTP_200_OK,
)
async def compute_nmi_feature_weights(
    collection: list[FeatureVector] = ...,
    feature_types: list[str] | None = None,
    _: str = Security(_require_valid_api_key),
) -> dict[str, Any]:
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="collection must contain at least one vector.",
        )
    if len(collection) > MAX_COLLECTION_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"collection exceeds maximum size of {MAX_COLLECTION_SIZE}.",
        )

    dims = {len(v.features) for v in collection}
    if len(dims) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inconsistent feature dimensions in collection: {sorted(dims)}.",
        )

    raw_features = [v.features for v in collection]
    matrix = _engine._encode_features_to_numeric(raw_features)

    if feature_types is not None:
        d = matrix.shape[1]
        if len(feature_types) != d:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"feature_types length ({len(feature_types)}) does not match dimensionality ({d}).",
            )
        invalid = {t for t in feature_types if t not in ("continuous", "categorical")}
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid feature_types: {invalid}.",
            )
    else:
        feature_types = _engine._infer_feature_types(matrix)

    centroid = matrix.mean(axis=0)
    weights = _engine._compute_nmi_weights(matrix, centroid, feature_types)

    return {
        "nmi_weights": [round(float(w), 8) for w in weights.tolist()],
        "feature_types_used": feature_types,
        "dimensions": int(matrix.shape[1]),
        "collection_size": len(collection),
    }


@app.post(
    "/v1/similarity/cross-collection-nmi",
    summary="Compute pairwise NMI-weighted similarity matrix between two collections",
    status_code=status.HTTP_200_OK,
)
async def cross_collection_nmi_matrix(
    collection_a: list[FeatureVector] = ...,
    collection_b: list[FeatureVector] = ...,
    nmi_weight_alpha: float = 0.7,
    feature_types: list[str] | None = None,
    _: str = Security(_require_valid_api_key),
) -> dict[str, Any]:
    if not collection_a or not collection_b:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Both collection_a and collection_b must be non-empty.",
        )
    cross_size = len(collection_a) * len(collection_b)
    if cross_size > 250_000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cross-product size {cross_size} exceeds limit of 250,000. Reduce collection sizes.",
        )

    dims_a = {len(v.features) for v in collection_a}
    dims_b = {len(v.features) for v in collection_b}
    all_dims = dims_a | dims_b
    if len(all_dims) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dimension mismatch across collections: {sorted(all_dims)}.",
        )

    if not (0.0 <= nmi_weight_alpha <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="nmi_weight_alpha must be in [0.0, 1.0].",
        )

    combined_raw = [v.features for v in collection_a] + [v.features for v in collection_b]
    combined_matrix = _engine._encode_features_to_numeric(combined_raw)

    na = len(collection_a)
    matrix_a = combined_matrix[:na]
    matrix_b = combined_matrix[na:]

    if feature_types is not None:
        d = combined_matrix.shape[1]
        if len(feature_types) != d:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"feature_types length mismatch.",
            )
    else:
        feature_types = _engine._infer_feature_types(combined_matrix)

    similarity_matrix: list[list[float]] = []

    for i, row_a in enumerate(matrix_a):
        weights = _engine._compute_nmi_weights(matrix_b, row_a, feature_types)
        nmi_scores = _engine._cosine_similarity_weighted(row_a, matrix_b, weights)
        pure_scores = _engine._pure_cosine_similarity(row_a, matrix_b)
        blended = nmi_weight_alpha * nmi_scores + (1.0 - nmi_weight_alpha) * pure_scores
        similarity_matrix.append([round(float(s), 8) for s in blended.tolist()])

    return {
        "similarity_matrix": similarity_matrix,
        "row_ids": [v.id for v in collection_a],
        "col_ids": [v.id for v in collection_b],
        "shape": [len(collection_a), len(collection_b)],
        "total_comparisons": cross_size,
    }


@app.get(
    "/v1/healthz",
    summary="API liveness check with NMI engine self-test",
    status_code=status.HTTP_200_OK,
)
async def nmi_engine_health_check() -> dict[str, Any]:
    try:
        probe_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        probe_b = np.array([2.0, 4.0, 1.0, 3.0, 5.0])
        nmi_val = _engine._nmi.compute_continuous(probe_a, probe_b, bins=3)
        engine_ok = isinstance(nmi_val, (float, int, np.floating)) and not np.isnan(nmi_val)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"NMI engine self-test failed: {exc}",
        )

    return {
        "status": "ok" if engine_ok else "degraded",
        "nmi_engine_probe": round(float(nmi_val), 6),
        "max_collection_size": MAX_COLLECTION_SIZE,
        "max_query_size": MAX_QUERY_SIZE,
        "max_dimensions": MAX_DIMENSIONS,
        "api_version": "1.0.0",
    }


@app.get(
    "/v1/pricing",
    summary="Per-call pricing schedule for NMI similarity operations",
    status_code=status.HTTP_200_OK,
)
async def nmi_similarity_pricing_schedule() -> dict[str, Any]:
    return {
        "model": "per_call",
        "currency": "USD",
        "tiers": [
            {
                "operation": "nmi-ranked",
                "description": "NMI-weighted cosine search over ad-hoc collection",
                "base_price_usd": 0.002,
                "unit": "per query vector",
                "overage_per_1k_collection_items": 0.0005,
            },
            {
                "operation": "nmi-weights-only",
                "description": "NMI weight vector computation only, no search",
                "base_price_usd": 0.0008,
                "unit": "per call",
                "overage_per_1k_collection_items": 0.0002,
            },
            {
                "operation": "cross-collection-nmi",
                "description": "Full pairwise NMI similarity matrix between two collections",
                "base_price_usd": 0.005,
                "unit": "per call",
                "overage_per_10k_comparisons": 0.001,
            },
        ],
        "free_tier": {
            "calls_per_day": 100,
            "max_collection_size": 200,
            "max_dimensions": 64,
        },
        "complexity_note": "O(n*d*log(d)) per query where n=collection_size, d=dimensions. Billed on actual comparisons.",
    }