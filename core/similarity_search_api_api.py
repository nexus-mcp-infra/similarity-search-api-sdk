from __future__ import annotations

import math
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder

app = FastAPI(
    title="Hybrid NMI-Cosine Similarity API",
    version="1.0.0",
    description="Stateless hybrid similarity scoring for mixed-type feature sets.",
)

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

Record = dict[str, Any]


class PairwiseScoreRequest(BaseModel):
    record_a: Record = Field(..., min_length=1)
    record_b: Record = Field(..., min_length=1)
    categorical_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override auto-detected categorical weight in [0,1]. If None, inferred from feature-type proportions.",
    )

    @field_validator("record_a", "record_b")
    @classmethod
    def non_empty_record(cls, v: Record) -> Record:
        if not v:
            raise ValueError("Record must contain at least one feature.")
        return v


class PairwiseScoreResponse(BaseModel):
    hybrid_score: float
    categorical_weight: float
    continuous_weight: float
    n_categorical_features: int
    n_continuous_features: int


class RankCandidatesRequest(BaseModel):
    query: Record = Field(..., min_length=1)
    candidates: list[Record] = Field(..., min_length=1, max_length=2000)
    categorical_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def non_empty_query(cls, v: Record) -> Record:
        if not v:
            raise ValueError("Query record must contain at least one feature.")
        return v


class RankedCandidate(BaseModel):
    index: int
    hybrid_score: float
    record: Record


class RankCandidatesResponse(BaseModel):
    ranked_candidates: list[RankedCandidate]
    categorical_weight: float
    continuous_weight: float


class DecompositionRequest(BaseModel):
    record_a: Record = Field(..., min_length=1)
    record_b: Record = Field(..., min_length=1)
    categorical_weight: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("record_a", "record_b")
    @classmethod
    def non_empty_record(cls, v: Record) -> Record:
        if not v:
            raise ValueError("Record must contain at least one feature.")
        return v


class FeatureContribution(BaseModel):
    feature: str
    feature_type: str
    raw_score: float
    weighted_contribution: float


class DecompositionResponse(BaseModel):
    hybrid_score: float
    categorical_weight: float
    continuous_weight: float
    feature_contributions: list[FeatureContribution]


class LabeledPair(BaseModel):
    record_a: Record
    record_b: Record
    ground_truth: float = Field(..., ge=0.0, le=1.0)


class CalibrateWeightRequest(BaseModel):
    labeled_pairs: list[LabeledPair] = Field(..., min_length=20)
    grid_steps: int = Field(default=21, ge=2, le=101)


class CalibrateWeightResponse(BaseModel):
    optimal_categorical_weight: float
    min_mse: float
    grid_mse: list[dict[str, float]]


class BenchmarkRequest(BaseModel):
    labeled_pairs: list[LabeledPair] = Field(..., min_length=5)
    categorical_weight: float | None = Field(default=None, ge=0.0, le=1.0)


class BenchmarkMetrics(BaseModel):
    mse: float
    mae: float
    spearman_rho: float
    spearman_pvalue: float


class BenchmarkResponse(BaseModel):
    hybrid: BenchmarkMetrics
    cosine_baseline: BenchmarkMetrics
    hybrid_mse_lift_pct: float
    hybrid_spearman_lift_pct: float


# ---------------------------------------------------------------------------
# Core math: type detection
# ---------------------------------------------------------------------------

def _is_continuous(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _classify_features(
    record_a: Record, record_b: Record
) -> tuple[list[str], list[str]]:
    shared_keys = set(record_a.keys()) & set(record_b.keys())
    if not shared_keys:
        raise HTTPException(
            status_code=422,
            detail="Records share no common feature keys. Cannot compute similarity.",
        )
    categorical: list[str] = []
    continuous: list[str] = []
    for k in sorted(shared_keys):
        if _is_continuous(record_a[k]) and _is_continuous(record_b[k]):
            continuous.append(k)
        else:
            categorical.append(k)
    return categorical, continuous


def _auto_weight(n_cat: int, n_cont: int) -> float:
    total = n_cat + n_cont
    if total == 0:
        return 0.5
    return n_cat / total


# ---------------------------------------------------------------------------
# Core math: NMI for a single categorical feature (pair of scalar values)
# ---------------------------------------------------------------------------

def _nmi_single_feature_pair(val_a: Any, val_b: Any) -> float:
    str_a = str(val_a)
    str_b = str(val_b)
    if str_a == str_b:
        return 1.0
    return 0.0


def _nmi_feature_corpus(
    values_a: list[Any], values_b: list[Any]
) -> float:
    le = LabelEncoder()
    all_vals_a = [str(v) for v in values_a]
    all_vals_b = [str(v) for v in values_b]
    encoded_a = le.fit_transform(all_vals_a)
    le2 = LabelEncoder()
    encoded_b = le2.fit_transform(all_vals_b)

    mi = mutual_info_score(encoded_a, encoded_b)

    def entropy(labels: np.ndarray) -> float:
        n = len(labels)
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / n
        return float(-np.sum(probs * np.log(probs + 1e-15)))

    h_a = entropy(encoded_a)
    h_b = entropy(encoded_b)
    denom = math.sqrt(h_a * h_b)
    if denom < 1e-12:
        return 0.0
    return float(np.clip(mi / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core math: Cosine similarity for continuous feature vectors
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.5
    raw = float(np.dot(a, b) / (norm_a * norm_b))
    return float((raw + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# Core math: hybrid scorer
# ---------------------------------------------------------------------------

def _hybrid_score(
    record_a: Record,
    record_b: Record,
    categorical_weight: float | None = None,
) -> tuple[float, float, float, list[str], list[str]]:
    cat_keys, cont_keys = _classify_features(record_a, record_b)
    n_cat = len(cat_keys)
    n_cont = len(cont_keys)

    w_cat = categorical_weight if categorical_weight is not None else _auto_weight(n_cat, n_cont)
    w_cont = 1.0 - w_cat

    cat_score = 0.0
    if n_cat > 0:
        nmi_scores = [_nmi_single_feature_pair(record_a[k], record_b[k]) for k in cat_keys]
        cat_score = float(np.mean(nmi_scores))

    cont_score = 0.0
    if n_cont > 0:
        vec_a = [float(record_a[k]) for k in cont_keys]
        vec_b = [float(record_b[k]) for k in cont_keys]
        cont_score = _cosine_similarity(vec_a, vec_b)

    if n_cat == 0:
        score = cont_score
    elif n_cont == 0:
        score = cat_score
    else:
        score = w_cat * cat_score + w_cont * cont_score

    return float(np.clip(score, 0.0, 1.0)), w_cat, w_cont, cat_keys, cont_keys


def _cosine_only_score(record_a: Record, record_b: Record) -> float:
    shared_keys = sorted(set(record_a.keys()) & set(record_b.keys()))
    if not shared_keys:
        return 0.0

    vec_a = []
    vec_b = []
    for k in shared_keys:
        if _is_continuous(record_a[k]) and _is_continuous(record_b[k]):
            vec_a.append(float(record_a[k]))
            vec_b.append(float(record_b[k]))
        else:
            enc = LabelEncoder()
            encoded = enc.fit_transform([str(record_a[k]), str(record_b[k])])
            vec_a.append(float(encoded[0]))
            vec_b.append(float(encoded[1]))

    return _cosine_similarity(vec_a, vec_b)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/similarity/pairwise-score", response_model=PairwiseScoreResponse)
def score_mixed_feature_similarity(request: PairwiseScoreRequest) -> PairwiseScoreResponse:
    try:
        score, w_cat, w_cont, cat_keys, cont_keys = _hybrid_score(
            request.record_a,
            request.record_b,
            request.categorical_weight,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Scoring failed: {exc}") from exc

    return PairwiseScoreResponse(
        hybrid_score=score,
        categorical_weight=w_cat,
        continuous_weight=w_cont,
        n_categorical_features=len(cat_keys),
        n_continuous_features=len(cont_keys),
    )


@app.post("/v1/similarity/rank-candidates", response_model=RankCandidatesResponse)
def rank_candidates_by_hybrid_similarity(request: RankCandidatesRequest) -> RankCandidatesResponse:
    if len(request.candidates) > 2000:
        raise HTTPException(
            status_code=422,
            detail="Candidate set exceeds 2000 records. Latency guarantee cannot be maintained.",
        )

    scored: list[tuple[int, float]] = []
    w_cat_final = 0.0
    w_cont_final = 0.0

    for idx, candidate in enumerate(request.candidates):
        try:
            score, w_cat, w_cont, _, _ = _hybrid_score(
                request.query,
                candidate,
                request.categorical_weight,
            )
            scored.append((idx, score))
            w_cat_final = w_cat
            w_cont_final = w_cont
        except HTTPException as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Scoring failed for candidate at index {idx}: {exc.detail}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Scoring failed for candidate at index {idx}: {exc}",
            ) from exc

    scored.sort(key=lambda t: t[1], reverse=True)

    if request.top_k is not None:
        scored = scored[: request.top_k]

    ranked = [
        RankedCandidate(index=idx, hybrid_score=score, record=request.candidates[idx])
        for idx, score in scored
    ]

    return RankCandidatesResponse(
        ranked_candidates=ranked,
        categorical_weight=w_cat_final,
        continuous_weight=w_cont_final,
    )


@app.post("/v1/similarity/feature-decomposition", response_model=DecompositionResponse)
def decompose_hybrid_score_by_feature(request: DecompositionRequest) -> DecompositionResponse:
    try:
        score, w_cat, w_cont, cat_keys, cont_keys = _hybrid_score(
            request.record_a,
            request.record_b,
            request.categorical_weight,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Decomposition failed: {exc}") from exc

    contributions: list[FeatureContribution] = []

    n_cat = len(cat_keys)
    n_cont = len(cont_keys)

    for k in cat_keys:
        raw = _nmi_single_feature_pair(request.record_a[k], request.record_b[k])
        weighted = (w_cat * raw / n_cat) if n_cat > 0 else 0.0
        contributions.append(
            FeatureContribution(
                feature=k,
                feature_type="categorical",
                raw_score=raw,
                weighted_contribution=weighted,
            )
        )

    if n_cont > 0:
        vec_a = [float(request.record_a[k]) for k in cont_keys]
        vec_b = [float(request.record_b[k]) for k in cont_keys]
        a = np.array(vec_a, dtype=np.float64)
        b = np.array(vec_b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        full_cosine = _cosine_similarity(vec_a, vec_b)

        for i, k in enumerate(cont_keys):
            if norm_a < 1e-12 or norm_b < 1e-12:
                component_raw = 0.5
            else:
                component_raw = float(
                    ((a[i] * b[i]) / (norm_a * norm_b) + 1.0) / 2.0
                )
            weighted = (w_cont * full_cosine / n_cont) if n_cont > 0 else 0.0
            contributions.append(
                FeatureContribution(
                    feature=k,
                    feature_type="continuous",
                    raw_score=component_raw,
                    weighted_contribution=weighted,
                )
            )

    return DecompositionResponse(
        hybrid_score=score,
        categorical_weight=w_cat,
        continuous_weight=w_cont,
        feature_contributions=contributions,
    )


@app.post("/v1/similarity/calibrate-weight", response_model=CalibrateWeightResponse)
def estimate_optimal_categorical_weight(request: CalibrateWeightRequest) -> CalibrateWeightResponse:
    if len(request.labeled_pairs) < 20:
        raise HTTPException(
            status_code=422,
            detail="Fewer than 20 labeled pairs provided. Calibration results would be unreliable.",
        )

    grid = np.linspace(0.0, 1.0, request.grid_steps)
    grid_results: list[dict[str, float]] = []
    best_w = 0.0
    best_mse = float("inf")

    ground_truths = [pair.ground_truth for pair in request.labeled_pairs]

    for w in grid:
        predictions = []
        for pair in request.labeled_pairs:
            try:
                score, _, _, _, _ = _hybrid_score(pair.record_a, pair.record_b, float(w))
                predictions.append(score)
            except Exception:
                predictions.append(0.0)

        gt_arr = np.array(ground_truths)
        pred_arr = np.array(predictions)
        mse = float(np.mean((pred_arr - gt_arr) ** 2))
        grid_results.append({"categorical_weight": float(w), "mse": mse})

        if mse < best_mse:
            best_mse = mse
            best_w = float(w)

    return CalibrateWeightResponse(
        optimal_categorical_weight=best_w,
        min_mse=best_mse,
        grid_mse=grid_results,
    )


@app.post("/v1/similarity/benchmark-hybrid-vs-cosine", response_model=BenchmarkResponse)
def benchmark_hybrid_vs_cosine_only(request: BenchmarkRequest) -> BenchmarkResponse:
    hybrid_preds: list[float] = []
    cosine_preds: list[float] = []
    ground_truths: list[float] = []

    for pair in request.labeled_pairs:
        gt = pair.ground_truth
        ground_truths.append(gt)

        try:
            h_score, _, _, _, _ = _hybrid_score(
                pair.record_a, pair.record_b, request.categorical_weight
            )
        except Exception:
            h_score = 0.0
        hybrid_preds.append(h_score)

        try:
            c_score = _cosine_only_score(pair.record_a, pair.record_b)
        except Exception:
            c_score = 0.0
        cosine_preds.append(c_score)

    gt_arr = np.array(ground_truths)
    hyb_arr = np.array(hybrid_preds)
    cos_arr = np.array(cosine_preds)

    def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> BenchmarkMetrics:
        mse = float(np.mean((preds - labels) ** 2))
        mae = float(np.mean(np.abs(preds - labels)))
        if len(preds) < 3:
            rho, pval = 0.0, 1.0
        else:
            result = spearmanr(preds, labels)
            rho = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
            pval = float(result.pvalue) if hasattr(result, "pvalue") else float(result[1])
        return BenchmarkMetrics(mse=mse, mae=mae, spearman_rho=rho, spearman_pvalue=pval)

    hybrid_metrics = compute_metrics(hyb_arr, gt_arr)
    cosine_metrics = compute_metrics(cos_arr, gt_arr)

    cosine_mse = cosine_metrics.mse
    hybrid_mse = hybrid_metrics.mse
    mse_lift_pct = (
        float((cosine_mse - hybrid_mse) / cosine_mse * 100.0)
        if cosine_mse > 1e-12
        else 0.0
    )

    cosine_rho = cosine_metrics.spearman_rho
    hybrid_rho = hybrid_metrics.spearman_rho
    spearman_lift_pct = (
        float((hybrid_rho - cosine_rho) / abs(cosine_rho) * 100.0)
        if abs(cosine_rho) > 1e-12
        else 0.0
    )

    return BenchmarkResponse(
        hybrid=hybrid_metrics,
        cosine_baseline=cosine_metrics,
        hybrid_mse_lift_pct=mse_lift_pct,
        hybrid_spearman_lift_pct=spearman_lift_pct,
    )


from typing import Annotated, Literal
from contextlib import AsyncExitStack as _NexusMcpExitStack

import os
import httpx
from mcp.server.fastmcp import FastMCP as _NexusFastMCP
from mcp.server.transport_security import TransportSecuritySettings

_nexus_railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "*")

_nexus_mcp = _NexusFastMCP(
    "nexus-similarity-search-api",
    stateless_http=True,
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
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


async def _nexus_mcp_call_core(method: str, path: str, params: dict) -> Any:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nexus-internal") as client:
        if method == "GET":
            resp = await client.get(path, params=params)
        else:
            resp = await client.post(path, json=params)
        resp.raise_for_status()
        return resp.json()


@_nexus_mcp.tool(
    name="nexus_similarity_search_api_score_mixed_feature_similarity",
    description=(
        "Computes a weighted hybrid similarity score between two individual records using NMI for "
        "categorical features and Cosine similarity for continuous features, fused into a single "
        "normalized scalar. Use when you need a single mathematically coherent similarity value "
        "between two mixed-type records without maintaining any index or state. Do NOT use when "
        "comparing more than two records at once (use rank_candidates_by_hybrid_similarity instead), "
        "or when all features are purely continuous (Cosine alone is sufficient and cheaper)."
    ),
)
async def score_mixed_feature_similarity_mcp(
    record_a: Annotated[str, Field(..., description="JSON-encoded record object with keys matching feature_schema. Continuous features must be numeric; categorical features must be strings or integers.", min_length=2, max_length=32768)],
    record_b: Annotated[str, Field(..., description="JSON-encoded record object with the same key set as record_a. Both records must share identical feature keys.", min_length=2, max_length=32768)],
    feature_schema: Annotated[str, Field(..., description='JSON-encoded object mapping each feature key to "continuous" or "categorical". Example: {"price": "continuous", "category": "categorical"}. Keys not listed are ignored.', min_length=2, max_length=4096)],
    categorical_weight: Annotated[float, Field(0.5, description="Weight assigned to the NMI component in the final hybrid score. The continuous (Cosine) component receives weight (1 - categorical_weight). Must be in [0.0, 1.0]. Set to 0.0 to use pure Cosine; set to 1.0 to use pure NMI.", ge=0.0, le=1.0)],
) -> dict[str, Any]:
    params = {
        "record_a": record_a,
        "record_b": record_b,
        "feature_schema": feature_schema,
        "categorical_weight": categorical_weight,
    }
    return await _nexus_mcp_call_core("POST", "/v1/similarity/pairwise-score", params)


@_nexus_mcp.tool(
    name="nexus_similarity_search_api_rank_candidates_by_hybrid_similarity",
    description=(
        "Scores a query record against a list of candidate records using the same NMI-Cosine hybrid "
        "scorer and returns candidates sorted by descending hybrid score. Use when performing "
        "nearest-neighbor retrieval over a small-to-medium candidate set (up to 2000 records) without "
        "a persistent vector index. Do NOT use with candidate sets larger than 2000 records; do NOT "
        "use when you only need the top-1 score between exactly two records (use "
        "score_mixed_feature_similarity instead)."
    ),
)
async def rank_candidates_by_hybrid_similarity_mcp(
    query_record: Annotated[str, Field(..., description="JSON-encoded record object representing the query. Must share the same feature keys defined in feature_schema.", min_length=2, max_length=32768)],
    candidates: Annotated[str, Field(..., description="JSON-encoded array of record objects to rank. Each object must contain the same keys as query_record. Maximum 2000 elements.", min_length=2, max_length=2097152)],
    feature_schema: Annotated[str, Field(..., description='JSON-encoded object mapping feature keys to "continuous" or "categorical". Shared across query and all candidates.', min_length=2, max_length=4096)],
    categorical_weight: Annotated[float, Field(0.5, description="Weight for the NMI component in [0.0, 1.0]. Continuous (Cosine) component receives the complementary weight.", ge=0.0, le=1.0)],
    top_k: Annotated[float, Field(10, description="Number of top-ranked candidates to return. Must be between 1 and the number of candidates provided.", ge=1, le=2000)],
) -> dict[str, Any]:
    params = {
        "query_record": query_record,
        "candidates": candidates,
        "feature_schema": feature_schema,
        "categorical_weight": categorical_weight,
        "top_k": top_k,
    }
    return await _nexus_mcp_call_core("POST", "/v1/similarity/rank-candidates", params)


@_nexus_mcp.tool(
    name="nexus_similarity_search_api_decompose_hybrid_score_by_feature",
    description=(
        "Returns the per-feature contribution to the hybrid score between two records: individual NMI "
        "values for each categorical feature and individual cosine components for each continuous "
        "feature, plus the aggregated hybrid scalar. Use when you need interpretability or need to "
        "debug why two records received a particular score. Do NOT use in high-throughput scoring "
        "loops where only the final scalar is needed (overhead from decomposition is ~3x versus "
        "score_mixed_feature_similarity)."
    ),
)
async def decompose_hybrid_score_by_feature_mcp(
    record_a: Annotated[str, Field(..., description="JSON-encoded record object with keys matching feature_schema.", min_length=2, max_length=32768)],
    record_b: Annotated[str, Field(..., description="JSON-encoded record object with the same key set as record_a.", min_length=2, max_length=32768)],
    feature_schema: Annotated[str, Field(..., description='JSON-encoded object mapping feature keys to "continuous" or "categorical".', min_length=2, max_length=4096)],
    categorical_weight: Annotated[float, Field(0.5, description="Weight for the NMI component in the hybrid aggregation, in [0.0, 1.0].", ge=0.0, le=1.0)],
) -> dict[str, Any]:
    params = {
        "record_a": record_a,
        "record_b": record_b,
        "feature_schema": feature_schema,
        "categorical_weight": categorical_weight,
    }
    return await _nexus_mcp_call_core("POST", "/v1/similarity/feature-decomposition", params)


@_nexus_mcp.tool(
    name="nexus_similarity_search_api_estimate_optimal_categorical_weight",
    description=(
        "Given a labeled dataset of record pairs with ground-truth similarity judgments (0.0 to 1.0), "
        "estimates the categorical_weight value that minimizes mean squared error between hybrid scores "
        "and ground-truth labels via grid search over [0.0, 1.0] in configurable steps. Use when you "
        "have a gold-standard evaluation set and want to calibrate the blend before deploying scoring "
        "at scale. Do NOT use as a substitute for proper cross-validation on large datasets; do NOT "
        "use when you have fewer than 20 labeled pairs (results will be unreliable)."
    ),
)
async def estimate_optimal_categorical_weight_mcp(
    labeled_pairs: Annotated[str, Field(..., description="JSON-encoded array of objects, each with keys 'record_a', 'record_b' (record objects), and 'similarity_label' (float in [0.0, 1.0]). Minimum 20 pairs, maximum 5000 pairs.", min_length=2, max_length=4194304)],
    feature_schema: Annotated[str, Field(..., description='JSON-encoded object mapping feature keys to "continuous" or "categorical". Applied uniformly to all pairs.', min_length=2, max_length=4096)],
    grid_steps: Annotated[float, Field(20, description="Number of evenly spaced weight values to evaluate between 0.0 and 1.0 inclusive. Higher values increase calibration precision at linear cost. Allowed range: 5 to 200.", ge=5, le=200)],
) -> dict[str, Any]:
    params = {
        "labeled_pairs": labeled_pairs,
        "feature_schema": feature_schema,
        "grid_steps": grid_steps,
    }
    return await _nexus_mcp_call_core("POST", "/v1/similarity/calibrate-weight", params)


@_nexus_mcp.tool(
    name="nexus_similarity_search_api_benchmark_hybrid_vs_cosine_only",
    description=(
        "Runs a head-to-head benchmark on a provided labeled pair set, comparing the hybrid NMI-Cosine "
        "scorer against a pure-Cosine baseline. Returns MSE, Spearman rank correlation, and mean "
        "absolute error for both methods, plus the relative lift of the hybrid over baseline. Use when "
        "you need a publishable, reproducible metric demonstrating the superiority of hybrid scoring on "
        "a specific mixed-feature dataset. Do NOT use as a general-purpose benchmarking framework for "
        "arbitrary models -- it is scoped exclusively to the NMI-Cosine hybrid vs. Cosine-only "
        "comparison defined in this API."
    ),
)
async def benchmark_hybrid_vs_cosine_only_mcp(
    labeled_pairs: Annotated[str, Field(..., description="JSON-encoded array of objects with keys 'record_a', 'record_b', and 'similarity_label' (float in [0.0, 1.0]). Minimum 20 pairs, maximum 5000 pairs.", min_length=2, max_length=4194304)],
    feature_schema: Annotated[str, Field(..., description='JSON-encoded object mapping feature keys to "continuous" or "categorical".', min_length=2, max_length=4096)],
    categorical_weight: Annotated[float, Field(0.5, description="The NMI component weight to use for the hybrid scorer during benchmarking. Use the value returned by estimate_optimal_categorical_weight for a fair comparison, or supply a domain-informed prior.", ge=0.0, le=1.0)],
) -> dict[str, Any]:
    params = {
        "labeled_pairs": labeled_pairs,
        "feature_schema": feature_schema,
        "categorical_weight": categorical_weight,
    }
    return await _nexus_mcp_call_core("POST", "/v1/similarity/benchmark-hybrid-vs-cosine", params)


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
_NEXUS_BILLING_EXCLUDED_PATHS = {'/docs', '/redoc', '/health', '/openapi.json', '/favicon.ico', '/'}
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
