from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, model_validator
from typing import Any
import numpy as np
import duckdb
import hashlib
import time
import os

from src.math.information import NormalizedMutualInformation, TransferEntropy
from src.math.causal import DoCalculus, CausalDAG, build_nexus_dag
from src.math.game_theory import NashEquilibrium, MarketEntryGame
from src.math.statistics import Statistics

app = FastAPI(
    title="Hybrid Similarity Search API",
    version="1.0.0",
    description="Stateless NMI+Cosine hybrid scoring with adaptive entropy-calibrated weights",
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
_VALID_API_KEYS: set[str] = set(
    k.strip() for k in os.environ.get("HYBRID_SEARCH_API_KEYS", "").split(",") if k.strip()
)
_ENTROPY_CATEGORICAL_THRESHOLD: float = 1.5
_DB_PATH: str = os.environ.get("HYBRID_SEARCH_DB", "hybrid_search_telemetry.duckdb")

_db_conn: duckdb.DuckDBPyConnection | None = None


def _get_db() -> duckdb.DuckDBPyConnection:
    global _db_conn
    if _db_conn is None:
        _db_conn = duckdb.connect(_DB_PATH)
        _db_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weight_telemetry (
                request_id TEXT,
                domain TEXT,
                w_nmi DOUBLE,
                w_cosine DOUBLE,
                n_categorical INTEGER,
                n_continuous INTEGER,
                ts DOUBLE
            )
            """
        )
    return _db_conn


def _require_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if not _VALID_API_KEYS:
        raise HTTPException(status_code=503, detail="Server has no API keys configured")
    if api_key not in _VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


class FeatureRecord(BaseModel):
    id: str = Field(..., min_length=1, max_length=256)
    features: dict[str, Any] = Field(..., min_length=1)

    @model_validator(mode="after")
    def features_not_empty(self) -> "FeatureRecord":
        if not self.features:
            raise ValueError("features dict must contain at least one key")
        return self


class SimilaritySearchRequest(BaseModel):
    query: FeatureRecord
    corpus: list[FeatureRecord] = Field(..., min_length=1, max_length=10000)
    top_k: int = Field(default=10, ge=1, le=500)
    domain: str = Field(default="generic", min_length=1, max_length=64)

    @model_validator(mode="after")
    def top_k_fits_corpus(self) -> "SimilaritySearchRequest":
        if self.top_k > len(self.corpus):
            self.top_k = len(self.corpus)
        return self


class ComponentScore(BaseModel):
    nmi_score: float
    cosine_score: float
    hybrid_score: float
    w_nmi: float
    w_cosine: float
    dominant_component: str


class SimilarityResult(BaseModel):
    id: str
    score: ComponentScore


class SimilaritySearchResponse(BaseModel):
    request_id: str
    results: list[SimilarityResult]
    calibration: dict[str, float]
    latency_ms: float


class WeightDistributionResponse(BaseModel):
    domain: str
    avg_w_nmi: float
    avg_w_cosine: float
    sample_count: int
    entropy_threshold_bits: float


def _marginal_entropy(values: list[Any]) -> float:
    stats = Statistics()
    arr = np.array([float(v) if isinstance(v, (int, float)) else hash(str(v)) % 1000 for v in values])
    return float(stats.entropy(arr))


def _classify_features(
    features: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    categorical: dict[str, Any] = {}
    continuous: dict[str, Any] = {}
    entropies: dict[str, float] = {}

    for key, value in features.items():
        if isinstance(value, bool):
            categorical[key] = value
            entropies[key] = _marginal_entropy([value])
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            h = _marginal_entropy([value])
            entropies[key] = h
            if h < _ENTROPY_CATEGORICAL_THRESHOLD:
                categorical[key] = value
            else:
                continuous[key] = value
        elif isinstance(value, str):
            categorical[key] = value
            entropies[key] = _marginal_entropy([value])
        elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
            h = _marginal_entropy(list(value))
            entropies[key] = h
            if h < _ENTROPY_CATEGORICAL_THRESHOLD:
                categorical[key] = value
            else:
                continuous[key] = value
        else:
            categorical[key] = str(value)
            entropies[key] = _marginal_entropy([str(value)])

    return categorical, continuous, entropies


def _adaptive_weights(
    query_features: dict[str, Any],
    entropies: dict[str, float],
    categorical_keys: set[str],
    continuous_keys: set[str],
) -> tuple[float, float]:
    h_cat = sum(entropies[k] for k in categorical_keys if k in entropies)
    h_cont = sum(entropies[k] for k in continuous_keys if k in entropies)
    total = h_cat + h_cont
    if total == 0.0:
        return 0.5, 0.5
    w_nmi = h_cat / total
    w_cosine = h_cont / total
    return float(w_nmi), float(w_cosine)


def _nmi_score_pair(
    query_cat: dict[str, Any],
    candidate_cat: dict[str, Any],
) -> float:
    if not query_cat and not candidate_cat:
        return 1.0
    nmi_calc = NormalizedMutualInformation()
    scores: list[float] = []
    all_keys = set(query_cat.keys()) | set(candidate_cat.keys())
    for key in all_keys:
        qv = query_cat.get(key)
        cv = candidate_cat.get(key)
        if qv is None or cv is None:
            scores.append(0.0)
            continue
        q_arr = np.array([hash(str(qv)) % 1000])
        c_arr = np.array([hash(str(cv)) % 1000])
        try:
            score = float(nmi_calc.compute(q_arr, c_arr))
        except Exception:
            score = 1.0 if str(qv) == str(cv) else 0.0
        scores.append(max(0.0, min(1.0, score)))
    return float(np.mean(scores)) if scores else 0.0


def _cosine_score_pair(
    query_cont: dict[str, Any],
    candidate_cont: dict[str, Any],
) -> float:
    if not query_cont and not candidate_cont:
        return 1.0
    all_keys = sorted(set(query_cont.keys()) | set(candidate_cont.keys()))
    if not all_keys:
        return 1.0

    def _flatten(d: dict[str, Any], keys: list[str]) -> np.ndarray:
        parts: list[float] = []
        for k in keys:
            v = d.get(k, 0.0)
            if isinstance(v, (list, tuple)):
                parts.extend(float(x) for x in v)
            else:
                parts.append(float(v) if isinstance(v, (int, float)) else 0.0)
        return np.array(parts, dtype=np.float64)

    q_vec = _flatten(query_cont, all_keys)
    c_vec = _flatten(candidate_cont, all_keys)

    if q_vec.shape != c_vec.shape:
        min_len = min(len(q_vec), len(c_vec))
        q_vec = q_vec[:min_len]
        c_vec = c_vec[:min_len]

    q_norm = np.linalg.norm(q_vec)
    c_norm = np.linalg.norm(c_vec)
    if q_norm == 0.0 or c_norm == 0.0:
        return 1.0 if (q_norm == 0.0 and c_norm == 0.0) else 0.0
    return float(np.clip(np.dot(q_vec, c_vec) / (q_norm * c_norm), -1.0, 1.0))


def _score_candidate(
    query_cat: dict[str, Any],
    query_cont: dict[str, Any],
    cand_cat: dict[str, Any],
    cand_cont: dict[str, Any],
    w_nmi: float,
    w_cosine: float,
) -> ComponentScore:
    nmi_s = _nmi_score_pair(query_cat, cand_cat)
    cos_s = _cosine_score_pair(query_cont, cand_cont)
    hybrid = w_nmi * nmi_s + w_cosine * cos_s
    dominant = "nmi" if w_nmi >= w_cosine else "cosine"
    return ComponentScore(
        nmi_score=round(nmi_s, 6),
        cosine_score=round(cos_s, 6),
        hybrid_score=round(hybrid, 6),
        w_nmi=round(w_nmi, 6),
        w_cosine=round(w_cosine, 6),
        dominant_component=dominant,
    )


def _record_telemetry(
    request_id: str,
    domain: str,
    w_nmi: float,
    w_cosine: float,
    n_cat: int,
    n_cont: int,
) -> None:
    try:
        db = _get_db()
        db.execute(
            "INSERT INTO weight_telemetry VALUES (?, ?, ?, ?, ?, ?, ?)",
            [request_id, domain, w_nmi, w_cosine, n_cat, n_cont, time.time()],
        )
    except Exception:
        pass


@app.post("/v1/similarity/search", response_model=SimilaritySearchResponse)
def hybrid_similarity_search(
    request: SimilaritySearchRequest,
    _api_key: str = Depends(_require_api_key),
) -> SimilaritySearchResponse:
    t0 = time.perf_counter()

    if not request.query.features:
        raise HTTPException(status_code=422, detail="query.features must not be empty")

    query_cat, query_cont, q_entropies = _classify_features(request.query.features)
    w_nmi, w_cosine = _adaptive_weights(
        request.query.features,
        q_entropies,
        set(query_cat.keys()),
        set(query_cont.keys()),
    )

    scored: list[tuple[float, SimilarityResult]] = []
    for item in request.corpus:
        cand_cat, cand_cont, _ = _classify_features(item.features)
        comp = _score_candidate(query_cat, query_cont, cand_cat, cand_cont, w_nmi, w_cosine)
        scored.append((comp.hybrid_score, SimilarityResult(id=item.id, score=comp)))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = [r for _, r in scored[: request.top_k]]

    request_id = hashlib.sha256(
        f"{request.query.id}{time.time()}".encode()
    ).hexdigest()[:16]

    _record_telemetry(
        request_id,
        request.domain,
        w_nmi,
        w_cosine,
        len(query_cat),
        len(query_cont),
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return SimilaritySearchResponse(
        request_id=request_id,
        results=top_results,
        calibration={
            "w_nmi": round(w_nmi, 6),
            "w_cosine": round(w_cosine, 6),
            "n_categorical_features": float(len(query_cat)),
            "n_continuous_features": float(len(query_cont)),
            "entropy_threshold_bits": _ENTROPY_CATEGORICAL_THRESHOLD,
        },
        latency_ms=round(latency_ms, 3),
    )


@app.post("/v1/similarity/explain", response_model=dict[str, Any])
def explain_feature_decomposition(
    record: FeatureRecord,
    _api_key: str = Depends(_require_api_key),
) -> dict[str, Any]:
    if not record.features:
        raise HTTPException(status_code=422, detail="features must not be empty")

    cat, cont, entropies = _classify_features(record.features)
    w_nmi, w_cosine = _adaptive_weights(record.features, entropies, set(cat.keys()), set(cont.keys()))

    nmi_inst = NormalizedMutualInformation()
    dag = build_nexus_dag(list(record.features.keys()))
    causal = CausalDAG(dag)
    do_calc = DoCalculus(causal)

    feature_report: dict[str, dict[str, Any]] = {}
    for key, value in record.features.items():
        h = entropies.get(key, 0.0)
        regime = "categorical_nmi" if h < _ENTROPY_CATEGORICAL_THRESHOLD else "continuous_cosine"
        intervention_effect = None
        try:
            intervention_effect = float(do_calc.effect(key, list(record.features.keys())))
        except Exception:
            intervention_effect = None
        feature_report[key] = {
            "entropy_bits": round(h, 6),
            "regime": regime,
            "value_type": type(value).__name__,
            "causal_intervention_effect": intervention_effect,
        }

    return {
        "id": record.id,
        "feature_decomposition": feature_report,
        "adaptive_weights": {"w_nmi": round(w_nmi, 6), "w_cosine": round(w_cosine, 6)},
        "dominant_regime": "nmi" if w_nmi >= w_cosine else "cosine",
        "dag_node_count": len(dag.nodes) if hasattr(dag, "nodes") else len(record.features),
    }


@app.get("/v1/similarity/weight-distribution", response_model=WeightDistributionResponse)
def query_weight_distribution(
    domain: str = "generic",
    _api_key: str = Depends(_require_api_key),
) -> WeightDistributionResponse:
    if not domain or len(domain) > 64:
        raise HTTPException(status_code=422, detail="domain must be 1-64 characters")

    db = _get_db()
    row = db.execute(
        """
        SELECT
            AVG(w_nmi) as avg_w_nmi,
            AVG(w_cosine) as avg_w_cosine,
            COUNT(*) as sample_count
        FROM weight_telemetry
        WHERE domain = ?
        """,
        [domain],
    ).fetchone()

    if row is None or row[2] == 0:
        return WeightDistributionResponse(
            domain=domain,
            avg_w_nmi=0.5,
            avg_w_cosine=0.5,
            sample_count=0,
            entropy_threshold_bits=_ENTROPY_CATEGORICAL_THRESHOLD,
        )

    return WeightDistributionResponse(
        domain=domain,
        avg_w_nmi=round(float(row[0]), 6),
        avg_w_cosine=round(float(row[1]), 6),
        sample_count=int(row[2]),
        entropy_threshold_bits=_ENTROPY_CATEGORICAL_THRESHOLD,
    )


@app.get("/v1/similarity/health")
def readiness_probe() -> dict[str, str]:
    try:
        db = _get_db()
        db.execute("SELECT 1").fetchone()
        db_status = "ok"
    except Exception as exc:
        db_status = f"degraded: {exc}"
    return {"status": "ok", "db": db_status, "version": "1.0.0"}