import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests

from src.math.causal import CausalDAG, DoCalculus
from src.math.game_theory import NashEquilibrium
from src.math.information import NormalizedMutualInformation, TransferEntropy


@dataclass
class FeaturePartition:
    categorical_keys: list[str]
    continuous_keys: list[str]
    marginal_entropies: dict[str, float]
    w_nmi: float
    w_cosine: float


@dataclass
class ComponentScore:
    nmi_score: float
    cosine_score: float
    hybrid_score: float
    w_nmi: float
    w_cosine: float
    dominant_component: str
    feature_partition: dict[str, list[str]]


@dataclass
class SimilarityMatch:
    candidate_id: str
    candidate_payload: dict[str, Any]
    score: ComponentScore
    rank: int


@dataclass
class HybridSearchResult:
    query_id: str
    matches: list[SimilarityMatch]
    latency_ms: float
    payload_signature: str
    calibration_metadata: dict[str, Any]


class FeatureTypeClassifier:
    ENTROPY_THRESHOLD_BITS: float = 1.5

    def __init__(self):
        self._nmi_calculator = NormalizedMutualInformation()
        self._transfer_entropy = TransferEntropy()

    def _compute_marginal_entropy(self, values: list[Any]) -> float:
        if not values:
            return 0.0
        try:
            numeric = np.array([float(v) for v in values], dtype=np.float64)
            if np.all(np.isnan(numeric)):
                raise ValueError("all nan")
            bins = max(2, min(int(np.sqrt(len(numeric))), 50))
            hist, _ = np.histogram(numeric[~np.isnan(numeric)], bins=bins, density=False)
            hist = hist[hist > 0].astype(np.float64)
            hist /= hist.sum()
            return float(-np.sum(hist * np.log2(hist + 1e-12)))
        except (ValueError, TypeError):
            unique, counts = np.unique([str(v) for v in values], return_counts=True)
            probs = counts.astype(np.float64) / counts.sum()
            return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def partition_payload_features(
        self,
        query_payload: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> FeaturePartition:
        all_keys = set(query_payload.keys())
        for c in candidates:
            all_keys.update(c.keys())
        all_keys = sorted(all_keys)

        marginal_entropies: dict[str, float] = {}
        for key in all_keys:
            all_values = [query_payload.get(key)] + [c.get(key) for c in candidates]
            all_values = [v for v in all_values if v is not None]
            if not all_values:
                marginal_entropies[key] = 0.0
                continue
            marginal_entropies[key] = self._compute_marginal_entropy(all_values)

        categorical_keys = [
            k for k, h in marginal_entropies.items()
            if h < self.ENTROPY_THRESHOLD_BITS
        ]
        continuous_keys = [
            k for k, h in marginal_entropies.items()
            if h >= self.ENTROPY_THRESHOLD_BITS
        ]

        sum_h_cat = sum(marginal_entropies[k] for k in categorical_keys)
        sum_h_cont = sum(marginal_entropies[k] for k in continuous_keys)
        total = sum_h_cat + sum_h_cont

        if total < 1e-12:
            w_nmi, w_cosine = 0.5, 0.5
        else:
            w_nmi = sum_h_cat / total
            w_cosine = sum_h_cont / total

        return FeaturePartition(
            categorical_keys=categorical_keys,
            continuous_keys=continuous_keys,
            marginal_entropies=marginal_entropies,
            w_nmi=w_nmi,
            w_cosine=w_cosine,
        )


class HybridSimilarityScorer:
    def __init__(self):
        self._nmi_calculator = NormalizedMutualInformation()
        self._causal_dag = CausalDAG()
        self._do_calculus = DoCalculus(dag=self._causal_dag)
        self._nash = NashEquilibrium()

    def _extract_categorical_vectors(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
        keys: list[str],
    ) -> tuple[list[Any], list[Any]]:
        q_vals = [str(query.get(k, "")) for k in keys]
        c_vals = [str(candidate.get(k, "")) for k in keys]
        return q_vals, c_vals

    def _extract_continuous_vectors(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
        keys: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        q_vec, c_vec = [], []
        for k in keys:
            try:
                q_vec.append(float(query.get(k, 0.0) or 0.0))
                c_vec.append(float(candidate.get(k, 0.0) or 0.0))
            except (TypeError, ValueError):
                q_vec.append(0.0)
                c_vec.append(0.0)
        return np.array(q_vec, dtype=np.float64), np.array(c_vec, dtype=np.float64)

    def _nmi_score_from_categorical_vectors(
        self,
        q_vals: list[Any],
        c_vals: list[Any],
    ) -> float:
        if not q_vals:
            return 0.0
        all_cats = list(set(q_vals) | set(c_vals))
        if len(all_cats) == 1:
            return 1.0 if q_vals == c_vals else 0.0
        cat_to_idx = {c: i for i, c in enumerate(all_cats)}
        q_enc = np.array([cat_to_idx[v] for v in q_vals], dtype=np.int32)
        c_enc = np.array([cat_to_idx[v] for v in c_vals], dtype=np.int32)
        try:
            return float(self._nmi_calculator.compute(q_enc, c_enc))
        except Exception:
            matches = sum(1 for a, b in zip(q_vals, c_vals) if a == b)
            return matches / len(q_vals)

    def _cosine_score(self, q_vec: np.ndarray, c_vec: np.ndarray) -> float:
        if q_vec.size == 0:
            return 0.0
        q_norm = np.linalg.norm(q_vec)
        c_norm = np.linalg.norm(c_vec)
        if q_norm < 1e-12 or c_norm < 1e-12:
            return 0.0
        raw = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
        return (raw + 1.0) / 2.0

    def _resolve_weight_conflict_via_nash(
        self,
        w_nmi_entropy: float,
        w_cosine_entropy: float,
        nmi_raw: float,
        cosine_raw: float,
    ) -> tuple[float, float]:
        payoff_matrix = np.array([
            [nmi_raw * w_nmi_entropy, cosine_raw * w_cosine_entropy],
            [cosine_raw * w_nmi_entropy, nmi_raw * w_cosine_entropy],
        ])
        try:
            eq = self._nash.compute_mixed_strategy(payoff_matrix)
            w1 = float(eq[0])
            w2 = float(eq[1])
            total = w1 + w2
            if total < 1e-12:
                return w_nmi_entropy, w_cosine_entropy
            return w1 / total, w2 / total
        except Exception:
            return w_nmi_entropy, w_cosine_entropy

    def score(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
        partition: FeaturePartition,
    ) -> ComponentScore:
        q_cat, c_cat = self._extract_categorical_vectors(
            query, candidate, partition.categorical_keys
        )
        nmi_raw = self._nmi_score_from_categorical_vectors(q_cat, c_cat)

        q_cont, c_cont = self._extract_continuous_vectors(
            query, candidate, partition.continuous_keys
        )
        cosine_raw = self._cosine_score(q_cont, c_cont)

        w_nmi, w_cosine = self._resolve_weight_conflict_via_nash(
            partition.w_nmi,
            partition.w_cosine,
            nmi_raw,
            cosine_raw,
        )

        hybrid = w_nmi * nmi_raw + w_cosine * cosine_raw
        dominant = "nmi" if w_nmi >= w_cosine else "cosine"

        return ComponentScore(
            nmi_score=nmi_raw,
            cosine_score=cosine_raw,
            hybrid_score=hybrid,
            w_nmi=w_nmi,
            w_cosine=w_cosine,
            dominant_component=dominant,
            feature_partition={
                "categorical": partition.categorical_keys,
                "continuous": partition.continuous_keys,
            },
        )


def _payload_signature(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class SimilaritySearchAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class Client:
    BASE_URL: str = "https://api.similaritysearch.io/v1"
    DEFAULT_TIMEOUT_SECONDS: float = 30.0
    MAX_CANDIDATES: int = 10_000
    MIN_CANDIDATES: int = 1
    MAX_TOP_K: int = 500

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string")
        if len(api_key.strip()) == 0:
            raise ValueError("api_key must not be blank")
        self._api_key = api_key.strip()
        self._base_url = (base_url or self.BASE_URL).rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "similarity-search-sdk-python/1.0.0",
        })
        self._classifier = FeatureTypeClassifier()
        self._scorer = HybridSimilarityScorer()

    def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        try:
            response = self._session.post(url, json=body, timeout=self._timeout)
        except requests.exceptions.Timeout:
            raise SimilaritySearchAPIError(
                f"Request to {endpoint} timed out after {self._timeout}s"
            )
        except requests.exceptions.ConnectionError as exc:
            raise SimilaritySearchAPIError(
                f"Connection error reaching {url}: {exc}"
            )
        if response.status_code == 401:
            raise SimilaritySearchAPIError(
                "Authentication failed: invalid or missing api_key",
                status_code=401,
                response_body=response.text,
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise SimilaritySearchAPIError(
                f"Rate limit exceeded. Retry after {retry_after}s",
                status_code=429,
                response_body=response.text,
            )
        if not response.ok:
            raise SimilaritySearchAPIError(
                f"API error {response.status_code} at {endpoint}",
                status_code=response.status_code,
                response_body=response.text,
            )
        try:
            return response.json()
        except ValueError:
            raise SimilaritySearchAPIError(
                f"Non-JSON response from {endpoint}",
                status_code=response.status_code,
                response_body=response.text,
            )

    def _validate_search_inputs(
        self,
        query: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> None:
        if query is None or not isinstance(query, dict):
            raise TypeError("query must be a non-None dict")
        if not query:
            raise ValueError("query must not be an empty dict")
        if candidates is None or not isinstance(candidates, list):
            raise TypeError("candidates must be a non-None list of dicts")
        if len(candidates) < self.MIN_CANDIDATES:
            raise ValueError(
                f"candidates must contain at least {self.MIN_CANDIDATES} item"
            )
        if len(candidates) > self.MAX_CANDIDATES:
            raise ValueError(
                f"candidates exceeds maximum of {self.MAX_CANDIDATES} items; "
                f"got {len(candidates)}"
            )
        for i, c in enumerate(candidates):
            if not isinstance(c, dict):
                raise TypeError(
                    f"candidates[{i}] must be a dict, got {type(c).__name__}"
                )
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        if top_k < 1 or top_k > self.MAX_TOP_K:
            raise ValueError(
                f"top_k must be between 1 and {self.MAX_TOP_K}; got {top_k}"
            )

    def main_method(
        self,
        data: dict[str, Any],
    ) -> HybridSearchResult:
        if data is None or not isinstance(data, dict):
            raise TypeError("data must be a non-None dict")
        query = data.get("query")
        candidates = data.get("candidates")
        top_k = data.get("top_k", 10)
        candidate_ids = data.get("candidate_ids")

        if query is None:
            raise ValueError("data must contain key 'query' (dict)")
        if candidates is None:
            raise ValueError("data must contain key 'candidates' (list of dicts)")

        return self.hybrid_search(
            query=query,
            candidates=candidates,
            top_k=top_k,
            candidate_ids=candidate_ids,
        )

    def hybrid_search(
        self,
        query: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        candidate_ids: list[str] | None = None,
    ) -> HybridSearchResult:
        self._validate_search_inputs(query, candidates, top_k)

        if candidate_ids is not None:
            if len(candidate_ids) != len(candidates):
                raise ValueError(
                    f"candidate_ids length ({len(candidate_ids)}) must match "
                    f"candidates length ({len(candidates)})"
                )
            ids = candidate_ids
        else:
            ids = [str(i) for i in range(len(candidates))]

        t0 = time.perf_counter()

        partition = self._classifier.partition_payload_features(query, candidates)

        scores: list[tuple[int, ComponentScore]] = []
        for idx, candidate in enumerate(candidates):
            cs = self._scorer.score(query, candidate, partition)
            scores.append((idx, cs))

        scores.sort(key=lambda x: x[1].hybrid_score, reverse=True)
        top_scores = scores[:top_k]

        matches = [
            SimilarityMatch(
                candidate_id=ids[idx],
                candidate_payload=candidates[idx],
                score=component_score,
                rank=rank + 1,
            )
            for rank, (idx, component_score) in enumerate(top_scores)
        ]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        sig = _payload_signature(query)

        calibration_metadata = {
            "entropy_threshold_bits": FeatureTypeClassifier.ENTROPY_THRESHOLD_BITS,
            "w_nmi_entropy_derived": partition.w_nmi,
            "w_cosine_entropy_derived": partition.w_cosine,
            "n_categorical_features": len(partition.categorical_keys),
            "n_continuous_features": len(partition.continuous_keys),
            "marginal_entropies": partition.marginal_entropies,
            "total_candidates_scored": len(candidates),
        }

        return HybridSearchResult(
            query_id=sig,
            matches=matches,
            latency_ms=round(latency_ms, 3),
            payload_signature=sig,
            calibration_metadata=calibration_metadata,
        )

    def explain_score(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
    ) -> ComponentScore:
        if query is None or not isinstance(query, dict):
            raise TypeError("query must be a non-None dict")
        if not query:
            raise ValueError("query must not be empty")
        if candidate is None or not isinstance(candidate, dict):
            raise TypeError("candidate must be a non-None dict")
        if not candidate:
            raise ValueError("candidate must not be empty")

        partition = self._classifier.partition_payload_features(query, [candidate])
        return self._scorer.score(query, candidate, partition)

    def batch_hybrid_search(
        self,
        queries: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        candidate_ids: list[str] | None = None,
    ) -> list[HybridSearchResult]:
        if queries is None or not isinstance(queries, list):
            raise TypeError("queries must be a non-None list of dicts")
        if not queries:
            raise ValueError("queries must not be empty")
        if len(queries) > 100:
            raise ValueError(
                f"batch_hybrid_search supports at most 100 queries per call; "
                f"got {len(queries)}"
            )
        results = []
        for q in queries:
            results.append(
                self.hybrid_search(
                    query=q,
                    candidates=candidates,
                    top_k=top_k,
                    candidate_ids=candidate_ids,
                )
            )
        return results