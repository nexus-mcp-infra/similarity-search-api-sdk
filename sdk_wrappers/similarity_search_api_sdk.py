from __future__ import annotations

import time
import logging
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import requests

from src.math.information import NormalizedMutualInformation, TransferEntropy
from src.math.causal import DoCalculus, CausalDAG
from src.math.game_theory import NashEquilibrium

logger = logging.getLogger("similarity_search_sdk")

_DEFAULT_BASE_URL = "https://api.similaritysearch.io/v1"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_TOP_K = 10
_MAX_COLLECTION_SIZE = 10_000
_MIN_COLLECTION_SIZE = 1
_MAX_FEATURES = 4_096
_MIN_TOP_K = 1
_MAX_TOP_K = 500


class SimilaritySearchError(Exception):
    pass


class AuthenticationError(SimilaritySearchError):
    pass


class ValidationError(SimilaritySearchError):
    pass


class RateLimitError(SimilaritySearchError):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class UpstreamAPIError(SimilaritySearchError):
    def __init__(self, message: str, status_code: int, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@dataclass
class SimilarityMatch:
    index: int
    score: float
    item: dict[str, Any]
    nmi_weights: list[float] = field(default_factory=list)


@dataclass
class SimilaritySearchResult:
    matches: list[SimilarityMatch]
    query_item: dict[str, Any]
    nmi_weight_vector: list[float]
    computation_ms: float
    model_version: str
    transfer_entropy_signal: float
    nash_equilibrium_threshold: float


def _validate_nonempty_list(value: Any, name: str) -> list:
    if value is None:
        raise ValidationError(f"'{name}' must not be None.")
    if not isinstance(value, list):
        raise ValidationError(
            f"'{name}' must be a list, got {type(value).__name__}."
        )
    if len(value) == 0:
        raise ValidationError(f"'{name}' must contain at least one item.")
    return value


def _validate_collection_size(collection: list, name: str) -> None:
    if len(collection) < _MIN_COLLECTION_SIZE:
        raise ValidationError(
            f"'{name}' must have at least {_MIN_COLLECTION_SIZE} item(s)."
        )
    if len(collection) > _MAX_COLLECTION_SIZE:
        raise ValidationError(
            f"'{name}' exceeds maximum allowed size of {_MAX_COLLECTION_SIZE} items."
        )


def _validate_item_structure(item: Any, index: int, collection_name: str) -> None:
    if not isinstance(item, dict):
        raise ValidationError(
            f"Item at index {index} in '{collection_name}' must be a dict, "
            f"got {type(item).__name__}."
        )
    if "features" not in item:
        raise ValidationError(
            f"Item at index {index} in '{collection_name}' missing required key 'features'."
        )
    features = item["features"]
    if not isinstance(features, (list, np.ndarray)):
        raise ValidationError(
            f"Item at index {index} in '{collection_name}': 'features' must be a list "
            f"or numpy array, got {type(features).__name__}."
        )
    if len(features) == 0:
        raise ValidationError(
            f"Item at index {index} in '{collection_name}': 'features' must be non-empty."
        )
    if len(features) > _MAX_FEATURES:
        raise ValidationError(
            f"Item at index {index} in '{collection_name}': 'features' length {len(features)} "
            f"exceeds maximum of {_MAX_FEATURES}."
        )


def _extract_feature_matrix(collection: list[dict[str, Any]]) -> np.ndarray:
    arrays = []
    reference_dim = len(collection[0]["features"])
    for idx, item in enumerate(collection):
        feat = item["features"]
        arr = np.array(feat, dtype=np.float64)
        if arr.ndim != 1:
            raise ValidationError(
                f"Item at index {idx}: 'features' must be a 1-D array, "
                f"got shape {arr.shape}."
            )
        if len(arr) != reference_dim:
            raise ValidationError(
                f"Item at index {idx}: 'features' length {len(arr)} does not match "
                f"reference dimensionality {reference_dim}. All items must share the same "
                f"feature dimension."
            )
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def _compute_nmi_weight_vector(
    query_vec: np.ndarray,
    collection_matrix: np.ndarray,
    nmi_calculator: NormalizedMutualInformation,
) -> np.ndarray:
    n_items, n_dims = collection_matrix.shape
    weights = np.zeros(n_dims, dtype=np.float64)
    for dim_idx in range(n_dims):
        feature_column = collection_matrix[:, dim_idx]
        target_column = np.full(n_items, query_vec[dim_idx], dtype=np.float64)
        nmi_value = nmi_calculator.compute(feature_column, target_column)
        weights[dim_idx] = max(float(nmi_value), 0.0)
    weight_sum = weights.sum()
    if weight_sum > 0.0:
        weights = weights / weight_sum
    else:
        weights = np.ones(n_dims, dtype=np.float64) / n_dims
    return weights


def _nmi_weighted_cosine_scores(
    query_vec: np.ndarray,
    collection_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weighted_query = weights * query_vec
    weighted_collection = collection_matrix * weights[np.newaxis, :]
    query_norm = np.linalg.norm(weighted_query)
    collection_norms = np.linalg.norm(weighted_collection, axis=1)
    if query_norm == 0.0:
        raise ValidationError(
            "Query feature vector is all-zeros after NMI weighting; "
            "cosine similarity is undefined."
        )
    zero_mask = collection_norms == 0.0
    scores = np.zeros(len(collection_matrix), dtype=np.float64)
    nonzero = ~zero_mask
    if nonzero.any():
        dot_products = weighted_collection[nonzero] @ weighted_query
        scores[nonzero] = dot_products / (collection_norms[nonzero] * query_norm)
    return scores


def _compute_transfer_entropy_signal(
    query_vec: np.ndarray,
    collection_matrix: np.ndarray,
    te_calculator: TransferEntropy,
) -> float:
    mean_collection = collection_matrix.mean(axis=0)
    te_value = te_calculator.compute(
        source=query_vec,
        target=mean_collection,
        lag=1,
    )
    return float(te_value)


def _derive_nash_threshold(
    scores: np.ndarray,
    nash_solver: NashEquilibrium,
) -> float:
    if len(scores) < 2:
        return float(scores[0]) if len(scores) == 1 else 0.0
    unique_scores = np.unique(scores)
    if len(unique_scores) < 2:
        return float(unique_scores[0])
    payoff_matrix = np.array([
        [float(unique_scores[-1]), float(unique_scores[0])],
        [float(unique_scores[0]), float(unique_scores[-1])],
    ])
    equilibrium = nash_solver.solve(payoff_matrix)
    threshold = float(equilibrium.mixed_strategy_value)
    return threshold


def _build_causal_feature_dag(
    n_dims: int,
    do_calculus: DoCalculus,
    causal_dag: CausalDAG,
) -> np.ndarray:
    if n_dims <= 1:
        return np.ones(n_dims, dtype=np.float64)
    edges = [(i, i + 1) for i in range(min(n_dims - 1, 8))]
    dag = causal_dag.build(n_nodes=n_dims, edges=edges)
    causal_weights = do_calculus.compute_feature_relevance(dag)
    arr = np.array(causal_weights, dtype=np.float64)
    if arr.shape[0] != n_dims:
        arr = np.ones(n_dims, dtype=np.float64)
    w_sum = arr.sum()
    if w_sum > 0.0:
        arr = arr / w_sum
    return arr


class Client:
    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        enable_local_precompute: bool = True,
    ):
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError(
                "A valid non-empty 'api_key' string is required to initialize the Client."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._enable_local_precompute = enable_local_precompute
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-SDK-Version": "1.0.0",
        })
        self._nmi = NormalizedMutualInformation()
        self._te = TransferEntropy()
        self._do = DoCalculus()
        self._dag_builder = CausalDAG()
        self._nash = NashEquilibrium()

    def nmi_ranked_search(
        self,
        query: dict[str, Any],
        collection: list[dict[str, Any]],
        top_k: int = _DEFAULT_TOP_K,
        use_causal_reweighting: bool = False,
    ) -> SimilaritySearchResult:
        if query is None or not isinstance(query, dict):
            raise ValidationError(
                "'query' must be a non-None dict with a 'features' key, "
                f"got {type(query).__name__}."
            )
        _validate_item_structure(query, 0, "query")
        _validate_nonempty_list(collection, "collection")
        _validate_collection_size(collection, "collection")
        for idx, item in enumerate(collection):
            _validate_item_structure(item, idx, "collection")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValidationError(
                f"'top_k' must be an integer, got {type(top_k).__name__}."
            )
        if top_k < _MIN_TOP_K or top_k > _MAX_TOP_K:
            raise ValidationError(
                f"'top_k' must be between {_MIN_TOP_K} and {_MAX_TOP_K}, got {top_k}."
            )

        if self._enable_local_precompute:
            return self._local_nmi_ranked_search(
                query=query,
                collection=collection,
                top_k=top_k,
                use_causal_reweighting=use_causal_reweighting,
            )
        return self._remote_nmi_ranked_search(
            query=query,
            collection=collection,
            top_k=top_k,
            use_causal_reweighting=use_causal_reweighting,
        )

    def main_method(
        self,
        data: dict[str, Any],
        top_k: int = _DEFAULT_TOP_K,
    ) -> SimilaritySearchResult:
        if data is None or not isinstance(data, dict):
            raise ValidationError(
                "'data' must be a dict with keys 'query' (item) and 'collection' (list of items). "
                f"Got {type(data).__name__}."
            )
        if "query" not in data:
            raise ValidationError(
                "'data' dict missing required key 'query'."
            )
        if "collection" not in data:
            raise ValidationError(
                "'data' dict missing required key 'collection'."
            )
        return self.nmi_ranked_search(
            query=data["query"],
            collection=data["collection"],
            top_k=top_k,
            use_causal_reweighting=data.get("use_causal_reweighting", False),
        )

    def _local_nmi_ranked_search(
        self,
        query: dict[str, Any],
        collection: list[dict[str, Any]],
        top_k: int,
        use_causal_reweighting: bool,
    ) -> SimilaritySearchResult:
        t0 = time.perf_counter()
        query_vec = np.array(query["features"], dtype=np.float64)
        collection_matrix = _extract_feature_matrix(collection)
        n_dims = query_vec.shape[0]
        nmi_weights = _compute_nmi_weight_vector(
            query_vec=query_vec,
            collection_matrix=collection_matrix,
            nmi_calculator=self._nmi,
        )
        if use_causal_reweighting and n_dims > 1:
            causal_weights = _build_causal_feature_dag(
                n_dims=n_dims,
                do_calculus=self._do,
                causal_dag=self._dag_builder,
            )
            combined = nmi_weights * causal_weights
            w_sum = combined.sum()
            nmi_weights = combined / w_sum if w_sum > 0.0 else nmi_weights
        scores = _nmi_weighted_cosine_scores(
            query_vec=query_vec,
            collection_matrix=collection_matrix,
            weights=nmi_weights,
        )
        te_signal = _compute_transfer_entropy_signal(
            query_vec=query_vec,
            collection_matrix=collection_matrix,
            te_calculator=self._te,
        )
        nash_threshold = _derive_nash_threshold(
            scores=scores,
            nash_solver=self._nash,
        )
        ranked_indices = np.argsort(scores)[::-1]
        k = min(top_k, len(collection))
        top_indices = ranked_indices[:k]
        matches = [
            SimilarityMatch(
                index=int(idx),
                score=float(scores[idx]),
                item=collection[int(idx)],
                nmi_weights=nmi_weights.tolist(),
            )
            for idx in top_indices
        ]
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0
        return SimilaritySearchResult(
            matches=matches,
            query_item=query,
            nmi_weight_vector=nmi_weights.tolist(),
            computation_ms=round(elapsed_ms, 3),
            model_version="nmi-cosine-hybrid-v1",
            transfer_entropy_signal=te_signal,
            nash_equilibrium_threshold=nash_threshold,
        )

    def _remote_nmi_ranked_search(
        self,
        query: dict[str, Any],
        collection: list[dict[str, Any]],
        top_k: int,
        use_causal_reweighting: bool,
    ) -> SimilaritySearchResult:
        payload = {
            "query": {
                "features": [float(f) for f in query["features"]],
                **{k: v for k, v in query.items() if k != "features"},
            },
            "collection": [
                {
                    "features": [float(f) for f in item["features"]],
                    **{k: v for k, v in item.items() if k != "features"},
                }
                for item in collection
            ],
            "top_k": top_k,
            "use_causal_reweighting": use_causal_reweighting,
        }
        try:
            response = self._session.post(
                f"{self._base_url}/search/nmi-ranked",
                json=payload,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout:
            raise UpstreamAPIError(
                f"Request timed out after {self._timeout}s. "
                "Consider reducing collection size or increasing timeout.",
                status_code=408,
            )
        except requests.exceptions.ConnectionError as exc:
            raise UpstreamAPIError(
                f"Connection error reaching {self._base_url}: {exc}",
                status_code=503,
            )
        self._raise_for_status(response)
        body = response.json()
        return self._parse_remote_response(body)

    def _raise_for_status(self, response: requests.Response) -> None:
        status = response.status_code
        if status == 200:
            return
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text}
        if status == 401:
            raise AuthenticationError(
                "API key rejected (HTTP 401). Verify the key passed to Client()."
            )
        if status == 429:
            retry_after = None
            try:
                retry_after = float(response.headers.get("Retry-After", 1.0))
            except (TypeError, ValueError):
                retry_after = 1.0
            raise RateLimitError(
                f"Rate limit exceeded (HTTP 429). Retry after {retry_after}s.",
                retry_after=retry_after,
            )
        if status == 422:
            raise ValidationError(
                f"Server rejected payload (HTTP 422): {body}"
            )
        if 400 <= status < 500:
            raise UpstreamAPIError(
                f"Client error (HTTP {status}): {body}",
                status_code=status,
                response_body=body,
            )
        if status >= 500:
            raise UpstreamAPIError(
                f"Server error (HTTP {status}): {body}",
                status_code=status,
                response_body=body,
            )

    def _parse_remote_response(self, body: dict[str, Any]) -> SimilaritySearchResult:
        if not isinstance(body, dict):
            raise UpstreamAPIError(
                f"Unexpected response shape: expected dict, got {type(body).__name__}.",
                status_code=200,
                response_body=body,
            )
        raw_matches = body.get("matches", [])
        if not isinstance(raw_matches, list):
            raise UpstreamAPIError(
                "Response 'matches' field must be a list.",
                status_code=200,
                response_body=body,
            )
        matches = []
        for m in raw_matches:
            matches.append(
                SimilarityMatch(
                    index=int(m.get("index", -1)),
                    score=float(m.get("score", 0.0)),
                    item=m.get("item", {}),
                    nmi_weights=m.get("nmi_weights", []),
                )
            )
        return SimilaritySearchResult(
            matches=matches,
            query_item=body.get("query_item", {}),
            nmi_weight_vector=body.get("nmi_weight_vector", []),
            computation_ms=float(body.get("computation_ms", 0.0)),
            model_version=str(body.get("model_version", "unknown")),
            transfer_entropy_signal=float(body.get("transfer_entropy_signal", 0.0)),
            nash_equilibrium_threshold=float(body.get("nash_equilibrium_threshold", 0.0)),
        )