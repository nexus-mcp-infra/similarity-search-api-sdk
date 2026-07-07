from __future__ import annotations

import time
from typing import Any

import httpx


class SimilaritySearchError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response_body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body or {}


class AuthenticationError(SimilaritySearchError):
    pass


class RateLimitError(SimilaritySearchError):
    pass


class ValidationError(SimilaritySearchError):
    pass


class ServerError(SimilaritySearchError):
    pass


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 200:
        return
    try:
        body = response.json()
    except Exception:
        body = {"detail": response.text}
    message = body.get("detail", f"HTTP {response.status_code}")
    if response.status_code == 401:
        raise AuthenticationError(message, status_code=response.status_code, response_body=body)
    if response.status_code == 422:
        raise ValidationError(message, status_code=response.status_code, response_body=body)
    if response.status_code == 429:
        raise RateLimitError(message, status_code=response.status_code, response_body=body)
    if response.status_code >= 500:
        raise ServerError(message, status_code=response.status_code, response_body=body)
    raise SimilaritySearchError(message, status_code=response.status_code, response_body=body)


class HybridSimilarityResult:
    __slots__ = (
        "hybrid_score",
        "nmi_score",
        "cosine_score",
        "nmi_weight",
        "cosine_weight",
        "confidence_interval_low",
        "confidence_interval_high",
        "confidence_level",
        "categorical_feature_ratio",
        "bootstrap_n",
        "ranked_candidates",
        "metadata",
    )

    def __init__(self, raw: dict) -> None:
        self.hybrid_score: float = raw["hybrid_score"]
        self.nmi_score: float = raw["nmi_score"]
        self.cosine_score: float = raw["cosine_score"]
        self.nmi_weight: float = raw["nmi_weight"]
        self.cosine_weight: float = raw["cosine_weight"]
        self.confidence_interval_low: float = raw["confidence_interval"]["low"]
        self.confidence_interval_high: float = raw["confidence_interval"]["high"]
        self.confidence_level: float = raw["confidence_interval"]["level"]
        self.categorical_feature_ratio: float = raw["categorical_feature_ratio"]
        self.bootstrap_n: int = raw["bootstrap_n"]
        self.ranked_candidates: list[dict] = raw.get("ranked_candidates", [])
        self.metadata: dict = raw.get("metadata", {})

    def __repr__(self) -> str:
        return (
            f"HybridSimilarityResult("
            f"hybrid_score={self.hybrid_score:.4f}, "
            f"nmi_weight={self.nmi_weight:.3f}, "
            f"cosine_weight={self.cosine_weight:.3f}, "
            f"ci=[{self.confidence_interval_low:.4f}, {self.confidence_interval_high:.4f}])"
        )


class BatchSimilarityResult:
    __slots__ = ("results", "total", "metadata")

    def __init__(self, raw: dict) -> None:
        self.results: list[HybridSimilarityResult] = [
            HybridSimilarityResult(r) for r in raw["results"]
        ]
        self.total: int = raw["total"]
        self.metadata: dict = raw.get("metadata", {})

    def __repr__(self) -> str:
        return f"BatchSimilarityResult(total={self.total}, results={len(self.results)})"


def _validate_record(record: Any, label: str) -> None:
    if record is None:
        raise ValidationError(f"'{label}' must not be None")
    if not isinstance(record, dict):
        raise ValidationError(
            f"'{label}' must be a dict mapping feature names to values, got {type(record).__name__}"
        )
    if not record:
        raise ValidationError(f"'{label}' must not be an empty dict")


def _validate_candidates(candidates: Any) -> None:
    if candidates is None:
        raise ValidationError("'candidates' must not be None")
    if not isinstance(candidates, list):
        raise ValidationError(
            f"'candidates' must be a list of dicts, got {type(candidates).__name__}"
        )
    if not candidates:
        raise ValidationError("'candidates' must contain at least one record")
    if len(candidates) > 10_000:
        raise ValidationError(
            f"'candidates' exceeds maximum allowed size of 10,000 records (got {len(candidates)})"
        )
    for i, c in enumerate(candidates):
        if not isinstance(c, dict):
            raise ValidationError(
                f"'candidates[{i}]' must be a dict, got {type(c).__name__}"
            )


class Client:
    """
    Thin HTTP wrapper for the Similarity Search API.

    Computes a hybrid NMI+Cosine similarity score per-call, stateless,
    with bootstrap confidence intervals — no index, no infrastructure setup.

    Parameters
    ----------
    api_key : str
        Secret key issued at purchase. Required for every request.
    base_url : str
        API base URL. Defaults to the production endpoint.
    timeout : float
        Per-request timeout in seconds. Default is 30.0.
    max_retries : int
        Number of retries on transient server errors (5xx). Default is 2.
    """

    _DEFAULT_BASE_URL = "https://api.nexus-similarity.io/v1"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError(
                "'api_key' must be a non-empty string. "
                "Obtain one at https://nexus-similarity.io/keys"
            )
        self._api_key = api_key
        self._base_url = (base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "similarity-search-sdk-python/1.0.0",
            },
            timeout=self._timeout,
        )

    def _post_with_retries(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.post(url, json=payload)
                _raise_for_status(response)
                return response.json()
            except (RateLimitError, AuthenticationError, ValidationError):
                raise
            except ServerError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(0.5 * (2 ** attempt))
            except httpx.TimeoutException as exc:
                last_exc = SimilaritySearchError(
                    f"Request to {url} timed out after {self._timeout}s"
                )
                if attempt < self._max_retries:
                    time.sleep(0.5 * (2 ** attempt))
            except httpx.RequestError as exc:
                raise SimilaritySearchError(
                    f"Network error contacting {url}: {exc}"
                ) from exc
        raise last_exc

    def compute_hybrid_similarity(
        self,
        query: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        confidence_level: float = 0.95,
        nmi_weight_override: float | None = None,
    ) -> HybridSimilarityResult:
        """
        Compute hybrid NMI+Cosine similarity between one query record and a
        list of candidate records.

        The API infers feature types per-call and calibrates the NMI/Cosine
        weight split dynamically. No schema declaration required.

        Use this when: you have one query and want ranked candidates with
        confidence intervals on the top result.
        Do NOT use this when: you need pairwise similarity across an entire
        corpus — use `rank_candidates_by_hybrid_score` with `top_k` instead.

        Parameters
        ----------
        query : dict
            The reference record. Keys are feature names; values can be
            strings (categorical) or numbers (continuous).
        candidates : list[dict]
            Records to rank against the query. Maximum 10,000 per call.
        top_k : int
            Number of top-ranked candidates to return. Range: 1-100. Default 10.
        confidence_level : float
            Bootstrap confidence level for the score CI. Range: 0.80-0.99.
            Default 0.95.
        nmi_weight_override : float or None
            If provided, fixes the NMI component weight in [0.0, 1.0] and
            sets cosine_weight = 1 - nmi_weight_override, bypassing dynamic
            calibration. Use only when you have domain knowledge of the
            feature distribution.

        Returns
        -------
        HybridSimilarityResult
        """
        _validate_record(query, "query")
        _validate_candidates(candidates)
        if not isinstance(top_k, int) or not (1 <= top_k <= 100):
            raise ValidationError("'top_k' must be an integer in [1, 100]")
        if not isinstance(confidence_level, (int, float)) or not (0.80 <= confidence_level <= 0.99):
            raise ValidationError("'confidence_level' must be a float in [0.80, 0.99]")
        if nmi_weight_override is not None:
            if not isinstance(nmi_weight_override, (int, float)) or not (0.0 <= nmi_weight_override <= 1.0):
                raise ValidationError("'nmi_weight_override' must be a float in [0.0, 1.0]")

        payload: dict[str, Any] = {
            "query": query,
            "candidates": candidates,
            "top_k": top_k,
            "confidence_level": confidence_level,
        }
        if nmi_weight_override is not None:
            payload["nmi_weight_override"] = float(nmi_weight_override)

        raw = self._post_with_retries("/similarity/hybrid", payload)
        return HybridSimilarityResult(raw)

    def rank_candidates_by_hybrid_score(
        self,
        query: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        confidence_level: float = 0.95,
    ) -> list[HybridSimilarityResult]:
        """
        Rank all candidates against a query and return the top-k results,
        each with its own hybrid score and confidence interval.

        Use this when: you need a full ranked list with per-candidate
        confidence intervals, not just the aggregate top result.
        Do NOT use this when: candidates > 10,000 — split into batches
        and use `compute_hybrid_similarity` per shard.

        Parameters
        ----------
        query : dict
            The reference record.
        candidates : list[dict]
            Records to rank. Maximum 10,000.
        top_k : int
            Number of ranked results to return. Range: 1-100.
        confidence_level : float
            Bootstrap CI level applied to each candidate score. Range: 0.80-0.99.

        Returns
        -------
        list[HybridSimilarityResult], ordered descending by hybrid_score.
        """
        _validate_record(query, "query")
        _validate_candidates(candidates)
        if not isinstance(top_k, int) or not (1 <= top_k <= 100):
            raise ValidationError("'top_k' must be an integer in [1, 100]")
        if not isinstance(confidence_level, (int, float)) or not (0.80 <= confidence_level <= 0.99):
            raise ValidationError("'confidence_level' must be a float in [0.80, 0.99]")

        payload: dict[str, Any] = {
            "query": query,
            "candidates": candidates,
            "top_k": top_k,
            "confidence_level": confidence_level,
        }
        raw = self._post_with_retries("/similarity/rank", payload)
        return [HybridSimilarityResult(r) for r in raw["ranked_results"]]

    def compute_pairwise_nmi(
        self,
        record_a: dict[str, Any],
        record_b: dict[str, Any],
        categorical_features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compute raw NMI-only similarity between two records, normalized by
        joint entropy. Returns NMI score, joint entropy, and per-feature
        contributions.

        Use this when: you want to isolate the categorical dependency signal
        from the hybrid score, or when all features are categorical.
        Do NOT use this when: your data is predominantly continuous — cosine
        dominates in that regime and `compute_hybrid_similarity` is more
        appropriate.

        Parameters
        ----------
        record_a : dict
            First record.
        record_b : dict
            Second record.
        categorical_features : list[str] or None
            Explicit list of feature keys to treat as categorical. If None,
            the API infers types automatically.

        Returns
        -------
        dict with keys: nmi_score, joint_entropy, per_feature_nmi,
        detected_categorical_features.
        """
        _validate_record(record_a, "record_a")
        _validate_record(record_b, "record_b")
        if categorical_features is not None:
            if not isinstance(categorical_features, list):
                raise ValidationError("'categorical_features' must be a list of strings or None")
            for i, f in enumerate(categorical_features):
                if not isinstance(f, str) or not f:
                    raise ValidationError(
                        f"'categorical_features[{i}]' must be a non-empty string"
                    )

        payload: dict[str, Any] = {
            "record_a": record_a,
            "record_b": record_b,
        }
        if categorical_features is not None:
            payload["categorical_features"] = categorical_features

        return self._post_with_retries("/similarity/nmi-pairwise", payload)

    def estimate_score_confidence_interval(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
        bootstrap_n: int = 500,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """
        Run a standalone bootstrap confidence interval estimation on the
        hybrid score between one query and one candidate, with configurable
        resample count.

        Use this when: you need tighter or wider CI control (e.g., bootstrap_n
        > 500 for publication-grade estimates) on a single pair.
        Do NOT use this when: you are ranking multiple candidates — calling
        this per-candidate in a loop is wasteful; use `rank_candidates_by_hybrid_score`.

        Parameters
        ----------
        query : dict
            Reference record.
        candidate : dict
            Single candidate record to compare.
        bootstrap_n : int
            Number of bootstrap resamples. Range: 100-2000. Default 500.
        confidence_level : float
            Confidence level for the interval. Range: 0.80-0.99.

        Returns
        -------
        dict with keys: hybrid_score, ci_low, ci_high, ci_level,
        bootstrap_n, standard_error, nmi_weight, cosine_weight.
        """
        _validate_record(query, "query")
        _validate_record(candidate, "candidate")
        if not isinstance(bootstrap_n, int) or not (100 <= bootstrap_n <= 2000):
            raise ValidationError("'bootstrap_n' must be an integer in [100, 2000]")
        if not isinstance(confidence_level, (int, float)) or not (0.80 <= confidence_level <= 0.99):
            raise ValidationError("'confidence_level' must be a float in [0.80, 0.99]")

        payload: dict[str, Any] = {
            "query": query,
            "candidate": candidate,
            "bootstrap_n": bootstrap_n,
            "confidence_level": confidence_level,
        }
        return self._post_with_retries("/similarity/bootstrap-ci", payload)

    def main_method(
        self,
        data: dict[str, Any],
    ) -> HybridSimilarityResult:
        """
        Convenience entry point matching the canonical SDK invocation pattern.

        Expects `data` to contain:
            - 'query' (dict): the reference record
            - 'candidates' (list[dict]): records to rank
            - 'top_k' (int, optional): default 10
            - 'confidence_level' (float, optional): default 0.95
            - 'nmi_weight_override' (float, optional)

        Returns
        -------
        HybridSimilarityResult
        """
        if data is None:
            raise ValidationError("'data' must not be None")
        if not isinstance(data, dict):
            raise ValidationError(
                f"'data' must be a dict, got {type(data).__name__}"
            )
        query = data.get("query")
        candidates = data.get("candidates")
        top_k = data.get("top_k", 10)
        confidence_level = data.get("confidence_level", 0.95)
        nmi_weight_override = data.get("nmi_weight_override", None)

        return self.compute_hybrid_similarity(
            query=query,
            candidates=candidates,
            top_k=top_k,
            confidence_level=confidence_level,
            nmi_weight_override=nmi_weight_override,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()