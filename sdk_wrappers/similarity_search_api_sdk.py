from __future__ import annotations

import time
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.nexus-similarity.io/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_CORPUS_ITEMS = 10_000
MAX_VECTOR_DIMENSIONS = 4_096
MAX_TOP_K = 500


class SimilaritySearchError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(SimilaritySearchError):
    pass


class ValidationError(SimilaritySearchError):
    pass


class RateLimitError(SimilaritySearchError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ServerError(SimilaritySearchError):
    pass


class SimilaritySearchResult:
    def __init__(self, raw: dict):
        self._raw = raw

    @property
    def ranked_indices(self) -> list[int]:
        return [item["index"] for item in self._raw.get("ranking", [])]

    @property
    def ranking(self) -> list[dict]:
        return self._raw.get("ranking", [])

    @property
    def corpus_variance(self) -> float:
        return self._raw.get("corpus_variance", 0.0)

    @property
    def adaptive_nmi_weight(self) -> float:
        return self._raw.get("adaptive_nmi_weight", 0.5)

    @property
    def latency_ms(self) -> float:
        return self._raw.get("latency_ms", 0.0)

    @property
    def call_id(self) -> str:
        return self._raw.get("call_id", "")

    def __repr__(self) -> str:
        top = self.ranking[:3]
        return f"SimilaritySearchResult(top_results={top}, corpus_variance={self.corpus_variance:.4f}, nmi_weight={self.adaptive_nmi_weight:.4f})"


def _validate_vector(vec: Any, label: str) -> list[float]:
    if vec is None:
        raise ValidationError(f"'{label}' must not be None.")
    if not isinstance(vec, (list, tuple)):
        raise ValidationError(
            f"'{label}' must be a list or tuple of numbers, got {type(vec).__name__}."
        )
    if len(vec) == 0:
        raise ValidationError(f"'{label}' must not be empty.")
    if len(vec) > MAX_VECTOR_DIMENSIONS:
        raise ValidationError(
            f"'{label}' has {len(vec)} dimensions; maximum allowed is {MAX_VECTOR_DIMENSIONS}."
        )
    coerced: list[float] = []
    for i, v in enumerate(vec):
        if not isinstance(v, (int, float)):
            raise ValidationError(
                f"'{label}[{i}]' is not a number (got {type(v).__name__})."
            )
        coerced.append(float(v))
    return coerced


def _validate_corpus(corpus: Any) -> list[list[float]]:
    if corpus is None:
        raise ValidationError("'corpus' must not be None.")
    if not isinstance(corpus, (list, tuple)):
        raise ValidationError(
            f"'corpus' must be a list of vectors, got {type(corpus).__name__}."
        )
    if len(corpus) == 0:
        raise ValidationError("'corpus' must contain at least one vector.")
    if len(corpus) > MAX_CORPUS_ITEMS:
        raise ValidationError(
            f"'corpus' contains {len(corpus)} items; maximum allowed is {MAX_CORPUS_ITEMS}."
        )
    validated: list[list[float]] = []
    reference_dim: int | None = None
    for i, item in enumerate(corpus):
        vec = _validate_vector(item, f"corpus[{i}]")
        if reference_dim is None:
            reference_dim = len(vec)
        elif len(vec) != reference_dim:
            raise ValidationError(
                f"All corpus vectors must have the same dimensionality. "
                f"corpus[0] has {reference_dim} dims, corpus[{i}] has {len(vec)} dims."
            )
        validated.append(vec)
    return validated


def _parse_error_response(response: httpx.Response) -> SimilaritySearchError:
    status = response.status_code
    try:
        body = response.json()
        message = body.get("detail") or body.get("message") or response.text
    except Exception:
        message = response.text or f"HTTP {status}"

    if status == 401:
        return AuthenticationError(
            f"Authentication failed: {message}. Verify your API key.",
            status_code=status,
            response_body=body if "body" in dir() else None,
        )
    if status == 422:
        return ValidationError(
            f"Request validation failed: {message}",
            status_code=status,
            response_body=body if "body" in dir() else None,
        )
    if status == 429:
        retry_after: float | None = None
        ra_header = response.headers.get("Retry-After")
        if ra_header is not None:
            try:
                retry_after = float(ra_header)
            except ValueError:
                pass
        return RateLimitError(
            f"Rate limit exceeded: {message}. Retry after {retry_after}s.",
            retry_after=retry_after,
        )
    if status >= 500:
        return ServerError(
            f"Server error ({status}): {message}",
            status_code=status,
        )
    return SimilaritySearchError(
        f"Unexpected error ({status}): {message}",
        status_code=status,
    )


class Client:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 2,
    ):
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError(
                "A non-empty string 'api_key' is required to instantiate the Client."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "similarity-search-sdk/1.0.0 python-httpx",
            },
            timeout=self._timeout,
        )

    def _post_with_retry(self, endpoint: str, payload: dict) -> dict:
        url = f"{self._base_url}{endpoint}"
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.post(url, json=payload)
            except httpx.TimeoutException as exc:
                last_error = SimilaritySearchError(
                    f"Request timed out after {self._timeout}s (attempt {attempt + 1})."
                )
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise last_error from exc
            except httpx.RequestError as exc:
                last_error = SimilaritySearchError(
                    f"Network error on attempt {attempt + 1}: {exc}"
                )
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise last_error from exc

            if response.status_code == 200:
                try:
                    return response.json()
                except Exception as exc:
                    raise SimilaritySearchError(
                        f"Failed to parse JSON response: {response.text}"
                    ) from exc

            parsed_error = _parse_error_response(response)

            if isinstance(parsed_error, RateLimitError):
                if attempt < self._max_retries:
                    wait = parsed_error.retry_after or (1.0 * (attempt + 1))
                    time.sleep(wait)
                    last_error = parsed_error
                    continue
                raise parsed_error

            if isinstance(parsed_error, ServerError) and attempt < self._max_retries:
                time.sleep(0.5 * (attempt + 1))
                last_error = parsed_error
                continue

            raise parsed_error

        raise last_error or SimilaritySearchError("All retry attempts exhausted.")

    def rank_corpus_by_nmi_cosine_fusion(
        self,
        corpus: list[list[float]],
        query: list[float],
        top_k: int = 10,
        nmi_bandwidth: float | None = None,
    ) -> SimilaritySearchResult:
        validated_corpus = _validate_corpus(corpus)
        validated_query = _validate_vector(query, "query")

        if len(validated_query) != len(validated_corpus[0]):
            raise ValidationError(
                f"'query' dimensionality ({len(validated_query)}) must match "
                f"corpus item dimensionality ({len(validated_corpus[0])})."
            )

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValidationError(
                f"'top_k' must be an integer, got {type(top_k).__name__}."
            )
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValidationError(
                f"'top_k' must be between 1 and {MAX_TOP_K}, got {top_k}."
            )
        if top_k > len(validated_corpus):
            top_k = len(validated_corpus)

        payload: dict[str, Any] = {
            "corpus": validated_corpus,
            "query": validated_query,
            "top_k": top_k,
        }

        if nmi_bandwidth is not None:
            if not isinstance(nmi_bandwidth, (int, float)) or isinstance(nmi_bandwidth, bool):
                raise ValidationError(
                    f"'nmi_bandwidth' must be a positive float, got {type(nmi_bandwidth).__name__}."
                )
            if nmi_bandwidth <= 0.0:
                raise ValidationError(
                    f"'nmi_bandwidth' must be strictly positive, got {nmi_bandwidth}."
                )
            payload["nmi_bandwidth"] = float(nmi_bandwidth)

        raw = self._post_with_retry("/similarity/rank", payload)
        return SimilaritySearchResult(raw)

    def main_method(self, data: dict) -> SimilaritySearchResult:
        if data is None:
            raise ValidationError("'data' must not be None.")
        if not isinstance(data, dict):
            raise ValidationError(
                f"'data' must be a dict with keys 'corpus', 'query', and optionally "
                f"'top_k' and 'nmi_bandwidth', got {type(data).__name__}."
            )

        corpus = data.get("corpus")
        query = data.get("query")
        top_k = data.get("top_k", 10)
        nmi_bandwidth = data.get("nmi_bandwidth", None)

        if corpus is None:
            raise ValidationError(
                "'data' dict is missing required key 'corpus'."
            )
        if query is None:
            raise ValidationError(
                "'data' dict is missing required key 'query'."
            )

        return self.rank_corpus_by_nmi_cosine_fusion(
            corpus=corpus,
            query=query,
            top_k=top_k,
            nmi_bandwidth=nmi_bandwidth,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()