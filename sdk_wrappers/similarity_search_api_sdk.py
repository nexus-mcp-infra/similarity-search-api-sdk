import httpx
import time
from typing import Any

SIMILARITY_SEARCH_BASE_URL = "https://api.similaritysearch.nexus/v1"
SIMILARITY_SEARCH_DEFAULT_TIMEOUT = 30.0
SIMILARITY_SEARCH_MAX_RETRIES = 3
SIMILARITY_SEARCH_RETRY_BACKOFF = 1.5


class SimilaritySearchAuthError(Exception):
    pass


class SimilaritySearchValidationError(Exception):
    pass


class SimilaritySearchRateLimitError(Exception):
    pass


class SimilaritySearchAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def _validate_corpus_items(items: list[Any], param_name: str) -> None:
    if not isinstance(items, list):
        raise SimilaritySearchValidationError(
            f"'{param_name}' must be a list, got {type(items).__name__}"
        )
    if len(items) == 0:
        raise SimilaritySearchValidationError(
            f"'{param_name}' must contain at least one item"
        )
    if len(items) > 500_000:
        raise SimilaritySearchValidationError(
            f"'{param_name}' exceeds the 500,000-item corpus limit (got {len(items)})"
        )


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int):
        raise SimilaritySearchValidationError(
            f"'top_k' must be an int, got {type(top_k).__name__}"
        )
    if top_k < 1 or top_k > 1000:
        raise SimilaritySearchValidationError(
            f"'top_k' must be between 1 and 1000 (got {top_k})"
        )


def _validate_alpha_override(alpha: float | None) -> None:
    if alpha is None:
        return
    if not isinstance(alpha, (int, float)):
        raise SimilaritySearchValidationError(
            f"'alpha_override' must be a float between 0.0 and 1.0, got {type(alpha).__name__}"
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise SimilaritySearchValidationError(
            f"'alpha_override' must be between 0.0 and 1.0 (got {alpha})"
        )


class Client:
    def __init__(
        self,
        api_key: str,
        base_url: str = SIMILARITY_SEARCH_BASE_URL,
        timeout: float = SIMILARITY_SEARCH_DEFAULT_TIMEOUT,
        max_retries: int = SIMILARITY_SEARCH_MAX_RETRIES,
    ):
        if not api_key or not isinstance(api_key, str):
            raise SimilaritySearchAuthError(
                "A non-empty 'api_key' string is required to initialize the Client"
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
                "User-Agent": "similarity-search-sdk-python/1.0.0",
            },
            timeout=self._timeout,
        )

    def _post_with_retry(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        last_exception: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._http.post(url, json=payload)
            except httpx.TimeoutException as exc:
                last_exception = SimilaritySearchAPIError(
                    f"Request to {endpoint} timed out after {self._timeout}s (attempt {attempt + 1}/{self._max_retries})"
                )
                time.sleep(SIMILARITY_SEARCH_RETRY_BACKOFF ** attempt)
                continue
            except httpx.RequestError as exc:
                raise SimilaritySearchAPIError(
                    f"Network error reaching {endpoint}: {exc}"
                ) from exc

            if response.status_code == 401:
                raise SimilaritySearchAuthError(
                    "Invalid or expired API key. Verify the key passed to Client(api_key=...)"
                )
            if response.status_code == 422:
                raise SimilaritySearchValidationError(
                    f"Server rejected the request payload: {response.text}"
                )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "unknown")
                raise SimilaritySearchRateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after}s. "
                    "Consider batching requests or upgrading your plan."
                )
            if response.status_code >= 500:
                last_exception = SimilaritySearchAPIError(
                    f"Server error {response.status_code} on {endpoint} (attempt {attempt + 1}/{self._max_retries}): {response.text}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
                time.sleep(SIMILARITY_SEARCH_RETRY_BACKOFF ** attempt)
                continue
            if not response.is_success:
                raise SimilaritySearchAPIError(
                    f"Unexpected status {response.status_code} from {endpoint}: {response.text}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            try:
                return response.json()
            except Exception as exc:
                raise SimilaritySearchAPIError(
                    f"Could not parse JSON response from {endpoint}: {response.text}"
                ) from exc

        raise last_exception or SimilaritySearchAPIError(
            f"All {self._max_retries} attempts to {endpoint} failed"
        )

    def main_method(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise SimilaritySearchValidationError(
                f"'data' must be a dict, got {type(data).__name__}. "
                "Use search(), rank_corpus_by_nmi_cosine_fusion(), or compute_calibrated_alpha() for typed calls."
            )
        return self._post_with_retry("/search", data)

    def search(
        self,
        query: str | list[float],
        corpus: list[str | list[float]],
        top_k: int = 10,
        alpha_override: float | None = None,
    ) -> dict[str, Any]:
        if query is None:
            raise SimilaritySearchValidationError("'query' must not be None")
        if not isinstance(query, (str, list)):
            raise SimilaritySearchValidationError(
                f"'query' must be a string or a list of floats, got {type(query).__name__}"
            )
        if isinstance(query, str) and len(query.strip()) == 0:
            raise SimilaritySearchValidationError("'query' string must not be empty or whitespace")
        _validate_corpus_items(corpus, "corpus")
        _validate_top_k(top_k)
        _validate_alpha_override(alpha_override)

        payload: dict[str, Any] = {
            "query": query,
            "corpus": corpus,
            "top_k": top_k,
        }
        if alpha_override is not None:
            payload["alpha_override"] = float(alpha_override)

        return self._post_with_retry("/search", payload)

    def rank_corpus_by_nmi_cosine_fusion(
        self,
        query: str | list[float],
        corpus: list[str | list[float]],
        top_k: int = 10,
        alpha_override: float | None = None,
    ) -> list[dict[str, Any]]:
        response = self.search(
            query=query,
            corpus=corpus,
            top_k=top_k,
            alpha_override=alpha_override,
        )
        results = response.get("results")
        if not isinstance(results, list):
            raise SimilaritySearchAPIError(
                f"Unexpected response shape from /search: missing 'results' list. Got keys: {list(response.keys())}"
            )
        return results

    def compute_calibrated_alpha(
        self,
        corpus: list[str | list[float]],
    ) -> dict[str, Any]:
        _validate_corpus_items(corpus, "corpus")
        return self._post_with_retry("/alpha", {"corpus": corpus})

    def score_pair(
        self,
        item_a: str | list[float],
        item_b: str | list[float],
        alpha_override: float | None = None,
    ) -> dict[str, Any]:
        if item_a is None or item_b is None:
            raise SimilaritySearchValidationError("'item_a' and 'item_b' must not be None")
        for name, item in (("item_a", item_a), ("item_b", item_b)):
            if not isinstance(item, (str, list)):
                raise SimilaritySearchValidationError(
                    f"'{name}' must be a string or a list of floats, got {type(item).__name__}"
                )
            if isinstance(item, str) and len(item.strip()) == 0:
                raise SimilaritySearchValidationError(f"'{name}' string must not be empty or whitespace")
        _validate_alpha_override(alpha_override)

        payload: dict[str, Any] = {"item_a": item_a, "item_b": item_b}
        if alpha_override is not None:
            payload["alpha_override"] = float(alpha_override)

        return self._post_with_retry("/score", payload)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()