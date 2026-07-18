import httpx
from typing import Any

SIMILARITY_SEARCH_API_BASE_URL = "https://api.similaritysearch.io/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_BATCH_SIZE = 500


class SimilaritySearchAuthError(Exception):
    pass


class SimilaritySearchValidationError(Exception):
    pass


class SimilaritySearchAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"API error {status_code}: {message}")


class SimilaritySearchRateLimitError(SimilaritySearchAPIError):
    pass


class Client:
    def __init__(
        self,
        api_key: str,
        base_url: str = SIMILARITY_SEARCH_API_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if not api_key or not isinstance(api_key, str):
            raise SimilaritySearchAuthError(
                "api_key must be a non-empty string. "
                "Obtain one at https://api.similaritysearch.io/keys"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise SimilaritySearchAuthError(
                "Invalid or expired api_key. Check your credentials."
            )
        if response.status_code == 422:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise SimilaritySearchValidationError(
                f"Request payload rejected by the API: {detail}"
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise SimilaritySearchRateLimitError(
                response.status_code,
                f"Rate limit exceeded. Retry after {retry_after} seconds.",
            )
        if response.status_code >= 500:
            raise SimilaritySearchAPIError(
                response.status_code,
                f"Server error: {response.text[:200]}",
            )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise SimilaritySearchAPIError(response.status_code, str(detail))

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        response = self._http.post(url, json=payload)
        self._raise_for_status(response)
        return response.json()

    def hybrid_similarity_score(
        self,
        query: dict[str, Any],
        corpus: list[dict[str, Any]],
        top_k: int = 10,
        categorical_features: list[str] | None = None,
        continuous_features: list[str] | None = None,
    ) -> dict:
        if not query or not isinstance(query, dict):
            raise SimilaritySearchValidationError(
                "query must be a non-empty dict mapping feature names to values."
            )
        if not corpus or not isinstance(corpus, list):
            raise SimilaritySearchValidationError(
                "corpus must be a non-empty list of dicts."
            )
        if len(corpus) > MAX_BATCH_SIZE:
            raise SimilaritySearchValidationError(
                f"corpus exceeds the maximum allowed size of {MAX_BATCH_SIZE} items. "
                "Split into smaller batches."
            )
        if not all(isinstance(item, dict) for item in corpus):
            raise SimilaritySearchValidationError(
                "Every item in corpus must be a dict with the same schema as query."
            )
        if not isinstance(top_k, int) or top_k < 1:
            raise SimilaritySearchValidationError(
                "top_k must be a positive integer."
            )
        if top_k > len(corpus):
            top_k = len(corpus)

        payload: dict[str, Any] = {
            "query": query,
            "corpus": corpus,
            "top_k": top_k,
        }
        if categorical_features is not None:
            if not isinstance(categorical_features, list) or not all(
                isinstance(f, str) for f in categorical_features
            ):
                raise SimilaritySearchValidationError(
                    "categorical_features must be a list of feature name strings."
                )
            payload["categorical_features"] = categorical_features
        if continuous_features is not None:
            if not isinstance(continuous_features, list) or not all(
                isinstance(f, str) for f in continuous_features
            ):
                raise SimilaritySearchValidationError(
                    "continuous_features must be a list of feature name strings."
                )
            payload["continuous_features"] = continuous_features

        return self._post("/hybrid-score", payload)

    def batch_pairwise_hybrid_score(
        self,
        pairs: list[tuple[dict[str, Any], dict[str, Any]]],
        categorical_features: list[str] | None = None,
        continuous_features: list[str] | None = None,
    ) -> dict:
        if not pairs or not isinstance(pairs, list):
            raise SimilaritySearchValidationError(
                "pairs must be a non-empty list of (query, candidate) dict tuples."
            )
        if len(pairs) > MAX_BATCH_SIZE:
            raise SimilaritySearchValidationError(
                f"pairs exceeds the maximum allowed size of {MAX_BATCH_SIZE}. "
                "Split into smaller batches."
            )
        serialized_pairs = []
        for i, pair in enumerate(pairs):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not isinstance(pair[0], dict)
                or not isinstance(pair[1], dict)
            ):
                raise SimilaritySearchValidationError(
                    f"pairs[{i}] must be a 2-element tuple of dicts (query, candidate)."
                )
            if not pair[0] or not pair[1]:
                raise SimilaritySearchValidationError(
                    f"pairs[{i}] contains an empty dict. Both query and candidate must "
                    "have at least one feature."
                )
            serialized_pairs.append({"query": pair[0], "candidate": pair[1]})

        payload: dict[str, Any] = {"pairs": serialized_pairs}
        if categorical_features is not None:
            if not isinstance(categorical_features, list) or not all(
                isinstance(f, str) for f in categorical_features
            ):
                raise SimilaritySearchValidationError(
                    "categorical_features must be a list of feature name strings."
                )
            payload["categorical_features"] = categorical_features
        if continuous_features is not None:
            if not isinstance(continuous_features, list) or not all(
                isinstance(f, str) for f in continuous_features
            ):
                raise SimilaritySearchValidationError(
                    "continuous_features must be a list of feature name strings."
                )
            payload["continuous_features"] = continuous_features

        return self._post("/batch-pairwise-score", payload)

    def detect_feature_schema(self, sample: list[dict[str, Any]]) -> dict:
        if not sample or not isinstance(sample, list):
            raise SimilaritySearchValidationError(
                "sample must be a non-empty list of dicts representing corpus rows."
            )
        if len(sample) > MAX_BATCH_SIZE:
            raise SimilaritySearchValidationError(
                f"sample size exceeds {MAX_BATCH_SIZE}. Provide a representative subset."
            )
        if not all(isinstance(row, dict) for row in sample):
            raise SimilaritySearchValidationError(
                "Every item in sample must be a dict."
            )
        return self._post("/detect-schema", {"sample": sample})

    def main_method(
        self,
        data: dict[str, Any],
    ) -> dict:
        if not data or not isinstance(data, dict):
            raise SimilaritySearchValidationError(
                "data must be a non-empty dict with keys 'query', 'corpus', "
                "and optionally 'top_k', 'categorical_features', 'continuous_features'."
            )
        missing = [k for k in ("query", "corpus") if k not in data]
        if missing:
            raise SimilaritySearchValidationError(
                f"data is missing required keys: {missing}. "
                "Expected at minimum: 'query' (dict) and 'corpus' (list of dicts)."
            )
        return self.hybrid_similarity_score(
            query=data["query"],
            corpus=data["corpus"],
            top_k=data.get("top_k", 10),
            categorical_features=data.get("categorical_features"),
            continuous_features=data.get("continuous_features"),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()