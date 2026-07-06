import httpx
from typing import Any

BASE_URL = "https://api.nexus-similarity.io/v1"
DEFAULT_TIMEOUT = 30.0


class SimilaritySearchError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response_body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body or {}


class AuthenticationError(SimilaritySearchError):
    pass


class ValidationError(SimilaritySearchError):
    pass


class RateLimitError(SimilaritySearchError):
    pass


class Client:
    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError("api_key must be a non-empty string")
        if not api_key.strip():
            raise AuthenticationError("api_key cannot be whitespace-only")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    def _raise_for_status(self, response: httpx.Response) -> dict:
        try:
            body = response.json()
        except Exception:
            body = {"detail": response.text}

        if response.status_code == 401:
            raise AuthenticationError(
                "Invalid or missing API key",
                status_code=response.status_code,
                response_body=body,
            )
        if response.status_code == 422:
            raise ValidationError(
                f"Payload validation failed: {body.get('detail', body)}",
                status_code=response.status_code,
                response_body=body,
            )
        if response.status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded — retry after backing off",
                status_code=response.status_code,
                response_body=body,
            )
        if response.status_code >= 500:
            raise SimilaritySearchError(
                f"Server error {response.status_code}: {body.get('detail', body)}",
                status_code=response.status_code,
                response_body=body,
            )
        if response.status_code >= 400:
            raise SimilaritySearchError(
                f"Client error {response.status_code}: {body.get('detail', body)}",
                status_code=response.status_code,
                response_body=body,
            )
        return body

    def ranked_similarity_search(
        self,
        query: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        alpha: float = 0.6,
    ) -> dict[str, Any]:
        if not isinstance(query, dict) or not query:
            raise ValidationError("query must be a non-empty dict representing one item")
        if not isinstance(candidates, list) or len(candidates) == 0:
            raise ValidationError("candidates must be a non-empty list of dicts")
        if not all(isinstance(c, dict) for c in candidates):
            raise ValidationError("every element in candidates must be a dict")
        if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
            raise ValidationError("top_k must be an integer between 1 and 1000")
        if not isinstance(alpha, float) and not isinstance(alpha, int):
            raise ValidationError("alpha must be a float between 0.0 and 1.0")
        alpha = float(alpha)
        if alpha < 0.0 or alpha > 1.0:
            raise ValidationError("alpha must be between 0.0 and 1.0 — 1.0 is pure cosine, 0.0 is pure NMI")

        payload = {
            "query": query,
            "candidates": candidates,
            "top_k": top_k,
            "alpha": alpha,
        }

        try:
            response = self._client.post(f"{self._base_url}/search", json=payload)
        except httpx.TimeoutException as exc:
            raise SimilaritySearchError(f"Request timed out after {self._client.timeout.read}s") from exc
        except httpx.RequestError as exc:
            raise SimilaritySearchError(f"Network error contacting similarity API: {exc}") from exc

        return self._raise_for_status(response)

    def main_method(
        self,
        data: dict[str, Any],
        top_k: int = 10,
        alpha: float = 0.6,
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValidationError(
                "data must be a dict with keys 'query' (dict) and 'candidates' (list of dicts)"
            )
        query = data.get("query")
        candidates = data.get("candidates")
        if query is None or candidates is None:
            raise ValidationError(
                "data must contain 'query' (dict) and 'candidates' (list of dicts)"
            )
        return self.ranked_similarity_search(
            query=query,
            candidates=candidates,
            top_k=data.get("top_k", top_k),
            alpha=data.get("alpha", alpha),
        )

    def explain_score(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
        alpha: float = 0.6,
    ) -> dict[str, Any]:
        if not isinstance(query, dict) or not query:
            raise ValidationError("query must be a non-empty dict")
        if not isinstance(candidate, dict) or not candidate:
            raise ValidationError("candidate must be a non-empty dict")
        alpha = float(alpha)
        if alpha < 0.0 or alpha > 1.0:
            raise ValidationError("alpha must be between 0.0 and 1.0")

        payload = {
            "query": query,
            "candidate": candidate,
            "alpha": alpha,
        }

        try:
            response = self._client.post(f"{self._base_url}/explain", json=payload)
        except httpx.TimeoutException as exc:
            raise SimilaritySearchError(f"Request timed out") from exc
        except httpx.RequestError as exc:
            raise SimilaritySearchError(f"Network error: {exc}") from exc

        return self._raise_for_status(response)

    def batch_ranked_similarity_search(
        self,
        queries: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        alpha: float = 0.6,
    ) -> dict[str, Any]:
        if not isinstance(queries, list) or len(queries) == 0:
            raise ValidationError("queries must be a non-empty list of dicts")
        if not all(isinstance(q, dict) for q in queries):
            raise ValidationError("every element in queries must be a dict")
        if len(queries) > 50:
            raise ValidationError("batch queries is capped at 50 items per call — split into multiple requests")
        if not isinstance(candidates, list) or len(candidates) == 0:
            raise ValidationError("candidates must be a non-empty list of dicts")
        if not all(isinstance(c, dict) for c in candidates):
            raise ValidationError("every element in candidates must be a dict")
        if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
            raise ValidationError("top_k must be an integer between 1 and 1000")
        alpha = float(alpha)
        if alpha < 0.0 or alpha > 1.0:
            raise ValidationError("alpha must be between 0.0 and 1.0")

        payload = {
            "queries": queries,
            "candidates": candidates,
            "top_k": top_k,
            "alpha": alpha,
        }

        try:
            response = self._client.post(f"{self._base_url}/batch", json=payload)
        except httpx.TimeoutException as exc:
            raise SimilaritySearchError("Batch request timed out") from exc
        except httpx.RequestError as exc:
            raise SimilaritySearchError(f"Network error: {exc}") from exc

        return self._raise_for_status(response)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()