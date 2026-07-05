import httpx
from typing import Optional
import time


class SimilaritySearchError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(SimilaritySearchError):
    pass


class ValidationError(SimilaritySearchError):
    pass


class RateLimitError(SimilaritySearchError):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class Client:
    BASE_URL = "https://api.similarity-search.nexus/v1"
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.5

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError(
                "api_key must be a non-empty string. "
                "Obtain one at https://similarity-search.nexus/dashboard"
            )
        self._api_key = api_key
        self._base_url = (base_url or self.BASE_URL).rstrip("/")
        self._timeout = timeout
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "similarity-search-sdk-python/1.0.0",
            },
            timeout=self._timeout,
        )

    def rank_by_hybrid_nmi_cosine(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        top_k: int = 10,
        bootstrap_iterations: int = 500,
        confidence_level: float = 0.95,
        nmi_weight: float = 0.4,
    ) -> dict:
        if not query_embedding:
            raise ValidationError("query_embedding must be a non-empty list of floats.")
        if not isinstance(query_embedding, list) or not all(
            isinstance(v, (int, float)) for v in query_embedding
        ):
            raise ValidationError(
                "query_embedding must be a list of numeric values (list[float])."
            )
        if not candidate_embeddings or not isinstance(candidate_embeddings, list):
            raise ValidationError(
                "candidate_embeddings must be a non-empty list of embedding vectors."
            )
        if len(candidate_embeddings) < 1:
            raise ValidationError(
                "candidate_embeddings must contain at least one vector."
            )
        if len(candidate_embeddings) > 1000:
            raise ValidationError(
                f"candidate_embeddings exceeds the maximum of 1000 vectors per call "
                f"(received {len(candidate_embeddings)}). Batch your requests."
            )
        for idx, vec in enumerate(candidate_embeddings):
            if not isinstance(vec, list) or not all(isinstance(v, (int, float)) for v in vec):
                raise ValidationError(
                    f"candidate_embeddings[{idx}] must be a list of numeric values."
                )
            if len(vec) != len(query_embedding):
                raise ValidationError(
                    f"Dimension mismatch: query_embedding has {len(query_embedding)} dims, "
                    f"but candidate_embeddings[{idx}] has {len(vec)} dims."
                )
        if not isinstance(top_k, int) or top_k < 1 or top_k > len(candidate_embeddings):
            raise ValidationError(
                f"top_k must be an integer between 1 and len(candidate_embeddings) "
                f"(received top_k={top_k}, candidates={len(candidate_embeddings)})."
            )
        if not isinstance(bootstrap_iterations, int) or not (100 <= bootstrap_iterations <= 2000):
            raise ValidationError(
                "bootstrap_iterations must be an integer between 100 and 2000."
            )
        if not isinstance(confidence_level, float) or not (0.80 <= confidence_level <= 0.99):
            raise ValidationError(
                "confidence_level must be a float between 0.80 and 0.99 (e.g. 0.95)."
            )
        if not isinstance(nmi_weight, float) or not (0.0 <= nmi_weight <= 1.0):
            raise ValidationError(
                "nmi_weight must be a float between 0.0 and 1.0. "
                "It controls the blend: score = nmi_weight * NMI + (1 - nmi_weight) * cosine."
            )

        payload = {
            "query_embedding": query_embedding,
            "candidate_embeddings": candidate_embeddings,
            "top_k": top_k,
            "bootstrap_iterations": bootstrap_iterations,
            "confidence_level": confidence_level,
            "nmi_weight": nmi_weight,
        }

        return self._post_with_retry("/rank", payload)

    def estimate_embedding_nmi(
        self,
        embedding_a: list[float],
        embedding_b: list[float],
        bootstrap_iterations: int = 500,
    ) -> dict:
        if not embedding_a or not isinstance(embedding_a, list):
            raise ValidationError("embedding_a must be a non-empty list of floats.")
        if not embedding_b or not isinstance(embedding_b, list):
            raise ValidationError("embedding_b must be a non-empty list of floats.")
        if not all(isinstance(v, (int, float)) for v in embedding_a):
            raise ValidationError("embedding_a must contain only numeric values.")
        if not all(isinstance(v, (int, float)) for v in embedding_b):
            raise ValidationError("embedding_b must contain only numeric values.")
        if len(embedding_a) != len(embedding_b):
            raise ValidationError(
                f"Dimension mismatch: embedding_a has {len(embedding_a)} dims, "
                f"embedding_b has {len(embedding_b)} dims."
            )
        if not isinstance(bootstrap_iterations, int) or not (100 <= bootstrap_iterations <= 2000):
            raise ValidationError(
                "bootstrap_iterations must be an integer between 100 and 2000."
            )

        payload = {
            "embedding_a": embedding_a,
            "embedding_b": embedding_b,
            "bootstrap_iterations": bootstrap_iterations,
        }

        return self._post_with_retry("/nmi", payload)

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._http.post(url, json=payload)
            except httpx.TimeoutException as exc:
                last_error = SimilaritySearchError(
                    f"Request to {path} timed out after {self._timeout}s. "
                    "Consider increasing the timeout parameter."
                )
                wait = self.RETRY_BACKOFF_BASE ** attempt
                time.sleep(wait)
                continue
            except httpx.RequestError as exc:
                raise SimilaritySearchError(
                    f"Network error reaching {url}: {exc}"
                ) from exc

            if response.status_code == 200:
                try:
                    return response.json()
                except Exception as exc:
                    raise SimilaritySearchError(
                        f"API returned HTTP 200 but response is not valid JSON: {response.text[:200]}"
                    ) from exc

            try:
                body = response.json()
            except Exception:
                body = {"detail": response.text[:200]}

            if response.status_code == 401:
                raise AuthenticationError(
                    "Invalid or missing API key. Check your credentials at "
                    "https://similarity-search.nexus/dashboard",
                    status_code=401,
                    response_body=body,
                )

            if response.status_code == 422:
                detail = body.get("detail", body)
                raise ValidationError(
                    f"Request payload rejected by the API (422 Unprocessable Entity): {detail}",
                    status_code=422,
                    response_body=body,
                )

            if response.status_code == 429:
                retry_after_raw = response.headers.get("Retry-After")
                retry_after = float(retry_after_raw) if retry_after_raw else None
                if retry_after and attempt < self.MAX_RETRIES - 1:
                    time.sleep(retry_after)
                    continue
                raise RateLimitError(
                    "Rate limit exceeded. Reduce request frequency or upgrade your plan at "
                    "https://similarity-search.nexus/pricing",
                    retry_after=retry_after,
                )

            if response.status_code >= 500:
                last_error = SimilaritySearchError(
                    f"API server error ({response.status_code}) on {path}. "
                    f"Detail: {body.get('detail', 'no detail returned')}",
                    status_code=response.status_code,
                    response_body=body,
                )
                wait = self.RETRY_BACKOFF_BASE ** attempt
                time.sleep(wait)
                continue

            raise SimilaritySearchError(
                f"Unexpected HTTP {response.status_code} from {path}: {body}",
                status_code=response.status_code,
                response_body=body,
            )

        raise last_error or SimilaritySearchError(
            f"All {self.MAX_RETRIES} retry attempts failed for {path}."
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self._http.close()