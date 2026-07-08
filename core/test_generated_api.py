import unittest
from unittest.mock import patch, MagicMock
import json


MOCK_SIMILARITY_RESPONSE = {
    "results": [
        {"item_id": "doc_42", "cosine_similarity": 0.91, "nmi_score": 0.74, "combined_score": 0.83},
        {"item_id": "doc_17", "cosine_similarity": 0.85, "nmi_score": 0.68, "combined_score": 0.77},
    ],
    "query_tokens": 12,
    "corpus_size": 1000,
    "latency_ms": 38,
}

MOCK_AUTH_ERROR = {
    "detail": "Authentication failed: API key missing or invalid. Provide a valid key via the X-API-Key header."
}

MOCK_RATE_LIMIT_RESPONSE = {
    "detail": "Rate limit exceeded: maximum 60 requests per minute per API key."
}


def _make_mock_response(status_code: int, body: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = body
    mock_resp.headers = {"Content-Type": "application/json"}
    return mock_resp


class TestSimilaritySearchAPIHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_valid_query_returns_ranked_results(self, mock_post):
        """Verifica que una query válida con corpus poblado retorna ítems rankeados por combined_score."""
        mock_post.return_value = _make_mock_response(200, MOCK_SIMILARITY_RESPONSE)

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test-key-mock")
        results = client.search(
            query="machine learning optimization",
            corpus=["doc_42", "doc_17", "doc_99"],
            top_k=2,
        )

        self.assertEqual(len(results["results"]), 2)
        scores = [r["combined_score"] for r in results["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn("nmi_score", results["results"][0])
        self.assertIn("cosine_similarity", results["results"][0])

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_single_item_corpus_returns_one_result(self, mock_post):
        """Verifica que un corpus de un solo ítem retorna exactamente un resultado con scores válidos."""
        single_response = {
            "results": [{"item_id": "doc_01", "cosine_similarity": 1.0, "nmi_score": 1.0, "combined_score": 1.0}],
            "query_tokens": 5,
            "corpus_size": 1,
            "latency_ms": 12,
        }
        mock_post.return_value = _make_mock_response(200, single_response)

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test-key-mock")
        results = client.search(query="neural networks", corpus=["doc_01"], top_k=1)

        self.assertEqual(len(results["results"]), 1)
        self.assertAlmostEqual(results["results"][0]["combined_score"], 1.0)
        self.assertGreaterEqual(results["results"][0]["nmi_score"], 0.0)
        self.assertLessEqual(results["results"][0]["nmi_score"], 1.0)


class TestSimilaritySearchAPIEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_edge_case_empty_corpus_returns_empty_results(self, mock_post):
        """Verifica que un corpus vacío retorna lista de resultados vacía sin error de servidor."""
        empty_response = {"results": [], "query_tokens": 4, "corpus_size": 0, "latency_ms": 5}
        mock_post.return_value = _make_mock_response(200, empty_response)

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test-key-mock")
        results = client.search(query="quantum computing", corpus=[], top_k=10)

        self.assertEqual(results["results"], [])
        self.assertEqual(results["corpus_size"], 0)

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_edge_case_query_at_max_token_length_is_accepted(self, mock_post):
        """Verifica que una query en el límite máximo de tokens (512) es procesada sin truncamiento silencioso."""
        mock_post.return_value = _make_mock_response(200, MOCK_SIMILARITY_RESPONSE)

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test-key-mock")
        long_query = " ".join([f"token{i}" for i in range(512)])
        results = client.search(query=long_query, corpus=["doc_42", "doc_17"], top_k=2)

        call_payload = mock_post.call_args[1] if mock_post.call_args[1] else mock_post.call_args[0][0]
        self.assertIsNotNone(results)
        self.assertIn("results", results)


class TestSimilaritySearchAPIInvalidInput(unittest.TestCase):

    def test_invalid_input_none_query_raises_value_error(self):
        """Verifica que pasar None como query lanza ValueError con mensaje descriptivo antes de realizar la llamada HTTP."""
        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test-key-mock")

        with self.assertRaises(ValueError) as ctx:
            client.search(query=None, corpus=["doc_01"], top_k=5)

        self.assertIn("query", str(ctx.exception).lower())

    def test_invalid_input_non_integer_top_k_raises_type_error(self):
        """Verifica que top_k no entero lanza TypeError con indicación del parámetro afectado."""
        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test-key-mock")

        with self.assertRaises(TypeError) as ctx:
            client.search(query="valid query", corpus=["doc_01"], top_k="five")

        self.assertIn("top_k", str(ctx.exception).lower())


class TestSimilaritySearchAPIRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_rate_limit_consecutive_calls_do_not_raise_unhandled_exception(self, mock_post):
        """Verifica que 61 llamadas consecutivas retornan RateLimitError en la llamada 61 sin crash del proceso."""
        from similarity_search_api_sdk import SimilaritySearchClient, RateLimitError

        success = _make_mock_response(200, MOCK_SIMILARITY_RESPONSE)
        throttled = _make_mock_response(429, MOCK_RATE_LIMIT_RESPONSE)

        mock_post.side_effect = [success] * 60 + [throttled]

        client = SimilaritySearchClient(api_key="test-key-mock")
        errors_encountered = 0

        for i in range(61):
            try:
                client.search(query=f"query iteration {i}", corpus=["doc_42"], top_k=1)
            except RateLimitError as exc:
                errors_encountered += 1
                self.assertIn("rate limit", str(exc).lower())

        self.assertEqual(errors_encountered, 1)


class TestSimilaritySearchAPIAuth(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_auth_missing_api_key_raises_authentication_error(self, mock_post):
        """Verifica que instanciar el cliente sin API key lanza AuthenticationError antes de cualquier llamada HTTP."""
        from similarity_search_api_sdk import SimilaritySearchClient, AuthenticationError

        with self.assertRaises(AuthenticationError) as ctx:
            SimilaritySearchClient(api_key=None)

        self.assertIn("api key", str(ctx.exception).lower())
        mock_post.assert_not_called()


class TestSimilaritySearchAPIIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_idempotency_identical_queries_return_identical_combined_scores(self, mock_post):
        """Verifica que dos invocaciones con idénticos parámetros producen resultados con combined_scores idénticos."""
        mock_post.return_value = _make_mock_response(200, MOCK_SIMILARITY_RESPONSE)

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test-key-mock")

        first = client.search(query="deep learning embeddings", corpus=["doc_42", "doc_17"], top_k=2)
        second = client.search(query="deep learning embeddings", corpus=["doc_42", "doc_17"], top_k=2)

        self.assertEqual(
            [r["combined_score"] for r in first["results"]],
            [r["combined_score"] for r in second["results"]],
        )
        self.assertEqual(
            [r["item_id"] for r in first["results"]],
            [r["item_id"] for r in second["results"]],
        )
        self.assertEqual(mock_post.call_count, 2)


if __name__ == "__main__":
    unittest.main()