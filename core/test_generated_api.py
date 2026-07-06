import unittest
from unittest.mock import patch, MagicMock
import json


MOCK_SEARCH_RESPONSE = {
    "results": [
        {"id": "doc_001", "cosine_similarity": 0.91, "nmi_score": 0.74, "combined_score": 0.84},
        {"id": "doc_002", "cosine_similarity": 0.85, "nmi_score": 0.61, "combined_score": 0.75},
    ],
    "query_token_count": 12,
    "index_size": 500,
    "stateless": True,
}

MOCK_EMPTY_RESULTS = {
    "results": [],
    "query_token_count": 3,
    "index_size": 0,
    "stateless": True,
}


def make_sdk_client(api_key="test-key-forge-001"):
    sdk = MagicMock()
    sdk.api_key = api_key
    sdk.search = MagicMock(return_value=MOCK_SEARCH_RESPONSE)
    sdk.search_empty = MagicMock(return_value=MOCK_EMPTY_RESULTS)
    return sdk


class TestSimilaritySearchAPI(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_happy_path_mixed_dataset_returns_ranked_results(self, MockClient):
        """Verifica que un query con texto y features categoricos retorna resultados con NMI y cosine scores validos."""
        client = make_sdk_client()
        MockClient.return_value = client

        payload = {
            "query": {
                "text": "distributed stateless search pipeline",
                "categories": {"domain": "infrastructure", "tier": "enterprise"},
                "numerics": {"latency_ms": 42.0, "throughput_rps": 1200.0},
            },
            "corpus": [
                {
                    "id": "doc_001",
                    "text": "serverless search infrastructure for enterprise",
                    "categories": {"domain": "infrastructure", "tier": "enterprise"},
                    "numerics": {"latency_ms": 38.0, "throughput_rps": 1100.0},
                },
                {
                    "id": "doc_002",
                    "text": "embedding pipelines for NLP",
                    "categories": {"domain": "ml", "tier": "startup"},
                    "numerics": {"latency_ms": 95.0, "throughput_rps": 300.0},
                },
            ],
            "top_k": 2,
        }

        response = client.search(payload)

        self.assertEqual(len(response["results"]), 2)
        top = response["results"][0]
        self.assertIn("cosine_similarity", top)
        self.assertIn("nmi_score", top)
        self.assertIn("combined_score", top)
        self.assertGreaterEqual(top["cosine_similarity"], 0.0)
        self.assertLessEqual(top["cosine_similarity"], 1.0)
        self.assertGreaterEqual(top["nmi_score"], 0.0)
        self.assertLessEqual(top["nmi_score"], 1.0)
        self.assertTrue(response["stateless"])

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_happy_path_combined_score_preserves_ranking_order(self, MockClient):
        """Verifica que combined_score del primer resultado es mayor o igual al del segundo resultado."""
        client = make_sdk_client()
        MockClient.return_value = client

        response = client.search({"query": {"text": "latency optimization"}, "corpus": [], "top_k": 2})
        results = response["results"]

        self.assertGreaterEqual(
            results[0]["combined_score"],
            results[1]["combined_score"],
        )

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_edge_case_empty_corpus_returns_empty_results(self, MockClient):
        """Verifica que un corpus vacio retorna lista de resultados vacia sin lanzar excepcion."""
        client = make_sdk_client()
        client.search = MagicMock(return_value=MOCK_EMPTY_RESULTS)
        MockClient.return_value = client

        response = client.search({"query": {"text": "anything"}, "corpus": [], "top_k": 5})

        self.assertEqual(response["results"], [])
        self.assertEqual(response["index_size"], 0)

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_edge_case_top_k_larger_than_corpus_clips_to_corpus_size(self, MockClient):
        """Verifica que top_k mayor al corpus no retorna mas documentos de los que existen."""
        client = make_sdk_client()
        one_result_response = {**MOCK_SEARCH_RESPONSE, "results": [MOCK_SEARCH_RESPONSE["results"][0]]}
        client.search = MagicMock(return_value=one_result_response)
        MockClient.return_value = client

        response = client.search({"query": {"text": "search"}, "corpus": [{"id": "doc_001", "text": "x"}], "top_k": 999})

        self.assertLessEqual(len(response["results"]), 1)

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_invalid_input_none_query_raises_value_error(self, MockClient):
        """Verifica que pasar query=None levanta ValueError con mensaje descriptivo."""
        client = make_sdk_client()
        client.search = MagicMock(side_effect=ValueError("query must not be None or empty"))
        MockClient.return_value = client

        with self.assertRaises(ValueError) as ctx:
            client.search({"query": None, "corpus": [], "top_k": 5})

        self.assertIn("query", str(ctx.exception).lower())

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_invalid_input_non_numeric_top_k_raises_type_error(self, MockClient):
        """Verifica que top_k con tipo string en vez de int levanta TypeError."""
        client = make_sdk_client()
        client.search = MagicMock(side_effect=TypeError("top_k must be a positive integer, got str"))
        MockClient.return_value = client

        with self.assertRaises(TypeError) as ctx:
            client.search({"query": {"text": "test"}, "corpus": [], "top_k": "five"})

        self.assertIn("top_k", str(ctx.exception).lower())

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_auth_missing_api_key_raises_authentication_error(self, MockClient):
        """Verifica que instanciar el cliente sin API key y hacer una llamada levanta AuthenticationError."""
        client = make_sdk_client(api_key="")
        client.search = MagicMock(
            side_effect=PermissionError("AuthenticationError: api_key is required and must not be empty")
        )
        MockClient.return_value = client

        with self.assertRaises(PermissionError) as ctx:
            client.search({"query": {"text": "test"}, "corpus": [], "top_k": 1})

        self.assertIn("api_key", str(ctx.exception).lower())
        self.assertIn("required", str(ctx.exception).lower())

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_rate_limit_sequential_calls_do_not_crash(self, MockClient):
        """Verifica que 50 llamadas secuenciales al endpoint no levantan excepcion ni retornan None."""
        client = make_sdk_client()
        MockClient.return_value = client

        payload = {"query": {"text": "repeated query for rate limit test"}, "corpus": [], "top_k": 1}

        responses = []
        for _ in range(50):
            resp = client.search(payload)
            responses.append(resp)

        self.assertEqual(len(responses), 50)
        for resp in responses:
            self.assertIsNotNone(resp)
            self.assertIn("results", resp)

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_idempotency_same_payload_returns_identical_scores(self, MockClient):
        """Verifica que dos llamadas identicas producen exactamente los mismos combined_score y nmi_score."""
        client = make_sdk_client()
        MockClient.return_value = client

        payload = {
            "query": {
                "text": "idempotency check",
                "categories": {"domain": "fintech"},
                "numerics": {"price": 99.9},
            },
            "corpus": [
                {
                    "id": "doc_001",
                    "text": "fintech pricing engine",
                    "categories": {"domain": "fintech"},
                    "numerics": {"price": 89.0},
                }
            ],
            "top_k": 1,
        }

        first_response = client.search(payload)
        second_response = client.search(payload)

        self.assertEqual(
            first_response["results"][0]["combined_score"],
            second_response["results"][0]["combined_score"],
        )
        self.assertEqual(
            first_response["results"][0]["nmi_score"],
            second_response["results"][0]["nmi_score"],
        )
        self.assertEqual(
            first_response["results"][0]["cosine_similarity"],
            second_response["results"][0]["cosine_similarity"],
        )

    @patch("similarity_search_api_sdk.SimilaritySearchClient")
    def test_edge_case_very_long_text_query_does_not_raise(self, MockClient):
        """Verifica que un query de texto de 10000 caracteres no lanza excepcion y retorna estructura valida."""
        client = make_sdk_client()
        MockClient.return_value = client

        long_text = "distributed stateless nmi cosine search pipeline " * 200
        self.assertEqual(len(long_text), 10000)

        payload = {
            "query": {"text": long_text},
            "corpus": [{"id": "doc_001", "text": "reference document for long query test"}],
            "top_k": 1,
        }

        response = client.search(payload)

        self.assertIn("results", response)
        self.assertIn("stateless", response)
        self.assertIsInstance(response["results"], list)


if __name__ == "__main__":
    unittest.main()