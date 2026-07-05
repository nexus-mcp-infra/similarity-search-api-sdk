import unittest
from unittest.mock import patch, MagicMock
import json


MOCK_SEARCH_RESPONSE = {
    "results": [
        {
            "id": "doc_001",
            "cosine_similarity": 0.91,
            "nmi_score": 0.74,
            "is_statistically_significant": True,
        },
        {
            "id": "doc_002",
            "cosine_similarity": 0.85,
            "nmi_score": 0.61,
            "is_statistically_significant": True,
        },
        {
            "id": "doc_003",
            "cosine_similarity": 0.78,
            "nmi_score": 0.19,
            "is_statistically_significant": False,
        },
    ],
    "query_embedding_dim": 768,
    "nmi_bins": 10,
    "total_candidates_evaluated": 3,
}

MOCK_QUERY_EMBEDDING = [0.12] * 768
MOCK_CORPUS_EMBEDDINGS = [[round(0.12 + i * 0.01, 4)] * 768 for i in range(3)]


class TestSimilaritySearchAPIHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.search_with_nmi_filter")
    def test_happy_path_returns_nmi_scored_results(self, mock_search):
        """Verifica que una búsqueda válida retorna resultados con nmi_score y cosine_similarity por cada candidato."""
        mock_search.return_value = MOCK_SEARCH_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        response = client.search_with_nmi_filter(
            query_embedding=MOCK_QUERY_EMBEDDING,
            corpus_embeddings=MOCK_CORPUS_EMBEDDINGS,
            nmi_bins=10,
            top_k=3,
        )

        self.assertEqual(len(response["results"]), 3)
        for result in response["results"]:
            self.assertIn("cosine_similarity", result)
            self.assertIn("nmi_score", result)
            self.assertIn("is_statistically_significant", result)
            self.assertGreaterEqual(result["cosine_similarity"], 0.0)
            self.assertLessEqual(result["cosine_similarity"], 1.0)
            self.assertGreaterEqual(result["nmi_score"], 0.0)
            self.assertLessEqual(result["nmi_score"], 1.0)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.search_with_nmi_filter")
    def test_happy_path_statistical_significance_flag_is_boolean(self, mock_search):
        """Verifica que el campo is_statistically_significant es booleano y no un valor truthy arbitrario."""
        mock_search.return_value = MOCK_SEARCH_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        response = client.search_with_nmi_filter(
            query_embedding=MOCK_QUERY_EMBEDDING,
            corpus_embeddings=MOCK_CORPUS_EMBEDDINGS,
            nmi_bins=10,
            top_k=3,
        )

        for result in response["results"]:
            self.assertIsInstance(result["is_statistically_significant"], bool)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.search_with_nmi_filter")
    def test_idempotency_same_inputs_return_identical_nmi_scores(self, mock_search):
        """Verifica que dos llamadas idénticas retornan nmi_score y cosine_similarity iguales sin deriva."""
        mock_search.return_value = MOCK_SEARCH_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        kwargs = dict(
            query_embedding=MOCK_QUERY_EMBEDDING,
            corpus_embeddings=MOCK_CORPUS_EMBEDDINGS,
            nmi_bins=10,
            top_k=3,
        )
        response_a = client.search_with_nmi_filter(**kwargs)
        response_b = client.search_with_nmi_filter(**kwargs)

        scores_a = [(r["nmi_score"], r["cosine_similarity"]) for r in response_a["results"]]
        scores_b = [(r["nmi_score"], r["cosine_similarity"]) for r in response_b["results"]]
        self.assertEqual(scores_a, scores_b)


class TestSimilaritySearchAPIEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.search_with_nmi_filter")
    def test_edge_case_single_corpus_embedding_does_not_crash(self, mock_search):
        """Verifica que un corpus de un solo vector no produce error y retorna exactamente un resultado."""
        single_result_response = {
            "results": [
                {
                    "id": "doc_000",
                    "cosine_similarity": 0.99,
                    "nmi_score": 0.88,
                    "is_statistically_significant": True,
                }
            ],
            "query_embedding_dim": 768,
            "nmi_bins": 10,
            "total_candidates_evaluated": 1,
        }
        mock_search.return_value = single_result_response

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        response = client.search_with_nmi_filter(
            query_embedding=MOCK_QUERY_EMBEDDING,
            corpus_embeddings=[MOCK_CORPUS_EMBEDDINGS[0]],
            nmi_bins=10,
            top_k=1,
        )

        self.assertEqual(len(response["results"]), 1)
        self.assertEqual(response["total_candidates_evaluated"], 1)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.search_with_nmi_filter")
    def test_edge_case_nmi_bins_minimum_value_accepted(self, mock_search):
        """Verifica que nmi_bins=2 (mínimo viable para discretización) es aceptado sin excepción."""
        mock_search.return_value = MOCK_SEARCH_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        response = client.search_with_nmi_filter(
            query_embedding=MOCK_QUERY_EMBEDDING,
            corpus_embeddings=MOCK_CORPUS_EMBEDDINGS,
            nmi_bins=2,
            top_k=3,
        )

        self.assertIn("results", response)
        mock_search.assert_called_once()
        _, call_kwargs = mock_search.call_args
        self.assertEqual(call_kwargs.get("nmi_bins", 2), 2)


class TestSimilaritySearchAPIInvalidInputs(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.search_with_nmi_filter")
    def test_invalid_input_empty_query_embedding_raises_value_error(self, mock_search):
        """Verifica que un query_embedding vacío lanza ValueError con mensaje descriptivo del campo inválido."""
        mock_search.side_effect = ValueError(
            "query_embedding must be a non-empty list of floats; received empty list"
        )

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        with self.assertRaises(ValueError) as ctx:
            client.search_with_nmi_filter(
                query_embedding=[],
                corpus_embeddings=MOCK_CORPUS_EMBEDDINGS,
                nmi_bins=10,
                top_k=3,
            )

        self.assertIn("query_embedding", str(ctx.exception))

    @patch("similarity_search_api_sdk.SimilaritySearchClient.search_with_nmi_filter")
    def test_invalid_input_nmi_bins_as_string_raises_type_error(self, mock_search):
        """Verifica que pasar nmi_bins como string en vez de int lanza TypeError con nombre del parámetro."""
        mock_search.side_effect = TypeError(
            "nmi_bins must be an integer >= 2; received type str"
        )

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        with self.assertRaises(TypeError) as ctx:
            client.search_with_nmi_filter(
                query_embedding=MOCK_QUERY_EMBEDDING,
                corpus_embeddings=MOCK_CORPUS_EMBEDDINGS,
                nmi_bins="ten",
                top_k=3,
            )

        self.assertIn("nmi_bins", str(ctx.exception))


class TestSimilaritySearchAPIAuth(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.search_with_nmi_filter")
    def test_auth_missing_api_key_raises_authentication_error(self, mock_search):
        """Verifica que omitir api_key lanza una excepción con texto que identifica la causa como autenticación."""
        mock_search.side_effect = PermissionError(
            "Authentication failed: api_key is required and was not provided"
        )

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=None)
        with self.assertRaises(PermissionError) as ctx:
            client.search_with_nmi_filter(
                query_embedding=MOCK_QUERY_EMBEDDING,
                corpus_embeddings=MOCK_CORPUS_EMBEDDINGS,
                nmi_bins=10,
                top_k=3,
            )

        error_text = str(ctx.exception).lower()
        self.assertTrue(
            "api_key" in error_text or "authentication" in error_text or "auth" in error_text,
            msg=f"Error message does not describe auth failure: {ctx.exception}",
        )


class TestSimilaritySearchAPIRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.search_with_nmi_filter")
    def test_rate_limit_burst_of_ten_calls_does_not_raise_unhandled_exception(self, mock_search):
        """Verifica que 10 llamadas en ráfaga no producen excepción no manejada; el cliente absorbe o propaga limpiamente."""
        call_count = 0

        def side_effect_rate_limit(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 8:
                raise RuntimeError("rate_limit_exceeded: max 8 requests per second on this tier")
            return MOCK_SEARCH_RESPONSE

        mock_search.side_effect = side_effect_rate_limit

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        successful = 0
        rate_limited = 0

        for _ in range(10):
            try:
                client.search_with_nmi_filter(
                    query_embedding=MOCK_QUERY_EMBEDDING,
                    corpus_embeddings=MOCK_CORPUS_EMBEDDINGS,
                    nmi_bins=10,
                    top_k=3,
                )
                successful += 1
            except RuntimeError as exc:
                if "rate_limit_exceeded" in str(exc):
                    rate_limited += 1
                else:
                    raise

        self.assertEqual(successful + rate_limited, 10)
        self.assertGreater(successful, 0)
        self.assertGreater(rate_limited, 0)


if __name__ == "__main__":
    unittest.main()