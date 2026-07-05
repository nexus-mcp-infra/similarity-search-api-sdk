import unittest
from unittest.mock import patch, MagicMock
import numpy as np


MOCK_SEARCH_RESPONSE = {
    "results": [
        {"id": "vec_001", "score": 0.97, "metadata": {"label": "alpha"}},
        {"id": "vec_002", "score": 0.84, "metadata": {"label": "beta"}},
    ],
    "nmi_filtered_dims": 12,
    "original_dims": 128,
    "query_time_ms": 4.2,
}

MOCK_INDEX_RESPONSE = {
    "indexed": 3,
    "index_id": "idx_test_001",
    "status": "ok",
}


def _make_sdk_client(api_key="test-key-abc"):
    from similarity_search_api_sdk import SimilaritySearchClient
    return SimilaritySearchClient(api_key=api_key, base_url="http://localhost:8000")


class TestSimilaritySearchHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_cosine_search_returns_ranked_results(self, mock_post):
        """Verifica que una query valida retorna resultados ordenados por score coseno."""
        mock_post.return_value = MOCK_SEARCH_RESPONSE
        client = _make_sdk_client()
        query_vector = np.random.rand(128).tolist()
        response = client.search(
            query_vector=query_vector,
            index_id="idx_test_001",
            top_k=2,
            nmi_threshold=0.05,
        )
        self.assertEqual(len(response["results"]), 2)
        scores = [r["score"] for r in response["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn("nmi_filtered_dims", response)
        self.assertLessEqual(response["nmi_filtered_dims"], response["original_dims"])

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_nmi_reduces_dimensionality(self, mock_post):
        """Verifica que nmi_filtered_dims es estrictamente menor que original_dims cuando hay features ruidosas."""
        mock_post.return_value = MOCK_SEARCH_RESPONSE
        client = _make_sdk_client()
        query_vector = np.random.rand(128).tolist()
        response = client.search(
            query_vector=query_vector,
            index_id="idx_test_001",
            top_k=2,
            nmi_threshold=0.05,
        )
        self.assertLess(response["nmi_filtered_dims"], response["original_dims"])
        self.assertEqual(response["original_dims"], 128)

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_index_vectors_returns_index_id(self, mock_post):
        """Verifica que indexar un batch de vectores devuelve un index_id valido y conteo correcto."""
        mock_post.return_value = MOCK_INDEX_RESPONSE
        client = _make_sdk_client()
        vectors = [
            {"id": "v1", "vector": np.random.rand(128).tolist()},
            {"id": "v2", "vector": np.random.rand(128).tolist()},
            {"id": "v3", "vector": np.random.rand(128).tolist()},
        ]
        response = client.index_vectors(vectors=vectors)
        self.assertEqual(response["indexed"], 3)
        self.assertTrue(response["index_id"].startswith("idx_"))
        self.assertEqual(response["status"], "ok")


class TestSimilaritySearchEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_edge_case_single_dimension_vector(self, mock_post):
        """Verifica que un vector de dimension 1 no causa crash y retorna respuesta con dims coherentes."""
        single_dim_response = {
            "results": [],
            "nmi_filtered_dims": 1,
            "original_dims": 1,
            "query_time_ms": 0.9,
        }
        mock_post.return_value = single_dim_response
        client = _make_sdk_client()
        response = client.search(
            query_vector=[0.5],
            index_id="idx_test_001",
            top_k=5,
            nmi_threshold=0.05,
        )
        self.assertEqual(response["original_dims"], 1)
        self.assertIsInstance(response["results"], list)

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_edge_case_top_k_larger_than_index_size(self, mock_post):
        """Verifica que pedir mas resultados que vectores indexados retorna solo los disponibles sin error."""
        truncated_response = {
            "results": [{"id": "vec_001", "score": 0.91, "metadata": {}}],
            "nmi_filtered_dims": 10,
            "original_dims": 64,
            "query_time_ms": 2.1,
        }
        mock_post.return_value = truncated_response
        client = _make_sdk_client()
        response = client.search(
            query_vector=np.random.rand(64).tolist(),
            index_id="idx_small",
            top_k=1000,
            nmi_threshold=0.05,
        )
        self.assertLessEqual(len(response["results"]), 1000)
        self.assertGreaterEqual(len(response["results"]), 0)


class TestSimilaritySearchInvalidInput(unittest.TestCase):

    def test_invalid_input_none_query_vector_raises_value_error(self):
        """Verifica que pasar None como query_vector lanza ValueError con mensaje descriptivo."""
        client = _make_sdk_client()
        with self.assertRaises(ValueError) as ctx:
            client.search(
                query_vector=None,
                index_id="idx_test_001",
                top_k=5,
                nmi_threshold=0.05,
            )
        self.assertIn("query_vector", str(ctx.exception).lower())

    def test_invalid_input_non_numeric_vector_raises_type_error(self):
        """Verifica que un vector con strings en lugar de floats lanza TypeError antes de llegar a la red."""
        client = _make_sdk_client()
        with self.assertRaises((TypeError, ValueError)):
            client.search(
                query_vector=["a", "b", "c"],
                index_id="idx_test_001",
                top_k=5,
                nmi_threshold=0.05,
            )


class TestSimilaritySearchRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_rate_limit_burst_of_calls_does_not_raise(self, mock_post):
        """Verifica que 20 llamadas consecutivas rapidas no generan crash en el cliente."""
        mock_post.return_value = MOCK_SEARCH_RESPONSE
        client = _make_sdk_client()
        query_vector = np.random.rand(128).tolist()
        exceptions_raised = 0
        for _ in range(20):
            try:
                client.search(
                    query_vector=query_vector,
                    index_id="idx_test_001",
                    top_k=2,
                    nmi_threshold=0.05,
                )
            except Exception:
                exceptions_raised += 1
        self.assertEqual(exceptions_raised, 0)


class TestSimilaritySearchAuth(unittest.TestCase):

    def test_auth_missing_api_key_raises_authentication_error(self):
        """Verifica que inicializar el cliente sin API key lanza un error de autenticacion descriptivo."""
        from similarity_search_api_sdk import SimilaritySearchClient, AuthenticationError
        with self.assertRaises((AuthenticationError, ValueError)) as ctx:
            client = SimilaritySearchClient(api_key=None, base_url="http://localhost:8000")
            client.search(
                query_vector=np.random.rand(128).tolist(),
                index_id="idx_test_001",
                top_k=2,
                nmi_threshold=0.05,
            )
        error_msg = str(ctx.exception).lower()
        self.assertTrue(
            any(kw in error_msg for kw in ("api_key", "auth", "key", "credential")),
            msg=f"Error message not descriptive enough: {ctx.exception}",
        )


class TestSimilaritySearchIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_idempotency_same_query_returns_identical_scores(self, mock_post):
        """Verifica que la misma query ejecutada dos veces produce scores identicos (pipeline sin estado)."""
        mock_post.return_value = MOCK_SEARCH_RESPONSE
        client = _make_sdk_client()
        query_vector = np.random.rand(128).tolist()
        search_kwargs = dict(
            query_vector=query_vector,
            index_id="idx_test_001",
            top_k=2,
            nmi_threshold=0.05,
        )
        first_response = client.search(**search_kwargs)
        second_response = client.search(**search_kwargs)
        self.assertEqual(
            [r["score"] for r in first_response["results"]],
            [r["score"] for r in second_response["results"]],
        )
        self.assertEqual(
            first_response["nmi_filtered_dims"],
            second_response["nmi_filtered_dims"],
        )


if __name__ == "__main__":
    unittest.main()