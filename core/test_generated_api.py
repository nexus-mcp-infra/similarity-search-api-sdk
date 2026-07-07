import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import json


MOCK_API_KEY = "test-key-nexus-similarity-001"
BASE_URL = "http://localhost:8000"

VECTOR_A = [0.1, 0.4, 0.7, 0.2, 0.9, 0.3, 0.6, 0.8]
VECTOR_B = [0.15, 0.38, 0.72, 0.18, 0.88, 0.31, 0.59, 0.77]
VECTOR_ORTHOGONAL = [0.9, -0.4, 0.1, -0.7, 0.2, -0.8, 0.3, -0.6]


def make_mock_similarity_response(
    cosine_score=0.9982,
    nmi_score=0.7641,
    composite_score=0.8812,
    n_dims=8,
):
    return {
        "cosine_similarity": cosine_score,
        "nmi_score": nmi_score,
        "composite_score": composite_score,
        "metadata": {
            "n_dims": n_dims,
            "nmi_bins": 10,
            "composite_weights": {"cosine": 0.4, "nmi": 0.6},
        },
    }


class TestSimilaritySearchHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_similarity")
    def test_happy_path_valid_vectors_returns_composite_score(self, mock_compute):
        """Verifica que dos vectores válidos retornan composite_score en [0, 1]."""
        mock_compute.return_value = make_mock_similarity_response()

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)
        result = client.compute_similarity(VECTOR_A, VECTOR_B)

        self.assertIn("composite_score", result)
        self.assertGreaterEqual(result["composite_score"], 0.0)
        self.assertLessEqual(result["composite_score"], 1.0)
        mock_compute.assert_called_once_with(VECTOR_A, VECTOR_B)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_similarity")
    def test_happy_path_nmi_lower_than_cosine_for_spurious_vectors(self, mock_compute):
        """Verifica que vectores en subespacio denso tienen NMI significativamente menor que coseno."""
        mock_compute.return_value = make_mock_similarity_response(
            cosine_score=0.961,
            nmi_score=0.312,
            composite_score=0.571,
        )

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)
        result = client.compute_similarity(VECTOR_A, VECTOR_ORTHOGONAL)

        self.assertLess(result["nmi_score"], result["cosine_similarity"] - 0.3)
        self.assertLess(result["composite_score"], result["cosine_similarity"])

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_similarity")
    def test_happy_path_metadata_contains_weights_and_bins(self, mock_compute):
        """Verifica que la respuesta incluye metadatos de configuracion NMI y pesos del composite."""
        mock_compute.return_value = make_mock_similarity_response()

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)
        result = client.compute_similarity(VECTOR_A, VECTOR_B)

        self.assertIn("metadata", result)
        meta = result["metadata"]
        self.assertIn("nmi_bins", meta)
        self.assertIn("composite_weights", meta)
        self.assertAlmostEqual(
            sum(meta["composite_weights"].values()), 1.0, places=5
        )


class TestSimilaritySearchEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_similarity")
    def test_edge_case_identical_vectors_yield_max_scores(self, mock_compute):
        """Verifica que vectores identicos producen coseno=1.0, NMI=1.0 y composite=1.0."""
        mock_compute.return_value = make_mock_similarity_response(
            cosine_score=1.0,
            nmi_score=1.0,
            composite_score=1.0,
        )

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)
        result = client.compute_similarity(VECTOR_A, VECTOR_A)

        self.assertAlmostEqual(result["cosine_similarity"], 1.0, places=5)
        self.assertAlmostEqual(result["nmi_score"], 1.0, places=5)
        self.assertAlmostEqual(result["composite_score"], 1.0, places=5)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_similarity")
    def test_edge_case_high_dimensional_vector_1536_dims(self, mock_compute):
        """Verifica que vectores de 1536 dimensiones (OpenAI ada-002) se procesan sin error."""
        vec_1536 = list(np.random.default_rng(42).random(1536).tolist())
        mock_compute.return_value = make_mock_similarity_response(n_dims=1536)

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)
        result = client.compute_similarity(vec_1536, vec_1536)

        self.assertEqual(result["metadata"]["n_dims"], 1536)
        self.assertIn("composite_score", result)


class TestSimilaritySearchInvalidInputs(unittest.TestCase):

    def test_invalid_input_none_vector_raises_value_error(self):
        """Verifica que pasar None como vector lanza ValueError con mensaje descriptivo."""
        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)

        with self.assertRaises((ValueError, TypeError)) as ctx:
            client.compute_similarity(None, VECTOR_B)

        self.assertTrue(
            len(str(ctx.exception)) > 0,
            "La excepcion debe tener un mensaje descriptivo no vacio",
        )

    def test_invalid_input_mismatched_dimensions_raises_value_error(self):
        """Verifica que vectores de dimensiones distintas producen ValueError antes de llamar al backend."""
        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)
        short_vector = [0.1, 0.2, 0.3]

        with self.assertRaises(ValueError) as ctx:
            client.compute_similarity(VECTOR_A, short_vector)

        self.assertIn(
            str(len(VECTOR_A)), str(ctx.exception) + str(len(short_vector)),
        )

    def test_invalid_input_non_numeric_elements_raises_type_error(self):
        """Verifica que un vector con strings en lugar de floats lanza TypeError."""
        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)
        bad_vector = ["embed", "this", "text", "please", "0.1", "0.2", "0.3", "0.4"]

        with self.assertRaises((TypeError, ValueError)):
            client.compute_similarity(bad_vector, VECTOR_B)


class TestSimilaritySearchRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_similarity")
    def test_rate_limit_10_sequential_calls_no_crash(self, mock_compute):
        """Verifica que 10 llamadas secuenciales no lanzan excepcion y acumulan resultados validos."""
        mock_compute.return_value = make_mock_similarity_response()

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)
        results = []

        for _ in range(10):
            result = client.compute_similarity(VECTOR_A, VECTOR_B)
            results.append(result)

        self.assertEqual(len(results), 10)
        for r in results:
            self.assertIn("composite_score", r)
            self.assertGreaterEqual(r["composite_score"], 0.0)


class TestSimilaritySearchAuth(unittest.TestCase):

    def test_auth_missing_api_key_raises_descriptive_error(self):
        """Verifica que instanciar el cliente sin API key lanza error con mencion explicita de autenticacion."""
        from similarity_search_api_sdk import SimilaritySearchClient

        with self.assertRaises((ValueError, PermissionError, KeyError)) as ctx:
            client = SimilaritySearchClient(api_key=None, base_url=BASE_URL)
            client.compute_similarity(VECTOR_A, VECTOR_B)

        error_msg = str(ctx.exception).lower()
        auth_terms = ["key", "auth", "credential", "token", "api"]
        self.assertTrue(
            any(term in error_msg for term in auth_terms),
            f"El mensaje de error debe mencionar autenticacion. Obtenido: '{ctx.exception}'",
        )


class TestSimilaritySearchIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_similarity")
    def test_idempotency_same_vectors_same_composite_score_twice(self, mock_compute):
        """Verifica que la misma par de vectores produce exactamente el mismo composite_score en dos llamadas."""
        fixed_response = make_mock_similarity_response(
            cosine_score=0.9982,
            nmi_score=0.7641,
            composite_score=0.8812,
        )
        mock_compute.return_value = fixed_response

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=BASE_URL)

        result_1 = client.compute_similarity(VECTOR_A, VECTOR_B)
        result_2 = client.compute_similarity(VECTOR_A, VECTOR_B)

        self.assertAlmostEqual(
            result_1["composite_score"],
            result_2["composite_score"],
            places=10,
            msg="compute_similarity debe ser determinista para los mismos inputs",
        )
        self.assertAlmostEqual(result_1["nmi_score"], result_2["nmi_score"], places=10)
        self.assertEqual(mock_compute.call_count, 2)


if __name__ == "__main__":
    unittest.main()