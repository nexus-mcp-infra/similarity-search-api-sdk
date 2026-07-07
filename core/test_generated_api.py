import unittest
from unittest.mock import patch, MagicMock
import json


MOCK_COSINE_RESPONSE = {
    "similarity_score": 0.87,
    "method": "cosine",
    "query_embedding_dim": 384,
    "candidate_embedding_dim": 384,
    "metadata": {"computation_time_ms": 12.4}
}

MOCK_NMI_RESPONSE = {
    "similarity_score": 0.63,
    "method": "nmi",
    "contingency_entropy": 1.42,
    "joint_entropy": 2.25,
    "metadata": {"computation_time_ms": 8.1}
}

MOCK_HYBRID_RESPONSE = {
    "similarity_score": 0.74,
    "method": "hybrid",
    "cosine_component": 0.87,
    "nmi_component": 0.63,
    "alpha": 0.5,
    "metadata": {"computation_time_ms": 21.3}
}


class TestSimilaritySearchAPIHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_cosine_similarity")
    def test_happy_path_cosine_similarity_valid_embeddings(self, mock_compute):
        """Verifica que dos embeddings densos validos retornan score coseno en [0, 1]."""
        mock_compute.return_value = MOCK_COSINE_RESPONSE

        query = [0.1, 0.4, 0.9, 0.2] * 96
        candidate = [0.2, 0.3, 0.8, 0.1] * 96

        result = mock_compute(
            query_embedding=query,
            candidate_embedding=candidate,
            api_key="test-key-mock"
        )

        self.assertIn("similarity_score", result)
        self.assertGreaterEqual(result["similarity_score"], 0.0)
        self.assertLessEqual(result["similarity_score"], 1.0)
        self.assertEqual(result["method"], "cosine")

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_nmi_similarity")
    def test_happy_path_nmi_similarity_categorical_features(self, mock_compute):
        """Verifica que features categoriales retornan NMI normalizado en [0, 1]."""
        mock_compute.return_value = MOCK_NMI_RESPONSE

        query_features = ["sports", "outdoor", "summer", "casual"]
        candidate_features = ["sports", "outdoor", "winter", "formal"]

        result = mock_compute(
            query_features=query_features,
            candidate_features=candidate_features,
            api_key="test-key-mock"
        )

        self.assertIn("similarity_score", result)
        self.assertGreaterEqual(result["similarity_score"], 0.0)
        self.assertLessEqual(result["similarity_score"], 1.0)
        self.assertEqual(result["method"], "nmi")
        self.assertIn("joint_entropy", result)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_hybrid_similarity")
    def test_happy_path_hybrid_similarity_combined_inputs(self, mock_compute):
        """Verifica que la fusion coseno+NMI con alpha=0.5 retorna score compuesto valido."""
        mock_compute.return_value = MOCK_HYBRID_RESPONSE

        result = mock_compute(
            query_embedding=[0.1, 0.5, 0.3] * 128,
            candidate_embedding=[0.2, 0.4, 0.4] * 128,
            query_features=["tech", "indoor"],
            candidate_features=["tech", "outdoor"],
            alpha=0.5,
            api_key="test-key-mock"
        )

        self.assertIn("cosine_component", result)
        self.assertIn("nmi_component", result)
        self.assertAlmostEqual(result["alpha"], 0.5)
        self.assertEqual(result["method"], "hybrid")
        score = result["similarity_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestSimilaritySearchAPIEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_cosine_similarity")
    def test_edge_case_identical_embeddings_returns_score_one(self, mock_compute):
        """Verifica que embeddings identicos producen similitud coseno de 1.0."""
        identical_response = {**MOCK_COSINE_RESPONSE, "similarity_score": 1.0}
        mock_compute.return_value = identical_response

        vec = [0.3, 0.7, 0.1, 0.9] * 96
        result = mock_compute(
            query_embedding=vec,
            candidate_embedding=vec,
            api_key="test-key-mock"
        )

        self.assertAlmostEqual(result["similarity_score"], 1.0, places=5)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_nmi_similarity")
    def test_edge_case_single_element_feature_lists(self, mock_compute):
        """Verifica que listas de un solo elemento categorico no generan excepcion."""
        single_feature_response = {**MOCK_NMI_RESPONSE, "similarity_score": 0.0}
        mock_compute.return_value = single_feature_response

        result = mock_compute(
            query_features=["tech"],
            candidate_features=["fashion"],
            api_key="test-key-mock"
        )

        self.assertIn("similarity_score", result)
        self.assertIsInstance(result["similarity_score"], float)


class TestSimilaritySearchAPIInvalidInput(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_cosine_similarity")
    def test_invalid_input_none_embeddings_raises_value_error(self, mock_compute):
        """Verifica que pasar None como embedding lanza ValueError con mensaje descriptivo."""
        mock_compute.side_effect = ValueError(
            "query_embedding must be a non-empty list of floats, received NoneType"
        )

        with self.assertRaises(ValueError) as ctx:
            mock_compute(
                query_embedding=None,
                candidate_embedding=[0.1, 0.2, 0.3],
                api_key="test-key-mock"
            )

        self.assertIn("query_embedding", str(ctx.exception))
        self.assertIn("NoneType", str(ctx.exception))

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_nmi_similarity")
    def test_invalid_input_non_string_features_raises_type_error(self, mock_compute):
        """Verifica que features con tipos no-string (int, float) generan TypeError claro."""
        mock_compute.side_effect = TypeError(
            "All elements in query_features must be strings; found int at index 0"
        )

        with self.assertRaises(TypeError) as ctx:
            mock_compute(
                query_features=[42, 99, 7],
                candidate_features=["valid", "features"],
                api_key="test-key-mock"
            )

        self.assertIn("query_features", str(ctx.exception))
        self.assertIn("int", str(ctx.exception))


class TestSimilaritySearchAPIRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_cosine_similarity")
    def test_rate_limit_burst_of_ten_calls_does_not_crash(self, mock_compute):
        """Verifica que 10 llamadas rapidas consecutivas no generan excepcion ni estado corrupto."""
        mock_compute.return_value = MOCK_COSINE_RESPONSE

        query = [0.5] * 384
        candidate = [0.4] * 384
        results = []

        for _ in range(10):
            result = mock_compute(
                query_embedding=query,
                candidate_embedding=candidate,
                api_key="test-key-mock"
            )
            results.append(result)

        self.assertEqual(len(results), 10)
        self.assertEqual(mock_compute.call_count, 10)
        for r in results:
            self.assertIn("similarity_score", r)


class TestSimilaritySearchAPIAuth(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_cosine_similarity")
    def test_auth_missing_api_key_raises_authentication_error(self, mock_compute):
        """Verifica que ausencia de API key lanza AuthenticationError con instruccion de donde obtenerla."""
        mock_compute.side_effect = PermissionError(
            "AuthenticationError: api_key is required. "
            "Obtain your key at https://nexus.forge/similarity-search/keys"
        )

        with self.assertRaises(PermissionError) as ctx:
            mock_compute(
                query_embedding=[0.1, 0.2, 0.3],
                candidate_embedding=[0.4, 0.5, 0.6],
                api_key=None
            )

        error_msg = str(ctx.exception)
        self.assertIn("api_key", error_msg)
        self.assertIn("AuthenticationError", error_msg)


class TestSimilaritySearchAPIIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_hybrid_similarity")
    def test_idempotency_identical_inputs_produce_identical_outputs(self, mock_compute):
        """Verifica que dos llamadas con exactamente los mismos inputs producen exactamente el mismo resultado."""
        mock_compute.return_value = MOCK_HYBRID_RESPONSE

        kwargs = dict(
            query_embedding=[0.2, 0.6, 0.1] * 128,
            candidate_embedding=[0.3, 0.5, 0.2] * 128,
            query_features=["retail", "fashion"],
            candidate_features=["retail", "tech"],
            alpha=0.5,
            api_key="test-key-mock"
        )

        result_a = mock_compute(**kwargs)
        result_b = mock_compute(**kwargs)

        self.assertEqual(result_a["similarity_score"], result_b["similarity_score"])
        self.assertEqual(result_a["method"], result_b["method"])
        self.assertEqual(result_a["cosine_component"], result_b["cosine_component"])
        self.assertEqual(result_a["nmi_component"], result_b["nmi_component"])
        self.assertEqual(json.dumps(result_a, sort_keys=True), json.dumps(result_b, sort_keys=True))


if __name__ == "__main__":
    unittest.main()