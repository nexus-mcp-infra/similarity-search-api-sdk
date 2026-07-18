import unittest
from unittest.mock import patch, MagicMock
import json


MOCK_UNIFIED_SCORE_RESPONSE = {
    "unified_score": 0.7843,
    "cosine_similarity": 0.9121,
    "nmi_score": 0.6564,
    "feature_count": {"continuous": 3, "categorical": 2},
    "request_id": "req_abc123"
}

MOCK_BATCH_RESPONSE = {
    "results": [
        {"pair_index": 0, "unified_score": 0.812, "cosine_similarity": 0.901, "nmi_score": 0.723},
        {"pair_index": 1, "unified_score": 0.634, "cosine_similarity": 0.711, "nmi_score": 0.557}
    ],
    "batch_size": 2
}


class TestSimilaritySearchAPIHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_unified_score")
    def test_happy_path_unified_score_mixed_features(self, mock_compute):
        """Verifica que un input con features continuas y categoricas retorna unified_score en [0,1]."""
        mock_compute.return_value = MOCK_UNIFIED_SCORE_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test_key_mock")

        result = client.compute_unified_score(
            continuous_a=[0.1, 0.5, 0.9],
            continuous_b=[0.2, 0.4, 0.85],
            categorical_a=["red", "large"],
            categorical_b=["red", "medium"]
        )

        self.assertIn("unified_score", result)
        self.assertGreaterEqual(result["unified_score"], 0.0)
        self.assertLessEqual(result["unified_score"], 1.0)
        self.assertIn("cosine_similarity", result)
        self.assertIn("nmi_score", result)
        mock_compute.assert_called_once()

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_batch_unified_scores")
    def test_happy_path_batch_scoring_returns_per_pair_results(self, mock_batch):
        """Verifica que batch scoring retorna un resultado por cada par enviado."""
        mock_batch.return_value = MOCK_BATCH_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test_key_mock")

        pairs = [
            {"continuous_a": [0.3, 0.6], "continuous_b": [0.4, 0.5],
             "categorical_a": ["blue"], "categorical_b": ["blue"]},
            {"continuous_a": [0.9, 0.1], "continuous_b": [0.2, 0.8],
             "categorical_a": ["green"], "categorical_b": ["red"]}
        ]
        result = client.compute_batch_unified_scores(pairs=pairs)

        self.assertEqual(result["batch_size"], 2)
        self.assertEqual(len(result["results"]), 2)
        for item in result["results"]:
            self.assertIn("unified_score", item)
            self.assertGreaterEqual(item["unified_score"], 0.0)
            self.assertLessEqual(item["unified_score"], 1.0)


class TestSimilaritySearchAPIEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_unified_score")
    def test_edge_case_single_element_vectors_accepted(self, mock_compute):
        """Verifica que vectores de un solo elemento no provocan error y retornan score valido."""
        mock_compute.return_value = {
            "unified_score": 1.0,
            "cosine_similarity": 1.0,
            "nmi_score": 1.0,
            "feature_count": {"continuous": 1, "categorical": 0},
            "request_id": "req_single_001"
        }

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test_key_mock")

        result = client.compute_unified_score(
            continuous_a=[0.5],
            continuous_b=[0.5],
            categorical_a=[],
            categorical_b=[]
        )

        self.assertEqual(result["unified_score"], 1.0)
        self.assertEqual(result["feature_count"]["continuous"], 1)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_unified_score")
    def test_edge_case_high_dimensionality_continuous_vectors(self, mock_compute):
        """Verifica que vectores de 512 dimensiones no exceden limites del API y retornan respuesta."""
        mock_compute.return_value = {
            "unified_score": 0.9934,
            "cosine_similarity": 0.9934,
            "nmi_score": None,
            "feature_count": {"continuous": 512, "categorical": 0},
            "request_id": "req_highdim_002"
        }

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test_key_mock")

        vec = [float(i) / 512 for i in range(512)]
        result = client.compute_unified_score(
            continuous_a=vec,
            continuous_b=vec,
            categorical_a=[],
            categorical_b=[]
        )

        self.assertEqual(result["feature_count"]["continuous"], 512)
        self.assertGreaterEqual(result["unified_score"], 0.99)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_unified_score")
    def test_edge_case_none_categorical_features_raises_type_error(self, mock_compute):
        """Verifica que pasar None como categorical_a lanza TypeError antes de llegar al API."""
        mock_compute.side_effect = TypeError(
            "categorical_a must be a list of strings, got NoneType"
        )

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test_key_mock")

        with self.assertRaises(TypeError) as ctx:
            client.compute_unified_score(
                continuous_a=[0.1, 0.2],
                continuous_b=[0.3, 0.4],
                categorical_a=None,
                categorical_b=["cat"]
            )

        self.assertIn("NoneType", str(ctx.exception))


class TestSimilaritySearchAPIInvalidInput(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_unified_score")
    def test_invalid_input_mismatched_vector_lengths_raises_value_error(self, mock_compute):
        """Verifica que vectores continuos de distinta longitud generan ValueError con mensaje descriptivo."""
        mock_compute.side_effect = ValueError(
            "continuous_a and continuous_b must have the same length: got 3 vs 2"
        )

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test_key_mock")

        with self.assertRaises(ValueError) as ctx:
            client.compute_unified_score(
                continuous_a=[0.1, 0.2, 0.3],
                continuous_b=[0.4, 0.5],
                categorical_a=[],
                categorical_b=[]
            )

        self.assertIn("same length", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))
        self.assertIn("2", str(ctx.exception))

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_unified_score")
    def test_invalid_input_non_numeric_continuous_features_raises_type_error(self, mock_compute):
        """Verifica que strings dentro de continuous_a producen TypeError con nombre del parametro afectado."""
        mock_compute.side_effect = TypeError(
            "continuous_a elements must be float or int, found str at index 1"
        )

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test_key_mock")

        with self.assertRaises(TypeError) as ctx:
            client.compute_unified_score(
                continuous_a=[0.1, "not_a_number"],
                continuous_b=[0.2, 0.3],
                categorical_a=[],
                categorical_b=[]
            )

        self.assertIn("continuous_a", str(ctx.exception))
        self.assertIn("str", str(ctx.exception))


class TestSimilaritySearchAPIRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_unified_score")
    def test_rate_limit_burst_of_calls_does_not_crash_client(self, mock_compute):
        """Verifica que 20 llamadas rapidas consecutivas no generan excepcion no controlada en el cliente."""
        mock_compute.return_value = MOCK_UNIFIED_SCORE_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test_key_mock")

        errors = []
        for i in range(20):
            try:
                result = client.compute_unified_score(
                    continuous_a=[float(i), float(i + 1)],
                    continuous_b=[float(i + 2), float(i + 3)],
                    categorical_a=["x"],
                    categorical_b=["y"]
                )
                self.assertIn("unified_score", result)
            except Exception as exc:
                errors.append(exc)

        self.assertEqual(len(errors), 0, f"Unexpected errors during burst: {errors}")
        self.assertEqual(mock_compute.call_count, 20)


class TestSimilaritySearchAPIAuth(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_unified_score")
    def test_auth_missing_api_key_raises_authentication_error(self, mock_compute):
        """Verifica que un cliente sin API key genera AuthenticationError con mensaje que indica el header faltante."""
        mock_compute.side_effect = PermissionError(
            "AuthenticationError: missing or invalid X-API-Key header — "
            "obtain a key at https://nexus.forge/similarity-search/keys"
        )

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="")

        with self.assertRaises(PermissionError) as ctx:
            client.compute_unified_score(
                continuous_a=[0.1, 0.2],
                continuous_b=[0.3, 0.4],
                categorical_a=["a"],
                categorical_b=["b"]
            )

        self.assertIn("AuthenticationError", str(ctx.exception))
        self.assertIn("X-API-Key", str(ctx.exception))


class TestSimilaritySearchAPIIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_unified_score")
    def test_idempotency_identical_inputs_produce_identical_scores(self, mock_compute):
        """Verifica que dos llamadas identicas retornan exactamente el mismo unified_score y nmi_score."""
        mock_compute.return_value = MOCK_UNIFIED_SCORE_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="test_key_mock")

        kwargs = dict(
            continuous_a=[0.25, 0.50, 0.75],
            continuous_b=[0.30, 0.45, 0.80],
            categorical_a=["alpha", "beta"],
            categorical_b=["alpha", "gamma"]
        )

        result_first = client.compute_unified_score(**kwargs)
        result_second = client.compute_unified_score(**kwargs)

        self.assertEqual(result_first["unified_score"], result_second["unified_score"])
        self.assertEqual(result_first["nmi_score"], result_second["nmi_score"])
        self.assertEqual(result_first["cosine_similarity"], result_second["cosine_similarity"])
        self.assertEqual(mock_compute.call_count, 2)


if __name__ == "__main__":
    unittest.main()