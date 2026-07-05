import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import json


MOCK_API_BASE = "http://localhost:8000"
MOCK_API_KEY = "test-key-nexus-forge-001"


def _make_embedding(dim: int = 128, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _mock_similarity_response(score: float = 0.847) -> dict:
    return {
        "composite_score": score,
        "cosine_similarity": 0.912,
        "nmi_score": 0.743,
        "calibrated": True,
        "dimensions": 128,
    }


class TestSimilaritySearchAPIHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_pairwise_similarity")
    def test_happy_path_valid_embeddings_return_composite_score(self, mock_compute):
        """Verifica que dos embeddings normalizados válidos devuelven un score compuesto en [0, 1]."""
        mock_compute.return_value = _mock_similarity_response(score=0.847)

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=MOCK_API_BASE)

        vec_a = _make_embedding(dim=128, seed=1)
        vec_b = _make_embedding(dim=128, seed=2)

        result = client.compute_pairwise_similarity(embedding_a=vec_a, embedding_b=vec_b)

        self.assertIn("composite_score", result)
        self.assertGreaterEqual(result["composite_score"], 0.0)
        self.assertLessEqual(result["composite_score"], 1.0)
        mock_compute.assert_called_once_with(embedding_a=vec_a, embedding_b=vec_b)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_pairwise_similarity")
    def test_happy_path_identical_embeddings_score_near_one(self, mock_compute):
        """Verifica que embeddings idénticos producen un composite_score >= 0.99."""
        mock_compute.return_value = _mock_similarity_response(score=1.0)

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=MOCK_API_BASE)

        vec = _make_embedding(dim=128, seed=42)
        result = client.compute_pairwise_similarity(embedding_a=vec, embedding_b=vec)

        self.assertGreaterEqual(result["composite_score"], 0.99)
        self.assertTrue(result["calibrated"])


class TestSimilaritySearchAPIEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_pairwise_similarity")
    def test_edge_case_high_dimensional_embeddings(self, mock_compute):
        """Verifica que embeddings de 1536 dimensiones (OpenAI-style) no crashean ni truncan."""
        mock_compute.return_value = _mock_similarity_response(score=0.731)
        mock_compute.return_value["dimensions"] = 1536

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=MOCK_API_BASE)

        vec_a = _make_embedding(dim=1536, seed=10)
        vec_b = _make_embedding(dim=1536, seed=11)

        result = client.compute_pairwise_similarity(embedding_a=vec_a, embedding_b=vec_b)

        self.assertEqual(result["dimensions"], 1536)
        self.assertIn("composite_score", result)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_pairwise_similarity")
    def test_edge_case_orthogonal_embeddings_score_near_zero(self, mock_compute):
        """Verifica que embeddings ortogonales (coseno=0) producen composite_score cercano a 0."""
        mock_compute.return_value = _mock_similarity_response(score=0.03)

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=MOCK_API_BASE)

        vec_a = [1.0] + [0.0] * 127
        vec_b = [0.0, 1.0] + [0.0] * 126

        result = client.compute_pairwise_similarity(embedding_a=vec_a, embedding_b=vec_b)

        self.assertLessEqual(result["composite_score"], 0.15)


class TestSimilaritySearchAPIInvalidInput(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_pairwise_similarity")
    def test_invalid_input_none_embedding_raises_value_error(self, mock_compute):
        """Verifica que pasar None como embedding_a lanza ValueError con mensaje descriptivo."""
        mock_compute.side_effect = ValueError(
            "embedding_a must be a non-empty list of floats; received NoneType"
        )

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=MOCK_API_BASE)

        with self.assertRaises(ValueError) as ctx:
            client.compute_pairwise_similarity(embedding_a=None, embedding_b=_make_embedding())

        self.assertIn("embedding_a", str(ctx.exception))
        self.assertIn("NoneType", str(ctx.exception))

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_pairwise_similarity")
    def test_invalid_input_mismatched_dimensions_raises_value_error(self, mock_compute):
        """Verifica que embeddings de dimensiones distintas lanzan ValueError antes de la llamada HTTP."""
        mock_compute.side_effect = ValueError(
            "Dimension mismatch: embedding_a has 128 dims, embedding_b has 256 dims"
        )

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=MOCK_API_BASE)

        vec_a = _make_embedding(dim=128, seed=1)
        vec_b = _make_embedding(dim=256, seed=2)

        with self.assertRaises(ValueError) as ctx:
            client.compute_pairwise_similarity(embedding_a=vec_a, embedding_b=vec_b)

        self.assertIn("mismatch", str(ctx.exception).lower())


class TestSimilaritySearchAPIRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_pairwise_similarity")
    def test_rate_limit_50_sequential_calls_no_crash(self, mock_compute):
        """Verifica que 50 llamadas secuenciales no producen excepción ni estado corrupto."""
        mock_compute.return_value = _mock_similarity_response(score=0.65)

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=MOCK_API_BASE)

        vec_a = _make_embedding(dim=128, seed=5)
        vec_b = _make_embedding(dim=128, seed=6)

        results = []
        for _ in range(50):
            r = client.compute_pairwise_similarity(embedding_a=vec_a, embedding_b=vec_b)
            results.append(r["composite_score"])

        self.assertEqual(len(results), 50)
        self.assertTrue(all(isinstance(s, float) for s in results))
        self.assertEqual(mock_compute.call_count, 50)


class TestSimilaritySearchAPIAuth(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_pairwise_similarity")
    def test_auth_missing_api_key_raises_authentication_error(self, mock_compute):
        """Verifica que la ausencia de API key lanza un error de autenticación con mensaje claro."""
        mock_compute.side_effect = PermissionError(
            "Authentication failed: X-API-Key header is missing or invalid. "
            "Obtain a key at https://nexus.forge/similarity-search/keys"
        )

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key="", base_url=MOCK_API_BASE)

        vec_a = _make_embedding(dim=128, seed=7)
        vec_b = _make_embedding(dim=128, seed=8)

        with self.assertRaises(PermissionError) as ctx:
            client.compute_pairwise_similarity(embedding_a=vec_a, embedding_b=vec_b)

        error_msg = str(ctx.exception)
        self.assertIn("Authentication failed", error_msg)
        self.assertIn("X-API-Key", error_msg)


class TestSimilaritySearchAPIIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.compute_pairwise_similarity")
    def test_idempotency_same_embeddings_twice_returns_identical_scores(self, mock_compute):
        """Verifica que la misma pareja de embeddings produce exactamente el mismo composite_score en dos llamadas."""
        fixed_response = _mock_similarity_response(score=0.773)
        mock_compute.return_value = fixed_response

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY, base_url=MOCK_API_BASE)

        vec_a = _make_embedding(dim=128, seed=99)
        vec_b = _make_embedding(dim=128, seed=100)

        result_first = client.compute_pairwise_similarity(embedding_a=vec_a, embedding_b=vec_b)
        result_second = client.compute_pairwise_similarity(embedding_a=vec_a, embedding_b=vec_b)

        self.assertEqual(result_first["composite_score"], result_second["composite_score"])
        self.assertEqual(result_first["nmi_score"], result_second["nmi_score"])
        self.assertEqual(result_first["cosine_similarity"], result_second["cosine_similarity"])
        self.assertEqual(mock_compute.call_count, 2)


if __name__ == "__main__":
    unittest.main()