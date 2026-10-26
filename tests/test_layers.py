"""Pruebas de embeddings, capa lineal y LayerNorm."""

import unittest

import numpy as np

from numpy_llm.layers import Embeddings, LayerNorm, Linear


class TestLayers(unittest.TestCase):
    def test_embeddings_shapes(self):
        rng = np.random.default_rng(0)
        emb = Embeddings(vocab_size=10, context_length=8, d_model=6, rng=rng)
        ids = np.array([[1, 2, 3], [4, 5, 6]])
        out = emb.forward(ids)
        self.assertEqual(out.shape, (2, 3, 6))
        # Indexar equivale a seleccionar filas + posición.
        expected = emb.token_embedding[ids] + emb.position_embedding[:3]
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_linear_matches_known_calculation(self):
        rng = np.random.default_rng(0)
        layer = Linear(2, 2, rng)
        layer.weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        layer.bias = np.array([10.0, 20.0], dtype=np.float32)
        x = np.array([[1.0, 1.0]])
        result = layer.forward(x)
        # [1,1] @ [[1,2],[3,4]] = [4,6]; + [10,20] = [14,26].
        np.testing.assert_allclose(result, [[14.0, 26.0]], atol=1e-6)
        # Dimensiones iniciales arbitrarias: [B, T, D].
        batch = layer.forward(np.ones((2, 3, 2)))
        self.assertEqual(batch.shape, (2, 3, 2))

    def test_layernorm_matches_direct_reference(self):
        ln = LayerNorm(4, eps=1e-5)
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5)
        result = ln.forward(x)
        np.testing.assert_allclose(result, expected, atol=1e-5)
        # Media ~0 y varianza ~1 por token.
        np.testing.assert_allclose(result.mean(axis=-1), [0.0], atol=1e-5)

    def test_layernorm_finite_with_constant_input(self):
        ln = LayerNorm(4)
        result = ln.forward(np.full((2, 3, 4), 5.0))
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, np.zeros((2, 3, 4)), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
