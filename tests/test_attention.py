"""Pruebas de la atención causal multicabeza."""

import unittest

import numpy as np

from numpy_llm.attention import CausalMultiHeadAttention


class TestAttention(unittest.TestCase):
    def test_shapes_with_multiple_heads(self):
        rng = np.random.default_rng(1)
        attn = CausalMultiHeadAttention(d_model=8, n_heads=4, rng=rng)
        x = rng.normal(size=(2, 5, 8)).astype(np.float32)
        out, weights = attn.forward(x, return_attention_weights=True)
        self.assertEqual(out.shape, (2, 5, 8))
        self.assertEqual(weights.shape, (2, 4, 5, 5))

    def test_future_probabilities_are_zero_and_rows_sum_to_one(self):
        rng = np.random.default_rng(2)
        attn = CausalMultiHeadAttention(d_model=8, n_heads=2, rng=rng)
        x = rng.normal(size=(1, 4, 8)).astype(np.float32)
        _, weights = attn.forward(x, return_attention_weights=True)
        # j > i debe ser exactamente 0.
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertEqual(weights[0, 0, i, j], 0.0)
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones((1, 2, 4)), atol=1e-6)

    def test_length_one_works(self):
        rng = np.random.default_rng(3)
        attn = CausalMultiHeadAttention(d_model=4, n_heads=2, rng=rng)
        x = rng.normal(size=(1, 1, 4)).astype(np.float32)
        out, weights = attn.forward(x, return_attention_weights=True)
        self.assertEqual(out.shape, (1, 1, 4))
        np.testing.assert_allclose(weights, np.ones((1, 2, 1, 1)), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
