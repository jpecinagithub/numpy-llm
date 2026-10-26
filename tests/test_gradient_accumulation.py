"""Acumulación de gradientes equivalente a un lote grande."""

import unittest

import numpy as np

from numpy_llm.config import ModelConfig
from numpy_llm.model import TransformerLanguageModel
from numpy_llm.training import compute_batch_gradients


def _model(seed=42):
    config = ModelConfig(
        context_length=8, d_model=4, n_heads=2, n_layers=1, d_ff=8,
        seed=seed, dtype="float64",
    ).with_vocab_size(5)
    return TransformerLanguageModel(config)


class TestGradientAccumulation(unittest.TestCase):
    def test_split_batch_equals_large_batch(self):
        big_x = np.array([[1, 2, 3], [4, 0, 1], [2, 3, 4], [0, 1, 2]])
        big_y = np.array([[2, 3, 4], [0, 1, 2], [3, 4, 0], [1, 2, 3]])
        model_big = _model()
        model_big.zero_grad()
        loss_big = compute_batch_gradients(model_big, [(big_x, big_y)])
        model_split = _model()
        model_split.zero_grad()
        loss_split = compute_batch_gradients(
            model_split, [(big_x[:2], big_y[:2]), (big_x[2:], big_y[2:])]
        )
        self.assertAlmostEqual(loss_big, loss_split, places=10)
        for (_, g_big), (_, g_split) in zip(model_big.named_gradients(), model_split.named_gradients()):
            np.testing.assert_allclose(g_big, g_split, rtol=1e-10, atol=1e-12)

    def test_unequal_microbatch_sizes_weighted_by_targets(self):
        # Microlotes de 2 y 1 ejemplos deben ponderarse por M_i/M_total,
        # no por 1/n_microlotes.
        big_x = np.array([[1, 2, 3], [4, 0, 1], [2, 3, 4]])
        big_y = np.array([[2, 3, 4], [0, 1, 2], [3, 4, 0]])
        model_big = _model()
        model_big.zero_grad()
        loss_big = compute_batch_gradients(model_big, [(big_x, big_y)])
        model_split = _model()
        model_split.zero_grad()
        loss_split = compute_batch_gradients(
            model_split, [(big_x[:2], big_y[:2]), (big_x[2:], big_y[2:])]
        )
        self.assertAlmostEqual(loss_big, loss_split, places=10)
        for (_, g_big), (_, g_split) in zip(model_big.named_gradients(), model_split.named_gradients()):
            np.testing.assert_allclose(g_big, g_split, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
