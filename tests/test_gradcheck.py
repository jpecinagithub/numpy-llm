"""Comprobación numérica de gradientes (gradcheck)."""

import unittest

import numpy as np

from numpy_llm.config import ModelConfig
from numpy_llm.gradcheck import central_difference, check_value, gradcheck_model
from numpy_llm.model import TransformerLanguageModel


class TestGradcheckHelpers(unittest.TestCase):
    def test_central_difference_quadratic(self):
        x = np.array([2.0, -1.0])
        func = lambda: float(np.sum(x ** 2))
        for idx in [(0,), (1,)]:
            numeric = central_difference(func, x, idx)
            self.assertAlmostEqual(numeric, 2 * x[idx], places=8)
        np.testing.assert_allclose(x, [2.0, -1.0])  # restaurado

    def test_check_value_criterion(self):
        self.assertTrue(check_value(1.0, 1.0 + 1e-9)["passed"])
        self.assertFalse(check_value(1.0, 1.1)["passed"])
        # Gradiente casi nulo: el error absoluto manda, no el relativo.
        self.assertTrue(check_value(0.0, 8e-22)["passed"])


class TestGradcheckModel(unittest.TestCase):
    def test_full_tiny_model(self):
        config = ModelConfig(
            context_length=8, d_model=4, n_heads=2, n_layers=1, d_ff=8,
            seed=42, dtype="float64",
        ).with_vocab_size(5)
        model = TransformerLanguageModel(config)
        x = np.array([[1, 2, 3], [4, 0, 1]])
        y = np.array([[2, 3, 4], [0, 1, 2]])
        report = gradcheck_model(model, x, y, sample_per_tensor=None, seed=42)
        self.assertTrue(report["passed"], f"fallos: {report['failures'][:3]}")
        self.assertEqual(report["checked_components"], 257)
        self.assertLess(report["max_abs_error"], 1e-7)
        # Todos los grupos de parámetros están cubiertos.
        self.assertEqual(len(report["per_parameter"]), len(model.named_parameters()))
        for info in report["per_parameter"].values():
            self.assertTrue(info["passed"])

    def test_fast_mode_covers_all_groups(self):
        config = ModelConfig(
            context_length=8, d_model=4, n_heads=2, n_layers=1, d_ff=8,
            seed=42, dtype="float64",
        ).with_vocab_size(5)
        model = TransformerLanguageModel(config)
        x = np.array([[1, 2, 3], [4, 0, 1]])
        y = np.array([[2, 3, 4], [0, 1, 2]])
        report = gradcheck_model(model, x, y, sample_per_tensor=3, seed=42)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["per_parameter"]), len(model.named_parameters()))

    def test_float32_model_is_rejected(self):
        config = ModelConfig(
            context_length=8, d_model=4, n_heads=2, n_layers=1, d_ff=8, seed=42
        ).with_vocab_size(5)
        model = TransformerLanguageModel(config)
        with self.assertRaises(ValueError):
            gradcheck_model(model, np.array([[1, 2, 3]]), np.array([[2, 3, 4]]), sample_per_tensor=1)


if __name__ == "__main__":
    unittest.main()
