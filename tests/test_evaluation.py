"""Evaluación: pérdida ponderada, perplexity, baselines, no-efectos."""

import math
import unittest

import numpy as np

from numpy_llm.config import ModelConfig
from numpy_llm.evaluation import evaluate_model, uniform_baseline, unigram_baseline
from numpy_llm.model import TransformerLanguageModel


class TestBaselines(unittest.TestCase):
    def test_uniform_baseline(self):
        base = uniform_baseline(10)
        self.assertAlmostEqual(base["mean_loss"], math.log(10))
        self.assertAlmostEqual(base["perplexity"], 10.0)

    def test_unigram_baseline_uses_only_train_counts(self):
        train = np.array([0, 0, 1, 2])
        eval_tokens = np.array([0, 1])
        base = unigram_baseline(train, eval_tokens, vocab_size=3, alpha=1.0)
        # p(0) = (2+1)/(4+3) = 3/7; p(1) = (1+1)/7 = 2/7.
        expected = -0.5 * (math.log(3 / 7) + math.log(2 / 7))
        self.assertAlmostEqual(base["mean_loss"], expected)
        self.assertAlmostEqual(base["perplexity"], math.exp(expected))
        self.assertEqual(base["total_targets"], 2)


class TestEvaluateModel(unittest.TestCase):
    def _model(self):
        config = ModelConfig(
            context_length=16, d_model=8, n_heads=2, n_layers=1, d_ff=16, seed=42
        ).with_vocab_size(6)
        return TransformerLanguageModel(config)

    def test_metrics_are_consistent_and_weighted(self):
        model = self._model()
        tokens = np.array([0, 1, 2, 3, 4, 5] * 5, dtype=np.int64)
        result = evaluate_model(model, tokens, sequence_length=8, batch_size=3)
        # Ventanas: inicios 0, 8, 16 (16+8=24 <= 30-? rango(0, 22, 8)=0,8,16)
        self.assertEqual(result["evaluated_windows"], 3)
        self.assertEqual(result["total_targets"], 24)
        self.assertEqual(result["coverage"], "full")
        self.assertAlmostEqual(result["perplexity"], math.exp(result["mean_loss"]), places=6)
        self.assertTrue(0.0 <= result["token_accuracy"] <= 1.0)

    def test_subset_is_not_presented_as_full(self):
        model = self._model()
        tokens = np.array([0, 1, 2, 3, 4, 5] * 5, dtype=np.int64)
        result = evaluate_model(model, tokens, sequence_length=8, batch_size=4, limit_windows=1)
        self.assertEqual(result["coverage"], "subset")
        self.assertEqual(result["evaluated_windows"], 1)
        self.assertEqual(result["total_targets"], 8)

    def test_evaluation_does_not_change_weights_or_gradients(self):
        model = self._model()
        before = {n: p.copy() for n, p in model.named_parameters()}
        tokens = np.array([0, 1, 2, 3, 4, 5] * 5, dtype=np.int64)
        evaluate_model(model, tokens, sequence_length=8, batch_size=2)
        for name, param in model.named_parameters():
            np.testing.assert_array_equal(param, before[name])
        for _, grad in model.named_gradients():
            np.testing.assert_allclose(grad, 0, atol=0)

    def test_deterministic(self):
        model = self._model()
        tokens = np.array([0, 1, 2, 3, 4, 5] * 5, dtype=np.int64)
        r1 = evaluate_model(model, tokens, sequence_length=8, batch_size=2)
        r2 = evaluate_model(model, tokens, sequence_length=8, batch_size=2)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
