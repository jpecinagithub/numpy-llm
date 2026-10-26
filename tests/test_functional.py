"""Pruebas de las funciones numéricas básicas."""

import math
import unittest

import numpy as np

from numpy_llm.data import build_examples
from numpy_llm.functional import cross_entropy_loss, log_softmax, relu, softmax


class TestFunctional(unittest.TestCase):
    def test_relu(self):
        result = relu(np.array([-2.0, 0.0, 3.0]))
        np.testing.assert_array_equal(result, np.array([0.0, 0.0, 3.0]))

    def test_softmax_sums_to_one_on_axis(self):
        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        result = softmax(x, axis=-1)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones(2), atol=1e-6)
        # Eje 0: cada columna suma 1.
        result0 = softmax(x, axis=0)
        np.testing.assert_allclose(result0.sum(axis=0), np.ones(3), atol=1e-6)

    def test_softmax_handles_large_logits_without_overflow(self):
        x = np.array([1000.0, 1001.0, 999.0])
        result = softmax(x)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_log_softmax_matches_log_of_softmax(self):
        x = np.array([[1.0, 2.0, -1.0]])
        np.testing.assert_allclose(log_softmax(x), np.log(softmax(x)), atol=1e-6)

    def test_uniform_loss_equals_log_vocab(self):
        vocab = 7
        logits = np.zeros((2, 3, vocab))
        targets = np.zeros((2, 3), dtype=np.int64)
        loss = cross_entropy_loss(logits, targets)
        self.assertAlmostEqual(loss, math.log(vocab), places=6)

    def test_small_manual_loss(self):
        # Un solo ejemplo: logits [0, log(3)] -> probs [1/4, 3/4].
        logits = np.array([[[0.0, math.log(3.0)]]])
        targets = np.array([[1]])
        loss = cross_entropy_loss(logits, targets)
        self.assertAlmostEqual(loss, -math.log(0.75), places=6)
        targets0 = np.array([[0]])
        loss0 = cross_entropy_loss(logits, targets0)
        self.assertAlmostEqual(loss0, -math.log(0.25), places=6)

    def test_loss_validates_shapes_and_targets(self):
        with self.assertRaises(ValueError):
            cross_entropy_loss(np.zeros((1, 2, 3)), np.zeros((1, 3), dtype=np.int64))
        with self.assertRaises(ValueError):
            cross_entropy_loss(np.zeros((1, 2, 3)), np.array([[0, 3]]))


class TestData(unittest.TestCase):
    def test_shift_is_correct(self):
        tokens = [10, 20, 30, 40, 50]
        x, y = build_examples(tokens, sequence_length=4)
        self.assertEqual(x.shape, (1, 4))
        self.assertEqual(y.shape, (1, 4))
        np.testing.assert_array_equal(x[0], [10, 20, 30, 40])
        np.testing.assert_array_equal(y[0], [20, 30, 40, 50])

    def test_multiple_windows_and_shapes(self):
        tokens = list(range(10))
        x, y = build_examples(tokens, sequence_length=3)
        self.assertEqual(x.shape, (7, 3))
        self.assertEqual(y.shape, (7, 3))
        np.testing.assert_array_equal(x[1], [1, 2, 3])
        np.testing.assert_array_equal(y[1], [2, 3, 4])

    def test_insufficient_corpus_raises(self):
        with self.assertRaises(ValueError):
            build_examples([], sequence_length=4)
        with self.assertRaises(ValueError):
            build_examples([1, 2, 3], sequence_length=4)


if __name__ == "__main__":
    unittest.main()
