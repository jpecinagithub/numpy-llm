"""Clipping por norma global."""

import unittest

import numpy as np

from numpy_llm.optimizers import clip_gradients, global_grad_norm


class TestGradientClipping(unittest.TestCase):
    def test_below_limit_unchanged(self):
        grads = [("a", np.array([0.3, 0.4])), ("b", np.array([0.0]))]
        before, after, clipped = clip_gradients(grads, 1.0)
        self.assertAlmostEqual(before, 0.5)
        self.assertFalse(clipped)
        self.assertAlmostEqual(after, 0.5)
        np.testing.assert_allclose(grads[0][1], [0.3, 0.4])

    def test_above_limit_scales_all_by_same_factor(self):
        # Cada gradiente por separado está por debajo de 1.0, pero la
        # norma global (sqrt(0.36+0.64)=1.0... usar valores que superen)
        # debe aplicar clipping global, no por matriz.
        g1 = np.array([0.9, 0.0])
        g2 = np.array([0.9, 0.0])
        grads = [("a", g1), ("b", g2)]
        before, after, clipped = clip_gradients(grads, 1.0)
        self.assertAlmostEqual(before, np.sqrt(1.62), places=10)
        self.assertTrue(clipped)
        self.assertAlmostEqual(after, 1.0, places=6)
        # Mismo factor para ambos (norma global, no por matriz).
        np.testing.assert_allclose(g1, g2, rtol=0, atol=1e-12)
        np.testing.assert_allclose(g1, [0.9 / before, 0.0], rtol=1e-9, atol=1e-12)

    def test_zero_norm_unchanged(self):
        grads = [("a", np.zeros(3))]
        before, after, clipped = clip_gradients(grads, 1.0)
        self.assertEqual(before, 0.0)
        self.assertFalse(clipped)

    def test_nonfinite_raises_and_is_not_silently_zeroed(self):
        grads = [("a", np.array([1.0, np.nan]))]
        with self.assertRaises(ValueError):
            clip_gradients(grads, 1.0)
        grads = [("a", np.array([1.0, np.inf]))]
        with self.assertRaises(ValueError):
            clip_gradients(grads, 1.0)

    def test_global_norm_in_float64(self):
        grads = [("a", np.array([3.0], dtype=np.float32)), ("b", np.array([4.0], dtype=np.float32))]
        self.assertAlmostEqual(global_grad_norm(grads), 5.0)


if __name__ == "__main__":
    unittest.main()
