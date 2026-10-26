"""Backward del modelo completo y del bloque Transformer."""

import unittest

import numpy as np

from numpy_llm.config import ModelConfig
from numpy_llm.functional import cross_entropy_with_grad
from numpy_llm.gradcheck import central_difference, check_value
from numpy_llm.model import TransformerLanguageModel


def _tiny_model(dtype=np.float64, seed=42):
    config = ModelConfig(
        context_length=8, d_model=4, n_heads=2, n_layers=1, d_ff=8,
        seed=seed, dtype=str(np.dtype(dtype)),
    ).with_vocab_size(5)
    return TransformerLanguageModel(config)


def _assert_close(test, analytic, numeric, msg):
    result = check_value(float(analytic), float(numeric))
    test.assertTrue(result["passed"], f"{msg}: analytic={analytic}, numeric={numeric}, abs_err={result['abs_error']}")


class TestModelBackward(unittest.TestCase):
    def test_named_gradients_match_parameters(self):
        model = _tiny_model()
        params = model.named_parameters()
        grads = model.named_gradients()
        self.assertEqual([n for n, _ in params], [n for n, _ in grads])
        for (name, param), (_, grad) in zip(params, grads):
            self.assertEqual(param.shape, grad.shape, name)
            self.assertEqual(param.dtype, grad.dtype, name)
            self.assertIsNot(param, grad, name)
            np.testing.assert_allclose(grad, np.zeros_like(grad), atol=0)

    def test_forward_and_backward_do_not_change_weights(self):
        model = _tiny_model()
        before = {n: p.copy() for n, p in model.named_parameters()}
        x = np.array([[1, 2, 3], [4, 0, 1]])
        y = np.array([[2, 3, 4], [0, 1, 2]])
        logits = model.forward(x, retain_cache=True)
        loss, dlogits = cross_entropy_with_grad(logits, y)
        model.backward(dlogits)
        for name, param in model.named_parameters():
            np.testing.assert_array_equal(param, before[name])

    def test_zero_grad_and_accumulation(self):
        model = _tiny_model()
        x = np.array([[1, 2, 3]])
        y = np.array([[2, 3, 4]])
        logits = model.forward(x, retain_cache=True)
        _, d1 = cross_entropy_with_grad(logits, y)
        model.backward(d1)
        g_after_first = {n: g.copy() for n, g in model.named_gradients()}
        logits = model.forward(x, retain_cache=True)
        _, d2 = cross_entropy_with_grad(logits, y)
        model.backward(d2)
        for name, grad in model.named_gradients():
            np.testing.assert_allclose(grad, 2 * g_after_first[name], rtol=1e-12, atol=1e-12)
        model.zero_grad()
        for _, grad in model.named_gradients():
            np.testing.assert_allclose(grad, 0, atol=0)

    def test_accumulated_equals_combined_gradient(self):
        # Dos backward acumulados == un backward con la suma de dlogits.
        model_a = _tiny_model()
        model_b = _tiny_model()
        x = np.array([[1, 2, 3], [4, 0, 1]])
        y1 = np.array([[2, 3, 4], [0, 1, 2]])
        y2 = np.array([[3, 4, 0], [1, 2, 3]])
        logits = model_a.forward(x, retain_cache=True)
        _, d1 = cross_entropy_with_grad(logits, y1)
        model_a.backward(d1)
        logits = model_a.forward(x, retain_cache=True)
        _, d2 = cross_entropy_with_grad(logits, y2)
        model_a.backward(d2)
        logits = model_b.forward(x, retain_cache=True)
        _, d1b = cross_entropy_with_grad(logits, y1)
        _, d2b = cross_entropy_with_grad(logits, y2)
        model_b.backward(d1b + d2b)
        for (_, ga), (_, gb) in zip(model_a.named_gradients(), model_b.named_gradients()):
            np.testing.assert_allclose(ga, gb, rtol=1e-10, atol=1e-12)

    def test_block_backward_matches_finite_differences(self):
        model = _tiny_model()
        block = model.blocks[0]
        rng = np.random.default_rng(9)
        x = rng.normal(size=(2, 3, 4))
        u = rng.normal(size=(2, 3, 4))

        def scalar():
            return float(np.sum(block.forward(x) * u))

        block.forward(x, retain_cache=True)
        d_x = block.backward(u)
        self.assertEqual(d_x.shape, x.shape)
        # Muestra de cada grupo de parámetros del bloque.
        for prefix, layer in (
            ("ln1", block.ln1), ("ln2", block.ln2),
        ):
            _assert_close(self, layer.grad_gamma[(0,)], central_difference(scalar, layer.gamma, (0,)), f"{prefix}.gamma")
        for proj_name, proj in (
            ("q_proj", block.attention.q_proj), ("out_proj", block.attention.out_proj),
            ("fc1", block.feed_forward.fc1), ("fc2", block.feed_forward.fc2),
        ):
            _assert_close(self, proj.grad_weight[(0, 0)], central_difference(scalar, proj.weight, (0, 0)), f"{proj_name}.weight")
        x_copy = x.copy()
        for idx in [(0, 0, 0), (1, 2, 3)]:
            def scalar_x(x_arr=x_copy):
                return float(np.sum(block.forward(x_arr) * u))
            _assert_close(self, d_x[idx], central_difference(scalar_x, x_copy, idx), f"block dX{idx}")

    def test_inference_does_not_overwrite_training_cache(self):
        model = _tiny_model()
        x = np.array([[1, 2, 3]])
        y = np.array([[2, 3, 4]])
        logits = model.forward(x, retain_cache=True)
        _, dlogits = cross_entropy_with_grad(logits, y)
        model.forward(np.array([[4, 4, 4]]))  # inferencia intermedia
        model.backward(dlogits)  # debe seguir funcionando
        self.assertTrue(all(np.all(np.isfinite(g)) for _, g in model.named_gradients()))

    def test_double_backward_and_missing_forward_raise(self):
        model = _tiny_model()
        x = np.array([[1, 2, 3]])
        with self.assertRaises(ValueError):
            model.backward(np.ones((1, 3, 5)))
        logits = model.forward(x, retain_cache=True)
        _, dlogits = cross_entropy_with_grad(logits, np.array([[2, 3, 4]]))
        model.backward(dlogits)
        with self.assertRaises(ValueError):
            model.backward(dlogits)

    def test_to_dtype_float64(self):
        config = ModelConfig(
            context_length=8, d_model=4, n_heads=2, n_layers=1, d_ff=8, seed=42
        ).with_vocab_size(5)
        model = TransformerLanguageModel(config)
        self.assertEqual(str(model.dtype), "float32")
        model.to_dtype(np.float64)
        self.assertEqual(str(model.dtype), "float64")
        self.assertTrue(all(str(p.dtype) == "float64" for _, p in model.named_parameters()))
        self.assertTrue(all(str(g.dtype) == "float64" for _, g in model.named_gradients()))
        self.assertEqual([n for n, _ in model.named_parameters()], [n for n, _ in model.named_gradients()])

    def test_causal_prefix_invariance_preserved(self):
        # Prueba causal de la fase 1, conservada en la fase 2.
        model = _tiny_model(dtype=np.float32)
        x = np.array([[1, 2, 3, 4]])
        logits = model.forward(x)
        x2 = np.array([[1, 2, 0, 0]])
        logits2 = model.forward(x2)
        np.testing.assert_allclose(logits[0, :2], logits2[0, :2], rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
