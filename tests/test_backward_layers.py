"""Backward de capas sencillas: Linear, ReLU/MLP, LayerNorm, Embeddings, Softmax."""

import unittest

import numpy as np

from numpy_llm.functional import softmax, softmax_backward
from numpy_llm.gradcheck import central_difference, check_value
from numpy_llm.layers import Embeddings, LayerNorm, Linear
from numpy_llm.transformer import FeedForward


def _numeric_grad(func, array, index, h=1e-5):
    return central_difference(func, array, index, h=h)


def _assert_close(test, analytic, numeric, msg):
    result = check_value(float(analytic), float(numeric))
    test.assertTrue(result["passed"], f"{msg}: analytic={analytic}, numeric={numeric}, abs_err={result['abs_error']}")


class TestLinearBackward(unittest.TestCase):
    def test_weight_bias_and_input_gradients(self):
        rng = np.random.default_rng(0)
        layer = Linear(3, 2, rng, dtype=np.float64)
        x = rng.normal(size=(2, 3))
        u = rng.normal(size=(2, 2))

        def scalar():
            return float(np.sum(layer.forward(x) * u))

        out = layer.forward(x, retain_cache=True)
        d_x = layer.backward(u)
        self.assertEqual(d_x.shape, x.shape)
        self.assertEqual(layer.grad_weight.shape, layer.weight.shape)
        self.assertEqual(layer.grad_bias.shape, layer.bias.shape)
        for idx in [(0, 0), (1, 1), (2, 0)]:
            _assert_close(self, layer.grad_weight[idx], _numeric_grad(scalar, layer.weight, idx), f"dW{idx}")
        for idx in [(0,), (1,)]:
            _assert_close(self, layer.grad_bias[idx], _numeric_grad(scalar, layer.bias, idx), f"db{idx}")
        # Gradiente de entrada: perturbar x (la función usa el mismo array).
        x_copy = x.copy()
        for idx in [(0, 0), (1, 2)]:
            def scalar_x(x_arr=x_copy):
                return float(np.sum(layer.forward(x_arr) * u))
            # Recomputar el analítico no es necesario: d_x ya es el de x.
            _assert_close(self, d_x[idx], _numeric_grad(scalar_x, x_copy, idx), f"dX{idx}")

    def test_bias_sums_over_positions(self):
        # Un bias compartido por todas las posiciones: su gradiente es
        # la suma de dY sobre todas ellas.
        rng = np.random.default_rng(1)
        layer = Linear(2, 2, rng, dtype=np.float64)
        x = rng.normal(size=(3, 4, 2))
        d_y = np.ones((3, 4, 2))
        layer.forward(x, retain_cache=True)
        layer.backward(d_y)
        np.testing.assert_allclose(layer.grad_bias, np.full(2, 12.0), rtol=0, atol=1e-12)


class TestReluAndMLP(unittest.TestCase):
    def test_feedforward_all_parameters(self):
        rng = np.random.default_rng(2)
        ff = FeedForward(4, 6, rng, dtype=np.float64)
        # Región lineal de ReLU controlada: pesos positivos y entradas
        # positivas, así que toda pre-activación está lejos de cero y
        # ninguna perturbación de diferencias finitas cruza el kink.
        ff.fc1.weight[...] = 0.3 * np.ones((4, 6))
        ff.fc1.bias[...] = 0.0
        x = rng.uniform(0.5, 1.5, size=(2, 3, 4))
        z1 = ff.fc1.forward(x)
        self.assertTrue(np.min(np.abs(z1)) > 0.05, f"z1 demasiado cerca de 0: {np.min(np.abs(z1))}")
        u = rng.normal(size=(2, 3, 4))

        def scalar():
            return float(np.sum(ff.forward(x) * u))

        ff.forward(x, retain_cache=True)
        d_x = ff.backward(u)
        self.assertEqual(d_x.shape, x.shape)
        for layer, name in ((ff.fc1, "fc1"), (ff.fc2, "fc2")):
            for idx in [tuple([0] * layer.weight.ndim), (layer.weight.shape[0] - 1, layer.weight.shape[1] - 1)]:
                _assert_close(self, layer.grad_weight[idx], _numeric_grad(scalar, layer.weight, idx), f"{name}.weight{idx}")
            _assert_close(self, layer.grad_bias[(0,)], _numeric_grad(scalar, layer.bias, (0,)), f"{name}.bias")


class TestLayerNormBackward(unittest.TestCase):
    def test_input_gamma_beta(self):
        rng = np.random.default_rng(3)
        ln = LayerNorm(4, dtype=np.float64)
        x = rng.normal(size=(2, 3, 4))
        u = rng.normal(size=(2, 3, 4))

        def scalar():
            return float(np.sum(ln.forward(x) * u))

        ln.forward(x, retain_cache=True)
        d_x = ln.backward(u)
        self.assertEqual(d_x.shape, x.shape)
        for idx in [(0,), (3,)]:
            _assert_close(self, ln.grad_gamma[idx], _numeric_grad(scalar, ln.gamma, idx), f"gamma{idx}")
            _assert_close(self, ln.grad_beta[idx], _numeric_grad(scalar, ln.beta, idx), f"beta{idx}")
        x_copy = x.copy()
        for idx in [(0, 0, 0), (1, 2, 3)]:
            def scalar_x(x_arr=x_copy):
                return float(np.sum(ln.forward(x_arr) * u))
            _assert_close(self, d_x[idx], _numeric_grad(scalar_x, x_copy, idx), f"dX{idx}")

    def test_constant_and_near_constant_inputs_finite(self):
        ln = LayerNorm(4, dtype=np.float64)
        for x in (np.ones((2, 3, 4)), np.ones((2, 3, 4)) + 1e-9 * np.arange(24).reshape(2, 3, 4)):
            ln.forward(x, retain_cache=True)
            d_x = ln.backward(np.ones_like(x))
            self.assertTrue(np.all(np.isfinite(d_x)))
            self.assertTrue(np.all(np.isfinite(ln.grad_gamma)))
        # Entrada constante: x_hat = 0, así que dgamma = 0 (tras limpiar
        # los gradientes acumulados por las llamadas anteriores).
        ln.zero_grad()
        ln.forward(np.ones((1, 1, 4)), retain_cache=True)
        ln.backward(np.ones((1, 1, 4)))
        np.testing.assert_allclose(ln.grad_gamma, np.zeros(4), atol=1e-12)

    def test_no_mixing_between_positions(self):
        # Cambiar una posición no debe cambiar el gradiente de otra
        # (LayerNorm normaliza cada vector por separado).
        rng = np.random.default_rng(4)
        ln = LayerNorm(4, dtype=np.float64)
        x = rng.normal(size=(1, 2, 4))
        u = np.zeros((1, 2, 4)); u[0, 0] = 1.0
        ln.forward(x, retain_cache=True)
        d_x = ln.backward(u)
        # Solo la primera posición recibe gradiente no nulo en este caso
        # extremo (u es cero en la segunda, y LN no mezcla posiciones).
        np.testing.assert_allclose(d_x[0, 1], np.zeros(4), atol=1e-12)


class TestEmbeddingsBackward(unittest.TestCase):
    def test_repeated_tokens_accumulate(self):
        rng = np.random.default_rng(5)
        emb = Embeddings(5, 8, 4, rng, dtype=np.float64)
        token_ids = np.array([[1, 2, 1], [2, 2, 3]])
        d_x = rng.normal(size=(2, 3, 4))
        emb.forward(token_ids, retain_cache=True)
        emb.backward(d_x)
        expected = np.zeros((5, 4))
        np.add.at(expected, token_ids.reshape(-1), d_x.reshape(-1, 4))
        np.testing.assert_allclose(emb.grad_token_embedding, expected, rtol=0, atol=1e-12)
        # El token 1 aparece 2 veces: su gradiente es la suma, no la
        # última contribución (el fallo clásico de grad[ids] += dX).
        np.testing.assert_allclose(
            emb.grad_token_embedding[1], d_x[0, 0] + d_x[0, 2], rtol=0, atol=1e-12
        )
        # Posiciones: suma sobre el lote; las no usadas (3..7) son cero.
        np.testing.assert_allclose(emb.grad_position_embedding[:3], np.sum(d_x, axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(emb.grad_position_embedding[3:], np.zeros((5, 4)), atol=1e-12)


class TestSoftmaxBackward(unittest.TestCase):
    def test_softmax_backward_matches_finite_differences(self):
        rng = np.random.default_rng(6)
        s = rng.normal(size=(2, 3, 4))
        u = rng.normal(size=(2, 3, 4))
        probs = softmax(s, axis=-1)
        d_s = softmax_backward(probs, u)
        for idx in [(0, 0, 0), (1, 2, 3)]:
            def scalar(s_arr=s):
                return float(np.sum(softmax(s_arr, axis=-1) * u))
            _assert_close(self, d_s[idx], _numeric_grad(scalar, s, idx), f"dS{idx}")


class TestCacheContract(unittest.TestCase):
    def test_backward_without_forward_raises(self):
        rng = np.random.default_rng(7)
        layer = Linear(2, 2, rng, dtype=np.float64)
        with self.assertRaises(ValueError):
            layer.backward(np.ones((1, 2)))

    def test_double_backward_raises(self):
        rng = np.random.default_rng(7)
        layer = Linear(2, 2, rng, dtype=np.float64)
        x = rng.normal(size=(1, 2))
        layer.forward(x, retain_cache=True)
        layer.backward(np.ones((1, 2)))
        with self.assertRaises(ValueError):
            layer.backward(np.ones((1, 2)))

    def test_second_retained_forward_raises(self):
        rng = np.random.default_rng(7)
        layer = Linear(2, 2, rng, dtype=np.float64)
        x = rng.normal(size=(1, 2))
        layer.forward(x, retain_cache=True)
        with self.assertRaises(ValueError):
            layer.forward(x, retain_cache=True)

    def test_inference_does_not_overwrite_training_cache(self):
        rng = np.random.default_rng(7)
        layer = Linear(2, 2, rng, dtype=np.float64)
        x = rng.normal(size=(1, 2))
        layer.forward(x, retain_cache=True)
        layer.forward(x + 100.0)  # inferencia: no debe tocar la caché
        d_x = layer.backward(np.ones((1, 2)))
        # El gradiente corresponde a x, no a x+100.
        np.testing.assert_allclose(d_x, np.ones((1, 2)) @ layer.weight.T, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
