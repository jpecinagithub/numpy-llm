"""Optimizadores: SGD y AdamW, con cálculos manuales independientes."""

import unittest

import numpy as np

from numpy_llm.optimizers import AdamW, SGD, decay_for_name


class TestSGD(unittest.TestCase):
    def test_manual_update_in_place(self):
        param = np.array([1.0, 2.0])
        grad = np.array([0.5, -1.0])
        original_id = id(param)
        opt = SGD([(param, grad)], lr=0.1)
        opt.step()
        np.testing.assert_allclose(param, [0.95, 2.1], rtol=0, atol=1e-12)
        self.assertEqual(id(param), original_id)
        self.assertEqual(opt.steps, 1)


class TestAdamW(unittest.TestCase):
    def test_first_step_bias_correction(self):
        # Primer paso: m_hat = g, v_hat = g^2, así que la actualización
        # es lr * g / (|g| + eps) (sin weight decay).
        param = np.array([1.0, -2.0])
        grad = np.array([0.5, -0.25])
        opt = AdamW([(param, grad)], lr=0.1)
        opt.step()
        expected = np.array([1.0, -2.0]) - 0.1 * grad / (np.abs(grad) + 1e-8)
        np.testing.assert_allclose(param, expected, rtol=1e-9, atol=1e-12)
        self.assertEqual(opt.t, 1)

    def test_several_steps_match_independent_implementation(self):
        rng = np.random.default_rng(0)
        params = [rng.normal(size=(3,)) for _ in range(2)]
        grads = [rng.normal(size=(3,)) for _ in range(2)]
        params_ref = [p.copy() for p in params]
        opt = AdamW(list(zip(params, grads)), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.01
        m = [np.zeros(3) for _ in range(2)]
        v = [np.zeros(3) for _ in range(2)]
        for t in range(1, 6):
            opt.step()
            for i in range(2):
                m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
                v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                params_ref[i] = params_ref[i] - lr * m_hat / (np.sqrt(v_hat) + eps)
            for i in range(2):
                np.testing.assert_allclose(params[i], params_ref[i], rtol=1e-10, atol=1e-12)
        self.assertEqual(opt.t, 5)  # t aumenta una vez por update, no por parámetro

    def test_weight_decay_only_on_linear_matrices(self):
        self.assertTrue(decay_for_name("blocks.0.attention.q_proj.weight"))
        self.assertTrue(decay_for_name("lm_head.weight"))
        self.assertFalse(decay_for_name("blocks.0.attention.q_proj.bias"))
        self.assertFalse(decay_for_name("final_ln.gamma"))
        self.assertFalse(decay_for_name("final_ln.beta"))
        self.assertFalse(decay_for_name("embeddings.token_embedding"))
        self.assertFalse(decay_for_name("embeddings.position_embedding"))
        # Comportamiento: con gradiente cero, solo la matriz con decay cambia.
        weight = np.array([2.0])
        bias = np.array([2.0])
        opt = AdamW(
            [("fc.weight", weight, np.zeros(1)), ("fc.bias", bias, np.zeros(1))],
            lr=0.1, weight_decay=0.1,
        )
        opt.step()
        np.testing.assert_allclose(weight, [2.0 - 0.1 * 0.1 * 2.0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(bias, [2.0], rtol=0, atol=1e-12)

    def test_decay_is_decoupled_from_gradient(self):
        # Con weight decay, el gradiente usado para m/v no incluye wd*theta:
        # con g = 0, m y v quedan en cero tras el paso.
        param = np.array([3.0])
        opt = AdamW([("fc.weight", param, np.zeros(1))], lr=0.1, weight_decay=0.5)
        opt.step()
        np.testing.assert_allclose(opt.m["fc.weight"], [0.0], atol=0)
        np.testing.assert_allclose(opt.v["fc.weight"], [0.0], atol=0)

    def test_state_save_and_load(self):
        param = np.array([1.0, 2.0])
        grad = np.array([0.3, -0.2])
        opt = AdamW([("p", param, grad)], lr=0.01)
        opt.step(); opt.step()
        state = opt.state_dict()
        param2 = np.array([1.0, 2.0])
        opt2 = AdamW([("p", param2, grad)], lr=0.01)
        opt2.load_state_dict(state)
        self.assertEqual(opt2.t, 2)
        np.testing.assert_allclose(opt2.m["p"], opt.m["p"], rtol=0, atol=0)
        np.testing.assert_allclose(opt2.v["p"], opt.v["p"], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            opt2.load_state_dict({"type": "SGD"})

    def test_invalid_hyperparameters_rejected(self):
        param = np.array([1.0])
        with self.assertRaises(ValueError):
            AdamW([(param, param)], lr=0.0)
        with self.assertRaises(ValueError):
            AdamW([(param, param)], lr=0.01, betas=(1.0, 0.999))
        with self.assertRaises(ValueError):
            AdamW([(param, param)], lr=0.01, weight_decay=-0.1)


if __name__ == "__main__":
    unittest.main()
