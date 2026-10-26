"""Backward de la atención causal multicabeza."""

import unittest

import numpy as np

from numpy_llm.attention import CausalMultiHeadAttention
from numpy_llm.gradcheck import central_difference, check_value


def _assert_close(test, analytic, numeric, msg):
    result = check_value(float(analytic), float(numeric))
    test.assertTrue(result["passed"], f"{msg}: analytic={analytic}, numeric={numeric}, abs_err={result['abs_error']}")


class TestAttentionBackward(unittest.TestCase):
    def _make(self, seed=0):
        rng = np.random.default_rng(seed)
        attn = CausalMultiHeadAttention(4, 2, rng, dtype=np.float64)
        return attn, rng

    def test_projection_and_input_gradients(self):
        attn, rng = self._make(0)
        x = rng.normal(size=(2, 3, 4))
        u = rng.normal(size=(2, 3, 4))

        def scalar():
            return float(np.sum(attn.forward(x) * u))

        attn.forward(x, retain_cache=True)
        d_x = attn.backward(u)
        self.assertEqual(d_x.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(d_x)))
        for proj, name in (
            (attn.q_proj, "q_proj"), (attn.k_proj, "k_proj"),
            (attn.v_proj, "v_proj"), (attn.out_proj, "out_proj"),
        ):
            for idx in [(0, 0), (3, 3)]:
                _assert_close(self, proj.grad_weight[idx], central_difference(scalar, proj.weight, idx), f"{name}.weight{idx}")
            _assert_close(self, proj.grad_bias[(0,)], central_difference(scalar, proj.bias, (0,)), f"{name}.bias")
        x_copy = x.copy()
        for idx in [(0, 0, 0), (1, 2, 3)]:
            def scalar_x(x_arr=x_copy):
                return float(np.sum(attn.forward(x_arr) * u))
            _assert_close(self, d_x[idx], central_difference(scalar_x, x_copy, idx), f"dX{idx}")

    def test_causal_mask_in_forward_and_backward(self):
        attn, rng = self._make(1)
        x = rng.normal(size=(1, 4, 4))
        out, weights = attn.forward(x, return_attention_weights=True)
        # Posiciones futuras tienen probabilidad exactamente 0.
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertEqual(weights[0, 0, i, j], 0.0)
        # Cambiar el sufijo no altera el prefijo (causalidad).
        x2 = x.copy()
        x2[0, 3] = x2[0, 3] + 5.0
        out2 = attn.forward(x2)
        np.testing.assert_allclose(out[0, :3], out2[0, :3], rtol=0, atol=1e-12)
        # Backward con -inf en los puntajes: gradientes finitos.
        attn.forward(x, retain_cache=True)
        d_x = attn.backward(rng.normal(size=(1, 4, 4)))
        self.assertTrue(np.all(np.isfinite(d_x)))
        for _, grad in attn.named_gradients("attn"):
            self.assertTrue(np.all(np.isfinite(grad)))

    def test_split_merge_inverse(self):
        attn, rng = self._make(2)
        x = rng.normal(size=(2, 3, 4))
        merged = attn._merge_heads(attn._split_heads(x))
        np.testing.assert_allclose(merged, x, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
