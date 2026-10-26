"""Pruebas del modelo completo y de la generación."""

import unittest

import numpy as np

from numpy_llm.config import ModelConfig
from numpy_llm.generation import generate
from numpy_llm.model import TransformerLanguageModel


def make_model(**overrides):
    config_kwargs = dict(
        context_length=16, d_model=16, n_heads=4, n_layers=2, d_ff=32, seed=42, vocab_size=20
    )
    config_kwargs.update(overrides)
    return TransformerLanguageModel(ModelConfig(**config_kwargs))


class TestModel(unittest.TestCase):
    def test_logits_shape_and_finite(self):
        model = make_model()
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        logits = model.forward(x)
        self.assertEqual(logits.shape, (2, 4, 20))
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_same_seed_same_parameters(self):
        a = make_model()
        b = make_model()
        for (name_a, array_a), (name_b, array_b) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(name_a, name_b)
            np.testing.assert_array_equal(array_a, array_b)

    def test_forward_does_not_modify_parameters(self):
        model = make_model()
        before = [(name, array.copy()) for name, array in model.named_parameters()]
        x = np.array([[1, 2, 3]])
        model.forward(x)
        model.forward(x)
        for (name, old), (_, new) in zip(before, model.named_parameters()):
            np.testing.assert_array_equal(old, new)

    def test_repeated_forward_is_deterministic(self):
        model = make_model()
        x = np.array([[1, 2, 3, 4]])
        first = model.forward(x)
        second = model.forward(x)
        np.testing.assert_array_equal(first, second)

    def test_invalid_inputs_raise_clear_errors(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.forward(np.array([1, 2, 3]))  # No es [B, T].
        with self.assertRaises(ValueError):
            model.forward(np.array([[1.5, 2.0]]))  # No enteros.
        with self.assertRaises(ValueError):
            model.forward(np.array([[20]]))  # Fuera del vocabulario.
        with self.assertRaises(ValueError):
            model.forward(np.zeros((1, 17), dtype=np.int64))  # T > C.
        with self.assertRaises(ValueError):
            model.forward(np.zeros((0, 3), dtype=np.int64))  # B = 0.

    def test_causality_full_model(self):
        """Mismo prefijo, distinto sufijo: los logits del prefijo coinciden."""
        model = make_model()
        prefix = [1, 2, 3, 4]
        a = np.array([prefix + [5, 6, 7]])
        b = np.array([prefix + [15, 16, 17]])
        logits_a = model.forward(a)
        logits_b = model.forward(b)
        np.testing.assert_allclose(logits_a[:, :4, :], logits_b[:, :4, :], atol=1e-5)

    def test_parameter_count_is_real(self):
        model = make_model()
        counted = sum(array.size for _, array in model.named_parameters())
        self.assertEqual(model.count_parameters(), counted)
        self.assertGreater(counted, 0)
        # Nombres únicos.
        names = [name for name, _ in model.named_parameters()]
        self.assertEqual(len(names), len(set(names)))


class TestGeneration(unittest.TestCase):
    def test_output_length(self):
        model = make_model()
        result = generate(model, [1, 2, 3], max_new_tokens=5, temperature=1.0, rng=np.random.default_rng(0))
        self.assertEqual(len(result), 8)
        self.assertEqual(result[:3], [1, 2, 3])

    def test_zero_new_tokens_returns_prompt(self):
        model = make_model()
        result = generate(model, [1, 2], max_new_tokens=0, temperature=1.0, rng=np.random.default_rng(0))
        self.assertEqual(result, [1, 2])

    def test_temperature_zero_is_greedy(self):
        model = make_model()
        prompt = [1, 2, 3]
        result = generate(model, prompt, max_new_tokens=3, temperature=0.0, rng=np.random.default_rng(0))
        # Reproducir greedy manualmente.
        sequence = list(prompt)
        for _ in range(3):
            logits = model.forward(np.asarray([sequence[-16:]], dtype=np.int64))
            sequence.append(int(np.argmax(logits[0, -1])))
        self.assertEqual(result, sequence)

    def test_top_k_one_selects_max_logit(self):
        model = make_model()
        prompt = [1, 2]
        result = generate(model, prompt, max_new_tokens=1, temperature=1.0, top_k=1, rng=np.random.default_rng(0))
        logits = model.forward(np.asarray([prompt], dtype=np.int64))
        self.assertEqual(result[-1], int(np.argmax(logits[0, -1])))

    def test_context_is_truncated_beyond_c(self):
        model = make_model(context_length=4)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = generate(model, prompt, max_new_tokens=3, temperature=0.0, rng=np.random.default_rng(0))
        self.assertEqual(len(result), 13)

    def test_same_sampling_seed_reproduces_output(self):
        model = make_model()
        a = generate(model, [1, 2], max_new_tokens=10, temperature=1.0, top_k=5, rng=np.random.default_rng(7))
        b = generate(model, [1, 2], max_new_tokens=10, temperature=1.0, top_k=5, rng=np.random.default_rng(7))
        self.assertEqual(a, b)

    def test_generation_validates_args(self):
        model = make_model()
        with self.assertRaises(ValueError):
            generate(model, [], max_new_tokens=1, rng=np.random.default_rng(0))
        with self.assertRaises(ValueError):
            generate(model, [1], max_new_tokens=1, temperature=-1.0, rng=np.random.default_rng(0))
        with self.assertRaises(ValueError):
            generate(model, [1], max_new_tokens=1, temperature=1.0, top_k=0, rng=np.random.default_rng(0))
        with self.assertRaises(ValueError):
            generate(model, [1], max_new_tokens=1, temperature=1.0, top_k=21, rng=np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
