"""Pruebas de guardado y carga del modelo."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from numpy_llm.config import ModelConfig
from numpy_llm.model import TransformerLanguageModel
from numpy_llm.serialization import load_model, save_model
from numpy_llm.tokenizer import CharTokenizer


def make_model_and_tokenizer():
    tokenizer = CharTokenizer.from_text("Hola mundo, ¿qué tal?")
    config = ModelConfig(
        context_length=16, d_model=16, n_heads=4, n_layers=1, d_ff=32, seed=123
    ).with_vocab_size(tokenizer.vocab_size)
    return TransformerLanguageModel(config), tokenizer


class TestSerialization(unittest.TestCase):
    def test_save_and_load_preserves_vocab_config_and_logits(self):
        model, tokenizer = make_model_and_tokenizer()
        x = np.array([[1, 2, 3, 4]])
        before = model.forward(x)
        with tempfile.TemporaryDirectory() as tmp:
            save_model(model, tokenizer, Path(tmp) / "modelo")
            loaded_model, loaded_tokenizer = load_model(Path(tmp) / "modelo")
            self.assertEqual(loaded_tokenizer.tokens, tokenizer.tokens)
            self.assertEqual(loaded_model.config.to_dict(), model.config.to_dict())
            after = loaded_model.forward(x)
            np.testing.assert_allclose(before, after, atol=1e-6)

    def test_missing_parameter_is_detected(self):
        model, tokenizer = make_model_and_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "modelo"
            save_model(model, tokenizer, directory)
            # Reescribir weights.npz sin el primer parámetro.
            with np.load(directory / "weights.npz", allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays.pop("p0")
            np.savez(directory / "weights.npz", **arrays)
            with self.assertRaises(ValueError):
                load_model(directory)

    def test_incompatible_shape_is_detected(self):
        model, tokenizer = make_model_and_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "modelo"
            save_model(model, tokenizer, directory)
            with np.load(directory / "weights.npz", allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["p0"] = np.zeros((999,), dtype=np.float32)
            np.savez(directory / "weights.npz", **arrays)
            with self.assertRaises(ValueError):
                load_model(directory)

    def test_unexpected_parameter_is_detected(self):
        model, tokenizer = make_model_and_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "modelo"
            save_model(model, tokenizer, directory)
            with np.load(directory / "weights.npz", allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["p999"] = np.zeros((1,), dtype=np.float32)
            np.savez(directory / "weights.npz", **arrays)
            with self.assertRaises(ValueError):
                load_model(directory)

    def test_vocab_config_mismatch_is_detected(self):
        model, tokenizer = make_model_and_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "modelo"
            save_model(model, tokenizer, directory)
            vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
            vocab["tokens"] = vocab["tokens"][:-1]
            (directory / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(directory)


if __name__ == "__main__":
    unittest.main()
