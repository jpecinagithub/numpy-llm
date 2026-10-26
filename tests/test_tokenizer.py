"""Pruebas del tokenizador a nivel de carácter."""

import tempfile
import unittest
from pathlib import Path

from numpy_llm.tokenizer import UNK_ID, CharTokenizer


class TestCharTokenizer(unittest.TestCase):
    def setUp(self):
        self.text = "Hola, ¿qué tal?\nCanción árbol"
        self.tokenizer = CharTokenizer.from_text(self.text)

    def test_roundtrip_known_characters(self):
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(self.text)), self.text)

    def test_preserves_spaces_accents_and_newlines(self):
        sample = "a á\n "
        encoded = self.tokenizer.encode(sample)
        self.assertEqual(self.tokenizer.decode(encoded), sample)

    def test_unknown_characters_become_unk(self):
        encoded = self.tokenizer.encode("Hola€")
        self.assertEqual(encoded[-1], UNK_ID)
        self.assertEqual(self.tokenizer.decode(encoded), "Hola<UNK>")

    def test_vocab_is_deterministic(self):
        other = CharTokenizer.from_text(self.text)
        self.assertEqual(self.tokenizer.tokens, other.tokens)
        self.assertEqual(self.tokenizer.token_to_id if hasattr(self.tokenizer, "token_to_id") else self.tokenizer.tokens, other.tokens)

    def test_unk_is_zero(self):
        self.assertEqual(self.tokenizer.tokens[0], "<UNK>")
        self.assertEqual(self.tokenizer.vocab_size, len(set(self.text)) + 1)

    def test_decode_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            self.tokenizer.decode([self.tokenizer.vocab_size])

    def test_save_and_load_vocab(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            self.tokenizer.save_vocab(path)
            loaded = CharTokenizer.load_vocab(path)
            self.assertEqual(loaded.tokens, self.tokenizer.tokens)
            self.assertEqual(loaded.encode(self.text), self.tokenizer.encode(self.text))


if __name__ == "__main__":
    unittest.main()
