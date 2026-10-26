"""Datos: partición train/val, vocabulario solo de train, lotes, ventanas."""

import unittest

import numpy as np

from numpy_llm.data import eval_windows, sample_batch, split_corpus


class TestSplitCorpus(unittest.TestCase):
    def test_vocab_only_from_train_and_unk_for_val_only_chars(self):
        # El carácter 'Ω' solo aparece en el último 10 % del texto.
        text = "abcabc" * 30 + "ΩΩΩΩ"
        split = split_corpus(text, train_fraction=0.9)
        self.assertNotIn("Ω", split.tokenizer.tokens)
        self.assertGreater(split.val_unk_rate, 0.0)
        self.assertEqual(split.train_unk_rate, 0.0)
        # Ningún token de val distinto de <UNK> sale del vocabulario.
        self.assertTrue(np.all(split.val_tokens < split.tokenizer.vocab_size))
        # La partición se hace sobre el texto, no sobre ventanas.
        self.assertEqual(split.train_text + split.val_text, text)
        self.assertEqual(split.split_char_index, int(len(text) * 0.9))

    def test_insufficient_corpus_error_states_minimum(self):
        with self.assertRaises(ValueError) as ctx:
            split_corpus("abc", train_fraction=0.9, sequence_length=8)
        self.assertIn("9", str(ctx.exception))  # sequence_length + 1

    def test_split_does_not_copy_train_to_val(self):
        text = "abcdefghij" * 20
        split = split_corpus(text, train_fraction=0.9)
        self.assertEqual(len(split.train_text), int(len(text) * 0.9))
        self.assertTrue(len(split.val_text) > 0)
        self.assertNotEqual(split.train_text, split.val_text)


class TestSampleBatch(unittest.TestCase):
    def test_teacher_forcing_windows_are_consecutive_in_corpus(self):
        tokens = np.arange(100, dtype=np.int64) % 7
        rng = np.random.default_rng(0)
        x, y = sample_batch(tokens, batch_size=4, sequence_length=8, rng=rng)
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))
        for row_x, row_y in zip(x, y):
            found = False
            for start in range(0, 100 - 8):
                if np.array_equal(tokens[start : start + 8], row_x) and np.array_equal(
                    tokens[start + 1 : start + 9], row_y
                ):
                    found = True
                    break
            self.assertTrue(found, "cada (x, y) debe ser una ventana consecutiva desplazada")

    def test_sampling_is_reproducible_with_same_rng_seed(self):
        tokens = np.arange(50, dtype=np.int64)
        x1, y1 = sample_batch(tokens, 3, 8, np.random.default_rng(123))
        x2, y2 = sample_batch(tokens, 3, 8, np.random.default_rng(123))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


class TestEvalWindows(unittest.TestCase):
    def test_stride_and_last_partial_batch(self):
        tokens = np.arange(25, dtype=np.int64)
        batches = list(eval_windows(tokens, sequence_length=8, batch_size=2))
        # Inicios 0, 8, 16 -> 3 ventanas; lotes de 2 + 1.
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][0].shape[0], 2)
        self.assertEqual(batches[1][0].shape[0], 1)
        np.testing.assert_array_equal(batches[0][0][0], np.arange(8))
        np.testing.assert_array_equal(batches[0][1][0], np.arange(1, 9))
        # Objetivos evaluados: 3 ventanas * 8 = 24; fuera: 25 - (24+1) = 0.
        self.assertEqual(batches[-1][2], 24)
        self.assertEqual(batches[-1][3], 0)

    def test_leftover_fragment_is_reported(self):
        tokens = np.arange(30, dtype=np.int64)
        batches = list(eval_windows(tokens, sequence_length=8, batch_size=4))
        # Inicios 0, 8, 16 -> cubren hasta el token 24; fuera: 30-25=5.
        self.assertEqual(batches[-1][2], 24)
        self.assertEqual(batches[-1][3], 5)


if __name__ == "__main__":
    unittest.main()
