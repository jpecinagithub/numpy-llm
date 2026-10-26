"""Checkpoints y reanudación exacta."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from numpy_llm.checkpoint import load_checkpoint
from numpy_llm.config import ModelConfig
from numpy_llm.data import sample_batch, split_corpus
from numpy_llm.model import TransformerLanguageModel
from numpy_llm.optimizers import AdamW
from numpy_llm.tokenizer import CharTokenizer
from numpy_llm.training import Trainer, TrainingConfig

CORPUS = (
    "La inteligencia artificial aprende a partir de ejemplos. "
    "Un modelo de lenguaje observa texto y trata de predecir la palabra siguiente. "
    "Cada letra se convierte en un número, y cada número en un vector. "
) * 2


def _make_trainer(total_updates=6):
    split = split_corpus(CORPUS, train_fraction=0.9, sequence_length=8)
    config = ModelConfig(
        context_length=16, d_model=16, n_heads=2, n_layers=1, d_ff=32, seed=42
    ).with_vocab_size(split.tokenizer.vocab_size)
    model = TransformerLanguageModel(config)
    training = TrainingConfig(
        total_updates=total_updates, batch_size=2, sequence_length=8,
        optimizer="adamw", schedule_kind="constant", lr_peak=1e-3, lr_min=1e-3,
        warmup_steps=0, weight_decay=0.0, eval_every=0, checkpoint_every=0, seed=42,
    )
    return Trainer(model, split.tokenizer, split, training, outdir=None), split


class TestCheckpointRoundtrip(unittest.TestCase):
    def test_save_and_load_preserves_weights_optimizer_and_rng(self):
        trainer, _split = _make_trainer()
        trainer.run(limit_updates=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.npz"
            trainer.save(path)
            ckpt = load_checkpoint(path)
            self.assertEqual(ckpt.tokenizer.tokens, trainer.tokenizer.tokens)
            self.assertEqual(ckpt.completed_updates, 2)
            self.assertEqual(ckpt.optimizer_state["t"], 2)
            for (name, param), (_, loaded) in zip(trainer.model.named_parameters(), ckpt.model.named_parameters()):
                np.testing.assert_array_equal(param, loaded)
            for name in ckpt.optimizer_state["m"]:
                np.testing.assert_array_equal(
                    ckpt.optimizer_state["m"][name], trainer.optimizer.m[name]
                )
            # Cargar pesos no es reanudar: un optimizador nuevo empieza en t=0.
            fresh = AdamW(
                [(n, p, g) for (n, p), (_, g) in zip(ckpt.model.named_parameters(), ckpt.model.named_gradients())]
            )
            self.assertEqual(fresh.t, 0)
            # Los gradientes se inicializan a cero al cargar.
            for _, grad in ckpt.model.named_gradients():
                np.testing.assert_allclose(grad, 0, atol=0)

    def test_corpus_identity_mismatch_is_rejected(self):
        trainer, _split = _make_trainer()
        trainer.run(limit_updates=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.npz"
            trainer.save(path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, expected_corpus_sha256="0" * 64)

    def test_vocab_mismatch_is_rejected(self):
        trainer, _split = _make_trainer()
        trainer.run(limit_updates=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.npz"
            trainer.save(path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, expected_vocab_tokens=["<UNK>", "x"])


class TestResumeEquivalence(unittest.TestCase):
    def test_resume_matches_uninterrupted_run(self):
        # Ejecución A: 6 actualizaciones consecutivas.
        trainer_a, split_a = _make_trainer(total_updates=6)
        trainer_a.run()
        # Ejecución B: 2 actualizaciones, guardar, cargar, continuar hasta 6
        # con el MISMO horizonte total (6) desde el comienzo.
        trainer_b, _split_b = _make_trainer(total_updates=6)
        trainer_b.run(limit_updates=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.npz"
            trainer_b.save(path)
            resumed = Trainer.from_checkpoint(path, CORPUS, outdir=None)
            self.assertEqual(resumed.completed_updates, 2)
            self.assertEqual(resumed.optimizer.t, 2)
            resumed.run()
        self.assertEqual(resumed.completed_updates, 6)
        self.assertEqual(resumed.tokens_processed, trainer_a.tokens_processed)
        self.assertEqual(resumed.lr_history, trainer_a.lr_history)
        self.assertEqual(resumed.optimizer.t, trainer_a.optimizer.t)
        for (name, pa), (_, pb) in zip(trainer_a.model.named_parameters(), resumed.model.named_parameters()):
            np.testing.assert_allclose(pa, pb, rtol=1e-6, atol=1e-7, err_msg=name)
        for name in trainer_a.optimizer.m:
            np.testing.assert_allclose(
                trainer_a.optimizer.m[name], resumed.optimizer.m[name], rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                trainer_a.optimizer.v[name], resumed.optimizer.v[name], rtol=1e-6, atol=1e-7
            )
        # Próximos lotes muestreados: idénticos tras la reanudación.
        next_a = sample_batch(split_a.train_tokens, 2, 8, trainer_a.train_rng)
        next_b = sample_batch(split_a.train_tokens, 2, 8, resumed.train_rng)
        np.testing.assert_array_equal(next_a[0], next_b[0])
        np.testing.assert_array_equal(next_a[1], next_b[1])
        # Pérdidas reproducibles: misma evaluación final.
        self.assertAlmostEqual(
            trainer_a.evaluate("val")["mean_loss"],
            resumed.evaluate("val")["mean_loss"],
            places=6,
        )

    def test_completed_training_is_not_restarted(self):
        trainer, _split = _make_trainer(total_updates=2)
        trainer.run()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.npz"
            trainer.save(path)
            resumed = Trainer.from_checkpoint(path, CORPUS, outdir=None)
            before = {n: p.copy() for n, p in resumed.model.named_parameters()}
            resumed.run()  # no debe dar ningún paso más
            self.assertEqual(resumed.completed_updates, 2)
            for name, param in resumed.model.named_parameters():
                np.testing.assert_array_equal(param, before[name])


if __name__ == "__main__":
    unittest.main()
