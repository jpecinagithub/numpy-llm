"""Evaluación: pérdida, perplexity y exactitud por token, más baselines.

Reglas
------
- Forward sin retener intermediarios, sin backward, sin modificar
  pesos ni estado del optimizador, sin consumir el generador de
  entrenamiento. Ventanas deterministas con stride T (ver
  ``data.eval_windows``); el último lote puede ser más pequeño.
- Las estadísticas se acumulan en float64 y se ponderan por número de
  objetivos: nunca se promedian sin ponderar pérdidas de lotes de
  distinto tamaño.
- ``mean_loss = total_nll / total_targets``;
  ``perplexity = exp(mean_loss)``; ``token_accuracy = correct / total``.
- Si ``exp(mean_loss)`` desborda, ``perplexity`` es ``None`` y se
  marca ``perplexity_overflow=True`` (sin sustituir el valor por uno
  arbitrario y manteniendo el JSON válido).
- La perplexity depende del tokenizador y del protocolo de evaluación;
  una estimación parcial nunca se presenta como cobertura total:
  el resultado indica siempre cuántas ventanas y objetivos se evaluaron.

Baselines
---------
- Uniforme: ``loss = log(V)``, ``perplexity = V``.
- Unigrama: frecuencias SOLO de train, con suavizado aditivo
  (alpha = 1.0 por defecto): ``p(c) = (count(c) + alpha) / (N + alpha*V)``.
"""

from __future__ import annotations

import math

import numpy as np

from .data import eval_windows
from .functional import log_softmax


def _batch_nll_and_correct(logits: np.ndarray, targets: np.ndarray) -> tuple[float, int, int]:
    """Suma de NLL (float64), nº de objetivos y nº de aciertos de un lote."""
    log_probs = log_softmax(logits.astype(np.float64), axis=-1)
    batch, seq_len, _ = logits.shape
    batch_index = np.arange(batch)[:, None]
    time_index = np.arange(seq_len)[None, :]
    selected = log_probs[batch_index, time_index, targets]
    total_nll = float(-np.sum(selected))
    predicted = np.argmax(logits, axis=-1)
    correct = int(np.sum(predicted == targets))
    return total_nll, int(targets.size), correct


def evaluate_model(
    model,
    tokens: np.ndarray,
    sequence_length: int,
    batch_size: int = 4,
    limit_windows: int | None = None,
) -> dict:
    """Evalúa el modelo sobre una partición completa o un subconjunto fijo.

    Returns:
        Dict con mean_loss, perplexity (o None si desborda),
        perplexity_overflow, token_accuracy, total_targets,
        evaluated_windows, total_windows_available, leftover_tokens y
        coverage ("full" si se evaluaron todas las ventanas, "subset"
        en caso contrario).
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    all_starts = list(range(0, tokens.size - sequence_length, sequence_length))
    total_windows_available = len(all_starts)
    total_nll = 0.0
    total_targets = 0
    total_correct = 0
    evaluated_windows = 0
    leftover = int(tokens.size)
    for x, y, _evaluated, leftover in eval_windows(
        tokens, sequence_length, batch_size=batch_size, limit=limit_windows
    ):
        logits = model.forward(x)
        if not isinstance(logits, np.ndarray):
            raise ValueError("evaluate_model espera logits sin pesos de atención.")
        nll, n_targets, correct = _batch_nll_and_correct(logits, y)
        total_nll += nll
        total_targets += n_targets
        total_correct += correct
        evaluated_windows += int(x.shape[0])
    if total_targets == 0:
        raise ValueError(
            f"No hay ventanas de evaluación: la partición tiene {tokens.size} "
            f"tokens y se necesitan al menos {sequence_length + 1} para una ventana."
        )
    mean_loss = total_nll / total_targets
    try:
        perplexity: float | None = float(math.exp(mean_loss))
        if not math.isfinite(perplexity):
            raise OverflowError
        overflow = False
    except OverflowError:
        perplexity = None
        overflow = True
    return {
        "mean_loss": float(mean_loss),
        "perplexity": perplexity,
        "perplexity_overflow": overflow,
        "token_accuracy": float(total_correct / total_targets),
        "total_targets": int(total_targets),
        "correct_predictions": int(total_correct),
        "evaluated_windows": int(evaluated_windows),
        "total_windows_available": int(total_windows_available),
        "leftover_tokens": int(leftover),
        "total_tokens": int(tokens.size),
        "coverage": "full" if evaluated_windows == total_windows_available else "subset",
    }


def uniform_baseline(vocab_size: int) -> dict:
    """Baseline uniforme: loss = log(V), perplexity = V."""
    if vocab_size <= 0:
        raise ValueError("vocab_size debe ser positivo.")
    return {"mean_loss": float(math.log(vocab_size)), "perplexity": float(vocab_size)}


def unigram_baseline(
    train_tokens: np.ndarray, eval_tokens: np.ndarray, vocab_size: int, alpha: float = 1.0
) -> dict:
    """Baseline unigrama con frecuencias solo de train y suavizado aditivo.

    Evalúa la NLL media sobre eval_tokens (cada token es un objetivo,
    sin ventanas): ``-mean(log p(token))`` con
    ``p(c) = (count_train(c) + alpha) / (N_train + alpha * V)``.
    """
    train_tokens = np.asarray(train_tokens, dtype=np.int64)
    eval_tokens = np.asarray(eval_tokens, dtype=np.int64)
    if train_tokens.size == 0 or eval_tokens.size == 0:
        raise ValueError("train y eval deben ser no vacíos para el baseline unigrama.")
    if alpha <= 0:
        raise ValueError("alpha debe ser positivo.")
    counts = np.bincount(train_tokens, minlength=vocab_size).astype(np.float64)
    probs = (counts + alpha) / (train_tokens.size + alpha * vocab_size)
    nll = float(-np.mean(np.log(probs[eval_tokens])))
    try:
        perplexity: float | None = float(math.exp(nll))
        overflow = not math.isfinite(perplexity)
        if overflow:
            perplexity = None
    except OverflowError:
        perplexity = None
        overflow = True
    return {
        "mean_loss": nll,
        "perplexity": perplexity,
        "perplexity_overflow": overflow,
        "alpha": float(alpha),
        "total_targets": int(eval_tokens.size),
    }
