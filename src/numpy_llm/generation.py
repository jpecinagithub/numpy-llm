"""Generación autorregresiva.

En cada paso:

1. Se toman como entrada los últimos C tokens disponibles.
2. Se ejecuta el modelo.
3. Se extraen los logits de la última posición.
4. Se aplica temperatura y, cuando corresponde, filtrado top-k.
5. Se obtienen probabilidades estables.
6. Se selecciona el siguiente token.
7. Se añade a la secuencia.
8. Se repite hasta generar ``max_new_tokens``.

Definiciones explícitas:

- ``temperature > 0``: muestreo con ``logits / temperature``.
- ``temperature == 0``: selección greedy mediante argmax, sin dividir
  por cero.
- ``temperature`` negativa o no finita: error.
- ``top_k`` ausente: se permite todo el vocabulario.
- ``top_k`` presente: entero entre 1 y V; se conservan exactamente k
  candidatos y el resto se enmascara antes de softmax.
- ``max_new_tokens == 0``: devuelve la secuencia original.
- Prompt vacío: error claro en esta versión.

Se usa un generador de muestreo separado del generador de
inicialización del modelo. Con el mismo estado del modelo y la misma
semilla, la generación es reproducible en el mismo entorno.

Simplificación documentada: cuando se recorta el contexto, las
posiciones de la ventana activa son 0, 1, ..., T-1 (los embeddings de
posición se aplican a la ventana, no a la posición absoluta en el texto
completo). No se implementan KV cache, beam search, EOS ni mecanismos
avanzados de generación.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .functional import softmax
from .model import TransformerLanguageModel


def _validate_generation_args(
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    vocab_size: int,
) -> None:
    if len(prompt_token_ids) == 0:
        raise ValueError("El prompt no puede estar vacío en esta versión.")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError(f"max_new_tokens debe ser un entero >= 0, no {max_new_tokens!r}.")
    if not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or float(temperature) < 0:
        raise ValueError(
            f"temperature debe ser un número finito >= 0, no {temperature!r}."
        )
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or top_k > vocab_size:
            raise ValueError(f"top_k debe ser un entero entre 1 y {vocab_size}, no {top_k!r}.")
    for token_id in prompt_token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"Token del prompt {token_id!r} fuera del vocabulario [0, {vocab_size})."
            )


def generate(
    model: TransformerLanguageModel,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Genera tokens de forma autorregresiva a partir de un prompt.

    Args:
        model: Modelo ya construido (sus pesos no se modifican).
        prompt_token_ids: Tokens iniciales, no vacíos.
        max_new_tokens: Número de tokens nuevos que se añadirán.
        temperature: 0 para greedy; > 0 para muestreo.
        top_k: Si se indica, solo se conservan los k logits mayores.
        rng: Generador de muestreo. Si es None, se crea uno sin semilla
            (generación no reproducible); para reproducibilidad, pasa
            ``np.random.default_rng(semilla)``.

    Returns:
        La secuencia completa: prompt + tokens generados.
    """
    config = model.config
    vocab_size = config.vocab_size
    assert vocab_size is not None
    _validate_generation_args(prompt_token_ids, max_new_tokens, temperature, top_k, vocab_size)
    if rng is None:
        rng = np.random.default_rng()

    sequence: list[int] = list(prompt_token_ids)
    for _ in range(max_new_tokens):
        # Últimos C tokens disponibles (recorte de contexto).
        window = sequence[-config.context_length :]
        inputs = np.asarray([window], dtype=np.int64)  # [1, T]
        logits = model.forward(inputs)  # [1, T, V]
        assert isinstance(logits, np.ndarray)
        last_logits = logits[0, -1]  # [V], float32

        if float(temperature) == 0.0:
            next_token = int(np.argmax(last_logits))
        else:
            # Trabajar en float64 para el muestreo: más precisión y
            # probabilidades finitas, no negativas y normalizadas.
            scaled = last_logits.astype(np.float64) / float(temperature)
            if top_k is not None:
                # Conservar exactamente k candidatos: los k mayores.
                order = np.argsort(-scaled, kind="stable")
                keep = set(order[:top_k].tolist())
                mask = np.array([i not in keep for i in range(vocab_size)])
                scaled = np.where(mask, -np.inf, scaled)
            probs = softmax(scaled, axis=-1)
            if not np.all(np.isfinite(probs)) or np.any(probs < 0):
                raise ValueError("Las probabilidades de muestreo no son finitas y no negativas.")
            probs = probs / probs.sum()  # renormalización por redondeo
            next_token = int(rng.choice(vocab_size, p=probs))
        sequence.append(next_token)
    return sequence
