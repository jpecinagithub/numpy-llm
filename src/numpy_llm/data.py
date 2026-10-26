"""Preparación de secuencias de entrenamiento (solo datos, sin entrenar).

Dada una ventana de T + 1 tokens consecutivos:

- ``x`` contiene los primeros T tokens.
- ``y`` contiene los últimos T tokens.

Ejemplo: secuencia A B C D E, con T = 4:

- Entrada x: A B C D
- Objetivo y: B C D E

Cada posición predice, por tanto, el token inmediatamente siguiente.
Las ventanas son completas y de longitud fija: no hay padding ni máscaras
de padding en esta fase. Un corpus vacío o demasiado corto es un error
explicativo; nunca se repite silenciosamente el corpus para fabricar
ejemplos.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def build_examples(
    token_ids: Sequence[int], sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construye pares (x, y) desplazados a partir de tokens consecutivos.

    Args:
        token_ids: Secuencia completa de identificadores de token.
        sequence_length: T, longitud de cada entrada (y de cada objetivo).

    Returns:
        Una tupla ``(x, y)`` con formas ``[B, T]`` y ``[B, T]``, donde B es
        el número de ventanas completas que caben en el corpus.

    Raises:
        ValueError: si el corpus está vacío, es demasiado corto para una
            sola ventana de T + 1 tokens, o si T no es positivo.
    """
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length <= 0:
        raise ValueError(f"sequence_length debe ser un entero positivo, no {sequence_length!r}.")
    tokens = np.asarray(list(token_ids), dtype=np.int64)
    if tokens.size == 0:
        raise ValueError(
            "El corpus está vacío: no se puede construir ningún ejemplo. "
            "Proporciona texto con al menos sequence_length + 1 tokens."
        )
    if tokens.size < sequence_length + 1:
        raise ValueError(
            f"Corpus demasiado corto: tiene {tokens.size} tokens, pero se "
            f"necesitan al menos {sequence_length + 1} para una ventana de "
            f"{sequence_length} entradas y {sequence_length} objetivos."
        )
    # Ventanas deslizantes de longitud T + 1, sin repetir el corpus.
    windows = np.lib.stride_tricks.sliding_window_view(tokens, sequence_length + 1)
    x = np.ascontiguousarray(windows[:, :sequence_length])
    y = np.ascontiguousarray(windows[:, 1:])
    return x, y
