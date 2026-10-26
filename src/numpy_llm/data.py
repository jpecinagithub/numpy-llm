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

import hashlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .tokenizer import CharTokenizer


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


# ---------------------------------------------------------------------------
# Fase 2: partición train/validation, muestreo y ventanas de evaluación.
# ---------------------------------------------------------------------------

def corpus_sha256(text: str) -> str:
    """Huella SHA-256 del texto UTF-8 (identidad del corpus)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unk_rate(token_ids: Sequence[int]) -> float:
    """Proporción de <UNK> (id 0) en una secuencia de tokens."""
    ids = np.asarray(list(token_ids), dtype=np.int64)
    if ids.size == 0:
        return 0.0
    return float(np.mean(ids == 0))


@dataclass
class SplitData:
    """Partición determinista de un único archivo de texto.

    La partición se hace SOBRE EL TEXTO, antes de construir ventanas:
    ninguna ventana cruza la frontera. El vocabulario se construye
    EXCLUSIVAMENTE con train; validation se codifica con ese mismo
    vocabulario fijo, y sus caracteres ausentes en train se convierten
    en <UNK>. Una partición sin índices compartidos no garantiza
    independencia semántica si el corpus contiene fragmentos duplicados.
    """

    train_text: str
    val_text: str
    train_tokens: np.ndarray  # [N_train]
    val_tokens: np.ndarray  # [N_val]
    tokenizer: CharTokenizer
    train_fraction: float
    split_char_index: int
    sha256: str
    train_unk_rate: float
    val_unk_rate: float

    def to_dict(self) -> dict:
        return {
            "train_fraction": self.train_fraction,
            "split_char_index": self.split_char_index,
            "train_chars": len(self.train_text),
            "val_chars": len(self.val_text),
            "train_tokens": int(self.train_tokens.size),
            "val_tokens": int(self.val_tokens.size),
            "vocab_size": self.tokenizer.vocab_size,
            "sha256": self.sha256,
            "train_unk_rate": self.train_unk_rate,
            "val_unk_rate": self.val_unk_rate,
        }


def split_corpus(text: str, train_fraction: float = 0.9, sequence_length: int | None = None) -> SplitData:
    """Divide el texto en train/val y codifica con vocabulario de train.

    Raises:
        ValueError: si el texto está vacío, la fracción no está en
            (0, 1), o alguna partición tiene menos de
            ``sequence_length + 1`` tokens (cuando se indica). En ese
            caso se indica el tamaño mínimo necesario; nunca se copia
            train a validation, ni se repite el texto, ni se reduce el
            contexto sin comunicarlo.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("El corpus debe ser un texto no vacío.")
    if not isinstance(train_fraction, (int, float)) or not (0.0 < float(train_fraction) < 1.0):
        raise ValueError(f"train_fraction debe estar en (0, 1), no {train_fraction!r}.")
    cut = int(len(text) * float(train_fraction))
    if cut == 0 or cut >= len(text):
        raise ValueError(
            f"La partición con train_fraction={train_fraction} deja una "
            f"partición vacía para un texto de {len(text)} caracteres."
        )
    train_text, val_text = text[:cut], text[cut:]
    tokenizer = CharTokenizer.from_text(train_text)
    train_tokens = np.asarray(tokenizer.encode(train_text), dtype=np.int64)
    val_tokens = np.asarray(tokenizer.encode(val_text), dtype=np.int64)
    if sequence_length is not None:
        needed = sequence_length + 1
        for name, tokens in (("train", train_tokens), ("validation", val_tokens)):
            if tokens.size < needed:
                raise ValueError(
                    f"La partición {name} tiene {tokens.size} tokens, pero se "
                    f"necesitan al menos {needed} (sequence_length + 1) para "
                    f"una ventana. Corpus total: {len(text)} caracteres; "
                    f"partición {name}: {tokens.size} tokens. No se copia "
                    f"train a validation ni se repite el texto."
                )
    return SplitData(
        train_text=train_text,
        val_text=val_text,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        tokenizer=tokenizer,
        train_fraction=float(train_fraction),
        split_char_index=cut,
        sha256=corpus_sha256(text),
        train_unk_rate=unk_rate(train_tokens),
        val_unk_rate=unk_rate(val_tokens),
    )


def sample_batch(
    tokens: np.ndarray, batch_size: int, sequence_length: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Un lote de entrenamiento con teacher forcing.

    Para una secuencia de N tokens, un inicio válido cumple
    ``0 <= start <= N - T - 1``. Se eligen B inicios uniformemente,
    CON reemplazo, con el generador indicado (``Generator.integers``
    tiene el límite superior exclusivo, así que el límite es N - T).
    Teacher forcing: las entradas contienen tokens reales del corpus
    y los objetivos son esos mismos tokens desplazados una posición;
    nunca se introducen predicciones del modelo como entradas, y la
    máscara causal impide que una posición consulte su objetivo futuro.

    Returns:
        ``(x, y)`` con ``x = tokens[start:start+T]`` e
        ``y = tokens[start+1:start+T+1]``, formas ``[B, T]``.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size y sequence_length deben ser positivos.")
    if tokens.size < sequence_length + 1:
        raise ValueError(
            f"Se necesitan al menos {sequence_length + 1} tokens para "
            f"muestrear, pero hay {tokens.size}."
        )
    starts = rng.integers(0, tokens.size - sequence_length, size=batch_size)
    x = np.stack([tokens[s : s + sequence_length] for s in starts])
    y = np.stack([tokens[s + 1 : s + sequence_length + 1] for s in starts])
    return x, y


def eval_windows(
    tokens: np.ndarray, sequence_length: int, batch_size: int = 4, limit: int | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray, int, int]]:
    """Ventanas de evaluación con stride T, sin solapamiento de objetivos.

    Los inicios son ``0, T, 2T, ...``; cada ventana necesita T + 1
    tokens. Los tokens objetivo de ventanas consecutivas no se cuentan
    repetidamente. El fragmento final que no completa una ventana se
    omite (se informa cuántos tokens quedaron fuera). El último lote
    puede tener menos ejemplos.

    Yields:
        ``(x, y, evaluated_targets, leftover_tokens)`` por lote, donde
        los dos últimos valores son acumulados hasta ese lote (totales
        parciales): objetivos evaluados y tokens no evaluados al final
        de la partición.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if sequence_length <= 0 or batch_size <= 0:
        raise ValueError("sequence_length y batch_size deben ser positivos.")
    starts = list(range(0, tokens.size - sequence_length, sequence_length))
    if limit is not None:
        starts = starts[:limit]
    total_windows = len(starts)
    evaluated = 0
    # Tokens distintos cubiertos por las ventanas: windows*T + 1 (las
    # ventanas comparten el token frontera). El resto queda fuera.
    leftover = int(tokens.size - (total_windows * sequence_length + 1)) if total_windows else int(tokens.size)
    for i in range(0, total_windows, batch_size):
        chunk = starts[i : i + batch_size]
        x = np.stack([tokens[s : s + sequence_length] for s in chunk])
        y = np.stack([tokens[s + 1 : s + sequence_length + 1] for s in chunk])
        evaluated += int(y.size)
        yield x, y, evaluated, leftover


def eval_window_starts(tokens: np.ndarray, sequence_length: int, limit: int | None = None) -> list[int]:
    """Inicios de ventanas de evaluación (para guardarlos en checkpoints)."""
    tokens = np.asarray(tokens, dtype=np.int64)
    starts = list(range(0, tokens.size - sequence_length, sequence_length))
    return starts[:limit] if limit is not None else starts
