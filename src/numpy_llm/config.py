"""Configuración del modelo.

Notación usada en todo el proyecto:

- B: tamaño del lote.
- T: longitud de la secuencia actual.
- C: longitud máxima de contexto (``context_length``).
- V: tamaño del vocabulario (``vocab_size``).
- D: dimensión del embedding (``d_model``).
- H: número de cabezas de atención (``n_heads``).
- Dh: dimensión de cada cabeza, D / H.
- F: dimensión interna de la red feed-forward (``d_ff``).
- L: número de bloques Transformer (``n_layers``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    """Hiperparámetros del modelo.

    Los hiperparámetros son decisiones de diseño elegidas antes de crear el
    modelo; no se aprenden. Los parámetros (pesos, biases, gammas, betas) son
    los arrays que sí se aprenderían en una fase de entrenamiento.

    ``vocab_size`` se calcula a partir del tokenizador. Puede dejarse en
    ``None`` mientras se construye la configuración y fijarse después con
    :meth:`with_vocab_size`; el modelo exige un valor positivo.
    """

    context_length: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02
    seed: int = 42
    vocab_size: int | None = None
    # Dtype de todos los parámetros y activaciones. "float32" es el
    # comportamiento de la fase 1; "float64" es obligatorio para las
    # comprobaciones numéricas de gradientes (fase 2), porque ningún
    # cast interno a float32 debe invalidarlas silenciosamente.
    dtype: str = "float32"

    def __post_init__(self) -> None:
        """Valida la configuración en cuanto se construye."""
        for name in ("context_length", "d_model", "n_heads", "n_layers", "d_ff"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} debe ser un entero positivo, no {value!r}.")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) debe ser divisible entre "
                f"n_heads ({self.n_heads})."
            )
        for name in ("layer_norm_eps", "init_std"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} debe ser un número positivo y finito, no {value!r}.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed debe ser un entero, no {self.seed!r}.")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"dtype debe ser 'float32' o 'float64', no {self.dtype!r}.")
        if self.vocab_size is not None:
            if not isinstance(self.vocab_size, int) or isinstance(self.vocab_size, bool) or self.vocab_size <= 0:
                raise ValueError(f"vocab_size debe ser un entero positivo, no {self.vocab_size!r}.")

    @property
    def head_dim(self) -> int:
        """Dh: dimensión de cada cabeza de atención (D / H)."""
        return self.d_model // self.n_heads

    def with_vocab_size(self, vocab_size: int) -> "ModelConfig":
        """Devuelve una copia de la configuración con el vocabulario fijado."""
        data = asdict(self)
        data["vocab_size"] = vocab_size
        return ModelConfig(**data)

    def to_dict(self) -> dict:
        """Representación JSON-compatible de la configuración."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Reconstruye la configuración desde :meth:`to_dict`."""
        if not isinstance(data, dict):
            raise ValueError("La configuración guardada debe ser un objeto JSON.")
        allowed = {
            "context_length",
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "layer_norm_eps",
            "init_std",
            "seed",
            "vocab_size",
            "dtype",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Campos desconocidos en la configuración: {sorted(unknown)}.")
        # "dtype" es opcional por compatibilidad con configuraciones
        # guardadas en la fase 1 (siempre float32); si falta, se usa
        # el valor por defecto en lugar de rechazar el archivo.
        missing = (allowed - {"dtype"}) - set(data)
        if missing:
            raise ValueError(f"Campos ausentes en la configuración: {sorted(missing)}.")
        data = dict(data)
        data.setdefault("dtype", "float32")
        return cls(**data)
