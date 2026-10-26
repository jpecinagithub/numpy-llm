"""Embeddings, capa lineal y Layer Normalization.

Embeddings
----------
Indexar una matriz de embeddings equivale a seleccionar filas: si
``E`` tiene forma ``[V, D]`` y ``token_ids`` es ``[B, T]``, entonces
``E[token_ids]`` tiene forma ``[B, T, D]``. No necesitamos construir
vectores one-hot (de tamaño V, casi todo ceros) para luego
multiplicarlos por ``E``; la indexación hace exactamente esa selección.

Las posiciones se suman por broadcasting: ``P[0:T]`` tiene forma
``[T, D]`` y se suma a ``[B, T, D]`` replicándose sobre el lote. La
atención por sí sola es invariante a permutaciones (no distingue el
orden), así que sin información de posición el modelo no sabría si un
token va antes o después de otro.

Los embeddings inicialmente aleatorios todavía no contienen
representaciones semánticas aprendidas: son solo puntos de partida.

Capa lineal
------------
``y = x @ W + b`` con ``W: [input_dim, output_dim]`` y ``b:
[output_dim]``. Opera sobre la última dimensión, así que acepta tensores
con dimensiones iniciales arbitrarias (por ejemplo ``[B, T, D]``).

Layer Normalization
--------------------
Normaliza las características de cada token por separado, sobre la
última dimensión:

- ``mean = mean(x, axis=-1, keepdims=True)``
- ``variance = mean((x - mean) ** 2, axis=-1, keepdims=True)``
- ``normalized = (x - mean) / sqrt(variance + eps)``
- ``output = gamma * normalized + beta``

No mezcla posiciones ni ejemplos del lote: cada vector de dimensión D
se normaliza con su propia media y varianza. No debe confundirse con
BatchNorm, que normaliza a través del lote.
"""

from __future__ import annotations

import numpy as np


def _init_normal(rng: np.random.Generator, shape: tuple[int, ...], std: float) -> np.ndarray:
    """Matriz float32 con distribución normal (media 0, desviación std)."""
    return rng.normal(0.0, std, size=shape).astype(np.float32)


class Linear:
    """Capa lineal ``y = x @ W + b``."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Las dimensiones de una capa lineal deben ser positivas.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        # W: [input_dim, output_dim]; b: [output_dim], a cero.
        self.weight = _init_normal(rng, (input_dim, output_dim), init_std)
        self.bias = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica la capa sobre la última dimensión de ``x``."""
        x = np.asarray(x)
        if x.ndim == 0 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"La última dimensión debe ser {self.input_dim}, no {x.shape}."
            )
        return x @ self.weight + self.bias

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        """Parámetros con nombres únicos y estables."""
        return [(f"{prefix}.weight", self.weight), (f"{prefix}.bias", self.bias)]


class LayerNorm:
    """Layer Normalization sobre la última dimensión."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        if dim <= 0:
            raise ValueError("La dimensión de LayerNorm debe ser positiva.")
        self.dim = dim
        self.eps = eps
        # gamma a uno, beta a cero: al inicio no cambia la escala.
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normaliza cada vector de la última dimensión por separado."""
        x = np.asarray(x)
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(
                f"La última dimensión debe ser {self.dim}, no {x.shape}."
            )
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        return [(f"{prefix}.gamma", self.gamma), (f"{prefix}.beta", self.beta)]


class Embeddings:
    """Embeddings de tokens + embeddings de posición aprendibles.

    - Token embeddings ``E``: ``[V, D]``.
    - Position embeddings ``P``: ``[C, D]`` (posiciones aprendibles;
      en esta fase no se usan codificaciones sinusoidales ni RoPE).

    Salida: ``x = E[token_ids] + P[0:T]`` con forma ``[B, T, D]``.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
    ) -> None:
        if vocab_size <= 0 or context_length <= 0 or d_model <= 0:
            raise ValueError("vocab_size, context_length y d_model deben ser positivos.")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embedding = _init_normal(rng, (vocab_size, d_model), init_std)
        self.position_embedding = _init_normal(rng, (context_length, d_model), init_std)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Convierte ids ``[B, T]`` en representaciones ``[B, T, D]``."""
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids debe tener forma [B, T], no {token_ids.shape}.")
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError("El lote y la longitud de secuencia deben ser mayores que cero.")
        if token_ids.dtype.kind not in "iu":
            raise ValueError("Los identificadores de token deben ser enteros.")
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            raise ValueError(f"Todos los tokens deben estar en [0, {self.vocab_size}).")
        seq_len = token_ids.shape[1]
        if seq_len > self.context_length:
            raise ValueError(
                f"La secuencia tiene longitud {seq_len}, superior al contexto "
                f"máximo {self.context_length}."
            )
        token_part = self.token_embedding[token_ids]  # [B, T, D]
        # P[0:T] es [T, D]; broadcasting lo suma a cada ejemplo del lote.
        position_part = self.position_embedding[:seq_len]  # [T, D]
        return token_part + position_part

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        return [
            (f"{prefix}.token_embedding", self.token_embedding),
            (f"{prefix}.position_embedding", self.position_embedding),
        ]
