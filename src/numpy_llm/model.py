"""Modelo completo: Transformer decoder-only.

Recorrido (forward)::

    token_ids                     [B, T]
    -> embeddings de tokens + posiciones   [B, T, D]
    -> bloque Transformer 1                [B, T, D]
    -> ...
    -> bloque Transformer L                [B, T, D]
    -> LayerNorm final                     [B, T, D]
    -> proyección al vocabulario           [B, T, V]  (logits)

La proyección final es ``logits = normalized_hidden @ W_vocab +
b_vocab`` con ``W_vocab: [D, V]`` y ``b_vocab: [V]``. En esta fase NO
se comparten los pesos de ``W_vocab`` con los embeddings de tokens.

El forward devuelve logits, sin aplicar softmax final de forma
obligatoria. Se separan claramente tres cosas:

- Cálculo de logits (aquí).
- Cálculo de la pérdida (``functional.cross_entropy_loss``).
- Selección del siguiente token durante la generación
  (``generation.generate``).

Inicialización
--------------
Una única instancia del generador de inicialización se consume
progresivamente al crear las capas (embeddings, bloques en orden y
proyección final). Nunca se reinicia la misma semilla dentro de cada
capa: si todas las capas empezaran con la misma semilla, empezarían
idénticas y el modelo perdería diversidad.

Esta inicialización (normal, media 0, desviación ``init_std``) es una
elección sencilla para la demostración, no una receta universal para
entrenar modelos grandes.
"""

from __future__ import annotations

import numpy as np

from .config import ModelConfig
from .layers import Embeddings, LayerNorm, Linear
from .transformer import TransformerBlock


class TransformerLanguageModel:
    """Pequeño modelo de lenguaje Transformer decoder-only."""

    def __init__(self, config: ModelConfig) -> None:
        if config.vocab_size is None:
            raise ValueError(
                "ModelConfig.vocab_size debe estar fijado (calcúlalo a partir "
                "del tokenizador con config.with_vocab_size(...)) antes de "
                "crear el modelo."
            )
        self.config = config
        # Generador explícito: no se usa el generador global de NumPy.
        rng = np.random.default_rng(config.seed)
        self.embeddings = Embeddings(
            config.vocab_size, config.context_length, config.d_model, rng, config.init_std
        )
        self.blocks = [
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                rng,
                config.init_std,
                config.layer_norm_eps,
            )
            for _ in range(config.n_layers)
        ]
        self.final_ln = LayerNorm(config.d_model, config.layer_norm_eps)
        self.lm_head = Linear(config.d_model, config.vocab_size, rng, config.init_std)

    def forward(
        self, token_ids: np.ndarray, return_attention_weights: bool = False
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """Calcula los logits para una entrada ``[B, T]``.

        Valida forma, tipo entero, rango de tokens, B y T mayores que
        cero y T dentro del contexto máximo. No modifica los parámetros.
        """
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"La entrada debe tener forma [B, T], no {token_ids.shape}.")
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError("B y T deben ser mayores que cero.")
        if token_ids.dtype.kind not in "iu":
            raise ValueError("Los identificadores de token deben ser enteros.")
        if np.any(token_ids < 0) or np.any(token_ids >= self.config.vocab_size):  # type: ignore[operator]
            raise ValueError(
                f"Todos los tokens deben estar en [0, {self.config.vocab_size})."
            )
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError(
                f"T ({token_ids.shape[1]}) supera context_length "
                f"({self.config.context_length})."
            )
        hidden = self.embeddings.forward(token_ids)  # [B, T, D]
        all_weights: list[np.ndarray] = []
        for block in self.blocks:
            hidden, weights = block.forward(hidden, return_attention_weights=True)
            all_weights.append(weights)
        hidden = self.final_ln.forward(hidden)  # [B, T, D]
        logits = self.lm_head.forward(hidden)  # [B, T, V]
        if return_attention_weights:
            return logits, all_weights
        return logits

    def named_parameters(self) -> list[tuple[str, np.ndarray]]:
        """Todos los parámetros con nombres únicos y estables."""
        params: list[tuple[str, np.ndarray]] = []
        params += self.embeddings.named_parameters("embeddings")
        for i, block in enumerate(self.blocks):
            params += block.named_parameters(f"blocks.{i}")
        params += self.final_ln.named_parameters("final_ln")
        params += self.lm_head.named_parameters("lm_head")
        return params

    def count_parameters(self) -> int:
        """Número total de parámetros, sumando el tamaño real de los arrays."""
        return int(sum(array.size for _, array in self.named_parameters()))
