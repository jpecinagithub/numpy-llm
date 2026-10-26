"""Red feed-forward y bloque Transformer (arquitectura pre-norm).

Feed-forward (MLP por posición)::

    hidden = relu(x @ W1 + b1)   # expansión D -> F
    output = hidden @ W2 + b2    # reducción F -> D

La expansión permite representar más combinaciones de características
que la dimensión D original; la reducción devuelve a D para poder sumar
el residuo. Sin la activación no lineal, dos capas lineales seguidas
equivaldrían a una sola capa lineal y la red perdería expresividad.
Los mismos parámetros se usan en todas las posiciones, y esta MLP no
intercambia directamente información entre tokens: ese intercambio lo
hace la atención.

Bloque Transformer (pre-norm)::

    h = x + attention(layer_norm_1(x))
    output = h + feed_forward(layer_norm_2(h))

Las conexiones residuales (``x + ...``) permiten que la información
fluya directamente a través del bloque sin pasar por la transformación;
más adelante, en entrenamiento, también facilitan el flujo de
gradientes. "Pre-norm" significa que la normalización se aplica antes
de cada subcapa, no después.
"""

from __future__ import annotations

import numpy as np

from .attention import CausalMultiHeadAttention
from .functional import relu
from .layers import LayerNorm, Linear


class FeedForward:
    """MLP aplicada independientemente a cada posición."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
    ) -> None:
        self.fc1 = Linear(d_model, d_ff, rng, init_std)  # W1: [D, F]
        self.fc2 = Linear(d_ff, d_model, rng, init_std)  # W2: [F, D]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """``relu(x @ W1 + b1) @ W2 + b2``."""
        hidden = relu(self.fc1.forward(x))
        return self.fc2.forward(hidden)

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        params: list[tuple[str, np.ndarray]] = []
        params += self.fc1.named_parameters(f"{prefix}.fc1")
        params += self.fc2.named_parameters(f"{prefix}.fc2")
        return params


class TransformerBlock:
    """Un bloque Transformer con atención causal y feed-forward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        self.ln1 = LayerNorm(d_model, layer_norm_eps)
        self.attention = CausalMultiHeadAttention(d_model, n_heads, rng, init_std)
        self.ln2 = LayerNorm(d_model, layer_norm_eps)
        self.feed_forward = FeedForward(d_model, d_ff, rng, init_std)

    def forward(
        self, x: np.ndarray, return_attention_weights: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Aplica el bloque pre-norm completo."""
        attended, weights = self.attention.forward(
            self.ln1.forward(x), return_attention_weights=True
        )
        h = x + attended
        output = h + self.feed_forward.forward(self.ln2.forward(h))
        if return_attention_weights:
            return output, weights
        return output

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        """Parámetros del bloque con nombres únicos y estables."""
        params: list[tuple[str, np.ndarray]] = []
        params += self.ln1.named_parameters(f"{prefix}.ln1")
        params += self.attention.named_parameters(f"{prefix}.attention")
        params += self.ln2.named_parameters(f"{prefix}.ln2")
        params += self.feed_forward.named_parameters(f"{prefix}.feed_forward")
        return params
