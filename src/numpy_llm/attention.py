"""Atención causal multicabeza, con operaciones vectorizadas de NumPy.

Entrada: ``x`` con forma ``[B, T, D]``.

Proyecciones (cada W es ``[D, D]`` y cada bias es ``[D]``)::

    Q = x @ Wq + bq
    K = x @ Wk + bk
    V_values = x @ Wv + bv

``V_values`` se llama así para no confundirlo con V, el tamaño del
vocabulario.

Las cabezas se separan con reshape y transpose::

    [B, T, D] -> [B, H, T, Dh]

Puntajes::

    scores = Q @ transpose(K, last_two_axes) / sqrt(Dh)   # [B, H, T, T]

Qué representan Query, Key y Value
-----------------------------------
Son analogías útiles, no funciones lingüísticas programadas a mano:
Query es "lo que busca" cada posición, Key es "lo que ofrece" cada
posición y Value es "el contenido" que se recupera. El modelo aprende
las proyecciones; nosotros no escribimos reglas del tipo "el sujeto
atiende al verbo".

Por qué dividir entre sqrt(Dh)
-------------------------------
Sin escalar, la varianza de los productos escalares crece con Dh y los
puntajes se vuelven muy grandes; softmax se satura (casi todo 0 o 1) y
más adelante los gradientes se desvanecen. Dividir entre sqrt(Dh)
mantiene los puntajes en una escala razonable.

Máscara causal
--------------
La posición i puede atender a j <= i, pero no a j > i. Los puntajes
prohibidos se sustituyen por -inf ANTES de softmax: después de softmax
se convierten en probabilidad exactamente 0. Si se enmascarara después,
las probabilidades ya no sumarían 1 correctamente. Cada fila conserva
al menos su diagonal, así que ninguna fila queda completamente
enmascarada.

Softmax se aplica sobre el eje de las claves (último eje de scores):
cada fila es una distribución sobre "a qué posiciones pasadas miro".

Por qué varias cabezas y una proyección final
----------------------------------------------
Cada cabeza tiene sus propias proyecciones Q, K, V, así que aunque
partan de la misma entrada aprenden relaciones distintas (una puede
fijarse en el token anterior, otra en una estructura más lejana, etc.).
Al concatenarlas se obtiene de nuevo dimensión D, y la proyección de
salida Wo mezcla la información de todas las cabezas en una única
representación.

No se usan bucles de Python sobre tokens ni cabezas: todo el cálculo es
vectorizado con reshape, transpose y multiplicaciones de matrices.
"""

from __future__ import annotations

import numpy as np

from .functional import softmax
from .layers import Linear


class CausalMultiHeadAttention:
    """Atención causal multicabeza (self-attention)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
    ) -> None:
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) debe ser positivo y divisible entre n_heads ({n_heads})."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Se consumen en un orden fijo: Q, K, V, salida.
        self.q_proj = Linear(d_model, d_model, rng, init_std)
        self.k_proj = Linear(d_model, d_model, rng, init_std)
        self.v_proj = Linear(d_model, d_model, rng, init_std)
        self.out_proj = Linear(d_model, d_model, rng, init_std)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """[B, T, D] -> [B, H, T, Dh]."""
        batch, seq_len, _ = x.shape
        reshaped = x.reshape(batch, seq_len, self.n_heads, self.head_dim)
        return reshaped.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """[B, H, T, Dh] -> [B, T, D]."""
        batch, _, seq_len, _ = x.shape
        transposed = x.transpose(0, 2, 1, 3)
        return transposed.reshape(batch, seq_len, self.d_model)

    def forward(
        self, x: np.ndarray, return_attention_weights: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Calcula la atención causal.

        Args:
            x: Activaciones de forma ``[B, T, D]``.
            return_attention_weights: si es True, devuelve también los
                pesos de atención ``[B, H, T, T]``. En uso normal es
                False: los pesos no se conservan permanentemente.

        Returns:
            La salida ``[B, T, D]`` o, si se pide inspección, la tupla
            ``(salida, pesos_de_atencion)``.
        """
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"La entrada debe tener forma [B, T, {self.d_model}], no {x.shape}.")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("El lote y la longitud de secuencia deben ser mayores que cero.")
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj.forward(x))  # [B, H, T, Dh]
        k = self._split_heads(self.k_proj.forward(x))  # [B, H, T, Dh]
        v_values = self._split_heads(self.v_proj.forward(x))  # [B, H, T, Dh]

        # Puntajes: [B, H, T, T]; el último eje recorre las claves.
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(float(self.head_dim))

        # Máscara causal: True donde j > i (futuro, prohibido).
        positions = np.arange(seq_len)
        causal_mask = positions[None, :] > positions[:, None]  # [T, T]
        # Broadcasting a [B, H, T, T]; -inf antes de softmax -> 0 después.
        scores = np.where(causal_mask, -np.inf, scores)

        attention_weights = softmax(scores, axis=-1)  # [B, H, T, T]
        context = attention_weights @ v_values  # [B, H, T, Dh]
        merged = self._merge_heads(context)  # [B, T, D]
        output = self.out_proj.forward(merged)  # [B, T, D]

        if return_attention_weights:
            return output, attention_weights
        return output

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        """Parámetros con nombres únicos y estables."""
        params: list[tuple[str, np.ndarray]] = []
        params += self.q_proj.named_parameters(f"{prefix}.q_proj")
        params += self.k_proj.named_parameters(f"{prefix}.k_proj")
        params += self.v_proj.named_parameters(f"{prefix}.v_proj")
        params += self.out_proj.named_parameters(f"{prefix}.out_proj")
        return params
