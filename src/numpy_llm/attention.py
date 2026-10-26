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

from .functional import softmax, softmax_backward
from .layers import Linear, _require_cache, _store_cache


class CausalMultiHeadAttention:
    """Atención causal multicabeza (self-attention) con backward manual.

    Formas: Q, K, V_values: [B, H, T, Dh]; S, A: [B, H, T, T];
    R (contexto): [B, H, T, Dh].

    Backward, en orden:
    a) proyección de salida (Linear.backward) -> gradiente de las
       cabezas concatenadas y gradientes de Wo, bo.
    b) separación inversa de cabezas -> dR [B, H, T, Dh].
    c) ``dA = dR @ V^T``; ``dV = A^T @ dR``.
    d) softmax: ``dS_masked = A * (dA - sum(dA * A))``.
    e) máscara causal: gradiente cero en posiciones prohibidas. Como
       A ya es exactamente 0 ahí, dS_masked ya es 0 ahí; el cero
       explícito evita además cualquier NaN procedente de los -inf
       de los puntajes enmascarados (nunca se opera con ellos: el
       backward usa A, no S_masked).
    f) ``dQ = (dS @ K) / sqrt(Dh)``; ``dK = (dS^T @ Q) / sqrt(Dh)``.
    g) backward de las tres proyecciones; la entrada original alimentó
       tres ramas, así que ``dX = dX_Q + dX_K + dX_V`` (suma, nunca
       sobrescritura).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) debe ser positivo y divisible entre n_heads ({n_heads})."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dtype = np.dtype(dtype)
        self.q_proj = Linear(d_model, d_model, rng, init_std, self.dtype)
        self.k_proj = Linear(d_model, d_model, rng, init_std, self.dtype)
        self.v_proj = Linear(d_model, d_model, rng, init_std, self.dtype)
        self.out_proj = Linear(d_model, d_model, rng, init_std, self.dtype)
        self._cache: dict | None = None

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
        self,
        x: np.ndarray,
        retain_cache: bool = False,
        return_attention_weights: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Calcula la atención causal."""
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"La entrada debe tener forma [B, T, {self.d_model}], no {x.shape}.")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("El lote y la longitud de secuencia deben ser mayores que cero.")
        batch, seq_len, _ = x.shape
        # Propagar retain_cache a las proyecciones: si este forward es
        # de entrenamiento, cada Linear guarda su entrada; si es de
        # inferencia, ninguno guarda nada (no se sobrescribe caché).
        # El control de "pendiente" se hace aquí, antes de llamarlas,
        # para no dejar cachés parciales si ya había una pendiente.
        if retain_cache and self._cache is not None:
            raise ValueError(
                "CausalMultiHeadAttention: ya hay un forward de entrenamiento "
                "pendiente. Llama a backward() o a clear_cache() antes de un "
                "nuevo forward con retain_cache=True."
            )
        q = self._split_heads(self.q_proj.forward(x, retain_cache=retain_cache))
        k = self._split_heads(self.k_proj.forward(x, retain_cache=retain_cache))
        v_values = self._split_heads(self.v_proj.forward(x, retain_cache=retain_cache))
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(float(self.head_dim))
        positions = np.arange(seq_len)
        causal_mask = positions[None, :] > positions[:, None]  # [T, T], True = futuro
        scores = np.where(causal_mask, -np.inf, scores)
        attention_weights = softmax(scores, axis=-1)
        context = attention_weights @ v_values
        merged = self._merge_heads(context)
        output = self.out_proj.forward(merged, retain_cache=retain_cache)
        if retain_cache:
            self._cache = {
                "q": q, "k": k, "v_values": v_values,
                "attention_weights": attention_weights,
                "causal_mask": causal_mask,
            }
        if return_attention_weights:
            return output, attention_weights
        return output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """Devuelve dX. Consume los intermediarios y los de las proyecciones."""
        cache = _require_cache("CausalMultiHeadAttention", self._cache)
        self._cache = None
        q = cache["q"]; k = cache["k"]; v_values = cache["v_values"]
        attention_weights = cache["attention_weights"]
        causal_mask = cache["causal_mask"]
        scale = np.sqrt(float(self.head_dim))
        # a) Proyección de salida.
        d_merged = self.out_proj.backward(d_output)  # [B, T, D]
        # b) Separación inversa de cabezas: invertir merge exactamente.
        # merge: [B,H,T,Dh] -> transpose(0,2,1,3) -> reshape [B,T,D].
        # Inversa: reshape [B,T,H,Dh] -> transpose(0,2,1,3).
        batch, seq_len, _ = d_merged.shape
        d_context = d_merged.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # c) Producto atención-valores.
        d_attention = d_context @ v_values.transpose(0, 1, 3, 2)  # [B,H,T,T]
        d_v = attention_weights.transpose(0, 1, 3, 2) @ d_context  # [B,H,T,Dh]
        # d) Softmax.
        d_scores_masked = softmax_backward(attention_weights, d_attention)
        # e) Máscara causal: cero en posiciones prohibidas.
        d_scores = np.where(causal_mask, 0, d_scores_masked)
        # f) Producto queries-keys (con el factor de escalado).
        d_q = (d_scores @ k) / scale
        d_k = (d_scores.transpose(0, 1, 3, 2) @ q) / scale
        # g) Proyecciones Q, K, V: invertir split (reshape+transpose
        # inversos) y sumar las tres contribuciones a la misma entrada.
        d_x_q = self.q_proj.backward(self._merge_heads(d_q))
        d_x_k = self.k_proj.backward(self._merge_heads(d_k))
        d_x_v = self.v_proj.backward(self._merge_heads(d_v))
        return d_x_q + d_x_k + d_x_v

    def zero_grad(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.zero_grad()

    def clear_cache(self) -> None:
        self._cache = None
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.clear_cache()

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        """Parámetros con nombres únicos y estables."""
        params: list[tuple[str, np.ndarray]] = []
        params += self.q_proj.named_parameters(f"{prefix}.q_proj")
        params += self.k_proj.named_parameters(f"{prefix}.k_proj")
        params += self.v_proj.named_parameters(f"{prefix}.v_proj")
        params += self.out_proj.named_parameters(f"{prefix}.out_proj")
        return params

    def named_gradients(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        grads: list[tuple[str, np.ndarray]] = []
        grads += self.q_proj.named_gradients(f"{prefix}.q_proj")
        grads += self.k_proj.named_gradients(f"{prefix}.k_proj")
        grads += self.v_proj.named_gradients(f"{prefix}.v_proj")
        grads += self.out_proj.named_gradients(f"{prefix}.out_proj")
        return grads
