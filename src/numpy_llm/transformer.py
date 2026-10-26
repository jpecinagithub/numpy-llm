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
from .layers import LayerNorm, Linear, _require_cache, _store_cache


class FeedForward:
    """MLP por posición ``relu(x @ W1 + b1) @ W2 + b2`` con backward.

    Backward en orden inverso: ``dA1 = fc2.backward(dY)``,
    ``dZ1 = dA1 * (Z1 > 0)`` (derivada en 0 definida como 0),
    ``dX = fc1.backward(dZ1)``. Se actualizan TODOS los parámetros de
    la MLP, no solo los de la segunda capa. La máscara de ReLU se
    guarda en el forward de entrenamiento; ReLU no es diferenciable
    exactamente en 0, por lo que las pruebas de diferencias finitas
    deben evitar ese punto.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.fc1 = Linear(d_model, d_ff, rng, init_std, self.dtype)  # W1: [D, F]
        self.fc2 = Linear(d_ff, d_model, rng, init_std, self.dtype)  # W2: [F, D]
        self._cache: dict | None = None

    def forward(self, x: np.ndarray, retain_cache: bool = False) -> np.ndarray:
        """``relu(x @ W1 + b1) @ W2 + b2``."""
        if retain_cache and self._cache is not None:
            raise ValueError(
                "FeedForward: ya hay un forward de entrenamiento pendiente. "
                "Llama a backward() o a clear_cache() antes de otro forward "
                "con retain_cache=True."
            )
        z1 = self.fc1.forward(x, retain_cache=retain_cache)
        hidden = relu(z1)
        if retain_cache:
            self._cache = {"relu_mask": z1 > 0}
        return self.fc2.forward(hidden, retain_cache=retain_cache)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Devuelve dX. Consume los intermediarios."""
        cache = _require_cache("FeedForward", self._cache)
        self._cache = None
        d_a1 = self.fc2.backward(dY)
        d_z1 = d_a1 * cache["relu_mask"]
        return self.fc1.backward(d_z1)

    def zero_grad(self) -> None:
        self.fc1.zero_grad()
        self.fc2.zero_grad()

    def clear_cache(self) -> None:
        self._cache = None
        self.fc1.clear_cache()
        self.fc2.clear_cache()

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        params: list[tuple[str, np.ndarray]] = []
        params += self.fc1.named_parameters(f"{prefix}.fc1")
        params += self.fc2.named_parameters(f"{prefix}.fc2")
        return params

    def named_gradients(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        grads: list[tuple[str, np.ndarray]] = []
        grads += self.fc1.named_gradients(f"{prefix}.fc1")
        grads += self.fc2.named_gradients(f"{prefix}.fc2")
        return grads


class TransformerBlock:
    """Un bloque Transformer pre-norm con backward completo.

    Forward::

        A = LN1(X); B = Attention(A); H = X + B
        C = LN2(H); D_ff = FFN(C); Y = H + D_ff

    Backward (cada suma residual reparte el gradiente por sus dos
    ramas; ninguna contribución sobrescribe a otra)::

        dD_ff = dY; dC = FFN.backward(dD_ff)
        dH = dY + LN2.backward(dC)
        dB = dH; dA = Attention.backward(dB)
        dX = dH + LN1.backward(dA)

    No se hacen operaciones in-place sobre activaciones compartidas:
    las sumas crean arrays nuevos, así que modificar una rama no
    altera el gradiente que necesita la otra.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
        layer_norm_eps: float = 1e-5,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.ln1 = LayerNorm(d_model, layer_norm_eps, self.dtype)
        self.attention = CausalMultiHeadAttention(d_model, n_heads, rng, init_std, self.dtype)
        self.ln2 = LayerNorm(d_model, layer_norm_eps, self.dtype)
        self.feed_forward = FeedForward(d_model, d_ff, rng, init_std, self.dtype)
        self._cache: dict | None = None

    def forward(
        self,
        x: np.ndarray,
        retain_cache: bool = False,
        return_attention_weights: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Aplica el bloque pre-norm completo."""
        if retain_cache and self._cache is not None:
            raise ValueError(
                "TransformerBlock: ya hay un forward de entrenamiento "
                "pendiente. Llama a backward() o a clear_cache() antes de "
                "otro forward con retain_cache=True."
            )
        x = np.asarray(x)
        a = self.ln1.forward(x, retain_cache=retain_cache)
        attended, weights = self.attention.forward(
            a, retain_cache=retain_cache, return_attention_weights=True
        )
        h = x + attended
        c = self.ln2.forward(h, retain_cache=retain_cache)
        d_ff = self.feed_forward.forward(c, retain_cache=retain_cache)
        output = h + d_ff
        if retain_cache:
            # Solo se necesita saber que el forward de entrenamiento
            # existe; los intermediarios viven en cada subcomponente.
            # Guardar las formas permite validar dY en backward.
            self._cache = {"shape": x.shape}
        if return_attention_weights:
            return output, weights
        return output

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Devuelve dX siguiendo el orden inverso del forward."""
        cache = _require_cache("TransformerBlock", self._cache)
        self._cache = None
        dY = np.asarray(dY)
        if dY.shape != cache["shape"]:
            raise ValueError(f"dY debe tener forma {cache['shape']}, no {dY.shape}.")
        d_dff = dY
        d_c = self.feed_forward.backward(d_dff)
        d_h = dY + self.ln2.backward(d_c)
        d_b = d_h
        d_a = self.attention.backward(d_b)
        d_x = d_h + self.ln1.backward(d_a)
        return d_x

    def zero_grad(self) -> None:
        self.ln1.zero_grad()
        self.attention.zero_grad()
        self.ln2.zero_grad()
        self.feed_forward.zero_grad()

    def clear_cache(self) -> None:
        self._cache = None
        self.ln1.clear_cache()
        self.attention.clear_cache()
        self.ln2.clear_cache()
        self.feed_forward.clear_cache()

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        """Parámetros del bloque con nombres únicos y estables."""
        params: list[tuple[str, np.ndarray]] = []
        params += self.ln1.named_parameters(f"{prefix}.ln1")
        params += self.attention.named_parameters(f"{prefix}.attention")
        params += self.ln2.named_parameters(f"{prefix}.ln2")
        params += self.feed_forward.named_parameters(f"{prefix}.feed_forward")
        return params

    def named_gradients(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        grads: list[tuple[str, np.ndarray]] = []
        grads += self.ln1.named_gradients(f"{prefix}.ln1")
        grads += self.attention.named_gradients(f"{prefix}.attention")
        grads += self.ln2.named_gradients(f"{prefix}.ln2")
        grads += self.feed_forward.named_gradients(f"{prefix}.feed_forward")
        return grads
