"""Embeddings, capa lineal y Layer Normalization, con backward manual.

Contrato de intermediarios (fase 2, común a todos los componentes)
------------------------------------------------------------------
- ``forward(..., retain_cache=False)``: con ``retain_cache=False``
  (inferencia) no crea ni sobrescribe intermediarios de entrenamiento.
  Con ``retain_cache=True`` guarda solo lo necesario para ``backward``
  y rechaza con ``ValueError`` explicativo si ya hay un forward de
  entrenamiento pendiente (no lo sobrescribe silenciosamente).
- ``backward(dY)``: consume los intermediarios (después se liberan) y
  ACUMULA los gradientes de parámetros (``grad += contribución``).
  Sin un forward válido previo, o tras un backward ya consumido,
  produce un error explicativo. No modifica los pesos.
- ``clear_cache()``: libera los intermediarios sin tocar gradientes.
- ``zero_grad()``: pone a cero los buffers de gradiente.
- Cada gradiente es un array separado, con la misma forma y dtype que
  su parámetro, inicializado a cero.

Forward (fase 1, sin cambios de arquitectura)
---------------------------------------------
Embeddings: indexar ``E`` selecciona filas; las posiciones se suman
por broadcasting. Capa lineal: ``y = x @ W + b`` sobre la última
dimensión. LayerNorm: normaliza cada vector de dimensión D por
separado (no mezcla posiciones ni ejemplos del lote).
"""

from __future__ import annotations

import numpy as np


def _init_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    std: float,
    dtype: np.dtype | str = np.float32,
) -> np.ndarray:
    """Matriz con distribución normal (media 0, desviación std).

    El dtype se propaga explícitamente: un modelo float64 debe
    inicializarse en float64, no en float32 convertido después, para
    que las comprobaciones numéricas de gradientes sean válidas.
    """
    return rng.normal(0.0, std, size=shape).astype(dtype, copy=False)


def _require_cache(component: str, cache: dict | None) -> dict:
    if cache is None:
        raise ValueError(
            f"{component}.backward() sin un forward válido previo: ejecuta "
            f"primero forward(..., retain_cache=True) y no llames a backward "
            f"dos veces sobre los mismos intermediarios (se consumen)."
        )
    return cache


def _store_cache(component: str, holder: object, cache: dict, retain: bool) -> None:
    if not retain:
        return
    if getattr(holder, "_cache", None) is not None:
        raise ValueError(
            f"{component}: ya hay un forward de entrenamiento pendiente. "
            f"Llama a backward() o a clear_cache() antes de un nuevo "
            f"forward con retain_cache=True; no se sobrescribe en silencio."
        )
    holder._cache = cache  # type: ignore[attr-defined]


class Linear:
    """Capa lineal ``y = x @ W + b`` con backward manual.

    Backward (``Y = X @ W + b``, ``X: [..., Din]``, ``W: [Din, Dout]``)::

        dX = dY @ W.T
        dW = X_flat.T @ dY_flat
        db = sum(dY_flat, axis=0)

    - dX conserva la forma de X porque la derivada de una operación
      lineal respecto a su entrada es la transpuesta de la matriz, y
      al aplicarla se recupera exactamente la forma de entrada.
    - dW tiene la forma de W porque cada peso W[i, j] multiplica a
      X[..., i] para producir Y[..., j]; su gradiente es la suma
      sobre todas las posiciones del producto X[..., i] * dY[..., j],
      es decir, el producto matricial X_flat.T @ dY_flat.
    - db suma sobre todas las posiciones porque el mismo bias se
      reutiliza (broadcasting) en cada posición: el backward invierte
      ese broadcasting sumando las contribuciones de todas ellas.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Las dimensiones de una capa lineal deben ser positivas.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = np.dtype(dtype)
        self.weight = _init_normal(rng, (input_dim, output_dim), init_std, self.dtype)
        self.bias = np.zeros(output_dim, dtype=self.dtype)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._cache: dict | None = None

    def forward(self, x: np.ndarray, retain_cache: bool = False) -> np.ndarray:
        """Aplica la capa sobre la última dimensión de ``x``."""
        x = np.asarray(x)
        if x.ndim == 0 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"La última dimensión debe ser {self.input_dim}, no {x.shape}."
            )
        _store_cache("Linear", self, {"x": x}, retain_cache)
        return x @ self.weight + self.bias

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Devuelve dX y acumula dW, db. Consume los intermediarios."""
        cache = _require_cache("Linear", self._cache)
        self._cache = None
        x = cache["x"]
        dY = np.asarray(dY)
        if dY.shape != x.shape[:-1] + (self.output_dim,):
            raise ValueError(
                f"dY debe tener forma {x.shape[:-1] + (self.output_dim,)}, no {dY.shape}."
            )
        dX = dY @ self.weight.T
        x_flat = x.reshape(-1, self.input_dim)
        dY_flat = dY.reshape(-1, self.output_dim)
        self.grad_weight += x_flat.T @ dY_flat
        self.grad_bias += np.sum(dY_flat, axis=0)
        return dX

    def zero_grad(self) -> None:
        self.grad_weight[...] = 0
        self.grad_bias[...] = 0

    def clear_cache(self) -> None:
        self._cache = None

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        """Parámetros con nombres únicos y estables."""
        return [(f"{prefix}.weight", self.weight), (f"{prefix}.bias", self.bias)]

    def named_gradients(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        return [(f"{prefix}.weight", self.grad_weight), (f"{prefix}.bias", self.grad_bias)]


class LayerNorm:
    """Layer Normalization sobre la última dimensión, con backward.

    Forward::

        mu = mean(X, axis=-1, keepdims=True)
        variance = mean((X - mu) ** 2, axis=-1, keepdims=True)
        inv_std = 1 / sqrt(variance + eps)
        X_hat = (X - mu) * inv_std
        Y = gamma * X_hat + beta

    Gradientes de parámetros (suma sobre todos los ejes salvo el último)::

        dgamma = sum(dY * X_hat)
        dbeta = sum(dY)

    Con ``G = dY * gamma``::

        dX = inv_std * (G - mean(G) - X_hat * mean(G * X_hat))

    Esta expresión compacta ES la regla de la cadena completa: mu y
    variance dependen de X, así que no se tratan como constantes. El
    término ``- mean(G)`` recoge el camino a través de mu y el término
    ``- X_hat * mean(G * X_hat)`` recoge el camino a través de la
    varianza; ``inv_std`` es el factor común de derivar X_hat respecto
    a X. Se conserva el mismo eps del forward.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        if dim <= 0:
            raise ValueError("La dimensión de LayerNorm debe ser positiva.")
        self.dim = dim
        self.eps = eps
        self.dtype = np.dtype(dtype)
        self.gamma = np.ones(dim, dtype=self.dtype)
        self.beta = np.zeros(dim, dtype=self.dtype)
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)
        self._cache: dict | None = None

    def forward(self, x: np.ndarray, retain_cache: bool = False) -> np.ndarray:
        """Normaliza cada vector de la última dimensión por separado."""
        x = np.asarray(x)
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(
                f"La última dimensión debe ser {self.dim}, no {x.shape}."
            )
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(variance + self.eps)
        x_hat = (x - mean) * inv_std
        _store_cache("LayerNorm", self, {"x_hat": x_hat, "inv_std": inv_std}, retain_cache)
        return self.gamma * x_hat + self.beta

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Devuelve dX y acumula dgamma, dbeta. Consume los intermediarios."""
        cache = _require_cache("LayerNorm", self._cache)
        self._cache = None
        x_hat = cache["x_hat"]
        inv_std = cache["inv_std"]
        dY = np.asarray(dY)
        if dY.shape != x_hat.shape:
            raise ValueError(f"dY debe tener forma {x_hat.shape}, no {dY.shape}.")
        axes = tuple(range(dY.ndim - 1))
        self.grad_gamma += np.sum(dY * x_hat, axis=axes) if axes else dY * x_hat
        self.grad_beta += np.sum(dY, axis=axes) if axes else dY
        g = dY * self.gamma
        dX = inv_std * (
            g
            - np.mean(g, axis=-1, keepdims=True)
            - x_hat * np.mean(g * x_hat, axis=-1, keepdims=True)
        )
        return dX

    def zero_grad(self) -> None:
        self.grad_gamma[...] = 0
        self.grad_beta[...] = 0

    def clear_cache(self) -> None:
        self._cache = None

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        return [(f"{prefix}.gamma", self.gamma), (f"{prefix}.beta", self.beta)]

    def named_gradients(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        return [(f"{prefix}.gamma", self.grad_gamma), (f"{prefix}.beta", self.grad_beta)]


class Embeddings:
    """Embeddings de tokens + posiciones aprendibles, con backward.

    - Token embeddings ``E``: ``[V, D]``; posición ``P``: ``[C, D]``.
    - Salida: ``x = E[token_ids] + P[0:T]`` con forma ``[B, T, D]``.

    Backward: los identificadores enteros no se derivan; lo que se
    aprende son los valores de las filas seleccionadas. Cada aparición
    de un token contribuye al gradiente de su fila y las contribuciones
    se SUMAN (``np.add.at``, nunca ``grad[ids] += dX``, que pierde
    contribuciones con índices repetidos por el buffering de NumPy).
    Para posiciones: ``grad_P[:T] += sum(dX, axis=0)``; las posiciones
    no utilizadas no reciben gradiente. La suma de embeddings reparte
    el mismo dX a ambas tablas.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        rng: np.random.Generator,
        init_std: float = 0.02,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        if vocab_size <= 0 or context_length <= 0 or d_model <= 0:
            raise ValueError("vocab_size, context_length y d_model deben ser positivos.")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.dtype = np.dtype(dtype)
        self.token_embedding = _init_normal(rng, (vocab_size, d_model), init_std, self.dtype)
        self.position_embedding = _init_normal(rng, (context_length, d_model), init_std, self.dtype)
        self.grad_token_embedding = np.zeros_like(self.token_embedding)
        self.grad_position_embedding = np.zeros_like(self.position_embedding)
        self._cache: dict | None = None

    def forward(self, token_ids: np.ndarray, retain_cache: bool = False) -> np.ndarray:
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
        _store_cache(
            "Embeddings", self, {"token_ids": token_ids.copy(), "seq_len": seq_len}, retain_cache
        )
        token_part = self.token_embedding[token_ids]  # [B, T, D]
        position_part = self.position_embedding[:seq_len]  # [T, D]
        return token_part + position_part

    def backward(self, dX: np.ndarray) -> None:
        """Acumula gradientes de ambas tablas. No devuelve gradiente
        de los identificadores enteros (no existe). Consume la caché."""
        cache = _require_cache("Embeddings", self._cache)
        self._cache = None
        token_ids = cache["token_ids"]
        seq_len = cache["seq_len"]
        dX = np.asarray(dX)
        if dX.shape != (token_ids.shape[0], seq_len, self.d_model):
            raise ValueError(
                f"dX debe tener forma {(token_ids.shape[0], seq_len, self.d_model)}, no {dX.shape}."
            )
        np.add.at(self.grad_token_embedding, token_ids.reshape(-1), dX.reshape(-1, self.d_model))
        self.grad_position_embedding[:seq_len] += np.sum(dX, axis=0)
        return None

    def zero_grad(self) -> None:
        self.grad_token_embedding[...] = 0
        self.grad_position_embedding[...] = 0

    def clear_cache(self) -> None:
        self._cache = None

    def named_parameters(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        return [
            (f"{prefix}.token_embedding", self.token_embedding),
            (f"{prefix}.position_embedding", self.position_embedding),
        ]

    def named_gradients(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        return [
            (f"{prefix}.token_embedding", self.grad_token_embedding),
            (f"{prefix}.position_embedding", self.grad_position_embedding),
        ]
