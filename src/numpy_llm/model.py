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
from dataclasses import replace

from .config import ModelConfig
from .layers import Embeddings, LayerNorm, Linear
from .transformer import TransformerBlock


class TransformerLanguageModel:
    """Pequeño modelo de lenguaje Transformer decoder-only, con backward.

    Reglas de la fase 2:
    - ``forward`` no modifica parámetros.
    - ``backward(dlogits)`` calcula y ACUMULA gradientes; no modifica
      pesos. Recorre el modelo en orden inverso: proyección al
      vocabulario -> LayerNorm final -> bloques (del último al primero)
      -> embeddings de tokens y posiciones.
    - ``zero_grad()`` pone a cero todos los gradientes.
    - Solo ``optimizer.step`` modifica los parámetros, in-place.
    - ``named_parameters()`` y ``named_gradients()`` usan exactamente
      los mismos nombres; cada gradiente tiene la misma forma y dtype
      que su parámetro, en un array separado, inicializado a cero.
    - No se devuelve ningún gradiente de los identificadores enteros.
    - Un gradiente legítimamente nulo (p. ej. ciertas simetrías del
      softmax en algunos biases) no es por sí solo un error.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.vocab_size is None:
            raise ValueError(
                "ModelConfig.vocab_size debe estar fijado (calcúlalo a partir "
                "del tokenizador con config.with_vocab_size(...)) antes de "
                "crear el modelo."
            )
        self.config = config
        self.dtype = np.dtype(config.dtype)
        rng = np.random.default_rng(config.seed)
        self.embeddings = Embeddings(
            config.vocab_size, config.context_length, config.d_model, rng, config.init_std, self.dtype
        )
        self.blocks = [
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                rng,
                config.init_std,
                config.layer_norm_eps,
                self.dtype,
            )
            for _ in range(config.n_layers)
        ]
        self.final_ln = LayerNorm(config.d_model, config.layer_norm_eps, self.dtype)
        self.lm_head = Linear(config.d_model, config.vocab_size, rng, config.init_std, self.dtype)
        self._cache: dict | None = None

    def forward(
        self,
        token_ids: np.ndarray,
        retain_cache: bool = False,
        return_attention_weights: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """Calcula los logits para una entrada ``[B, T]``.

        Con ``retain_cache=False`` (inferencia) no crea ni sobrescribe
        intermediarios de entrenamiento. Con ``retain_cache=True``,
        un segundo forward pendiente produce un error explicativo.
        No modifica los parámetros.
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
        if retain_cache and self._cache is not None:
            raise ValueError(
                "TransformerLanguageModel: ya hay un forward de entrenamiento "
                "pendiente. Llama a backward() o a clear_cache() antes de un "
                "nuevo forward con retain_cache=True."
            )
        hidden = self.embeddings.forward(token_ids, retain_cache=retain_cache)
        all_weights: list[np.ndarray] = []
        for block in self.blocks:
            hidden, weights = block.forward(
                hidden, retain_cache=retain_cache, return_attention_weights=True
            )
            all_weights.append(weights)
        hidden = self.final_ln.forward(hidden, retain_cache=retain_cache)
        logits = self.lm_head.forward(hidden, retain_cache=retain_cache)
        if retain_cache:
            self._cache = {"shape": logits.shape}
        if return_attention_weights:
            return logits, all_weights
        return logits

    def backward(self, dlogits: np.ndarray) -> None:
        """Acumula gradientes de todos los parámetros. Consume la caché."""
        if self._cache is None:
            raise ValueError(
                "TransformerLanguageModel.backward() sin un forward válido "
                "previo: ejecuta primero forward(..., retain_cache=True) y "
                "no llames a backward dos veces sobre los mismos "
                "intermediarios (se consumen tras el primer backward)."
            )
        self._cache = None
        dlogits = np.asarray(dlogits)
        d_hidden = self.lm_head.backward(dlogits)
        d_hidden = self.final_ln.backward(d_hidden)
        for block in reversed(self.blocks):
            d_hidden = block.backward(d_hidden)
        self.embeddings.backward(d_hidden)

    def zero_grad(self) -> None:
        """Pone a cero todos los buffers de gradiente."""
        self.embeddings.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.final_ln.zero_grad()
        self.lm_head.zero_grad()

    def clear_cache(self) -> None:
        """Libera todos los intermediarios de forward, sin tocar gradientes."""
        self._cache = None
        self.embeddings.clear_cache()
        for block in self.blocks:
            block.clear_cache()
        self.final_ln.clear_cache()
        self.lm_head.clear_cache()

    def named_parameters(self) -> list[tuple[str, np.ndarray]]:
        """Todos los parámetros con nombres únicos y estables."""
        params: list[tuple[str, np.ndarray]] = []
        params += self.embeddings.named_parameters("embeddings")
        for i, block in enumerate(self.blocks):
            params += block.named_parameters(f"blocks.{i}")
        params += self.final_ln.named_parameters("final_ln")
        params += self.lm_head.named_parameters("lm_head")
        return params

    def named_gradients(self) -> list[tuple[str, np.ndarray]]:
        """Gradientes con exactamente los mismos nombres que los parámetros."""
        grads: list[tuple[str, np.ndarray]] = []
        grads += self.embeddings.named_gradients("embeddings")
        for i, block in enumerate(self.blocks):
            grads += block.named_gradients(f"blocks.{i}")
        grads += self.final_ln.named_gradients("final_ln")
        grads += self.lm_head.named_gradients("lm_head")
        return grads

    def count_parameters(self) -> int:
        """Número total de parámetros, sumando el tamaño real de los arrays."""
        return int(sum(array.size for _, array in self.named_parameters()))

    def to_dtype(self, dtype: np.dtype | str) -> "TransformerLanguageModel":
        """Convierte TODOS los parámetros y gradientes al dtype indicado.

        Sustituye los arrays de cada componente (los nombres y las
        formas no cambian). Debe llamarse ANTES de crear un optimizador,
        porque el optimizador conserva referencias a los arrays: tras
        la conversión, esas referencias quedarían obsoletas. Devuelve
        ``self`` para encadenar. También actualiza ``config.dtype``.
        """
        target = np.dtype(dtype)
        if target not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"dtype debe ser float32 o float64, no {target}.")
        self.dtype = target
        self.config = replace(self.config, dtype=str(target))
        for component, names in (
            (self.embeddings, ("token_embedding", "position_embedding")),
            (self.final_ln, ("gamma", "beta")),
            (self.lm_head, ("weight", "bias")),
        ):
            for name in names:
                arr = getattr(component, name)
                setattr(component, name, arr.astype(target, copy=True))
                grad = getattr(component, "grad_" + name)
                setattr(component, "grad_" + name, grad.astype(target, copy=True))
        for block in self.blocks:
            for layer in (block.ln1, block.ln2):
                for name in ("gamma", "beta"):
                    setattr(layer, name, getattr(layer, name).astype(target, copy=True))
                    setattr(layer, "grad_" + name, getattr(layer, "grad_" + name).astype(target, copy=True))
            for proj in (
                block.attention.q_proj, block.attention.k_proj,
                block.attention.v_proj, block.attention.out_proj,
                block.feed_forward.fc1, block.feed_forward.fc2,
            ):
                for name in ("weight", "bias"):
                    setattr(proj, name, getattr(proj, name).astype(target, copy=True))
                    setattr(proj, "grad_" + name, getattr(proj, "grad_" + name).astype(target, copy=True))
        return self

    # Alias legible.
    astype = to_dtype
