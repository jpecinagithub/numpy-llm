"""Pérdidas de la fase 2.

La única pérdida de entrenamiento es la entropía cruzada media sobre
la predicción del siguiente token. Su implementación y derivación
completas viven en :mod:`numpy_llm.functional` (para no duplicar
funciones equivalentes); este módulo es el punto de entrada con nombre
explícito que usan el entrenador y las pruebas.

Derivación resumida (ver docs/derivadas.md)::

    log_probs = logits - logsumexp(logits)
    loss = -sum(log_probs[target]) / M,   M = B * T
    dlogits = softmax(logits); dlogits[target] -= 1; dlogits /= M

Es decir, ``d loss / d logit = probabilidad_predicha -
indicador_del_token_correcto``, dividido una sola vez entre M.
"""

from __future__ import annotations

from .functional import cross_entropy_loss, cross_entropy_with_grad, softmax_backward

__all__ = ["cross_entropy_loss", "cross_entropy_with_grad", "softmax_backward"]
