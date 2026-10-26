"""Funciones numéricas básicas, implementadas explícitamente.

- ``relu``: activación no lineal sencilla.
- ``softmax``: convierte logits en probabilidades, de forma estable.
- ``log_softmax``: logaritmo de softmax, calculado con log-sum-exp
  estable; nunca como ``log(softmax(z))``, que puede perder precisión.
- ``cross_entropy_loss``: pérdida media sobre todas las posiciones.

Un logit es la puntuación sin normalizar que el modelo asigna a cada
token del vocabulario antes de convertirla en probabilidad. A diferencia
de una probabilidad, un logit puede ser negativo, mayor que 1, y el
conjunto de logits de una posición no suma 1. Softmax los convierte en
probabilidades (positivas y que suman 1).

Usamos logaritmos porque trabajar en escala logarítmica convierte
productos de probabilidades en sumas, evita el underflow de
probabilidades muy pequeñas y hace que la pérdida sea la log-probabilidad
negativa del token correcto. Minimizarla empuja al modelo a asignar más
probabilidad al siguiente token correcto y menos al resto.

Una distribución exactamente uniforme sobre V tokens asigna probabilidad
1/V a cada uno; su pérdida es -log(1/V) = log(V) (logaritmo natural).
Un modelo inicializado al azar produce una pérdida cercana a log(V),
pero no tiene por qué coincidir exactamente: sus logits no son
exactamente uniformes.
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: ``maximum(0, x)``, elemento a elemento."""
    x = np.asarray(x)
    return np.maximum(0, x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax estable sobre el eje indicado.

    Se resta el máximo de cada vector antes de exponenciar, de modo que
    el mayor exponente es 0 y no hay overflow aunque los logits sean
    grandes (por ejemplo, 1000). Los valores ``-inf`` (usados por la
    máscara causal) se convierten en probabilidad exactamente 0.
    """
    x = np.asarray(x)
    if x.dtype.kind not in "fc":
        x = x.astype(np.float64)
    # Máximo por vector; si una fila fuera toda -inf, max sería -inf y
    # el resultado sería NaN, pero la máscara causal siempre deja al
    # menos la diagonal, así que ese caso no ocurre en el modelo.
    maximum = np.max(x, axis=axis, keepdims=True)
    shifted = x - maximum
    exp_shifted = np.exp(shifted)
    total = np.sum(exp_shifted, axis=axis, keepdims=True)
    return exp_shifted / total


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log-softmax estable mediante log-sum-exp.

    ``log_softmax(z) = z - max(z) - log(sum(exp(z - max(z))))``.

    No se calcula como ``log(softmax(z))``: si una probabilidad es tan
    pequeña que se redondea a 0, su logaritmo sería -inf y se perdería
    información.
    """
    x = np.asarray(x)
    if x.dtype.kind not in "fc":
        x = x.astype(np.float64)
    maximum = np.max(x, axis=axis, keepdims=True)
    shifted = x - maximum
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


def softmax_backward(probs: np.ndarray, dprobs: np.ndarray) -> np.ndarray:
    """Backward general de softmax, sin materializar el Jacobiano.

    Si ``A = softmax(S)`` y llega ``dA``, entonces
    ``dS = A * (dA - sum(dA * A, axis=-1, keepdims=True))``. El término
    ``sum(dA * A)`` es el Jacobiano aplicado al gradiente entrante:
    el Jacobiano es ``diag(A) - A A^T`` por fila, y multiplicarlo por
    dA da exactamente esa expresión, sin construir ninguna matriz
    T x T por fila. El eje de reducción coincide con el eje
    normalizado en el forward.
    """
    probs = np.asarray(probs)
    dprobs = np.asarray(dprobs)
    if probs.shape != dprobs.shape:
        raise ValueError(f"probs y dprobs deben tener la misma forma, no {probs.shape} y {dprobs.shape}.")
    return probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))


def cross_entropy_with_grad(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, np.ndarray]:
    """Entropía cruzada media y su gradiente respecto a los logits.

    Derivación: con ``p = softmax(logits)`` y log-softmax estable,
    ``loss = -sum(log_probs[target]) / M`` con ``M = B * T``. Derivando
    el log-softmax, ``d loss / d logit_j = p_j - 1{j == target}`` en
    cada posición, y el factor ``1/M`` aparece UNA sola vez, al dividir
    la suma de la pérdida (y por tanto su gradiente) entre el número
    total de objetivos. Por eso ninguna capa vuelve a dividir por B o
    T: la normalización se aplica una sola vez, aquí.

    Returns:
        ``(loss, dlogits)`` con ``dlogits`` de la misma forma y dtype
        que ``logits`` (una copia: nunca se modifican probabilidades
        que se usen en otra parte). No se construye un one-hot
        completo: se resta 1 mediante indexación en la posición del
        token correcto. No se aplica temperatura ni top-k.
    """
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    if logits.ndim != 3:
        raise ValueError(f"logits debe tener forma [B, T, V], no {logits.shape}.")
    batch, seq_len, vocab_size = logits.shape
    if batch == 0 or seq_len == 0:
        raise ValueError("logits no puede tener lote o secuencia de longitud cero.")
    if targets.shape != (batch, seq_len):
        raise ValueError(f"targets debe tener forma {(batch, seq_len)}, no {targets.shape}.")
    if targets.dtype.kind not in "iu":
        raise ValueError("targets debe contener enteros.")
    if np.any(targets < 0) or np.any(targets >= vocab_size):
        raise ValueError(f"Todos los objetivos deben estar en [0, {vocab_size}).")
    # Cálculo en el dtype de los logits (float64 en gradcheck, float32
    # en entrenamiento ordinario); la suma final se hace en float64.
    log_probs = log_softmax(logits, axis=-1)
    batch_index = np.arange(batch)[:, None]
    time_index = np.arange(seq_len)[None, :]
    selected = log_probs[batch_index, time_index, targets]
    total = float(batch * seq_len)
    loss = float(-np.sum(selected.astype(np.float64)) / total)
    probs = softmax(logits, axis=-1)
    dlogits = np.array(probs, copy=True)
    dlogits[batch_index, time_index, targets] -= 1
    dlogits = dlogits / total
    # Conservar el dtype de entrada (para float32, la división por el
    # escalar Python no cambia el dtype con NEP 50; se fuerza por
    # seguridad sin alterar el valor en float64).
    if dlogits.dtype != logits.dtype:
        dlogits = dlogits.astype(logits.dtype, copy=False)
    return loss, dlogits


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Entropía cruzada media: ``-mean(log_softmax(logits)[b, t, target])``.

    Args:
        logits: Array de forma ``[B, T, V]``. No debe aplicarse softmax
            previamente: la función calcula el log-softmax internamente.
        targets: Array de enteros de forma ``[B, T]`` con ids en [0, V).

    Returns:
        La pérdida media sobre las B x T posiciones, como ``float``.

    Nota: el cálculo interno se hace en float64 para reducir el error de
    acumulación en la media, aunque el forward del modelo use float32.
    Por eso las comprobaciones numéricas de referencia usan float64.
    """
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    if logits.ndim != 3:
        raise ValueError(f"logits debe tener forma [B, T, V], no {logits.shape}.")
    batch, seq_len, vocab_size = logits.shape
    if batch == 0 or seq_len == 0:
        raise ValueError("logits no puede tener lote o secuencia de longitud cero.")
    if targets.shape != (batch, seq_len):
        raise ValueError(
            f"targets debe tener forma {(batch, seq_len)}, no {targets.shape}."
        )
    if targets.dtype.kind not in "iu":
        raise ValueError("targets debe contener enteros.")
    if np.any(targets < 0) or np.any(targets >= vocab_size):
        raise ValueError(f"Todos los objetivos deben estar en [0, {vocab_size}).")
    # float64 aquí: referencia numérica más precisa, sin cambiar el
    # forward del modelo, que sigue siendo float32.
    log_probs = log_softmax(logits.astype(np.float64), axis=-1)
    batch_index = np.arange(batch)[:, None]
    time_index = np.arange(seq_len)[None, :]
    selected = log_probs[batch_index, time_index, targets]
    return float(-np.mean(selected))
