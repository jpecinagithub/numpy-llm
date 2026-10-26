"""Optimizadores manuales: SGD sin momentum y AdamW.

Reglas comunes
--------------
- Solo ``step()`` modifica los parámetros, y lo hace IN-PLACE
  (``param[...] = ...``), para no romper las referencias que conserva
  el optimizador.
- Los gradientes se leen, nunca se ponen a cero aquí: eso lo hace
  ``model.zero_grad()`` en el bucle de entrenamiento.
- Un "parámetro" se describe como ``(nombre, param, grad)`` o como
  ``(param, grad)`` (en ese caso el nombre es su índice). El nombre
  sirve para decidir el weight decay y para guardar/restaurar estado.

SGD (referencia)::

    θ = θ - η * g

AdamW (entrenamiento principal), con t empezando en 1 y aumentando
UNA vez por actualización completa (no por parámetro ni microlote)::

    m = β1 * m + (1 - β1) * g
    v = β2 * v + (1 - β2) * g²
    m_hat = m / (1 - β1**t);  v_hat = v / (1 - β2**t)
    θ_new = θ_old - η * m_hat / (sqrt(v_hat) + eps) - η * wd * θ_old

El weight decay está DESACOPLADO del gradiente: nunca se suma
``wd * θ`` a ``g``. En este proyecto se aplica solo a las matrices de
las capas lineales (nombres que terminan en ``.weight``) y se excluye
de biases, parámetros de LayerNorm y embeddings. La función
:func:`decay_for_name` documenta esa decisión y el entrenador la usa
para construir los grupos.
"""

from __future__ import annotations

import math

import numpy as np


def decay_for_name(name: str) -> bool:
    """True solo para matrices de capas lineales (``*.weight``).

    Excluye: biases (``*.bias``), LayerNorm (``*.gamma``/``*.beta``) y
    embeddings (``*.token_embedding``/``*.position_embedding``).
    """
    return name.endswith(".weight")


def _normalize_params(params):
    """Normaliza entradas a listas (nombre, param, grad)."""
    normalized = []
    for i, entry in enumerate(params):
        if len(entry) == 3:
            name, param, grad = entry
        elif len(entry) == 2:
            param, grad = entry
            name = str(i)
        else:
            raise ValueError(
                "Cada parámetro debe ser (nombre, param, grad) o (param, grad)."
            )
        param = np.asarray(param)
        grad = np.asarray(grad)
        if param.shape != grad.shape:
            raise ValueError(
                f"El parámetro {name!r} y su gradiente deben tener la misma "
                f"forma, no {param.shape} y {grad.shape}."
            )
        normalized.append((str(name), param, grad))
    return normalized


def _validate_lr(lr: float) -> float:
    if not isinstance(lr, (int, float)) or not math.isfinite(float(lr)) or float(lr) <= 0:
        raise ValueError(f"La tasa de aprendizaje debe ser positiva y finita, no {lr!r}.")
    return float(lr)


class SGD:
    """Descenso de gradiente sin momentum, como referencia."""

    def __init__(self, params, lr: float = 1e-3) -> None:
        self.params = _normalize_params(params)
        self.lr = _validate_lr(lr)
        self.steps = 0

    def step(self, lr: float | None = None) -> None:
        """Una actualización in-place con la tasa indicada (o la propia)."""
        eta = _validate_lr(lr) if lr is not None else self.lr
        for _name, param, grad in self.params:
            param[...] = param - eta * grad
        self.steps += 1

    def state_dict(self) -> dict:
        return {"type": "SGD", "lr": self.lr, "steps": self.steps}

    def load_state_dict(self, state: dict) -> None:
        if state.get("type") != "SGD":
            raise ValueError("El estado guardado no corresponde a SGD.")
        self.lr = _validate_lr(state["lr"])
        self.steps = int(state["steps"])


class AdamW:
    """Adam con weight decay desacoplado."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decay: dict[str, bool] | set[str] | None = None,
    ) -> None:
        self.params = _normalize_params(params)
        self.lr = _validate_lr(lr)
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"betas deben estar en [0, 1), no {betas!r}.")
        if not (isinstance(eps, (int, float)) and math.isfinite(float(eps)) and float(eps) > 0):
            raise ValueError(f"eps debe ser positivo y finito, no {eps!r}.")
        if not (isinstance(weight_decay, (int, float)) and math.isfinite(float(weight_decay)) and float(weight_decay) >= 0):
            raise ValueError(f"weight_decay debe ser >= 0 y finito, no {weight_decay!r}.")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        # Decidir decay por nombre: un dict/set explícito gana; si no
        # se indica nada, se aplica la regla del proyecto.
        if decay is None:
            self.decay = {name: decay_for_name(name) for name, _, _ in self.params}
        elif isinstance(decay, dict):
            self.decay = {name: bool(decay.get(name, decay_for_name(name))) for name, _, _ in self.params}
        else:
            decay_set = set(decay)
            self.decay = {name: name in decay_set for name, _, _ in self.params}
        self.t = 0
        self.m = {name: np.zeros_like(param) for name, param, _ in self.params}
        self.v = {name: np.zeros_like(param) for name, param, _ in self.params}

    def step(self, lr: float | None = None) -> None:
        """Una actualización completa; t aumenta una sola vez."""
        eta = _validate_lr(lr) if lr is not None else self.lr
        self.t += 1
        b1t = 1.0 - self.beta1 ** self.t
        b2t = 1.0 - self.beta2 ** self.t
        for name, param, grad in self.params:
            m = self.m[name]
            v = self.v[name]
            # Cálculo en float64 para estabilidad; el resultado se
            # escribe in-place en el array del parámetro.
            g = grad.astype(np.float64, copy=False)
            m[...] = self.beta1 * m + (1.0 - self.beta1) * g
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            m_hat = m.astype(np.float64, copy=False) / b1t
            v_hat = v.astype(np.float64, copy=False) / b2t
            update = eta * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay != 0.0 and self.decay.get(name, False):
                update = update + eta * self.weight_decay * param.astype(np.float64, copy=False)
            param[...] = param - update

    def state_dict(self) -> dict:
        return {
            "type": "AdamW",
            "lr": self.lr,
            "betas": [self.beta1, self.beta2],
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "decay": dict(self.decay),
            "t": self.t,
            "m": {name: arr.copy() for name, arr in self.m.items()},
            "v": {name: arr.copy() for name, arr in self.v.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("type") != "AdamW":
            raise ValueError("El estado guardado no corresponde a AdamW.")
        self.lr = _validate_lr(state["lr"])
        self.beta1, self.beta2 = (float(b) for b in state["betas"])
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        self.decay = {str(k): bool(v) for k, v in state["decay"].items()}
        self.t = int(state["t"])
        for name, _, _ in self.params:
            if name not in state["m"] or name not in state["v"]:
                raise ValueError(f"Falta el estado de AdamW para el parámetro {name!r}.")
            m = np.asarray(state["m"][name])
            v = np.asarray(state["v"][name])
            if m.shape != self.m[name].shape or v.shape != self.v[name].shape:
                raise ValueError(f"Forma incompatible en el estado de AdamW para {name!r}.")
            self.m[name][...] = m
            self.v[name][...] = v


def clip_gradients(named_gradients, max_grad_norm: float = 1.0) -> tuple[float, float, bool]:
    """Clipping por norma global, en el orden obligatorio.

    1) Los gradientes ya están calculados/acumulados.
    2) Se comprueba que sean finitos (NaN/inf detienen la ejecución
       con un diagnóstico; nunca se sustituyen silenciosamente por 0).
    3) Se mide la norma global en float64.
    4) Si supera el límite, TODOS los gradientes se multiplican por
       ``max_grad_norm / global_norm`` (nunca clipping por matriz).
    5) El optimizador se ejecuta después.

    Returns:
        ``(norma_antes, norma_despues, se_aplico_clipping)``. Si la
        norma es 0 o ya está por debajo del límite, no se modifica nada.
    """
    if not (isinstance(max_grad_norm, (int, float)) and math.isfinite(float(max_grad_norm)) and float(max_grad_norm) > 0):
        raise ValueError(f"max_grad_norm debe ser positivo y finito, no {max_grad_norm!r}.")
    grads = [np.asarray(g) for _, g in named_gradients]
    for (name, _), grad in zip(named_gradients, grads):
        if not np.all(np.isfinite(grad)):
            raise ValueError(
                f"Gradiente no finito en {name!r}: contiene NaN o infinitos. "
                f"Deteniendo antes del clipping; revisa la pérdida, la "
                f"tasa de aprendizaje y el backward (ver docs/problemas_frecuentes.md)."
            )
    total = 0.0
    for grad in grads:
        flat = grad.astype(np.float64, copy=False).ravel()
        total += float(np.sum(flat * flat))
    norm_before = math.sqrt(total)
    if norm_before == 0.0 or norm_before <= float(max_grad_norm):
        return norm_before, norm_before, False
    scale = float(max_grad_norm) / norm_before
    for _, grad in named_gradients:
        grad *= scale
    # Norma después: por construcción es max_grad_norm (salvo redondeo).
    total_after = 0.0
    for grad in grads:
        flat = grad.astype(np.float64, copy=False).ravel()
        total_after += float(np.sum(flat * flat))
    return norm_before, math.sqrt(total_after), True


def global_grad_norm(named_gradients) -> float:
    """Norma global en float64, sin modificar los gradientes."""
    total = 0.0
    for _, grad in named_gradients:
        flat = np.asarray(grad).astype(np.float64, copy=False).ravel()
        total += float(np.sum(flat * flat))
    return math.sqrt(total)
