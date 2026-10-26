"""Programación de la tasa de aprendizaje (por actualización).

``t`` es el índice de la actualización del optimizador, empezando en 1.
El scheduler depende de actualizaciones, NUNCA de microlotes.

a) Constante: ``lr(t) = lr_peak`` para todo t.

b) Warmup lineal + descenso coseno. Con S = total de actualizaciones,
W = actualizaciones de warmup, lr_peak y lr_min::

    t <= W:  lr = lr_peak * t / W
    t > W:   progress = (t - W) / (S - W)
             lr = lr_min + 0.5 * (lr_peak - lr_min) * (1 + cos(pi * progress))

Sin warmup (W = 0) y S > 1: ``progress = (t - 1) / (S - 1)`` con la misma
fórmula coseno. Para S = 1 se usa lr_peak y se exige W = 0.
Se exige ``0 <= W < S``.
"""

from __future__ import annotations

import math


def _validate_schedule(total_steps: int, warmup_steps: int, lr_peak: float, lr_min: float) -> None:
    if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps <= 0:
        raise ValueError(f"total_steps debe ser un entero positivo, no {total_steps!r}.")
    if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
        raise ValueError(f"warmup_steps debe ser un entero >= 0, no {warmup_steps!r}.")
    if not (0 <= warmup_steps < total_steps):
        raise ValueError(
            f"Se exige 0 <= warmup_steps < total_steps, no warmup={warmup_steps}, total={total_steps}."
        )
    for name, value in (("lr_peak", lr_peak), ("lr_min", lr_min)):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} debe ser finito y >= 0, no {value!r}.")
    if float(lr_min) > float(lr_peak):
        raise ValueError(f"lr_min ({lr_min}) no puede superar a lr_peak ({lr_peak}).")


def learning_rate(
    step: int,
    total_steps: int,
    lr_peak: float,
    lr_min: float | None = None,
    warmup_steps: int = 0,
    kind: str = "cosine",
) -> float:
    """Tasa de aprendizaje para la actualización ``step`` (desde 1)."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError(f"step debe ser un entero >= 1, no {step!r}.")
    if kind == "constant":
        if not isinstance(lr_peak, (int, float)) or not math.isfinite(float(lr_peak)) or float(lr_peak) <= 0:
            raise ValueError(f"lr_peak debe ser positivo y finito, no {lr_peak!r}.")
        return float(lr_peak)
    if kind != "cosine":
        raise ValueError(f"kind debe ser 'constant' o 'cosine', no {kind!r}.")
    lr_min_value = float(lr_peak) if lr_min is None else float(lr_min)
    _validate_schedule(total_steps, warmup_steps, lr_peak, lr_min_value)
    if total_steps == 1:
        return float(lr_peak)
    if warmup_steps > 0 and step <= warmup_steps:
        return float(lr_peak) * step / warmup_steps
    if warmup_steps > 0:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
    else:
        progress = (step - 1) / (total_steps - 1)
    progress = min(1.0, max(0.0, progress))
    return lr_min_value + 0.5 * (float(lr_peak) - lr_min_value) * (1.0 + math.cos(math.pi * progress))


class LRSchedule:
    """Scheduler con estado mínimo: solo recuerda su configuración."""

    def __init__(
        self,
        total_steps: int,
        lr_peak: float,
        lr_min: float | None = None,
        warmup_steps: int = 0,
        kind: str = "cosine",
    ) -> None:
        if kind == "cosine":
            _validate_schedule(total_steps, warmup_steps, lr_peak, lr_min if lr_min is not None else lr_peak)
        elif kind == "constant":
            _validate_lr_peak = float(lr_peak)
            if not math.isfinite(_validate_lr_peak) or _validate_lr_peak <= 0:
                raise ValueError(f"lr_peak debe ser positivo y finito, no {lr_peak!r}.")
        else:
            raise ValueError(f"kind debe ser 'constant' o 'cosine', no {kind!r}.")
        self.total_steps = total_steps
        self.lr_peak = float(lr_peak)
        self.lr_min = float(lr_peak) if lr_min is None else float(lr_min)
        self.warmup_steps = warmup_steps
        self.kind = kind

    def __call__(self, step: int) -> float:
        return learning_rate(step, self.total_steps, self.lr_peak, self.lr_min, self.warmup_steps, self.kind)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "total_steps": self.total_steps,
            "lr_peak": self.lr_peak,
            "lr_min": self.lr_min,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LRSchedule":
        return cls(
            total_steps=int(data["total_steps"]),
            lr_peak=float(data["lr_peak"]),
            lr_min=float(data["lr_min"]),
            warmup_steps=int(data["warmup_steps"]),
            kind=str(data["kind"]),
        )
