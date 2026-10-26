"""Comprobación numérica de gradientes por diferencias finitas.

Para una función escalar f y una componente θ_i, con diferencias
centrales::

    g_numeric = (f(θ_i + h) - f(θ_i - h)) / (2 * h)

Valores por defecto: dtype float64, h = 1e-5, atol = 1e-7, rtol = 1e-4.
Criterio por componente::

    abs(g_analytic - g_numeric) <= atol + rtol * max(|g_analytic|, |g_numeric|)

Procedimiento (modelo): se calculan y copian los gradientes analíticos,
se perturba UNA componente, se ejecuta un forward nuevo, se evalúa la
pérdida, se restaura el valor original incluso si hay un error, se
repite con la perturbación negativa y se compara. No se usa ningún
optimizador durante estas pruebas.

Para probar una capa con salida no escalar se usa
``f(X, θ) = sum(layer(X, θ) * U)`` con U fijo y reproducible.

Advertencias: las diferencias finitas VERIFICAN, nunca entrenan. Si
una perturbación atraviesa un cambio de rama de ReLU (entrada cercana
a 0), el resultado numérico no es comparable: hay que usar una entrada
o perturbación apropiada, no marcarlo silenciosamente como correcto.
Tampoco se resuelven discrepancias ampliando tolerancias sin investigar.
"""

from __future__ import annotations

import numpy as np

from .functional import cross_entropy_loss


def check_value(
    analytic: float, numeric: float, atol: float = 1e-7, rtol: float = 1e-4
) -> dict:
    """Compara un gradiente analítico con su aproximación numérica."""
    abs_err = abs(float(analytic) - float(numeric))
    tol = atol + rtol * max(abs(float(analytic)), abs(float(numeric)))
    denom = max(abs(float(analytic)), abs(float(numeric)))
    rel_err = abs_err / denom if denom > 0 else abs_err
    return {
        "analytic": float(analytic),
        "numeric": float(numeric),
        "abs_error": float(abs_err),
        "rel_error": float(rel_err),
        "tolerance": float(tol),
        "passed": bool(abs_err <= tol),
    }


def central_difference(func, array: np.ndarray, index: tuple[int, ...], h: float = 1e-5) -> float:
    """Diferencia central de ``func`` respecto a ``array[index]``.

    Restaura el valor original incluso si ``func`` produce un error.
    ``func`` no debe modificar ``array`` salvo a través de la
    perturbación que aquí se aplica.
    """
    original = float(array[index])
    try:
        array[index] = original + h
        f_plus = float(func())
        array[index] = original - h
        f_minus = float(func())
    finally:
        array[index] = original
    return (f_plus - f_minus) / (2.0 * h)


def _select_indices(shape: tuple[int, ...], count: int | None, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Índices a comprobar: todos si count es None; si no, una muestra
    determinista que cubre principio, medio y final de cada tensor."""
    total = int(np.prod(shape)) if shape else 1
    if count is None or count >= total:
        return [tuple(idx) for idx in np.ndindex(shape)] if shape else [()]
    flat_choices: set[int] = {0, total // 2, total - 1}
    while len(flat_choices) < min(count, total):
        flat_choices.add(int(rng.integers(0, total)))
    indices = []
    for flat in sorted(flat_choices):
        indices.append(tuple(np.unravel_index(flat, shape)) if shape else ())
    return indices


def gradcheck_model(
    model,
    token_ids: np.ndarray,
    targets: np.ndarray,
    h: float = 1e-5,
    atol: float = 1e-7,
    rtol: float = 1e-4,
    sample_per_tensor: int | None = None,
    seed: int = 42,
) -> dict:
    """Comprueba los gradientes del modelo completo con entropía cruzada.

    El modelo debe estar en float64 (si no lo está, se rechaza: un
    cast interno a float32 invalidaría la comprobación). Devuelve un
    informe con el error máximo absoluto y relativo, errores por
    parámetro, nº de componentes comprobadas y fallos.

    Args:
        sample_per_tensor: None = todas las componentes de cada tensor
            (modo completo, solo para modelos diminutos); entero = nº
            de componentes por tensor (modo rápido). En ambos casos
            se cubren todos los grupos de parámetros, no solo matrices
            grandes.
    """
    if model.dtype != np.dtype(np.float64):
        raise ValueError(
            f"gradcheck_model exige un modelo float64, no {model.dtype}. "
            f"Crea el modelo con ModelConfig(dtype='float64') o usa "
            f"model.to_dtype(np.float64) antes de crear el optimizador."
        )
    token_ids = np.asarray(token_ids)
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    model.zero_grad()
    logits = model.forward(token_ids, retain_cache=True)
    if not isinstance(logits, np.ndarray):
        raise ValueError("gradcheck_model espera logits sin pesos de atención.")
    from .functional import cross_entropy_with_grad

    loss, dlogits = cross_entropy_with_grad(logits, targets)
    model.backward(dlogits)
    analytic = {name: grad.copy() for name, grad in model.named_gradients()}
    params = dict(model.named_parameters())

    def loss_fn() -> float:
        current_logits = model.forward(token_ids)
        assert isinstance(current_logits, np.ndarray)
        return cross_entropy_loss(current_logits, targets)

    per_param: dict[str, dict] = {}
    failures: list[dict] = []
    max_abs = 0.0
    max_rel = 0.0
    checked = 0
    for name, param in params.items():
        grad = analytic[name]
        indices = _select_indices(param.shape, sample_per_tensor, rng)
        worst_abs = 0.0
        worst_rel = 0.0
        param_failures = []
        for idx in indices:
            numeric = central_difference(loss_fn, param, idx, h=h)
            result = check_value(float(grad[idx]), numeric, atol=atol, rtol=rtol)
            checked += 1
            worst_abs = max(worst_abs, result["abs_error"])
            worst_rel = max(worst_rel, result["rel_error"])
            if not result["passed"]:
                detail = {"parameter": name, "index": idx, **result}
                param_failures.append(detail)
                failures.append(detail)
        per_param[name] = {
            "checked": len(indices),
            "max_abs_error": float(worst_abs),
            "max_rel_error": float(worst_rel),
            "passed": len(param_failures) == 0,
        }
        max_abs = max(max_abs, worst_abs)
        max_rel = max(max_rel, worst_rel)
    # Dejar el modelo limpio: gradientes analíticos restaurados no se
    # conservan como estado de entrenamiento; se ponen a cero.
    model.zero_grad()
    model.clear_cache()
    return {
        "loss": float(loss),
        "checked_components": int(checked),
        "max_abs_error": float(max_abs),
        "max_rel_error": float(max_rel),
        "passed": len(failures) == 0,
        "failures": failures,
        "per_parameter": per_param,
        "h": float(h),
        "atol": float(atol),
        "rtol": float(rtol),
    }
