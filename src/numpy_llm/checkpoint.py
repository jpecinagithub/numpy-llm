"""Checkpoints de entrenamiento en un único archivo npz.

Contenido (versión de formato 1):
- Configuración del modelo y vocabulario completo (orden de tokens).
- Todos los parámetros (nombres, formas y dtypes verificados al cargar).
- Configuración de entrenamiento, tipo y configuración del optimizador,
  momentos m y v de AdamW, contador t, actualizaciones completadas,
  tokens procesados, configuración del scheduler y horizonte total.
- Estado de los generadores relevantes (train y muestreo), mejor
  pérdida de validación y paso correspondiente, identidad del corpus
  y de las particiones, y selección fija de ventanas de evaluación.

Formato: un único ``.npz`` con arrays NumPy y un JSON guardado como
cadena Unicode en la clave ``metadata``. Sin pickle ni arrays de
objetos; la carga usa ``allow_pickle=False``. La escritura es atómica:
se escribe primero un archivo temporal en el mismo directorio y solo
cuando está completo se reemplaza el destino con ``os.replace``.
Solo se guarda en fronteras entre actualizaciones completas, nunca a
mitad de una acumulación de gradientes.

Distinción importante:
- REANUDAR: continuar con optimizador, scheduler y generadores.
- CARGAR PESOS: empezar un entrenamiento nuevo con estado de
  optimizador nuevo. Cargar solo los pesos NO es una reanudación exacta.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelConfig
from .model import TransformerLanguageModel
from .tokenizer import CharTokenizer

FORMAT_VERSION = 1


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _rng_state_to_json(rng: np.random.Generator) -> dict:
    return _to_jsonable(rng.bit_generator.state)


def rng_from_state(state: dict) -> np.random.Generator:
    """Reconstruye un Generator PCG64 desde un estado guardado en JSON."""
    if state.get("bit_generator") != "PCG64":
        raise ValueError(
            f"Solo se soporta la restauración de generadores PCG64, no {state.get('bit_generator')!r}."
        )
    rng = np.random.default_rng(0)
    restored = {
        "bit_generator": "PCG64",
        "state": {"state": int(state["state"]["state"]), "inc": int(state["state"]["inc"])},
        "has_uint32": int(state.get("has_uint32", 0)),
        "uinteger": int(state.get("uinteger", 0)),
    }
    rng.bit_generator.state = restored
    return rng


@dataclass
class CheckpointData:
    """Resultado de cargar un checkpoint."""

    model: TransformerLanguageModel
    tokenizer: CharTokenizer
    metadata: dict
    optimizer_state: dict | None
    training_state: dict

    @property
    def completed_updates(self) -> int:
        return int(self.training_state.get("completed_updates", 0))

    @property
    def tokens_processed(self) -> int:
        return int(self.training_state.get("tokens_processed", 0))


def save_checkpoint(
    path: str | Path,
    model: TransformerLanguageModel,
    tokenizer: CharTokenizer,
    training_config: dict,
    optimizer_state: dict | None,
    training_state: dict,
    corpus_info: dict,
    scheduler_info: dict,
    eval_window_starts: list[int] | None = None,
) -> Path:
    """Guarda un checkpoint completo de forma atómica."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = model.named_parameters()
    metadata = {
        "format_version": FORMAT_VERSION,
        "model_config": model.config.to_dict(),
        "vocab_tokens": tokenizer.tokens,
        "parameter_names": [name for name, _ in params],
        "parameter_shapes": {name: list(arr.shape) for name, arr in params},
        "parameter_dtypes": {name: str(arr.dtype) for name, arr in params},
        "training_config": _to_jsonable(training_config),
        "optimizer_state_meta": None,
        "training_state": _to_jsonable(training_state),
        "corpus_info": _to_jsonable(corpus_info),
        "scheduler": _to_jsonable(scheduler_info),
        "eval_window_starts": list(eval_window_starts or []),
    }
    arrays: dict[str, np.ndarray] = {}
    for i, (_, arr) in enumerate(params):
        arrays[f"p{i}"] = np.array(arr, copy=True)
    if optimizer_state is not None:
        meta: dict[str, Any] = {"type": optimizer_state.get("type")}
        for key in ("lr", "betas", "eps", "weight_decay", "decay", "t", "steps"):
            if key in optimizer_state:
                meta[key] = _to_jsonable(optimizer_state[key])
        metadata["optimizer_state_meta"] = meta
        if optimizer_state.get("type") == "AdamW":
            for i, name in enumerate(metadata["parameter_names"]):
                arrays[f"m{i}"] = np.array(optimizer_state["m"][name], copy=True)
                arrays[f"v{i}"] = np.array(optimizer_state["v"][name], copy=True)
    arrays["metadata"] = np.array(json.dumps(metadata, ensure_ascii=False))
    # Escritura atómica: temporal en el mismo directorio + os.replace.
    # suffix .npz: np.savez añade la extensión si falta, y os.replace
    # debe mover exactamente el archivo escrito, no el temporal vacío.
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", suffix=".npz", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return path


def load_checkpoint(
    path: str | Path,
    expected_corpus_sha256: str | None = None,
    expected_vocab_tokens: list[str] | None = None,
) -> CheckpointData:
    """Carga y verifica un checkpoint.

    Verifica nombres, formas y dtypes de los parámetros, la
    compatibilidad del vocabulario y, si se indica, la identidad del
    corpus. Inicializa los buffers de gradiente a cero y NO restaura
    intermediarios de forward. El estado del optimizador se devuelve
    en ``optimizer_state`` (meta + arrays m/v) para que el entrenador
    lo restaure en una instancia nueva.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"No existe el checkpoint {path}.")
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            raise ValueError("El checkpoint no contiene metadatos.")
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Versión de checkpoint no soportada: {metadata.get('format_version')!r}."
            )
        names: list[str] = metadata["parameter_names"]
        shapes: dict[str, list[int]] = metadata["parameter_shapes"]
        dtypes: dict[str, str] = metadata["parameter_dtypes"]
        config = ModelConfig.from_dict(metadata["model_config"])
        tokenizer = CharTokenizer.from_tokens(metadata["vocab_tokens"])
        if expected_vocab_tokens is not None and tokenizer.tokens != list(expected_vocab_tokens):
            raise ValueError(
                "El vocabulario del checkpoint no coincide con el esperado: "
                "no se pueden cargar pesos asociados a otro orden de tokens."
            )
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError("Incompatibilidad vocabulario/configuración en el checkpoint.")
        corpus_info = metadata.get("corpus_info", {})
        if expected_corpus_sha256 is not None and corpus_info.get("sha256") not in (None, expected_corpus_sha256):
            raise ValueError(
                "La identidad del corpus no coincide con la del checkpoint "
                "(SHA-256 distinto): reanudar con otro corpus cambiaría los "
                "lotes y la validación."
            )
        model = TransformerLanguageModel(config)
        current = dict(model.named_parameters())
        if sorted(current) != sorted(names):
            raise ValueError("Los nombres de parámetros del checkpoint no coinciden con el modelo.")
        for i, name in enumerate(names):
            if f"p{i}" not in data.files:
                raise ValueError(f"Falta el parámetro {name!r} en el checkpoint.")
            array = data[f"p{i}"]
            if list(array.shape) != shapes[name]:
                raise ValueError(
                    f"Forma incompatible para {name}: se esperaba {shapes[name]} y se encontró {list(array.shape)}."
                )
            if str(array.dtype) != dtypes[name]:
                raise ValueError(
                    f"Dtype incompatible para {name}: se esperaba {dtypes[name]} y se encontró {array.dtype}."
                )
            current[name][...] = array
        optimizer_state = None
        meta = metadata.get("optimizer_state_meta")
        if meta is not None:
            optimizer_state = dict(meta)
            if meta.get("type") == "AdamW":
                m: dict[str, np.ndarray] = {}
                v: dict[str, np.ndarray] = {}
                for i, name in enumerate(names):
                    if f"m{i}" not in data.files or f"v{i}" not in data.files:
                        raise ValueError(f"Falta el estado de AdamW para {name!r}.")
                    m[name] = np.array(data[f"m{i}"])
                    v[name] = np.array(data[f"v{i}"])
                    if list(m[name].shape) != shapes[name] or list(v[name].shape) != shapes[name]:
                        raise ValueError(f"Forma incompatible en el estado de AdamW para {name!r}.")
                optimizer_state["m"] = m
                optimizer_state["v"] = v
    model.zero_grad()
    model.clear_cache()
    return CheckpointData(
        model=model,
        tokenizer=tokenizer,
        metadata=metadata,
        optimizer_state=optimizer_state,
        training_state=metadata.get("training_state", {}),
    )
