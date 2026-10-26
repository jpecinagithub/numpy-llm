"""Guardado y carga del modelo.

Formato (un directorio):

- ``config.json``: configuración del modelo (JSON).
- ``vocab.json``: vocabulario en orden de identificador (JSON).
- ``weights.npz``: todos los parámetros (NumPy, sin pickle).

No se usa pickle ni arrays de objetos; la carga usa
``allow_pickle=False``. Las claves del npz son ``p0, p1, ...`` en el
mismo orden que ``named_parameters()``; ``config.json`` guarda además
la lista de nombres y formas, de modo que al restaurar se puede
verificar que estén todos los parámetros, rechazar los inesperados y
comprobar las formas.

Al restaurar se comprueba también la compatibilidad entre vocabulario
y configuración (``vocab_size`` debe coincidir).

Aclaración: guardar pesos aleatorios no significa que el modelo esté
entrenado. No se guarda el estado del generador de muestreo: esta fase
no implementa la reanudación exacta de una generación interrumpida.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .model import TransformerLanguageModel
from .tokenizer import CharTokenizer


def save_model(
    model: TransformerLanguageModel, tokenizer: CharTokenizer, output_dir: str | Path
) -> Path:
    """Guarda configuración, vocabulario y todos los parámetros."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = model.named_parameters()
    metadata = {
        "parameter_names": [name for name, _ in params],
        "parameter_shapes": {name: list(array.shape) for name, array in params},
    }
    config_data = model.config.to_dict()
    config_data["_parameters"] = metadata
    (output_dir / "config.json").write_text(
        json.dumps(config_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "vocab.json").write_text(
        json.dumps(tokenizer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    arrays = {f"p{i}": array for i, (_, array) in enumerate(params)}
    np.savez(output_dir / "weights.npz", **arrays)
    return output_dir


def load_model(model_dir: str | Path) -> tuple[TransformerLanguageModel, CharTokenizer]:
    """Restaura un modelo guardado con :func:`save_model`.

    Rechaza parámetros ausentes, parámetros inesperados, formas
    incompatibles y vocabulario incompatible con la configuración.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    vocab_path = model_dir / "vocab.json"
    weights_path = model_dir / "weights.npz"
    for path in (config_path, vocab_path, weights_path):
        if not path.is_file():
            raise ValueError(f"Falta el archivo {path} en el modelo guardado.")
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = config_data.pop("_parameters", None)
    if metadata is None or "parameter_names" not in metadata or "parameter_shapes" not in metadata:
        raise ValueError("La configuración guardada no contiene la lista de parámetros.")
    config = ModelConfig.from_dict(config_data)
    tokenizer = CharTokenizer.from_dict(
        json.loads(vocab_path.read_text(encoding="utf-8"))
    )
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Incompatibilidad vocabulario/configuración: vocab_size en la "
            f"configuración es {config.vocab_size} pero el vocabulario tiene "
            f"{tokenizer.vocab_size} tokens."
        )
    model = TransformerLanguageModel(config)
    expected_names: list[str] = metadata["parameter_names"]
    expected_shapes: dict[str, list[int]] = metadata["parameter_shapes"]
    current = dict(model.named_parameters())
    if sorted(current) != sorted(expected_names):
        raise ValueError(
            "Los nombres de parámetros del modelo no coinciden con los guardados."
        )
    with np.load(weights_path, allow_pickle=False) as data:
        stored_keys = set(data.files)
        expected_keys = {f"p{i}" for i in range(len(expected_names))}
        if stored_keys != expected_keys:
            missing = [
                expected_names[int(key[1:])]
                for key in sorted(expected_keys - stored_keys)
            ]
            unexpected = sorted(stored_keys - expected_keys)
            details = []
            if missing:
                details.append(f"parámetros ausentes: {missing}")
            if unexpected:
                details.append(f"parámetros inesperados: {unexpected}")
            raise ValueError("Archivo de pesos incompleto o con extras. " + "; ".join(details) + ".")
        for i, name in enumerate(expected_names):
            array = data[f"p{i}"]
            if list(array.shape) != expected_shapes[name]:
                raise ValueError(
                    f"Forma incompatible para el parámetro {name}: se esperaba "
                    f"{expected_shapes[name]} y se encontró {list(array.shape)}."
                )
            # Copiar en el array existente para mantener las referencias.
            current[name][...] = array
    return model, tokenizer
