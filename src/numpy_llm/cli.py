"""Interfaz de consola.

Comandos::

    python -m numpy_llm inspect --corpus data/ejemplo.txt --seed 42
    python -m numpy_llm generate --corpus data/ejemplo.txt --prompt "La inteligencia" --max-new-tokens 120 --temperature 1.0 --top-k 20 --seed 42
    python -m numpy_llm save --corpus data/ejemplo.txt --output artifacts/modelo_inicial --seed 42
    python -m numpy_llm generate --model-dir artifacts/modelo_inicial --prompt "La inteligencia" --max-new-tokens 120 --seed 42

No hay comandos de entrenamiento: esta fase no entrena.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .data import build_examples
from .functional import cross_entropy_loss
from .generation import generate
from .model import TransformerLanguageModel
from .serialization import load_model, save_model
from .tokenizer import CharTokenizer

UNTRAINED_NOTICE = (
    "Aviso: este modelo NO está entrenado. Sus pesos son aleatorios, "
    "así que el texto generado es esencialmente aleatorio. Esto no "
    "indica un fallo de implementación: es el resultado esperado antes "
    "de entrenar."
)


def _load_corpus(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _build_from_corpus(corpus_path: str, seed: int) -> tuple[TransformerLanguageModel, CharTokenizer]:
    text = _load_corpus(corpus_path)
    tokenizer = CharTokenizer.from_text(text)
    config = ModelConfig(seed=seed).with_vocab_size(tokenizer.vocab_size)
    model = TransformerLanguageModel(config)
    return model, tokenizer


def _load_model_or_corpus(args: argparse.Namespace) -> tuple[TransformerLanguageModel, CharTokenizer]:
    if getattr(args, "model_dir", None):
        return load_model(args.model_dir)
    if getattr(args, "corpus", None):
        return _build_from_corpus(args.corpus, args.seed)
    raise ValueError("Debes indicar --model-dir o --corpus.")


def _format_matrix(matrix: np.ndarray) -> str:
    lines = []
    for row in matrix:
        lines.append("  " + " ".join(f"{float(v):.2f}" for v in row))
    return "\n".join(lines)


def _format_causal_mask(seq_len: int) -> str:
    lines = []
    for i in range(seq_len):
        lines.append("  " + " ".join("1" if j <= i else "0" for j in range(seq_len)))
    return "\n".join(lines)


def cmd_inspect(args: argparse.Namespace) -> int:
    """Muestra dimensiones, parámetros y una inspección pequeña."""
    model, tokenizer = _load_model_or_corpus(args)
    config = model.config
    print("=== Inspección del modelo ===")
    print(f"Vocabulario (V): {tokenizer.vocab_size}")
    print("Configuración:")
    for key, value in config.to_dict().items():
        print(f"  {key} = {value}")
    print(f"Número total de parámetros: {model.count_parameters()}")
    print("Parámetros (nombre: forma):")
    for name, array in model.named_parameters():
        print(f"  {name}: {tuple(array.shape)}")

    # Secuencia pequeña de inspección (6-10 tokens).
    if getattr(args, "corpus", None) and not getattr(args, "model_dir", None):
        text = _load_corpus(args.corpus)
    else:
        # Sin corpus, la inspección usa los primeros tokens del vocabulario
        # que sean caracteres visibles (sin <UNK>).
        text = "".join(tokenizer.tokens[1:11])
    token_ids = tokenizer.encode(text[:10])
    # Asegurar entre 6 y 10 tokens.
    token_ids = token_ids[:8] if len(token_ids) >= 8 else token_ids
    if len(token_ids) < 6:
        print("Secuencia de inspección demasiado corta; se omite el detalle.")
        return 0
    inputs = np.asarray([token_ids], dtype=np.int64)
    print(f"\nEntrada: forma {tuple(inputs.shape)} [B, T]")
    print("Secuencia de inspección:")
    for position, token_id in enumerate(token_ids):
        char = tokenizer.tokens[token_id]
        print(f"  posición {position}: id {token_id}, carácter {char!r}")
    print("Máscara causal (1 = permitido, 0 = futuro prohibido):")
    print(_format_causal_mask(len(token_ids)))

    embeddings = model.embeddings.forward(inputs)
    print(f"Embeddings: forma {tuple(embeddings.shape)} [B, T, D]")
    logits, all_weights = model.forward(inputs, return_attention_weights=True)
    assert isinstance(logits, np.ndarray)
    print(f"Logits: forma {tuple(logits.shape)} [B, T, V]")
    # Las formas de Q/K/V_values tras separar cabezas son [B, H, T, Dh].
    print(
        "Primera atención (tras separar cabezas): "
        f"Q, K y V_values tienen forma "
        f"({inputs.shape[0]}, {config.n_heads}, {len(token_ids)}, {config.head_dim}) [B, H, T, Dh]"
    )
    print(
        "Puntajes de atención: forma "
        f"({inputs.shape[0]}, {config.n_heads}, {len(token_ids)}, {len(token_ids)}) [B, H, T, T]"
    )
    print(
        "Pesos de atención: forma "
        f"({inputs.shape[0]}, {config.n_heads}, {len(token_ids)}, {len(token_ids)}) [B, H, T, T]"
    )
    # Salida de cada bloque: se recorre de nuevo para mostrar formas.
    hidden = embeddings
    for i, block in enumerate(model.blocks):
        hidden = block.forward(hidden)  # type: ignore[assignment]
        print(f"Salida del bloque {i + 1}: forma {tuple(hidden.shape)} [B, T, D]")
    print("\nMatriz de atención, bloque 1, cabeza 0 (filas = posiciones que consultan, columnas = posiciones consultadas):")
    print(_format_matrix(np.asarray(all_weights[0][0, 0])))

    # Pérdida inicial sobre un lote de ejemplo.
    try:
        corpus_text = _load_corpus(args.corpus) if getattr(args, "corpus", None) else text
        corpus_ids = tokenizer.encode(corpus_text)
        x, y = build_examples(corpus_ids, sequence_length=16)
        batch_x = x[:4]
        batch_y = y[:4]
        batch_logits = model.forward(batch_x)
        assert isinstance(batch_logits, np.ndarray)
        loss = cross_entropy_loss(batch_logits, batch_y)
        print(f"\nPérdida inicial sobre un lote de ejemplo [4, 16]: {loss:.4f}")
        print(f"Referencia log(V) = log({tokenizer.vocab_size}) = {math.log(tokenizer.vocab_size):.4f}")
        print("Una pérdida cercana a log(V) es lo esperado con pesos aleatorios.")
    except Exception as exc:  # Corpus demasiado corto, etc.
        print(f"\nNo se pudo calcular la pérdida inicial de ejemplo: {exc}")
    print("\nNota: la inspección no modifica pesos ni consume aleatoriedad de muestreo.")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Genera texto de forma autorregresiva con un modelo sin entrenar."""
    model, tokenizer = _load_model_or_corpus(args)
    print(UNTRAINED_NOTICE)
    prompt_ids = tokenizer.encode(args.prompt)
    rng = np.random.default_rng(args.seed)
    full_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=rng,
    )
    generated_text = tokenizer.decode(full_ids)
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Texto generado ({len(full_ids)} tokens en total):")
    print(generated_text)
    return 0


def cmd_save(args: argparse.Namespace) -> int:
    """Guarda configuración, vocabulario y pesos en un directorio."""
    model, tokenizer = _build_from_corpus(args.corpus, args.seed)
    output = save_model(model, tokenizer, args.output)
    print(f"Modelo guardado en: {output}")
    print(f"Vocabulario (V): {tokenizer.vocab_size}")
    print(f"Número total de parámetros: {model.count_parameters()}")
    print("Recuerda: guardar pesos aleatorios no significa que el modelo esté entrenado.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numpy_llm",
        description="Pequeño modelo de lenguaje Transformer decoder-only en NumPy (fase 1, sin entrenamiento).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Muestra dimensiones, parámetros y pérdida inicial.")
    inspect_parser.add_argument("--corpus", help="Archivo de texto para construir el vocabulario.")
    inspect_parser.add_argument("--model-dir", help="Directorio de un modelo guardado.")
    inspect_parser.add_argument("--seed", type=int, default=42)
    inspect_parser.set_defaults(func=cmd_inspect)

    generate_parser = subparsers.add_parser("generate", help="Genera texto con un modelo sin entrenar.")
    generate_parser.add_argument("--corpus", help="Archivo de texto para construir el vocabulario.")
    generate_parser.add_argument("--model-dir", help="Directorio de un modelo guardado.")
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=120)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--top-k", type=int, default=None)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.set_defaults(func=cmd_generate)

    save_parser = subparsers.add_parser("save", help="Guarda un modelo inicial (pesos aleatorios).")
    save_parser.add_argument("--corpus", required=True)
    save_parser.add_argument("--output", required=True)
    save_parser.add_argument("--seed", type=int, default=42)
    save_parser.set_defaults(func=cmd_save)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
