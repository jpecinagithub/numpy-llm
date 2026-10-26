"""Interfaz de consola.

Comandos de la fase 1::

    python -m numpy_llm inspect --corpus data/ejemplo.txt --seed 42
    python -m numpy_llm generate --corpus data/ejemplo.txt --prompt "La inteligencia" --max-new-tokens 120 --temperature 1.0 --top-k 20 --seed 42
    python -m numpy_llm save --corpus data/ejemplo.txt --output artifacts/modelo_inicial --seed 42
    python -m numpy_llm generate --model-dir artifacts/modelo_inicial --prompt "La inteligencia" --max-new-tokens 120 --seed 42

Comandos de la fase 2 (entrenamiento)::

    python -m numpy_llm gradcheck --scope all --seed 42
    python -m numpy_llm trace-step --config configs/smoke.json --corpus data/ejemplo.txt
    python -m numpy_llm train --config configs/smoke.json --corpus data/ejemplo.txt --outdir runs/smoke
    python -m numpy_llm overfit --config configs/overfit.json --outdir runs/overfit
    python -m numpy_llm train --config configs/demo.json --corpus data/ejemplo.txt --outdir runs/demo
    python -m numpy_llm evaluate --checkpoint runs/demo/best.npz --corpus data/ejemplo.txt --split val --full
    python -m numpy_llm resume --checkpoint runs/demo/last.npz --corpus data/ejemplo.txt
    python -m numpy_llm generate --checkpoint runs/demo/best.npz --prompt "La inteligencia" --max-new-tokens 200 --temperature 0.8 --seed 42

Todos los comandos se ejecutan desde la raíz del proyecto con
``PYTHONPATH=src`` (el paquete no está instalado en el entorno).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .data import build_examples, corpus_sha256, sample_batch, split_corpus
from .evaluation import evaluate_model, uniform_baseline, unigram_baseline
from .functional import cross_entropy_loss
from .generation import generate
from .gradcheck import gradcheck_model
from .model import TransformerLanguageModel
from .serialization import load_model, save_model
from .tokenizer import CharTokenizer
from .training import Trainer, TrainingConfig, trace_step

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
    if getattr(args, "checkpoint", None):
        ckpt = load_checkpoint(args.checkpoint)
        return ckpt.model, ckpt.tokenizer
    if getattr(args, "model_dir", None):
        return load_model(args.model_dir)
    if getattr(args, "corpus", None):
        return _build_from_corpus(args.corpus, args.seed)
    raise ValueError("Debes indicar --checkpoint, --model-dir o --corpus.")


def _load_json_config(path: str) -> tuple[ModelConfig, TrainingConfig]:
    import json as _json

    data = _json.loads(Path(path).read_text(encoding="utf-8"))
    if "model" not in data or "training" not in data:
        raise ValueError("La configuración debe tener las claves 'model' y 'training'.")
    model_data = dict(data["model"])
    model_data.pop("vocab_size", None)  # se fija con el vocabulario real
    base_config = ModelConfig(**model_data)
    training_config = TrainingConfig.from_dict(data["training"])
    return base_config, training_config


def _build_trainer(config_path: str, corpus_path: str, outdir: str | None) -> Trainer:
    base_config, training_config = _load_json_config(config_path)
    text = _load_corpus(corpus_path)
    split = split_corpus(
        text,
        train_fraction=training_config.train_fraction,
        sequence_length=training_config.sequence_length,
    )
    model_config = base_config.with_vocab_size(split.tokenizer.vocab_size)
    model = TransformerLanguageModel(model_config)
    return Trainer(model, split.tokenizer, split, training_config, outdir=outdir)


def _print_eval(label: str, result: dict) -> None:
    print(f"{label}: loss={result['mean_loss']:.4f} "
          f"perplexity={result['perplexity']} "
          f"accuracy={result['token_accuracy']:.4f} "
          f"objetivos={result['total_targets']} cobertura={result.get('coverage', '?')}")


def cmd_gradcheck(args: argparse.Namespace) -> int:
    """Comprueba los gradientes del modelo diminuto de la especificación."""
    if args.scope not in ("all", "fast", "model"):
        raise ValueError(f"--scope debe ser 'all', 'fast' o 'model', no {args.scope!r}.")
    config = ModelConfig(
        context_length=8, d_model=4, n_heads=2, n_layers=1, d_ff=8,
        seed=args.seed, dtype="float64",
    ).with_vocab_size(5)
    model = TransformerLanguageModel(config)
    rng = np.random.default_rng(args.seed)
    token_ids = rng.integers(0, 5, size=(2, 3)).astype(np.int64)
    targets = rng.integers(0, 5, size=(2, 3)).astype(np.int64)
    sample = None if args.scope == "all" else 3
    report = gradcheck_model(model, token_ids, targets, sample_per_tensor=sample, seed=args.seed)
    print(f"Gradcheck ({args.scope}): passed={report['passed']} "
          f"componentes={report['checked_components']} "
          f"error_abs_max={report['max_abs_error']:.3e} "
          f"error_rel_max={report['max_rel_error']:.3e} "
          f"(h={report['h']}, atol={report['atol']}, rtol={report['rtol']})")
    if args.scope == "all":
        for name, info in report["per_parameter"].items():
            print(f"  {name}: {info['checked']} componentes, error_abs_max={info['max_abs_error']:.3e}")
    if not report["passed"]:
        for failure in report["failures"][:10]:
            print(f"  FALLO: {failure}")
        return 1
    return 0


def cmd_trace_step(args: argparse.Namespace) -> int:
    """Diagnóstico de un forward+backward pequeño, sin actualizar pesos."""
    trainer = _build_trainer(args.config, args.corpus, outdir=None)
    assert trainer.split is not None
    rng = np.random.default_rng(args.seed)
    x, y = sample_batch(
        trainer.split.train_tokens, 2, trainer.config.sequence_length, rng
    )
    result = trace_step(trainer.model, x, y)
    print(f"Entrada: {result['input_shape']}  Logits: {result['logits_shape']}")
    print(f"Pérdida: {result['loss']:.4f}  Todos finitos: {result['all_finite']}")
    print("Norma de gradiente por grupo:")
    for group, info in result["grad_norm_by_group"].items():
        print(f"  {group}: {info['num_tensors']} tensores, norma={info['grad_norm']:.4f}")
    print("Este diagnóstico no actualiza pesos.")
    return 0


def _print_train_summary(report: dict) -> None:
    print(f"Actualizaciones completadas: {report['completed_updates']}")
    print(f"Tokens objetivo procesados: {report['tokens_processed']}")
    if report["initial_eval"]:
        _print_eval("Evaluación inicial", report["initial_eval"])
    if report["final_eval"]:
        _print_eval("Evaluación final", report["final_eval"])
    print(f"Mejor pérdida de validación: {report['best_val_loss']} (paso {report['best_step']})")
    if report["baselines"]:
        print(f"Baselines: {report['baselines']}")
    print("Muestra antes:")
    print(report["samples_before"])
    print("Muestra después:")
    print(report["samples_after"])


def cmd_train(args: argparse.Namespace) -> int:
    """Entrena con partición train/validation y guarda en outdir."""
    trainer = _build_trainer(args.config, args.corpus, outdir=args.outdir)
    report = trainer.run()
    _print_train_summary(report)
    print(f"Resultados en: {args.outdir}")
    return 0


def cmd_overfit(args: argparse.Namespace) -> int:
    """Prueba de memorización: sobreajustar 'abcabc...' con lotes fijos."""
    base_config, training_config = _load_json_config(args.config)
    tokenizer = CharTokenizer.from_tokens(["<UNK>", "a", "b", "c"])
    tokens = np.asarray(tokenizer.encode("abc" * 40), dtype=np.int64)
    seq_len = training_config.sequence_length
    # Lote fijo: B ejemplos con inicios 0..B-1 (un solo lote por update).
    x = np.stack([tokens[s : s + seq_len] for s in range(training_config.batch_size)])
    y = np.stack([tokens[s + 1 : s + seq_len + 1] for s in range(training_config.batch_size)])
    fixed = [(x, y)]
    model_config = base_config.with_vocab_size(tokenizer.vocab_size)
    model = TransformerLanguageModel(model_config)
    trainer = Trainer(
        model, tokenizer, None, training_config, outdir=args.outdir,
        fixed_batches=fixed, use_validation=False,
    )
    initial = trainer.evaluate_fixed_batches()
    report = trainer.run()
    final = report["final_eval"]
    print(f"Pérdida inicial: {initial['mean_loss']:.4f}")
    print(f"Pérdida final: {final['mean_loss']:.4f} "
          f"({100.0 * final['mean_loss'] / initial['mean_loss']:.1f} % de la inicial)")
    print(f"Exactitud final sobre el lote memorizado: {final['token_accuracy']:.4f}")
    print(f"Actualizaciones: {report['completed_updates']}  Semilla: {training_config.seed}")
    print("Muestra greedy después:")
    print(report["greedy_after"])
    ok_loss = final["mean_loss"] < 0.20 * initial["mean_loss"]
    ok_acc = final["token_accuracy"] >= 0.95
    print(f"Criterio pérdida final < 20 % inicial: {'SUPERADO' if ok_loss else 'NO SUPERADO'}")
    print(f"Criterio exactitud >= 95 %: {'SUPERADO' if ok_acc else 'NO SUPERADO'}")
    print(f"Resultados en: {args.outdir}")
    return 0 if (ok_loss and ok_acc) else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evalúa un checkpoint sobre train o val, completo o subconjunto fijo."""
    text = _load_corpus(args.corpus)
    ckpt = load_checkpoint(args.checkpoint, expected_corpus_sha256=corpus_sha256(text))
    training_config = TrainingConfig.from_dict(ckpt.metadata["training_config"])
    split = split_corpus(
        text,
        train_fraction=training_config.train_fraction,
        sequence_length=training_config.sequence_length,
    )
    if split.tokenizer.tokens != ckpt.tokenizer.tokens:
        raise ValueError("El vocabulario del checkpoint no coincide con el reconstruido.")
    tokens = split.train_tokens if args.split == "train" else split.val_tokens
    limit = None if args.full else 32
    result = evaluate_model(
        ckpt.model, tokens, training_config.sequence_length,
        batch_size=training_config.batch_size, limit_windows=limit,
    )
    _print_eval(f"Evaluación {args.split} ({'completa' if args.full else 'subconjunto fijo de 32 ventanas'})", result)
    print(f"Ventanas evaluadas: {result['evaluated_windows']} / {result['total_windows_available']} "
          f" Tokens fuera: {result['leftover_tokens']}")
    print(f"Baseline uniforme: {uniform_baseline(ckpt.tokenizer.vocab_size)}")
    print(f"Baseline unigrama (train -> {args.split}): "
          f"{unigram_baseline(split.train_tokens, tokens, ckpt.tokenizer.vocab_size)}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Reanuda un entrenamiento desde last.npz, con el mismo horizonte."""
    text = _load_corpus(args.corpus)
    trainer = Trainer.from_checkpoint(args.checkpoint, text, outdir=args.outdir)
    if trainer.completed_updates >= trainer.config.total_updates:
        print(f"El entrenamiento ya completó su objetivo "
              f"({trainer.completed_updates}/{trainer.config.total_updates} actualizaciones). "
              f"No se reinicia ni se extiende el scheduler.")
        return 0
    print(f"Reanudando desde la actualización {trainer.completed_updates} "
          f"hasta {trainer.config.total_updates} (mismo horizonte del scheduler).")
    report = trainer.run()
    _print_train_summary(report)
    return 0


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
    """Genera texto de forma autorregresiva."""
    model, tokenizer = _load_model_or_corpus(args)
    if getattr(args, "checkpoint", None):
        print(f"Modelo cargado desde checkpoint: {args.checkpoint}")
    else:
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

    generate_parser = subparsers.add_parser("generate", help="Genera texto con un modelo (checkpoint, model-dir o corpus).")
    generate_parser.add_argument("--checkpoint", help="Checkpoint .npz de entrenamiento (fase 2).")
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

    gradcheck_parser = subparsers.add_parser("gradcheck", help="Comprueba gradientes por diferencias finitas (float64).")
    gradcheck_parser.add_argument("--scope", default="all", choices=["all", "fast", "model"])
    gradcheck_parser.add_argument("--seed", type=int, default=42)
    gradcheck_parser.set_defaults(func=cmd_gradcheck)

    trace_parser = subparsers.add_parser("trace-step", help="Diagnóstico forward+backward sin actualizar pesos.")
    trace_parser.add_argument("--config", required=True)
    trace_parser.add_argument("--corpus", required=True)
    trace_parser.add_argument("--seed", type=int, default=42)
    trace_parser.set_defaults(func=cmd_trace_step)

    train_parser = subparsers.add_parser("train", help="Entrena con partición train/validation.")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--corpus", required=True)
    train_parser.add_argument("--outdir", required=True)
    train_parser.set_defaults(func=cmd_train)

    overfit_parser = subparsers.add_parser("overfit", help="Prueba de memorización sobre 'abcabc...'.")
    overfit_parser.add_argument("--config", required=True)
    overfit_parser.add_argument("--outdir", required=True)
    overfit_parser.set_defaults(func=cmd_overfit)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evalúa un checkpoint sobre train o val.")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--corpus", required=True)
    evaluate_parser.add_argument("--split", default="val", choices=["train", "val"])
    evaluate_parser.add_argument("--full", action="store_true", help="Evalúa todas las ventanas (sin --full: subconjunto fijo de 32).")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    resume_parser = subparsers.add_parser("resume", help="Reanuda un entrenamiento desde un checkpoint.")
    resume_parser.add_argument("--checkpoint", required=True)
    resume_parser.add_argument("--corpus", required=True)
    resume_parser.add_argument("--outdir", help="Directorio de salida (por defecto, el del checkpoint).")
    resume_parser.set_defaults(func=cmd_resume)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
