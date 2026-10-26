"""Bucle de entrenamiento (Trainer), separado del modelo.

Flujo de ``Trainer.run()``:
1) validar configuración; 2) particiones y tokenizador (ya preparados
por quien construye el Trainer); 3) modelo/optimizador (ya creados o
restaurados); 4) generadores aleatorios independientes; 5) ventanas
fijas de evaluación; 6) evaluar el estado inicial; 7) actualizaciones;
8) evaluar y guardar periódicamente; 9) guardar el estado final;
10) generar muestras comparables.

Una actualización (pseudocódigo real en :meth:`Trainer.step`)::

    model.zero_grad()
    total_targets = sum(M_i)
    for x, y in microbatches:
        logits = model.forward(x, retain_cache=True)
        loss, dlogits = cross_entropy_with_grad(logits, y)
        model.backward(dlogits * (y.size / total_targets))
    validate + clip_gradients()   # una sola vez, tras acumular
    lr = scheduler(completed_updates + 1)
    optimizer.step(lr)            # una sola vez; el scheduler avanza
    completed_updates += 1        # una sola vez por actualización

La pérdida del lote se calcula ANTES de actualizar; las métricas de
evaluación se calculan DESPUÉS. Evaluar, generar muestras o escribir
logs nunca consume el generador de entrenamiento, así que no cambian
la secuencia futura de lotes. Como el muestreo es con reemplazo, el
entrenamiento se organiza por actualizaciones del optimizador, no por
"épocas": un número arbitrario de pasos no es una época. El contador
de tokens procesados puede incluir posiciones vistas repetidamente.

Generadores independientes (derivados con SeedSequence, documentados):
- inicialización del modelo: ``ModelConfig.seed`` (propio del modelo);
- muestreo de train: ``SeedSequence([seed, 1])``;
- generación de muestras: ``SeedSequence([seed, 2])`` para derivar la
  semilla de muestreo (cada muestra usa un Generator nuevo con esa
  semilla, así que antes/después son comparables y no alteran train).
Nunca se usa ``np.random.seed`` global.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .checkpoint import load_checkpoint, rng_from_state, save_checkpoint, _rng_state_to_json
from .data import SplitData, eval_window_starts, sample_batch, split_corpus
from .evaluation import evaluate_model, uniform_baseline, unigram_baseline
from .functional import cross_entropy_with_grad
from .generation import generate
from .model import TransformerLanguageModel
from .optimizers import AdamW, SGD, clip_gradients
from .schedules import LRSchedule
from .tokenizer import CharTokenizer


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento (reproducible y guardada en JSON)."""

    total_updates: int = 500
    batch_size: int = 4
    sequence_length: int = 64
    optimizer: str = "adamw"  # "adamw" | "sgd"
    schedule_kind: str = "cosine"  # "constant" | "cosine"
    lr_peak: float = 1e-3
    lr_min: float = 1e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1
    eval_every: int = 50
    checkpoint_every: int = 50
    seed: int = 42
    train_fraction: float = 0.9
    eval_subset_windows: int | None = None
    sample_prompt: str = "La inteligencia"
    sample_max_tokens: int = 100
    sample_temperature: float = 0.8

    def validate(self) -> None:
        """Valida hiperparámetros con errores explicativos."""
        for name in ("total_updates", "batch_size", "sequence_length", "grad_accumulation_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} debe ser un entero positivo, no {value!r}.")
        for name in ("eval_every", "checkpoint_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} debe ser un entero >= 0, no {value!r}.")
        if self.optimizer not in ("adamw", "sgd"):
            raise ValueError(f"optimizer debe ser 'adamw' o 'sgd', no {self.optimizer!r}.")
        if self.schedule_kind not in ("constant", "cosine"):
            raise ValueError(f"schedule_kind debe ser 'constant' o 'cosine', no {self.schedule_kind!r}.")
        if not (0.0 < float(self.lr_peak) and math.isfinite(float(self.lr_peak))):
            raise ValueError(f"lr_peak debe ser positivo y finito, no {self.lr_peak!r}.")
        if not (0.0 <= float(self.lr_min) <= float(self.lr_peak)):
            raise ValueError(f"lr_min debe estar en [0, lr_peak], no {self.lr_min!r}.")
        if not (0 <= self.warmup_steps < self.total_updates) and not (
            self.total_updates == 1 and self.warmup_steps == 0
        ):
            # LRSchedule exige 0 <= W < S; para S == 1 se exige W == 0.
            raise ValueError(
                f"Se exige 0 <= warmup_steps < total_updates, no warmup={self.warmup_steps}, "
                f"total={self.total_updates}."
            )
        if self.schedule_kind == "constant" and self.warmup_steps != 0:
            raise ValueError("Con schedule constante, warmup_steps debe ser 0.")
        if float(self.weight_decay) < 0 or not math.isfinite(float(self.weight_decay)):
            raise ValueError(f"weight_decay debe ser >= 0 y finito, no {self.weight_decay!r}.")
        if float(self.max_grad_norm) <= 0 or not math.isfinite(float(self.max_grad_norm)):
            raise ValueError(f"max_grad_norm debe ser positivo y finito, no {self.max_grad_norm!r}.")
        if not (0.0 < float(self.train_fraction) < 1.0):
            raise ValueError(f"train_fraction debe estar en (0, 1), no {self.train_fraction!r}.")
        if self.eval_subset_windows is not None and (not isinstance(self.eval_subset_windows, int) or self.eval_subset_windows <= 0):
            raise ValueError(f"eval_subset_windows debe ser un entero positivo o None, no {self.eval_subset_windows!r}.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        allowed = set(asdict(cls()).keys())
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Campos desconocidos en la configuración de entrenamiento: {sorted(unknown)}.")
        cfg = cls(**{k: v for k, v in data.items() if k in allowed})
        cfg.validate()
        return cfg


def compute_batch_gradients(
    model: TransformerLanguageModel,
    microbatches: Sequence[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Acumula gradientes de varios microlotes y devuelve la pérdida media.

    Cada microlote i, con M_i objetivos y pérdida media loss_i,
    contribuye con peso ``M_i / M_total``: así los gradientes
    acumulados corresponden a la pérdida media sobre TODOS los
    objetivos (con microlotes iguales, equivale a dividir entre el
    número de microlotes). No actualiza pesos, no aplica clipping.
    Libera los intermediarios tras cada backward (cada backward del
    modelo los consume). El llamador debe hacer ``zero_grad()`` antes.
    """
    if not microbatches:
        raise ValueError("Se necesita al menos un microlote.")
    total_targets = int(sum(int(y.size) for _, y in microbatches))
    if total_targets == 0:
        raise ValueError("Los microlotes no contienen objetivos.")
    weighted_loss = 0.0
    for x, y in microbatches:
        logits = model.forward(x, retain_cache=True)
        if not isinstance(logits, np.ndarray):
            raise ValueError("compute_batch_gradients espera logits sin pesos de atención.")
        loss, dlogits = cross_entropy_with_grad(logits, y)
        weight = float(y.size) / float(total_targets)
        weighted_loss += weight * loss
        model.backward(dlogits * weight)
    return float(weighted_loss)


def trace_step(
    model: TransformerLanguageModel, x: np.ndarray, y: np.ndarray
) -> dict:
    """Diagnóstico de un forward+backward pequeño, SIN actualizar pesos.

    Devuelve formas de activaciones y gradientes, norma del gradiente
    por grupo de parámetros, pérdida y comprobación de finitos. Al
    terminar, los gradientes se ponen a cero y la caché se libera, así
    que el diagnóstico no deja estado de entrenamiento pendiente.
    (Los -inf de los puntajes de atención enmascarados son
    intencionales y no se confunden con errores en logits/gradientes:
    aquí solo se comprueban logits, pérdida y gradientes.)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    model.zero_grad()
    logits = model.forward(x, retain_cache=True)
    assert isinstance(logits, np.ndarray)
    loss, dlogits = cross_entropy_with_grad(logits, y)
    model.backward(dlogits)
    groups: dict[str, dict] = {}
    for name, grad in model.named_gradients():
        group = name.split(".")[0]
        entry = groups.setdefault(group, {"params": 0, "norm_sq": 0.0})
        entry["params"] += 1
        entry["norm_sq"] += float(np.sum(grad.astype(np.float64) ** 2))
    grad_norms = {
        group: {"num_tensors": info["params"], "grad_norm": float(math.sqrt(info["norm_sq"]))}
        for group, info in groups.items()
    }
    all_finite = bool(
        np.all(np.isfinite(logits))
        and math.isfinite(loss)
        and all(np.all(np.isfinite(g)) for _, g in model.named_gradients())
    )
    result = {
        "input_shape": [int(v) for v in x.shape],
        "logits_shape": [int(v) for v in logits.shape],
        "loss": float(loss),
        "grad_norm_by_group": grad_norms,
        "all_finite": all_finite,
        "gradient_shapes": {name: [int(v) for v in grad.shape] for name, grad in model.named_gradients()},
    }
    model.zero_grad()
    model.clear_cache()
    return result


class Trainer:
    """Entrenador legible, separado del modelo."""

    def __init__(
        self,
        model: TransformerLanguageModel,
        tokenizer: CharTokenizer,
        split: SplitData | None,
        config: TrainingConfig,
        outdir: str | Path | None = None,
        train_rng: np.random.Generator | None = None,
        fixed_batches: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
        use_validation: bool = True,
    ) -> None:
        config.validate()
        self.model = model
        self.tokenizer = tokenizer
        self.split = split
        self.config = config
        self.outdir = Path(outdir) if outdir is not None else None
        self.fixed_batches = list(fixed_batches) if fixed_batches is not None else None
        self.use_validation = use_validation and split is not None
        self.train_rng = train_rng if train_rng is not None else np.random.default_rng(
            np.random.SeedSequence([config.seed, 1])
        )
        self.sample_seed = int(np.random.SeedSequence([config.seed, 2]).generate_state(1)[0])
        self.schedule = LRSchedule(
            total_steps=config.total_updates,
            lr_peak=config.lr_peak,
            lr_min=config.lr_min,
            warmup_steps=config.warmup_steps if config.schedule_kind == "cosine" else 0,
            kind=config.schedule_kind,
        )
        named = [
            (name, param, grad)
            for (name, param), (_, grad) in zip(model.named_parameters(), model.named_gradients())
        ]
        if config.optimizer == "adamw":
            self.optimizer: AdamW | SGD = AdamW(
                named, lr=config.lr_peak, weight_decay=config.weight_decay
            )
        else:
            self.optimizer = SGD(named, lr=config.lr_peak)
        self.completed_updates = 0
        self.tokens_processed = 0
        self.best_val_loss: float | None = None
        self.best_step: int | None = None
        self.history: list[dict] = []
        self.lr_history: list[float] = []
        self.start_time = time.time()

    # -- construcción / reanudación ------------------------------------
    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, corpus_text: str, outdir: str | Path | None = None
    ) -> "Trainer":
        """Reconstruye un Trainer para REANUDAR (no solo cargar pesos)."""
        split_probe = split_corpus(corpus_text, train_fraction=0.9)  # solo para sha; se rehace abajo
        ckpt = load_checkpoint(checkpoint_path, expected_corpus_sha256=split_probe.sha256)
        training_config = TrainingConfig.from_dict(ckpt.metadata["training_config"])
        split = split_corpus(
            corpus_text,
            train_fraction=training_config.train_fraction,
            sequence_length=training_config.sequence_length,
        )
        if split.tokenizer.tokens != ckpt.tokenizer.tokens:
            raise ValueError(
                "El vocabulario reconstruido desde el corpus no coincide con "
                "el del checkpoint: reanudar exige exactamente el vocabulario "
                "del checkpoint."
            )
        trainer = cls(
            ckpt.model,
            ckpt.tokenizer,
            split,
            training_config,
            outdir=outdir if outdir is not None else Path(checkpoint_path).parent,
        )
        state = ckpt.training_state
        trainer.completed_updates = int(state.get("completed_updates", 0))
        trainer.tokens_processed = int(state.get("tokens_processed", 0))
        trainer.best_val_loss = state.get("best_val_loss")
        trainer.best_step = state.get("best_step")
        trainer.history = list(state.get("history", []))
        trainer.lr_history = list(state.get("lr_history", []))
        if ckpt.optimizer_state is not None:
            if isinstance(trainer.optimizer, AdamW) and ckpt.optimizer_state.get("type") == "AdamW":
                trainer.optimizer.load_state_dict(ckpt.optimizer_state)
            elif isinstance(trainer.optimizer, SGD) and ckpt.optimizer_state.get("type") == "SGD":
                trainer.optimizer.load_state_dict(ckpt.optimizer_state)
            else:
                raise ValueError(
                    "El tipo de optimizador del checkpoint no coincide con el "
                    "de la configuración de entrenamiento."
                )
        if "train_rng_state" in state:
            trainer.train_rng = rng_from_state(state["train_rng_state"])
        return trainer

    # -- lotes ----------------------------------------------------------
    def _microbatches(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.fixed_batches is not None:
            return list(self.fixed_batches)
        if self.split is None:
            raise ValueError("Sin partición ni lotes fijos no se pueden muestrear lotes.")
        batches = []
        for _ in range(self.config.grad_accumulation_steps):
            batches.append(
                sample_batch(
                    self.split.train_tokens,
                    self.config.batch_size,
                    self.config.sequence_length,
                    self.train_rng,
                )
            )
        return batches

    # -- un paso --------------------------------------------------------
    def step(self) -> dict:
        """Una actualización completa del optimizador."""
        microbatches = self._microbatches()
        self.model.zero_grad()
        train_loss = compute_batch_gradients(self.model, microbatches)
        norm_before, norm_after, clipped = clip_gradients(
            self.model.named_gradients(), self.config.max_grad_norm
        )
        lr = self.schedule(self.completed_updates + 1)
        self.optimizer.step(lr)
        self.completed_updates += 1
        total_targets = int(sum(int(y.size) for _, y in microbatches))
        self.tokens_processed += total_targets
        self.lr_history.append(float(lr))
        metrics = {
            "update": self.completed_updates,
            "train_loss": float(train_loss),
            "lr": float(lr),
            "grad_norm_before": float(norm_before),
            "grad_norm_after": float(norm_after),
            "clipped": bool(clipped),
            "tokens_processed": int(self.tokens_processed),
            "microbatches": len(microbatches),
            "elapsed_seconds": float(time.time() - self.start_time),
        }
        self.history.append(metrics)
        return metrics

    # -- evaluación -----------------------------------------------------
    def evaluate(self, split_name: str = "val") -> dict:
        """Evaluación determinista (no consume el generador de train)."""
        if self.split is None:
            raise ValueError("Sin partición no se puede evaluar.")
        tokens = self.split.train_tokens if split_name == "train" else self.split.val_tokens
        return evaluate_model(
            self.model,
            tokens,
            self.config.sequence_length,
            batch_size=self.config.batch_size,
            limit_windows=self.config.eval_subset_windows,
        )

    def evaluate_fixed_batches(self) -> dict:
        """Pérdida y exactitud sobre los lotes fijos (modo overfit)."""
        if self.fixed_batches is None:
            raise ValueError("No hay lotes fijos.")
        total_nll = 0.0
        total_targets = 0
        total_correct = 0
        for x, y in self.fixed_batches:
            logits = self.model.forward(x)
            assert isinstance(logits, np.ndarray)
            loss, _ = cross_entropy_with_grad(logits, y)
            total_nll += loss * int(y.size)
            total_targets += int(y.size)
            total_correct += int(np.sum(np.argmax(logits, axis=-1) == y))
        mean_loss = total_nll / total_targets if total_targets else float("nan")
        return {
            "mean_loss": float(mean_loss),
            "perplexity": float(math.exp(mean_loss)) if math.isfinite(mean_loss) else None,
            "token_accuracy": float(total_correct / total_targets) if total_targets else 0.0,
            "total_targets": int(total_targets),
        }

    # -- muestras -------------------------------------------------------
    def sample_text(self, prompt: str | None = None, temperature: float | None = None, greedy: bool = False) -> str:
        """Muestra con un generador nuevo por llamada (comparable y sin
        alterar el entrenamiento)."""
        prompt = prompt if prompt is not None else self.config.sample_prompt
        prompt_ids = self.tokenizer.encode(prompt)
        rng = np.random.default_rng(self.sample_seed)
        full = generate(
            self.model,
            prompt_ids,
            max_new_tokens=self.config.sample_max_tokens,
            temperature=0.0 if greedy else (self.config.sample_temperature if temperature is None else temperature),
            top_k=None,
            rng=rng,
        )
        return self.tokenizer.decode(full)

    # -- checkpoints ----------------------------------------------------
    def _checkpoint_payload(self) -> dict:
        state: dict = {
            "completed_updates": self.completed_updates,
            "tokens_processed": self.tokens_processed,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "history": self.history,
            "lr_history": self.lr_history,
            "train_rng_state": _rng_state_to_json(self.train_rng),
        }
        return state

    def save(self, path: str | Path) -> Path:
        """Guarda last/best (el llamador elige el nombre)."""
        corpus_info = self.split.to_dict() if self.split is not None else {}
        eval_starts = (
            eval_window_starts(
                self.split.val_tokens,
                self.config.sequence_length,
                limit=self.config.eval_subset_windows,
            )
            if self.split is not None
            else []
        )
        optimizer_state = self.optimizer.state_dict() if hasattr(self.optimizer, "state_dict") else None
        return save_checkpoint(
            path,
            self.model,
            self.tokenizer,
            self.config.to_dict(),
            optimizer_state,
            self._checkpoint_payload(),
            corpus_info,
            self.schedule.to_dict(),
            eval_window_starts=eval_starts,
        )

    def _write_jsonl(self, metrics: dict) -> None:
        if self.outdir is None:
            return
        self.outdir.mkdir(parents=True, exist_ok=True)
        with open(self.outdir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def _write_config_and_dataset(self) -> None:
        if self.outdir is None:
            return
        self.outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_config": self.model.config.to_dict(),
            "training_config": self.config.to_dict(),
            "dataset": self.split.to_dict() if self.split is not None else None,
            "python_version": __import__("sys").version,
            "numpy_version": np.__version__,
            "dtype": str(self.model.dtype),
        }
        (self.outdir / "config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -- ejecución ------------------------------------------------------
    def run(self, limit_updates: int | None = None) -> dict:
        """Ejecuta hasta total_updates (o hasta limit_updates si se indica).

        Devuelve un informe con métricas iniciales/finales, baselines y
        muestras antes/después. Si outdir está definido, escribe
        config.json, metrics.jsonl, last.npz, best.npz, muestras y
        report.md.
        """
        self._write_config_and_dataset()
        target = self.config.total_updates if limit_updates is None else min(limit_updates, self.config.total_updates)
        initial: dict = {}
        if self.use_validation:
            initial = self.evaluate("val")
        elif self.fixed_batches is not None:
            initial = self.evaluate_fixed_batches()
        samples_before = self.sample_text()
        greedy_before = self.sample_text(greedy=True)
        if self.outdir is not None:
            (self.outdir / "samples_before.txt").write_text(samples_before, encoding="utf-8")
            (self.outdir / "samples_greedy_before.txt").write_text(greedy_before, encoding="utf-8")
        while self.completed_updates < target:
            metrics = self.step()
            val_metrics: dict = {}
            if self.use_validation and self.config.eval_every > 0 and (
                self.completed_updates % self.config.eval_every == 0 or self.completed_updates == target
            ):
                val = self.evaluate("val")
                val_metrics = {
                    "val_loss": val["mean_loss"],
                    "val_perplexity": val["perplexity"],
                    "val_accuracy": val["token_accuracy"],
                    "val_coverage": val["coverage"],
                }
                if self.best_val_loss is None or val["mean_loss"] < self.best_val_loss:
                    self.best_val_loss = val["mean_loss"]
                    self.best_step = self.completed_updates
                    if self.outdir is not None:
                        self.save(self.outdir / "best.npz")
                metrics.update(val_metrics)
            self._write_jsonl(metrics)
            if self.outdir is not None and self.config.checkpoint_every > 0 and (
                self.completed_updates % self.config.checkpoint_every == 0 or self.completed_updates == target
            ):
                self.save(self.outdir / "last.npz")
        final: dict = {}
        if self.use_validation:
            final = self.evaluate("val")
        elif self.fixed_batches is not None:
            final = self.evaluate_fixed_batches()
        samples_after = self.sample_text()
        greedy_after = self.sample_text(greedy=True)
        if self.outdir is not None:
            (self.outdir / "samples_after.txt").write_text(samples_after, encoding="utf-8")
            (self.outdir / "samples_greedy_after.txt").write_text(greedy_after, encoding="utf-8")
            self.save(self.outdir / "last.npz")
            if self.best_val_loss is None:
                self.save(self.outdir / "best.npz")
        baselines: dict = {}
        if self.split is not None:
            vocab_size = self.tokenizer.vocab_size
            baselines["uniform"] = uniform_baseline(vocab_size)
            baselines["unigram_val"] = unigram_baseline(
                self.split.train_tokens, self.split.val_tokens, vocab_size
            )
        report = {
            "completed_updates": self.completed_updates,
            "tokens_processed": self.tokens_processed,
            "initial_eval": initial,
            "final_eval": final,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "baselines": baselines,
            "samples_before": samples_before,
            "samples_after": samples_after,
            "greedy_before": greedy_before,
            "greedy_after": greedy_after,
            "history": self.history,
            "lr_history": self.lr_history,
        }
        if self.outdir is not None:
            self._write_report(report)
        return report

    def _write_report(self, report: dict) -> None:
        assert self.outdir is not None
        lines = [
            "# Informe de entrenamiento (fase 2)",
            "",
            f"- Actualizaciones completadas: {report['completed_updates']}",
            f"- Tokens objetivo procesados: {report['tokens_processed']}",
            f"- Mejor pérdida de validación: {report['best_val_loss']} (paso {report['best_step']})",
            "",
            "## Evaluación inicial",
            "",
            "```json",
            json.dumps(report["initial_eval"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Evaluación final",
            "",
            "```json",
            json.dumps(report["final_eval"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Baselines",
            "",
            "```json",
            json.dumps(report["baselines"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Muestra al inicio de esta ejecución",
            "",
            report["samples_before"],
            "",
            "## Muestra al final de esta ejecución",
            "",
            report["samples_after"],
            "",
            "## Muestra greedy al inicio de esta ejecución",
            "",
            report["greedy_before"],
            "",
            "## Muestra greedy al final de esta ejecución",
            "",
            report["greedy_after"],
            "",
            "Nota: que el texto generado sea más plausible no significa que "
            "el modelo 'entienda' el texto; describe patrones observados y "
            "sus límites. La métrica principal es la pérdida/perplexity.",
        ]
        (self.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
