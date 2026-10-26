"""Paquete educativo numpy_llm.

Implementa un pequeño modelo de lenguaje autorregresivo con arquitectura
Transformer decoder-only, usando únicamente Python y NumPy.

Fase 1: inferencia (forward) y generación con pesos aleatorios.
Fase 2: backward manual, gradientes, optimizadores (SGD/AdamW),
clipping, acumulación, scheduler, entrenamiento con partición
train/validation, evaluación, checkpoints con reanudación exacta y
comprobación numérica de gradientes. Sin autograd ni frameworks.
"""

from .attention import CausalMultiHeadAttention
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig
from .data import SplitData, build_examples, eval_windows, sample_batch, split_corpus
from .evaluation import evaluate_model, uniform_baseline, unigram_baseline
from .functional import cross_entropy_loss, cross_entropy_with_grad, log_softmax, relu, softmax, softmax_backward
from .generation import generate
from .gradcheck import central_difference, check_value, gradcheck_model
from .layers import Embeddings, LayerNorm, Linear
from .losses import cross_entropy_with_grad as loss_with_grad
from .model import TransformerLanguageModel
from .optimizers import AdamW, SGD, clip_gradients, decay_for_name, global_grad_norm
from .schedules import LRSchedule, learning_rate
from .serialization import load_model, save_model
from .tokenizer import CharTokenizer
from .training import Trainer, TrainingConfig, compute_batch_gradients, trace_step
from .transformer import FeedForward, TransformerBlock

__all__ = [
    "AdamW",
    "CausalMultiHeadAttention",
    "CharTokenizer",
    "Embeddings",
    "FeedForward",
    "LayerNorm",
    "Linear",
    "LRSchedule",
    "ModelConfig",
    "SGD",
    "SplitData",
    "Trainer",
    "TrainingConfig",
    "TransformerBlock",
    "TransformerLanguageModel",
    "build_examples",
    "central_difference",
    "check_value",
    "clip_gradients",
    "compute_batch_gradients",
    "cross_entropy_loss",
    "cross_entropy_with_grad",
    "decay_for_name",
    "eval_windows",
    "evaluate_model",
    "generate",
    "global_grad_norm",
    "gradcheck_model",
    "learning_rate",
    "load_checkpoint",
    "load_model",
    "log_softmax",
    "loss_with_grad",
    "relu",
    "sample_batch",
    "save_checkpoint",
    "save_model",
    "softmax",
    "softmax_backward",
    "split_corpus",
    "trace_step",
    "uniform_baseline",
    "unigram_baseline",
]
