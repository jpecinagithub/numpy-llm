"""Paquete educativo numpy_llm.

Implementa un pequeño modelo de lenguaje autorregresivo con arquitectura
Transformer decoder-only, usando únicamente Python y NumPy.

Esta primera fase solo hace inferencia (forward) y generación con pesos
aleatorios. No hay entrenamiento, gradientes ni optimizadores.
"""

from .attention import CausalMultiHeadAttention
from .config import ModelConfig
from .data import build_examples
from .functional import cross_entropy_loss, log_softmax, relu, softmax
from .generation import generate
from .layers import Embeddings, LayerNorm, Linear
from .model import TransformerLanguageModel
from .serialization import load_model, save_model
from .tokenizer import CharTokenizer
from .transformer import FeedForward, TransformerBlock

__all__ = [
    "CausalMultiHeadAttention",
    "CharTokenizer",
    "Embeddings",
    "FeedForward",
    "LayerNorm",
    "Linear",
    "ModelConfig",
    "TransformerBlock",
    "TransformerLanguageModel",
    "build_examples",
    "cross_entropy_loss",
    "generate",
    "load_model",
    "log_softmax",
    "relu",
    "save_model",
    "softmax",
]
