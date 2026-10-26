"""Backend local del visualizador educativo (Fase 1, MVP).

Sirve el modelo Python real (numpy_llm) mediante una API JSON pequeña,
usando solo la biblioteca estándar además de NumPy (vía el modelo).

No entrena, no modifica pesos y no reimplementa el modelo: cada
respuesta se calcula con el mismo código que usa la CLI.

Uso (desde la raíz del repositorio numpy-llm)::

    .venv/bin/python visualizer/server.py

    # o, en Windows (PowerShell):
    .venv\\Scripts\\python visualizer/server.py

El servidor escucha en http://localhost:8000 y el frontend de Vite
redirige /api a esa dirección.

Endpoints:
    GET  /api/health
    GET  /api/info
    POST /api/tokenize  {"text": "..."}
    POST /api/forward   {"text": "..."}
    POST /api/generate  {"prompt": "...", "max_new_tokens": 30,
                         "temperature": 1.0, "top_k": 20, "seed": 42}
    POST /api/loss-demo {"text": "..."}   (Fase 2: pérdida y gradiente
        de UNA ventana, con un modelo nuevo sin entrenar; no modifica
        el modelo principal del visualizador)
    POST /api/train-demo {"updates": 60}  (Fase 2: mini-entrenamiento
        real sobre "abcabc...", modelo diminuto aparte, con curva de
        pérdida y generación greedy antes/después)
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# El paquete numpy_llm vive en ../src respecto a este archivo.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402

from numpy_llm.config import ModelConfig  # noqa: E402
from numpy_llm.functional import cross_entropy_with_grad, softmax  # noqa: E402
from numpy_llm.generation import generate  # noqa: E402
from numpy_llm.model import TransformerLanguageModel  # noqa: E402
from numpy_llm.optimizers import AdamW, clip_gradients  # noqa: E402
from numpy_llm.tokenizer import CharTokenizer  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000
DEFAULT_SEED = 42
MAX_TEXT_LENGTH = 512
MAX_NEW_TOKENS = 200


def build_model_and_tokenizer() -> tuple[TransformerLanguageModel, CharTokenizer]:
    """Construye el modelo de la Fase 1 desde data/ejemplo.txt."""
    corpus = (REPO_ROOT / "data" / "ejemplo.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(corpus)
    config = ModelConfig(seed=DEFAULT_SEED).with_vocab_size(tokenizer.vocab_size)
    return TransformerLanguageModel(config), tokenizer


MODEL, TOKENIZER = build_model_and_tokenizer()


def parameter_breakdown() -> tuple[list[dict], list[dict]]:
    """Desglose de parámetros por componente, sumando tamaños reales."""
    groups: dict[str, int] = {}
    details: list[dict] = []
    for name, array in MODEL.named_parameters():
        details.append({"name": name, "shape": list(array.shape), "size": int(array.size)})
        group = name.split(".")[0]
        if group == "blocks":
            group = ".".join(name.split(".")[:2])
        groups[group] = groups.get(group, 0) + int(array.size)
    return [
        {"name": name, "size": size}
        for name, size in sorted(groups.items())
    ], details


def token_view(text: str) -> list[dict]:
    """Vista por carácter: carácter, id y si es desconocido (<UNK>)."""
    token_ids = TOKENIZER.encode(text)
    view = []
    for char, token_id in zip(text, token_ids):
        view.append(
            {
                "char": char,
                "id": token_id,
                "is_unknown": token_id == 0,
                "display": TOKENIZER.tokens[token_id],
            }
        )
    return view


def info_payload() -> dict:
    breakdown, details = parameter_breakdown()
    return {
        "config": MODEL.config.to_dict(),
        "vocab_size": TOKENIZER.vocab_size,
        "vocab": TOKENIZER.tokens,
        "total_parameters": MODEL.count_parameters(),
        "parameter_breakdown": breakdown,
        "parameters": details,
        "untrained_notice": (
            "Modelo sin entrenar: sus pesos son aleatorios y el texto "
            "generado es esencialmente aleatorio. Esto no indica un "
            "fallo de implementación."
        ),
    }


def forward_payload(text: str) -> dict:
    """Recorrido con formas, usando los mismos componentes del modelo."""
    if not text:
        raise ValueError("El texto no puede estar vacío.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"El texto debe tener como máximo {MAX_TEXT_LENGTH} caracteres.")
    token_ids = TOKENIZER.encode(text)
    inputs = np.asarray([token_ids], dtype=np.int64)
    # Se recorren los mismos componentes que model.forward(), para que
    # las formas mostradas sean las reales y no una lista escrita a mano.
    hidden = MODEL.embeddings.forward(inputs)
    steps = [
        {"step": "token_ids", "description": "Identificadores de token", "shape": list(inputs.shape)},
        {"step": "embeddings", "description": "Tokens + posiciones", "shape": list(hidden.shape)},
    ]
    for i, block in enumerate(MODEL.blocks):
        hidden = block.forward(hidden)  # type: ignore[assignment]
        steps.append(
            {
                "step": f"blocks.{i}",
                "description": f"Bloque Transformer {i + 1}",
                "shape": list(hidden.shape),
            }
        )
    hidden = MODEL.final_ln.forward(hidden)
    steps.append({"step": "final_ln", "description": "LayerNorm final", "shape": list(hidden.shape)})
    logits = MODEL.lm_head.forward(hidden)
    steps.append({"step": "logits", "description": "Proyección al vocabulario", "shape": list(logits.shape)})
    return {
        "text": text,
        "token_ids": token_ids,
        "tokens": token_view(text),
        "steps": steps,
        "logits_shape": list(logits.shape),
    }


def generate_payload(data: dict) -> dict:
    prompt = data.get("prompt", "")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("El prompt no puede estar vacío.")
    max_new_tokens = data.get("max_new_tokens", 30)
    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k")
    seed = data.get("seed", DEFAULT_SEED)
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens debe ser un entero.")
    if max_new_tokens < 0 or max_new_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens debe estar entre 0 y {MAX_NEW_TOKENS}.")
    prompt_ids = TOKENIZER.encode(prompt)
    full_ids = generate(
        MODEL,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        top_k=top_k,
        rng=np.random.default_rng(seed),
    )
    generated_ids = full_ids[len(prompt_ids):]
    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated_ids,
        "full_token_ids": full_ids,
        "generated_text": TOKENIZER.decode(generated_ids),
        "full_text": TOKENIZER.decode(full_ids),
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    }


def _group_name(name: str) -> str:
    """Agrupa un parámetro para mostrar su norma de gradiente."""
    parts = name.split(".")
    if parts[0] == "blocks":
        return ".".join(parts[:2])  # blocks.0, blocks.1, ...
    return parts[0]


def loss_demo_payload(data: dict) -> dict:
    """Pérdida y gradiente de UNA ventana, sin entrenar nada.

    Se crea un modelo nuevo con la misma semilla que el principal, se
    toma la primera ventana de 9 tokens del texto (x = 8 primeros,
    y = 8 siguientes, teacher forcing) y se hace forward + backward.
    El modelo principal del visualizador NO se toca: este laboratorio
    mide, no entrena. Devuelve pérdida, perplexity, exactitud y la
    norma del gradiente por grupo de parámetros, más la probabilidad
    que el modelo (sin entrenar) asigna al token correcto en cada
    posición, para que se vea qué significa "la pérdida es alta".
    """
    import math

    text = data.get("text", "")
    if not isinstance(text, str) or len(text) < 9:
        raise ValueError("Escribe al menos 9 caracteres para formar una ventana de 9 tokens.")
    token_ids = TOKENIZER.encode(text)[:9]
    if len(token_ids) < 9:
        raise ValueError("Se necesitan al menos 9 tokens.")
    x = np.asarray([token_ids[:8]], dtype=np.int64)
    y = np.asarray([token_ids[1:9]], dtype=np.int64)
    model = TransformerLanguageModel(
        ModelConfig(seed=DEFAULT_SEED).with_vocab_size(TOKENIZER.vocab_size)
    )
    model.zero_grad()
    logits = model.forward(x, retain_cache=True)
    assert isinstance(logits, np.ndarray)
    loss, dlogits = cross_entropy_with_grad(logits, y)
    model.backward(dlogits)
    probs = softmax(logits.astype(np.float64), axis=-1)
    predicted = logits.argmax(axis=-1)[0].tolist()
    correct = y[0].tolist()
    prob_correct = [float(probs[0, t, correct[t]]) for t in range(8)]
    groups: dict[str, float] = {}
    for name, grad in model.named_gradients():
        groups[_group_name(name)] = groups.get(_group_name(name), 0.0) + float(
            np.sum(grad.astype(np.float64) ** 2)
        )
    grad_norms = {name: float(np.sqrt(value)) for name, value in sorted(groups.items())}
    total_norm = float(np.sqrt(sum(value for value in groups.values())))
    accuracy = float(np.mean([p == c for p, c in zip(predicted, correct)]))
    model.zero_grad()
    model.clear_cache()
    return {
        "text": text[:9],
        "input_token_ids": x[0].tolist(),
        "target_token_ids": y[0].tolist(),
        "input_text": TOKENIZER.decode(x[0].tolist()),
        "target_text": TOKENIZER.decode(y[0].tolist()),
        "loss": float(loss),
        "perplexity": float(math.exp(loss)) if loss < 700 else float("inf"),
        "token_accuracy": accuracy,
        "predicted_token_ids": predicted,
        "prob_correct": prob_correct,
        "grad_norm_by_group": grad_norms,
        "grad_norm_total": total_norm,
        "all_finite": bool(np.all(np.isfinite(logits))),
    }


def _mini_decode(tokenizer: CharTokenizer, token_ids: list[int]) -> str:
    """Decodifica mostrando <UNK> como �, para que se lea en pantalla."""
    return "".join("�" if token_id == 0 else tokenizer.tokens[token_id] for token_id in token_ids)


def train_demo_payload(data: dict) -> dict:
    """Mini-entrenamiento real: memorizar "abcabc...".

    Modelo diminuto aparte (D=16, H=2, L=1, F=32, vocabulario de solo
    4 tokens: <UNK>, a, b, c), un único lote fijo de 8 posiciones y
    AdamW con lr constante 3e-3, la configuración OVERFIT del repo
    reducida a los pasos que pide el usuario. Es una PRUEBA DE
    MEMORIZACIÓN, no una demostración de generalización: sirve para
    comprobar que el sistema aprende, antes de entrenar con un corpus
    real. No toca el modelo principal del visualizador.
    """
    updates = data.get("updates", 60)
    if not isinstance(updates, int) or isinstance(updates, bool) or not 10 <= updates <= 300:
        raise ValueError("updates debe ser un entero entre 10 y 300.")
    tokenizer = CharTokenizer.from_tokens(["<UNK>", "a", "b", "c"])
    config = ModelConfig(
        context_length=8, d_model=16, n_heads=2, n_layers=1, d_ff=32, seed=42
    ).with_vocab_size(tokenizer.vocab_size)
    model = TransformerLanguageModel(config)
    sequence = np.asarray([tokenizer.encode("abcabcabcabcabcabc")], dtype=np.int64)
    x, y = sequence[:, :8], sequence[:, 1:9]

    def evaluate() -> tuple[float, float]:
        from numpy_llm.functional import cross_entropy_loss

        logits = model.forward(x)
        assert isinstance(logits, np.ndarray)
        loss = cross_entropy_loss(logits, y)
        accuracy = float(np.mean(logits.argmax(axis=-1) == y))
        return float(loss), accuracy

    def greedy_text() -> str:
        full = generate(
            model, [1, 2], 20, temperature=0.0, rng=np.random.default_rng(42)
        )
        return _mini_decode(tokenizer, full[2:])

    loss_before, acc_before = evaluate()
    greedy_before = greedy_text()
    params = [
        (name, param, grad)
        for (name, param), (_, grad) in zip(model.named_parameters(), model.named_gradients())
    ]
    optimizer = AdamW(params, lr=3e-3, weight_decay=0.0)
    curve: list[dict] = []
    for step in range(updates):
        model.zero_grad()
        logits = model.forward(x, retain_cache=True)
        assert isinstance(logits, np.ndarray)
        loss, dlogits = cross_entropy_with_grad(logits, y)
        model.backward(dlogits)
        clip_gradients(model.named_gradients(), 1.0)
        optimizer.step(3e-3)
        if step % 10 == 0 or step == updates - 1:
            curve.append({"update": step + 1, "loss_before_update": float(loss)})
    loss_after, acc_after = evaluate()
    greedy_after = greedy_text()
    return {
        "task": "Memorizar la secuencia abcabc... (un solo lote fijo, 8 posiciones)",
        "model": "D=16, H=2, L=1, F=32, vocabulario de 4 tokens",
        "optimizer": "AdamW, lr constante 3e-3, weight_decay 0, clipping 1.0",
        "updates": updates,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "loss_ratio": (loss_after / loss_before) if loss_before else None,
        "curve": curve,
        "greedy_before": greedy_before,
        "greedy_after": greedy_after,
        "prompt": "abc",
        "warning": (
            "Prueba de memorización, no de generalización: el modelo solo "
            "ha visto este lote. Que memorice no demuestra que entienda "
            "texto nuevo."
        ),
    }


class Handler(BaseHTTPRequestHandler):
    """Manejador JSON mínimo, solo para desarrollo local."""

    def log_message(self, *args):  # Silencia el log por defecto.
        pass

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors(self) -> None:
        # Desarrollo local: el frontend de Vite corre en localhost:5173.
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                payload: dict = {"status": "ok"}
            elif path == "/api/info":
                payload = info_payload()
            else:
                self._send_json(404, {"error": f"Ruta desconocida: {path}"})
                return
        except Exception as exc:  # noqa: BLE001 - API local educativa
            self._send_json(500, {"error": str(exc)})
            return
        # Reenviar con CORS: _send_json ya envió cabeceras, así que
        # para GET se usa una variante sencilla.
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("El cuerpo debe ser un objeto JSON.")
            if path == "/api/tokenize":
                text = data.get("text", "")
                if not isinstance(text, str):
                    raise ValueError("text debe ser una cadena.")
                payload: dict = {"text": text, "token_ids": TOKENIZER.encode(text), "tokens": token_view(text)}
            elif path == "/api/forward":
                payload = forward_payload(data.get("text", ""))
            elif path == "/api/generate":
                payload = generate_payload(data)
            elif path == "/api/loss-demo":
                payload = loss_demo_payload(data)
            elif path == "/api/train-demo":
                payload = train_demo_payload(data)
            else:
                self._send_json(404, {"error": f"Ruta desconocida: {path}"})
                return
        except Exception as exc:  # noqa: BLE001 - errores claros al cliente
            self._send_json(400, {"error": str(exc)})
            return
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Visualizador numpy-llm (backend) en http://{HOST}:{PORT}")
    print(f"Vocabulario: {TOKENIZER.vocab_size} tokens, parámetros: {MODEL.count_parameters()}")
    print("Pulsa Ctrl+C para detenerlo.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")


if __name__ == "__main__":
    main()
