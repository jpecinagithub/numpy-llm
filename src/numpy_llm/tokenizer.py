"""Tokenizador a nivel de carácter.

Un token ordinario es aquí un carácter Unicode tal como lo representa
Python: no es necesariamente una palabra ni un byte. El vocabulario se
construye a partir de un corpus y después queda fijo.

Reglas:

- El identificador 0 se reserva para el token especial ``<UNK>``.
- El resto de caracteres se ordena de forma determinista (por código
  Unicode), de modo que el mismo corpus produce siempre el mismo
  vocabulario.
- Se conservan mayúsculas, espacios, saltos de línea y tildes.
- Los caracteres desconocidos se codifican como ``<UNK>`` (id 0).

Consecuencia importante: para textos cuyos caracteres están todos en el
vocabulario se cumple ``decode(encode(texto)) == texto``. Cuando aparece
un carácter desconocido esa propiedad ya no puede recuperarse exactamente,
porque varios caracteres distintos colapsan en el mismo ``<UNK>``.

Crear el vocabulario NO equivale a entrenar los pesos del modelo: el
vocabulario solo enumera los caracteres posibles; los embeddings y demás
pesos siguen siendo aleatorios hasta que se entrenen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

UNK_TOKEN = "<UNK>"
UNK_ID = 0


class CharTokenizer:
    """Tokenizador determinista a nivel de carácter."""

    def __init__(self, tokens: Sequence[str]) -> None:
        """Crea el tokenizador desde la lista de tokens en orden de id.

        ``tokens[0]`` debe ser ``<UNK>`` y el resto deben ser caracteres
        únicos de longitud 1. Normalmente no se llama directamente: usa
        :meth:`from_text` o :meth:`from_tokens`.
        """
        tokens = list(tokens)
        if not tokens or tokens[0] != UNK_TOKEN:
            raise ValueError("El primer token del vocabulario debe ser '<UNK>'.")
        if len(set(tokens)) != len(tokens):
            raise ValueError("El vocabulario contiene tokens duplicados.")
        for token in tokens[1:]:
            if not isinstance(token, str) or len(token) != 1:
                raise ValueError(
                    f"Cada token ordinario debe ser un único carácter, no {token!r}."
                )
        self._tokens: list[str] = tokens
        self._token_to_id: dict[str, int] = {token: i for i, token in enumerate(tokens)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Construye el vocabulario a partir de un corpus de texto."""
        if not isinstance(text, str) or not text:
            raise ValueError("El corpus debe ser un texto no vacío.")
        # sorted() da un orden estable y determinista.
        characters = sorted(set(text))
        return cls([UNK_TOKEN] + characters)

    @classmethod
    def from_tokens(cls, tokens: Sequence[str]) -> "CharTokenizer":
        """Reconstruye el tokenizador desde una lista de tokens en orden."""
        return cls(tokens)

    @property
    def vocab_size(self) -> int:
        """V: número de tokens del vocabulario, incluido <UNK>."""
        return len(self._tokens)

    @property
    def tokens(self) -> list[str]:
        """Copia de la lista de tokens en orden de identificador."""
        return list(self._tokens)

    def encode(self, text: str) -> list[int]:
        """Convierte texto en identificadores de token.

        Los caracteres desconocidos se representan con el id 0 (<UNK>).
        """
        if not isinstance(text, str):
            raise ValueError(f"encode espera texto, no {type(text).__name__}.")
        return [self._token_to_id.get(char, UNK_ID) for char in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convierte identificadores de token de nuevo en texto."""
        ids = list(token_ids)
        characters: list[str] = []
        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError(f"Los identificadores deben ser enteros, no {token_id!r}.")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(
                    f"Identificador {token_id} fuera del vocabulario "
                    f"(tamaño {self.vocab_size})."
                )
            characters.append(self._tokens[token_id])
        return "".join(characters)

    def to_dict(self) -> dict:
        """Representación JSON-compatible del vocabulario."""
        return {"unk_token": UNK_TOKEN, "tokens": self.tokens}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        """Reconstruye el tokenizador desde :meth:`to_dict`."""
        if not isinstance(data, dict) or "tokens" not in data:
            raise ValueError("El vocabulario guardado debe contener la clave 'tokens'.")
        return cls(data["tokens"])

    def save_vocab(self, path: str | Path) -> Path:
        """Guarda el vocabulario en un archivo JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_vocab(cls, path: str | Path) -> "CharTokenizer":
        """Recupera un vocabulario guardado con :meth:`save_vocab`."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
