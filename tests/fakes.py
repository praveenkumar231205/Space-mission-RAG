"""Lightweight stand-ins so the test-suite needs no model downloads."""

from __future__ import annotations

import re
import zlib

import numpy as np

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "was", "were", "it",
    "by", "for", "with", "as", "at", "that", "this", "which", "who", "what", "how",
    "did", "does", "from", "be", "has", "have", "its",
}


class FakeEmbedder:
    """Hashed bag-of-words embedder: deterministic, lexical, L2-normalised."""

    dim = 1024

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok not in STOPWORDS:
                    out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class FakeLLM:
    """Records the prompt it was given and returns a canned answer."""

    def __init__(self, reply: str = "stub answer [1]"):
        self.reply = reply
        self.last_messages: list[dict] | None = None

    def generate(self, messages: list[dict]) -> str:
        self.last_messages = messages
        return self.reply
