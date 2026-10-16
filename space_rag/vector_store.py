"""A minimal in-memory vector store: a numpy matrix + cosine similarity.

For a few thousand chunks, brute-force search is instant and has zero extra
dependencies. For millions of chunks, swap this class for FAISS / Chroma / Qdrant;
the rest of the pipeline only needs `.search()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: list[Chunk], meta: dict | None = None):
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")
        emb = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = emb / norms
        self.chunks = chunks
        self.meta = meta or {}

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[Hit]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    # ---- persistence -------------------------------------------------------
    def save(self, directory: Path | str, meta: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (directory / "meta.json").write_text(json.dumps(meta or self.meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> "VectorStore":
        directory = Path(directory)
        embeddings = np.load(directory / "embeddings.npy")
        chunks = [Chunk(**d) for d in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        meta_path = directory / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(embeddings, chunks, meta)

    @staticmethod
    def exists(directory: Path | str) -> bool:
        d = Path(directory)
        return (d / "embeddings.npy").exists() and (d / "chunks.json").exists()
