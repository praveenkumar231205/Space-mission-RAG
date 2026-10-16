"""The RAG pipeline: retrieve relevant chunks, build a grounded prompt, generate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .chunking import build_chunks
from .vector_store import Hit, VectorStore

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the numbered "
    "context passages provided by the user. "
    f'If the answer is not in the context, reply exactly: "{NO_ANSWER}" '
    "Keep answers short (1-3 sentences) and cite the passages you used, like [1] or [2]."
)


class EmbedderLike(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LLMLike(Protocol):
    def generate(self, messages: list[dict]) -> str: ...


@dataclass
class Answer:
    text: str  # empty string when running retrieval-only
    hits: list[Hit] = field(default_factory=list)
    used_llm: bool = False


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n\n".join(f"[{i}] ({h.chunk.title}) {h.chunk.text}" for i, h in enumerate(hits, 1))
    user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_index(
    docs_dir: Path | str,
    embedder: EmbedderLike,
    chunk_words: int = 120,
    overlap: int = 30,
) -> VectorStore:
    chunks = build_chunks(docs_dir, chunk_words, overlap)
    if not chunks:
        raise ValueError(f"No .md/.txt documents with text found in {docs_dir}")
    vectors = embedder.encode([c.contextual_text for c in chunks])
    return VectorStore(vectors, chunks)


class RAGPipeline:
    def __init__(
        self,
        embedder: EmbedderLike,
        store: VectorStore,
        llm: LLMLike | None = None,
        top_k: int = 3,
        min_score: float = 0.2,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str) -> list[Hit]:
        query_vec = self.embedder.encode([question])[0]
        return self.store.search(query_vec, self.top_k)

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)

        # Guardrail: if nothing in the knowledge base is even loosely related,
        # don't let a tiny LLM improvise an answer.
        if not hits or hits[0].score < self.min_score:
            return Answer(NO_ANSWER, hits, used_llm=False)

        if self.llm is None:  # retrieval-only mode
            return Answer("", hits, used_llm=False)

        text = self.llm.generate(build_messages(question, hits))
        return Answer(text, hits, used_llm=True)
