"""Central configuration. Every value can be overridden with an env variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    docs_dir: Path = ROOT / "data"
    index_dir: Path = ROOT / "index"

    # Embedding model (~90 MB) and the small generator LLM (~1 GB download).
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Chunking (measured in words).
    chunk_words: int = 120
    chunk_overlap: int = 30

    # Retrieval.
    top_k: int = 3
    min_score: float = 0.2  # below this cosine similarity we refuse to answer

    # Generation.
    max_new_tokens: int = 256

    @classmethod
    def from_env(cls) -> "Config":
        d = cls()
        return cls(
            docs_dir=Path(os.getenv("RAG_DOCS_DIR", d.docs_dir)),
            index_dir=Path(os.getenv("RAG_INDEX_DIR", d.index_dir)),
            embed_model=os.getenv("RAG_EMBED_MODEL", d.embed_model),
            llm_model=os.getenv("RAG_LLM_MODEL", d.llm_model),
            chunk_words=int(os.getenv("RAG_CHUNK_WORDS", d.chunk_words)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", d.chunk_overlap)),
            top_k=int(os.getenv("RAG_TOP_K", d.top_k)),
            min_score=float(os.getenv("RAG_MIN_SCORE", d.min_score)),
            max_new_tokens=int(os.getenv("RAG_MAX_NEW_TOKENS", d.max_new_tokens)),
        )
