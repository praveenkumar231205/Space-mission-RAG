"""Load documents from disk and split them into overlapping word-window chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass
class Chunk:
    id: int
    source: str  # file name
    title: str  # document title (first "# " heading, or the file name)
    text: str

    @property
    def contextual_text(self) -> str:
        """Text that gets embedded: the title is prepended so that a chunk like
        'It launched in 1977' still carries the subject it belongs to."""
        return f"{self.title}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_documents(docs_dir: Path | str) -> list[tuple[str, str]]:
    """Return [(file_name, text), ...] for every .md / .txt file, sorted by name."""
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {docs_dir}")
    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [(p.name, p.read_text(encoding="utf-8")) for p in files]


def chunk_text(text: str, chunk_words: int = 120, overlap: int = 30) -> list[str]:
    """Split text into windows of `chunk_words` words that share `overlap` words."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must be >= 0 and smaller than chunk_words")

    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):  # window reached the end
            break
    return chunks


def build_chunks(docs_dir: Path | str, chunk_words: int = 120, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name, text in load_documents(docs_dir):
        title = extract_title(text, fallback=Path(name).stem.replace("_", " ").title())
        # Drop the "# Title" line itself; the title travels in Chunk.title instead.
        body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("# "))
        for piece in chunk_text(body, chunk_words, overlap):
            chunks.append(Chunk(id=len(chunks), source=name, title=title, text=piece))
    return chunks
