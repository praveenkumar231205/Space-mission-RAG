import pytest

from space_rag.chunking import build_chunks, chunk_text, extract_title
from space_rag.config import Config


def test_chunk_text_windows_and_overlap():
    text = "a b c d e f g h i j"
    chunks = chunk_text(text, chunk_words=4, overlap=1)
    assert chunks == ["a b c d", "d e f g", "g h i j"]


def test_short_text_is_single_chunk():
    assert chunk_text("just a few words", 50, 10) == ["just a few words"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n  ", 10, 2) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, 9), (5, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text here", size, overlap)


def test_extract_title():
    assert extract_title("intro\n# My Title\nbody", "fallback") == "My Title"
    assert extract_title("no heading", "fallback") == "fallback"


def test_bundled_documents_are_chunked():
    chunks = build_chunks(Config().docs_dir, 120, 30)
    assert len(chunks) >= 8
    assert all(c.title and c.text and c.source.endswith(".md") for c in chunks)
    assert [c.id for c in chunks] == list(range(len(chunks)))
    # the "# Title" line must not leak into the chunk body
    assert not any(c.text.startswith("# ") for c in chunks)
