import numpy as np
import pytest

from space_rag.chunking import Chunk
from space_rag.vector_store import VectorStore


def _store():
    chunks = [Chunk(i, f"{i}.md", f"T{i}", f"text {i}") for i in range(3)]
    emb = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    return VectorStore(emb, chunks, meta={"embed_model": "x"})


def test_search_orders_by_cosine_similarity():
    hits = _store().search(np.array([1.0, 0.1]), k=2)
    assert [h.chunk.id for h in hits] == [0, 2]
    assert hits[0].score > hits[1].score


def test_search_k_larger_than_store():
    assert len(_store().search(np.array([1.0, 0.0]), k=10)) == 3


def test_save_and_load_roundtrip(tmp_path):
    store = _store()
    assert not VectorStore.exists(tmp_path)
    store.save(tmp_path)
    assert VectorStore.exists(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 3
    assert loaded.meta == {"embed_model": "x"}
    assert loaded.chunks[1].title == "T1"
    assert np.allclose(loaded.embeddings, store.embeddings)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        VectorStore(np.zeros((2, 2)), [Chunk(0, "a", "t", "x")])
