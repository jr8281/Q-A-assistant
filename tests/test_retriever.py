import zlib

import numpy as np
import pytest

from app.config import settings
from app.retriever import HybridRetriever, min_max, tokenize


class FakeEmbedder:
    """Hashing-trick bag-of-words vectors: deterministic, tiny, no model download."""
    model_name = "fake"

    def generate_embeddings(self, texts, batch_size=32, is_query=False):
        vecs = np.zeros((len(texts), 64))
        for i, text in enumerate(texts):
            for token in tokenize(text):
                vecs[i, zlib.crc32(token.encode()) % 64] += 1
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


CHUNKS = [
    {"chunk_id": 0, "source": "a.pdf", "page": 1,
     "text": "govern map measure and manage are the core functions of the risk framework"},
    {"chunk_id": 1, "source": "a.pdf", "page": 2,
     "text": "confabulation means a model produces confident but false statements"},
    {"chunk_id": 2, "source": "b.pdf", "page": 1,
     "text": "profiles describe the current and target cybersecurity posture of an organization"},
]


@pytest.fixture
def retriever(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    return HybridRetriever(CHUNKS, FakeEmbedder())


def test_tokenize_lowercases_and_splits():
    assert tokenize("The GOVERN function, v2.0!") == ["the", "govern", "function", "v2", "0"]


def test_min_max_scales_to_unit_range():
    out = min_max(np.array([2.0, 4.0, 6.0]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_min_max_constant_input_does_not_divide_by_zero():
    assert (min_max(np.array([3.0, 3.0])) == 0).all()


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_best_chunk_is_found_in_every_mode(retriever, mode):
    top = retriever.search("what is confabulation", top_k=1, mode=mode)[0]
    assert (top["source"], top["page"]) == ("a.pdf", 2)


def test_top_k_is_respected_and_scores_are_sorted(retriever):
    results = retriever.search("cybersecurity profiles", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_embeddings_are_cached_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    HybridRetriever(CHUNKS, FakeEmbedder())
    assert len(list(tmp_path.glob("embeddings_*.npy"))) == 1

    class ExplodingEmbedder(FakeEmbedder):
        def generate_embeddings(self, texts, batch_size=32, is_query=False):
            raise AssertionError("cache was not used")

    HybridRetriever(CHUNKS, ExplodingEmbedder())   # must load from cache, not re-embed