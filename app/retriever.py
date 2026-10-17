import hashlib
import logging
import re
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.embedding import PyTorchEmbedder

logger = logging.getLogger(__name__)

SearchMode = Literal["bm25", "dense", "hybrid"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def min_max(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)


class HybridRetriever:
    """Search over chunks with BM25 (keywords), dense embeddings (meaning), or a weighted mix of both."""

    def __init__(self, chunks: list[dict], embedder: PyTorchEmbedder):
        self.chunks = chunks
        self.embedder = embedder

        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.vector_index = self._load_or_build_embeddings()

    def _cache_file(self):
        # The key covers the model name AND the chunk texts, so changing either rebuilds the cache
        digest = hashlib.sha256(self.embedder.model_name.encode())
        for chunk in self.chunks:
            digest.update(chunk["text"].encode())
        return settings.cache_dir / f"embeddings_{digest.hexdigest()[:16]}.npy"

    def _load_or_build_embeddings(self) -> np.ndarray:
        cache_file = self._cache_file()

        if cache_file.exists():
            logger.info("Loaded embeddings from cache")
            return np.load(cache_file)

        logger.info("Embedding %d chunks (first run only)...", len(self.chunks))
        vectors = self.embedder.generate_embeddings([c["text"] for c in self.chunks])
        settings.cache_dir.mkdir(exist_ok=True)
        np.save(cache_file, vectors)
        return vectors

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: SearchMode = "hybrid",
        alpha: float | None = None,
    ) -> list[dict]:
        top_k = top_k or settings.default_top_k
        alpha = settings.alpha if alpha is None else alpha

        bm25_scores = np.array(self.bm25.get_scores(tokenize(query)))
        query_vec = self.embedder.generate_embeddings([query], is_query=True)
        similarity = np.dot(self.vector_index, query_vec.T).ravel()

        if mode == "bm25":
            final = min_max(bm25_scores)
        elif mode == "dense":
            final = min_max(similarity)
        else:
            final = alpha * min_max(bm25_scores) + (1 - alpha) * min_max(similarity)

        top_indices = np.argsort(final)[::-1][:top_k]
        return [
            {
                "chunk_id": self.chunks[i]["chunk_id"],
                "text": self.chunks[i]["text"],
                "source": self.chunks[i]["source"],
                "page": self.chunks[i]["page"],
                "score": round(float(final[i]), 3),
                "similarity": round(float(similarity[i]), 3),
            }
            for i in top_indices
        ]