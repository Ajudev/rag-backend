"""Embedding backends used at ingest and query time."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

DEFAULT_DIM = 384


class Embedder(Protocol):
    """Produces dense vectors for a batch of texts."""

    dim: int

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed ``texts`` in batches.

        Args:
            texts: Passages or queries.
            batch_size: Maximum texts per model call.

        Returns:
            One vector per input text.
        """
        ...


class SentenceTransformerEmbedder:
    """Batched SentenceTransformer encoder."""

    def __init__(self, model_name: str, dim: int = DEFAULT_DIM) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = dim

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Encode texts with L2-normalized BGE-style embeddings."""
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [row.tolist() for row in vectors]


class FakeEmbedder:
    """Deterministic hash embeddings for tests (no model download)."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Return L2-normalized hash vectors, honoring ``batch_size``."""
        del batch_size
        return [_hash_vector(text, self.dim) for text in texts]


def _hash_vector(text: str, dim: int) -> list[float]:
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    raw: list[int] = []
    while len(raw) < dim:
        seed = hashlib.sha256(seed).digest()
        raw.extend(seed)
    values = [(byte / 255.0) - 0.5 for byte in raw[:dim]]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
