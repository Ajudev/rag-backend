"""Cross-encoder reranking of query–passage pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from app.helpers import tokenize

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker(Protocol):
    """Scores (query, passage) pairs; higher is more relevant.

    Raw scores are ranking utilities, not calibrated probabilities.
    """

    def score_pairs(
        self,
        query: str,
        passages: Sequence[str],
        batch_size: int = 16,
    ) -> list[float]:
        """Return one score per passage, in the same order as ``passages``."""
        ...


class CrossEncoderReranker:
    """Sentence-Transformers CrossEncoder loaded once and reused."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device
        self._model = CrossEncoder(model_name, device=device)

    def score_pairs(
        self,
        query: str,
        passages: Sequence[str],
        batch_size: int = 16,
    ) -> list[float]:
        """Score query–passage pairs with a batched cross-encoder predict call.

        Scores are not probabilities and should not be interpreted as such.
        """
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs, batch_size=batch_size)
        return [float(score) for score in scores]


class FakeReranker:
    """Deterministic reranker for tests (no model download).

    Default scores are query-term overlap counts. Optional ``score_map`` keys
    are passage strings; ``prefer_substring`` adds a large bonus when present.
    """

    def __init__(
        self,
        score_map: Mapping[str, float] | None = None,
        prefer_substring: str | None = None,
    ) -> None:
        self.score_map = dict(score_map or {})
        self.prefer_substring = prefer_substring
        self.load_count = 1
        self.predict_calls = 0

    def score_pairs(
        self,
        query: str,
        passages: Sequence[str],
        batch_size: int = 16,
    ) -> list[float]:
        """Return overlap (or mapped) scores without downloading a model."""
        del batch_size
        self.predict_calls += 1
        query_tokens = set(tokenize(query))
        scored: list[float] = []
        for passage in passages:
            if passage in self.score_map:
                score = float(self.score_map[passage])
            else:
                passage_tokens = tokenize(passage)
                score = float(sum(1 for token in passage_tokens if token in query_tokens))
            if self.prefer_substring and self.prefer_substring in passage:
                score += 100.0
            scored.append(score)
        return scored
