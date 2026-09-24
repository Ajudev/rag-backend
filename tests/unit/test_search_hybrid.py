from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import InvalidSearchParamsError
from app.services.reranker import FakeReranker
from app.services.search import SearchService

pytestmark = pytest.mark.unit


def _payload(chunk_id: str, text: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc",
        "text": text,
        "score": score,
        "source": "src.md",
        "content_type": "markdown",
        "page_num": 1,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": len(text),
    }


def _service(reranker: FakeReranker | None = None) -> SearchService:
    settings = Settings(
        _env_file=None,
        qdrant_url=":memory:",
        qdrant_collection="test_chunks",
        embedding_model="fake",
        reranker_model="fake-reranker",
    )
    qdrant = MagicMock()
    qdrant.search.return_value = [
        _payload("c1", "alpha passage about handshake", 0.9),
        _payload("c2", "beta passage ignored by overlap", 0.8),
        _payload("c3", "gamma extra candidate", 0.7),
    ]
    bm25 = MagicMock()
    bm25.search.return_value = [
        _payload("c2", "beta passage ignored by overlap", 4.0),
        _payload("c1", "alpha passage about handshake", 2.0),
    ]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0] * 8]
    return SearchService(settings, qdrant, bm25, embedder, reranker=reranker or FakeReranker())


def test_fake_reranker_reorders_and_keeps_rrf_score() -> None:
    reranker = FakeReranker(prefer_substring="beta")
    service = _service(reranker)
    hybrid = service.search("handshake", mode="hybrid", top_k=2)
    reranked = service.search("handshake", mode="hybrid_rerank", top_k=2)
    assert (
        hybrid.results[0].chunk_id != reranked.results[0].chunk_id
        or hybrid.results[0].score != reranked.results[0].score
    )
    assert reranked.results[0].chunk_id == "c2"
    assert reranked.results[0].rerank_score is not None
    assert reranked.results[0].rrf_score is not None
    c2_hybrid = next(hit for hit in hybrid.results if hit.chunk_id == "c2")
    c2_rerank = next(hit for hit in reranked.results if hit.chunk_id == "c2")
    assert c2_rerank.rrf_score == pytest.approx(c2_hybrid.rrf_score)


def test_hybrid_truncates_to_top_k() -> None:
    service = _service()
    response = service.search("handshake", mode="hybrid", top_k=1)
    assert len(response.results) == 1
    assert response.results[0].rank == 1


def test_dense_only_timing_has_null_fusion_and_rerank() -> None:
    service = _service()
    response = service.search("handshake", mode="dense", top_k=2)
    assert response.timing.dense_ms is not None
    assert response.timing.bm25_ms is None
    assert response.timing.fusion_ms is None
    assert response.timing.rerank_ms is None
    assert response.timing.total_ms is not None
    assert response.latency_ms == response.timing.total_ms


def test_rerank_ms_excludes_fusion_sleep() -> None:
    class SlowReranker(FakeReranker):
        def score_pairs(self, query, passages, batch_size=16):
            time.sleep(0.05)
            return super().score_pairs(query, passages, batch_size=batch_size)

    service = _service(SlowReranker())
    response = service.search("handshake", mode="hybrid_rerank", top_k=2)
    assert response.timing.fusion_ms is not None
    assert response.timing.rerank_ms is not None
    assert response.timing.rerank_ms >= 40
    assert response.timing.fusion_ms < response.timing.rerank_ms


def test_zero_weights_raise_invalid_params() -> None:
    service = _service()
    with pytest.raises(InvalidSearchParamsError, match="both be 0"):
        service.search("handshake", mode="hybrid", dense_weight=0, bm25_weight=0)
