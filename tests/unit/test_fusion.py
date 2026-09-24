from __future__ import annotations

import pytest

from app.services.fusion import normalize_fusion_weights, reciprocal_rank_fusion

pytestmark = pytest.mark.unit


def _hit(chunk_id: str, score: float, document_id: str = "doc") -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "score": score,
        "text": chunk_id,
        "source": "src.md",
        "content_type": "markdown",
        "page_num": 1,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 1,
    }


def test_equal_weight_rrf_scores_are_exact() -> None:
    dense = [_hit("A", 0.9), _hit("B", 0.8)]
    bm25 = [_hit("B", 5.0), _hit("C", 3.0)]
    fused = reciprocal_rank_fusion(dense, bm25, k=60, dense_weight=1.0, bm25_weight=1.0)
    by_id = {hit.chunk_id: hit for hit in fused}
    assert by_id["A"].rrf_score == pytest.approx(0.5 / 61)
    assert by_id["B"].rrf_score == pytest.approx(0.5 / 62 + 0.5 / 61)
    assert by_id["C"].rrf_score == pytest.approx(0.5 / 62)
    assert [hit.chunk_id for hit in fused] == ["B", "A", "C"]
    assert by_id["B"].contributed_by == ("dense", "bm25")
    assert by_id["A"].contributed_by == ("dense",)
    assert by_id["A"].dense_rank == 1
    assert by_id["A"].bm25_rank is None
    assert by_id["B"].dense_score == 0.8
    assert by_id["B"].bm25_score == 5.0


def test_weighted_rrf_scores_are_exact() -> None:
    dense = [_hit("A", 0.9), _hit("B", 0.8)]
    bm25 = [_hit("B", 5.0), _hit("C", 3.0)]
    fused = reciprocal_rank_fusion(dense, bm25, k=60, dense_weight=2.0, bm25_weight=1.0)
    by_id = {hit.chunk_id: hit for hit in fused}
    w_dense, w_bm25 = 2 / 3, 1 / 3
    assert by_id["A"].rrf_score == pytest.approx(w_dense / 61)
    assert by_id["B"].rrf_score == pytest.approx(w_dense / 62 + w_bm25 / 61)
    assert by_id["C"].rrf_score == pytest.approx(w_bm25 / 62)


def test_duplicate_chunk_ids_keep_first_rank() -> None:
    dense = [_hit("A", 0.9), _hit("A", 0.1), _hit("B", 0.5)]
    fused = reciprocal_rank_fusion(dense, [], k=60, dense_weight=1.0, bm25_weight=1.0)
    assert [hit.chunk_id for hit in fused] == ["A", "B"]
    assert fused[0].dense_rank == 1
    assert fused[0].dense_score == 0.9


def test_tie_break_is_chunk_id_lexicographic() -> None:
    dense = [_hit("zeta", 1.0)]
    bm25 = [_hit("alpha", 9.0)]
    fused = reciprocal_rank_fusion(dense, bm25, k=60, dense_weight=1.0, bm25_weight=1.0)
    assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)
    assert [hit.chunk_id for hit in fused] == ["alpha", "zeta"]


def test_zero_weights_raise() -> None:
    with pytest.raises(ValueError, match="both be 0"):
        normalize_fusion_weights(0.0, 0.0)
    with pytest.raises(ValueError, match="both be 0"):
        reciprocal_rank_fusion([_hit("A", 1.0)], [], dense_weight=0.0, bm25_weight=0.0)
