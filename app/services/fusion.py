"""Weighted Reciprocal Rank Fusion over dense and BM25 ranked lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One chunk after rank fusion, with per-retriever provenance."""

    chunk_id: str
    payload: dict[str, Any]
    rrf_score: float
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    contributed_by: tuple[str, ...]


def normalize_fusion_weights(dense_weight: float, bm25_weight: float) -> tuple[float, float]:
    """Return ``(w'_dense, w'_bm25)`` that sum to 1.

    Args:
        dense_weight: Non-negative dense weight before normalization.
        bm25_weight: Non-negative BM25 weight before normalization.

    Returns:
        Normalized weights used in the RRF sum.

    Raises:
        ValueError: If a weight is negative or both weights are zero.
    """
    if dense_weight < 0 or bm25_weight < 0:
        raise ValueError("fusion weights must be greater than or equal to 0")
    total = dense_weight + bm25_weight
    if total == 0:
        raise ValueError("dense_weight and bm25_weight cannot both be 0")
    return dense_weight / total, bm25_weight / total


def reciprocal_rank_fusion(
    dense_hits: Sequence[Mapping[str, Any]],
    bm25_hits: Sequence[Mapping[str, Any]],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[FusedHit]:
    """Fuse two ranked lists with weighted reciprocal rank fusion.

    Ranks are 1-based from list order. A chunk missing from one list contributes
    0 for that retriever. Duplicate ``chunk_id`` values keep the first occurrence.

    Score formula::

        rrf(d) = w'_dense / (k + rank_dense(d)) + w'_bm25 / (k + rank_bm25(d))

    Raw cosine and BM25 scores are never averaged; only ranks enter the sum.

    Args:
        dense_hits: Dense retriever payloads in rank order, each with ``chunk_id``.
        bm25_hits: BM25 retriever payloads in rank order, each with ``chunk_id``.
        k: RRF smoothing constant (default 60).
        dense_weight: Unnormalized dense weight.
        bm25_weight: Unnormalized BM25 weight.

    Returns:
        Deduplicated hits sorted by descending RRF score, then ``chunk_id`` ascending.

    Raises:
        ValueError: If ``k`` is less than 1 or weights are invalid.
    """
    if k < 1:
        raise ValueError("rrf_k must be >= 1")
    w_dense, w_bm25 = normalize_fusion_weights(dense_weight, bm25_weight)
    dense_index = _first_ranks(dense_hits)
    bm25_index = _first_ranks(bm25_hits)
    chunk_ids = set(dense_index) | set(bm25_index)
    fused: list[FusedHit] = []
    for chunk_id in chunk_ids:
        dense_rank, dense_score, dense_payload = dense_index.get(chunk_id, (None, None, None))
        bm25_rank, bm25_score, bm25_payload = bm25_index.get(chunk_id, (None, None, None))
        rrf_score = 0.0
        contributed: list[str] = []
        if dense_rank is not None:
            rrf_score += w_dense / (k + dense_rank)
            contributed.append("dense")
        if bm25_rank is not None:
            rrf_score += w_bm25 / (k + bm25_rank)
            contributed.append("bm25")
        payload = dict(dense_payload or bm25_payload or {})
        fused.append(
            FusedHit(
                chunk_id=chunk_id,
                payload=payload,
                rrf_score=rrf_score,
                dense_rank=dense_rank,
                dense_score=dense_score,
                bm25_rank=bm25_rank,
                bm25_score=bm25_score,
                contributed_by=tuple(contributed),
            )
        )
    fused.sort(key=lambda hit: (-hit.rrf_score, hit.chunk_id))
    return fused


def _first_ranks(
    hits: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, float, dict[str, Any]]]:
    """Map chunk_id to (1-based rank, score, payload), first occurrence only."""
    indexed: dict[str, tuple[int, float, dict[str, Any]]] = {}
    for offset, hit in enumerate(hits):
        chunk_id = str(hit["chunk_id"])
        if chunk_id in indexed:
            continue
        payload = dict(hit)
        score = float(payload.get("score", 0.0))
        indexed[chunk_id] = (offset + 1, score, payload)
    return indexed
