"""Dense, BM25, hybrid RRF, and optional cross-encoder rerank search."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from app.core.config import Settings
from app.core.exceptions import InvalidSearchParamsError, RerankerUnavailableError
from app.schemas import (
    ChunkMetadata,
    SearchDebug,
    SearchFilters,
    SearchHit,
    SearchMode,
    SearchResponse,
    SearchTiming,
)
from app.services.bm25_index import BM25Index
from app.services.embedder import Embedder
from app.services.fusion import FusedHit, normalize_fusion_weights, reciprocal_rank_fusion
from app.services.qdrant_store import QdrantStore
from app.services.reranker import Reranker

logger = logging.getLogger(__name__)


class SearchService:
    """Runs dense, BM25, hybrid RRF, and hybrid-plus-rerank retrieval."""

    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantStore,
        bm25: BM25Index,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.qdrant = qdrant
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker

    def search(
        self,
        query: str,
        mode: SearchMode = "dense",
        top_k: int = 10,
        filters: SearchFilters | None = None,
        dense_candidate_count: int | None = None,
        bm25_candidate_count: int | None = None,
        rerank_candidate_count: int | None = None,
        rrf_k: int | None = None,
        dense_weight: float | None = None,
        bm25_weight: float | None = None,
        debug: bool = False,
    ) -> SearchResponse:
        """Search the dual indexes and optionally fuse or rerank.

        Args:
            query: Non-empty query string.
            mode: ``dense``, ``bm25``, ``hybrid``, or ``hybrid_rerank``.
            top_k: Hit cap, already clamped to 1..100 by the schema.
            filters: Optional metadata constraints applied to both retrievers.
            dense_candidate_count: Dense pool size for fusion; defaults to settings.
            bm25_candidate_count: BM25 pool size for fusion; defaults to settings.
            rerank_candidate_count: Fused pool size to rerank; defaults to settings.
            rrf_k: RRF smoothing constant; defaults to settings.
            dense_weight: Unnormalized dense fusion weight.
            bm25_weight: Unnormalized BM25 fusion weight.
            debug: When true, echo resolved fusion parameters on the response.

        Returns:
            Ranked hits, end-to-end ``latency_ms``, and per-stage timing.

        Raises:
            InvalidSearchParamsError: Invalid weights or rerank pool smaller than ``top_k``.
            RerankerUnavailableError: ``hybrid_rerank`` without a loaded reranker.
        """
        started = time.perf_counter()
        dense_count = (
            dense_candidate_count if dense_candidate_count is not None else self.settings.dense_candidate_count
        )
        bm25_count = bm25_candidate_count if bm25_candidate_count is not None else self.settings.bm25_candidate_count
        rerank_count = (
            rerank_candidate_count if rerank_candidate_count is not None else self.settings.rerank_candidate_count
        )
        k = rrf_k if rrf_k is not None else self.settings.rrf_k
        weight_dense = dense_weight if dense_weight is not None else self.settings.dense_weight
        weight_bm25 = bm25_weight if bm25_weight is not None else self.settings.bm25_weight

        dense_ms: float | None = None
        bm25_ms: float | None = None
        fusion_ms: float | None = None
        rerank_ms: float | None = None
        results: list[SearchHit]
        normalized: tuple[float, float] | None = None

        if mode == "dense":
            dense_hits, dense_ms = self._timed_dense(query, top_k, filters)
            results = [
                _hit_from_payload(rank, payload, contributed_by=["dense"], retriever="dense")
                for rank, payload in enumerate(dense_hits, start=1)
            ]
        elif mode == "bm25":
            bm25_hits, bm25_ms = self._timed_bm25(query, top_k, filters)
            results = [
                _hit_from_payload(rank, payload, contributed_by=["bm25"], retriever="bm25")
                for rank, payload in enumerate(bm25_hits, start=1)
            ]
        elif mode in {"hybrid", "hybrid_rerank"}:
            try:
                normalized = normalize_fusion_weights(weight_dense, weight_bm25)
            except ValueError as exc:
                raise InvalidSearchParamsError(str(exc)) from exc
            if mode == "hybrid_rerank" and rerank_count < top_k:
                raise InvalidSearchParamsError("rerank_candidate_count must be greater than or equal to top_k")
            dense_hits, dense_ms = self._timed_dense(query, dense_count, filters)
            bm25_hits, bm25_ms = self._timed_bm25(query, bm25_count, filters)
            fusion_started = time.perf_counter()
            fused = reciprocal_rank_fusion(
                dense_hits,
                bm25_hits,
                k=k,
                dense_weight=weight_dense,
                bm25_weight=weight_bm25,
            )
            fusion_ms = (time.perf_counter() - fusion_started) * 1000
            if mode == "hybrid_rerank":
                results, rerank_ms = self._rerank(query, fused[:rerank_count], top_k)
            else:
                results = [
                    _hit_from_fused(rank, hit, final_score=hit.rrf_score)
                    for rank, hit in enumerate(fused[:top_k], start=1)
                ]
        else:
            raise InvalidSearchParamsError(f"unsupported search mode: {mode}")

        total_ms = (time.perf_counter() - started) * 1000
        debug_block = None
        if debug:
            debug_block = SearchDebug(
                dense_candidate_count=dense_count,
                bm25_candidate_count=bm25_count,
                rerank_candidate_count=rerank_count,
                rrf_k=k,
                dense_weight=weight_dense,
                bm25_weight=weight_bm25,
                normalized_dense_weight=None if normalized is None else normalized[0],
                normalized_bm25_weight=None if normalized is None else normalized[1],
            )
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        logger.info(
            "search mode=%s top_k=%s dense_candidates=%s bm25_candidates=%s rerank_candidates=%s "
            "query_len=%s query_hash=%s result_count=%s dense_ms=%s bm25_ms=%s fusion_ms=%s rerank_ms=%s total_ms=%s",
            mode,
            top_k,
            dense_count,
            bm25_count,
            rerank_count,
            len(query),
            query_hash,
            len(results),
            _round_ms(dense_ms),
            _round_ms(bm25_ms),
            _round_ms(fusion_ms),
            _round_ms(rerank_ms),
            round(total_ms, 2),
        )
        return SearchResponse(
            query=query,
            mode=mode,
            top_k=top_k,
            latency_ms=round(total_ms, 2),
            results=results,
            timing=SearchTiming(
                dense_ms=_round_ms(dense_ms),
                bm25_ms=_round_ms(bm25_ms),
                fusion_ms=_round_ms(fusion_ms),
                rerank_ms=_round_ms(rerank_ms),
                total_ms=round(total_ms, 2),
            ),
            debug=debug_block,
        )

    def _timed_dense(
        self,
        query: str,
        top_k: int,
        filters: SearchFilters | None,
    ) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        vector = self.embedder.embed([query], batch_size=1)[0]
        hits = self.qdrant.search(vector, top_k=top_k, filters=filters)
        return hits, (time.perf_counter() - started) * 1000

    def _timed_bm25(
        self,
        query: str,
        top_k: int,
        filters: SearchFilters | None,
    ) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        hits = self.bm25.search(query, top_k=top_k, filters=filters)
        return hits, (time.perf_counter() - started) * 1000

    def _rerank(self, query: str, fused: list[FusedHit], top_k: int) -> tuple[list[SearchHit], float]:
        if self.reranker is None:
            raise RerankerUnavailableError("cross-encoder reranker is not loaded")
        started = time.perf_counter()
        passages = [str(hit.payload.get("text", "")) for hit in fused]
        scores = self.reranker.score_pairs(
            query,
            passages,
            batch_size=self.settings.reranker_batch_size,
        )
        rerank_ms = (time.perf_counter() - started) * 1000
        if len(scores) != len(fused):
            raise RuntimeError("reranker returned a different number of scores than passages")
        ordered = sorted(
            zip(fused, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].chunk_id),
        )
        results = [
            _hit_from_fused(rank, hit, final_score=float(score), rerank_score=float(score))
            for rank, (hit, score) in enumerate(ordered[:top_k], start=1)
        ]
        return results, rerank_ms


def _round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)


def _metadata_from_payload(payload: dict[str, Any]) -> ChunkMetadata:
    return ChunkMetadata(
        source=str(payload.get("source", "")),
        content_type=payload.get("content_type", "txt"),
        page_num=int(payload.get("page_num", 1)),
        chunk_index=int(payload.get("chunk_index", 0)),
        char_start=int(payload.get("char_start", 0)),
        char_end=int(payload.get("char_end", 0)),
    )


def _hit_from_payload(
    rank: int,
    payload: dict[str, Any],
    *,
    contributed_by: list[str],
    retriever: SearchMode,
) -> SearchHit:
    score = float(payload.get("score", 0.0))
    return SearchHit(
        rank=rank,
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        score=score,
        text=str(payload.get("text", "")),
        metadata=_metadata_from_payload(payload),
        dense_rank=rank if retriever == "dense" else None,
        dense_score=score if retriever == "dense" else None,
        bm25_rank=rank if retriever == "bm25" else None,
        bm25_score=score if retriever == "bm25" else None,
        rrf_score=None,
        rerank_score=None,
        contributed_by=contributed_by,
    )


def _hit_from_fused(
    rank: int,
    hit: FusedHit,
    *,
    final_score: float,
    rerank_score: float | None = None,
) -> SearchHit:
    payload = hit.payload
    return SearchHit(
        rank=rank,
        chunk_id=hit.chunk_id,
        document_id=str(payload.get("document_id", "")),
        score=final_score,
        text=str(payload.get("text", "")),
        metadata=_metadata_from_payload(payload),
        dense_rank=hit.dense_rank,
        dense_score=hit.dense_score,
        bm25_rank=hit.bm25_rank,
        bm25_score=hit.bm25_score,
        rrf_score=hit.rrf_score,
        rerank_score=rerank_score,
        contributed_by=list(hit.contributed_by),
    )
