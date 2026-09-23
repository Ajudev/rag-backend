"""Dense (Qdrant) and BM25 search over the same chunk identifiers."""

from __future__ import annotations

import time

from app.core.config import Settings
from app.schemas import ChunkMetadata, SearchFilters, SearchHit, SearchMode, SearchResponse
from app.services.bm25_index import BM25Index
from app.services.embedder import Embedder
from app.services.qdrant_store import QdrantStore


class SearchService:
    """Runs baseline dense or BM25 retrieval (no fusion or rerank)."""

    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantStore,
        bm25: BM25Index,
        embedder: Embedder,
    ) -> None:
        self.settings = settings
        self.qdrant = qdrant
        self.bm25 = bm25
        self.embedder = embedder

    def search(
        self,
        query: str,
        mode: SearchMode,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        """Search the dual indexes and return ranked passages.

        Args:
            query: Non-empty query string.
            mode: ``dense`` or ``bm25``.
            top_k: Hit cap, already clamped to 1..100 by the schema.
            filters: Optional metadata constraints applied to both modes.

        Returns:
            Ranked hits plus wall-clock latency in milliseconds.
        """
        started = time.perf_counter()
        if mode == "dense":
            vector = self.embedder.embed([query], batch_size=1)[0]
            raw_hits = self.qdrant.search(vector, top_k=top_k, filters=filters)
        else:
            raw_hits = self.bm25.search(query, top_k=top_k, filters=filters)
        latency_ms = (time.perf_counter() - started) * 1000
        results = [_to_hit(rank, payload) for rank, payload in enumerate(raw_hits, start=1)]
        return SearchResponse(
            query=query,
            mode=mode,
            top_k=top_k,
            latency_ms=round(latency_ms, 2),
            results=results,
        )


def _to_hit(rank: int, payload: dict) -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        score=float(payload.get("score", 0.0)),
        text=str(payload.get("text", "")),
        metadata=ChunkMetadata(
            source=str(payload.get("source", "")),
            content_type=payload.get("content_type", "txt"),
            page_num=int(payload.get("page_num", 1)),
            chunk_index=int(payload.get("chunk_index", 0)),
            char_start=int(payload.get("char_start", 0)),
            char_end=int(payload.get("char_end", 0)),
        ),
    )
