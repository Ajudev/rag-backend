"""Qdrant collection management, upsert, search, and deletes."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings
from app.core.exceptions import QdrantUnavailableError
from app.schemas import SearchFilters
from app.services.chunker import PreparedChunk
from app.services.ids import qdrant_point_id

logger = logging.getLogger(__name__)


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Construct a Qdrant client from settings.

    Args:
        settings: Application settings.

    Returns:
        Connected or in-memory ``QdrantClient``.
    """
    url = settings.qdrant_url.strip()
    if url in {":memory:", "memory"}:
        return QdrantClient(location=":memory:")
    return QdrantClient(url=url, timeout=settings.qdrant_timeout)


def metadata_filter(filters: SearchFilters | None) -> Filter | None:
    """Build a Qdrant payload filter from search filters."""
    if filters is None:
        return None
    conditions: list[FieldCondition] = []
    if filters.source:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=filters.source)))
    if filters.document_id:
        conditions.append(FieldCondition(key="document_id", match=MatchValue(value=filters.document_id)))
    if filters.content_type:
        conditions.append(FieldCondition(key="content_type", match=MatchValue(value=filters.content_type)))
    if not conditions:
        return None
    return Filter(must=conditions)


class QdrantStore:
    """Thin wrapper around Qdrant collection operations."""

    def __init__(self, client: QdrantClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.collection = settings.qdrant_collection

    def ping(self) -> bool:
        """Return True if Qdrant responds to a collections listing."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            logger.exception("Qdrant ping failed")
            return False

    def ensure_collection(self) -> None:
        """Create the cosine collection and payload indexes if missing.

        Raises:
            QdrantUnavailableError: If Qdrant cannot be reached.
        """
        try:
            existing = {item.name for item in self.client.get_collections().collections}
            if self.collection not in existing:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.settings.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
            if self.settings.qdrant_url.strip() not in {":memory:", "memory"}:
                for field_name in ("source", "document_id", "content_type"):
                    try:
                        self.client.create_payload_index(
                            collection_name=self.collection,
                            field_name=field_name,
                            field_schema=PayloadSchemaType.KEYWORD,
                        )
                    except Exception:
                        logger.debug("Payload index for %s already exists or is unsupported", field_name)
        except QdrantUnavailableError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError("Unable to initialize Qdrant collection") from exc

    def points_for_document(self, document_id: str) -> list[dict[str, Any]]:
        """Scroll all payloads for ``document_id``."""
        return self._scroll(Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]))

    def points_for_source(self, source: str) -> list[dict[str, Any]]:
        """Scroll all payloads for ``source`` filename."""
        return self._scroll(Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]))

    def delete_by_source(self, source: str) -> None:
        """Delete all points whose payload source equals ``source``."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
            ),
        )

    def update_source(self, document_id: str, source: str) -> None:
        """Set payload ``source`` on every point for ``document_id``.

        Args:
            document_id: Content-addressed parent document hash.
            source: Filename to store as the current source.
        """
        self.client.set_payload(
            collection_name=self.collection,
            payload={"source": source},
            points=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            ),
        )

    def upsert_chunks(self, chunks: Sequence[PreparedChunk], vectors: Sequence[Sequence[float]]) -> None:
        """Batch-upsert chunk points.

        Args:
            chunks: Prepared chunks sharing IDs with BM25.
            vectors: Embedding for each chunk, same order and length.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        batch_size = self.settings.upsert_batch_size
        points = [
            PointStruct(
                id=qdrant_point_id(chunk.chunk_id),
                vector=list(vector),
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(collection_name=self.collection, points=points[start : start + batch_size])

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[dict[str, Any]]:
        """Dense kNN search with optional payload filters."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            query_filter=metadata_filter(filters),
            limit=top_k,
            with_payload=True,
        )
        results: list[dict[str, Any]] = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            payload["score"] = float(hit.score)
            results.append(payload)
        return results

    def scroll_all(self) -> list[dict[str, Any]]:
        """Return every stored payload (used to rebuild BM25)."""
        return self._scroll(None)

    def _scroll(self, query_filter: Filter | None) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if record.payload:
                    payloads.append(dict(record.payload))
            if offset is None:
                break
        return payloads
