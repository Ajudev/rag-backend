"""Document ingestion: parse, chunk, embed, and index into Qdrant + BM25."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.exceptions import EmptyDocumentError, InvalidChunkParamsError, PayloadTooLargeError
from app.schemas import ContentTypeName
from app.services.bm25_index import BM25Index
from app.services.chunker import prepare_chunks
from app.services.embedder import Embedder
from app.services.ids import document_id_from_bytes
from app.services.parser import detect_content_type, parse_document
from app.services.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of ingesting one file."""

    document_id: str
    filename: str
    content_type: ContentTypeName
    chunk_count: int
    chunk_ids: list[str]
    idempotent_replay: bool


class IngestService:
    """Coordinates parsing, chunking, embedding, and dual-index writes."""

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

    def ingest(
        self,
        filename: str,
        declared_content_type: str | None,
        data: bytes,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> IngestResult:
        """Ingest a single uploaded file.

        Same bytes are content-addressed and do not duplicate Qdrant points.
        Re-uploading those bytes under a new filename updates stored ``source``.
        Same filename with different bytes replaces prior chunks for that source.

        Args:
            filename: Original upload name stored as ``source``.
            declared_content_type: Multipart content type header.
            data: Raw file bytes.
            chunk_size: Optional override for word window size.
            overlap: Optional override for window overlap.

        Returns:
            Ingest identifiers and whether this was an idempotent replay.

        Raises:
            PayloadTooLargeError: Bytes exceed ``MAX_UPLOAD_BYTES``.
            InvalidChunkParamsError: Bad chunk_size/overlap.
            EmptyDocumentError: Empty or non-indexable content.
        """
        if len(data) > self.settings.max_upload_bytes:
            raise PayloadTooLargeError(
                f"File exceeds max upload size of {self.settings.max_upload_bytes} bytes"
            )
        if not data:
            raise EmptyDocumentError("Uploaded file is empty")

        resolved_chunk_size = chunk_size if chunk_size is not None else self.settings.chunk_size
        resolved_overlap = overlap if overlap is not None else self.settings.chunk_overlap
        _validate_chunk_params(resolved_chunk_size, resolved_overlap)

        detect_content_type(filename, declared_content_type, data)
        document_id = document_id_from_bytes(data)
        existing = self.qdrant.points_for_document(document_id)
        if existing:
            return self._replay_existing(filename, document_id, existing)

        parsed = parse_document(filename, declared_content_type, data)
        prior_source = self.qdrant.points_for_source(parsed.filename)
        if prior_source:
            logger.info("Replacing prior chunks for source=%s", parsed.filename)
            self.qdrant.delete_by_source(parsed.filename)
            self.bm25.delete_by_source(parsed.filename)

        chunks = prepare_chunks(
            parsed.pages,
            document_id=document_id,
            content_type=parsed.content_type,
            chunk_size=resolved_chunk_size,
            overlap=resolved_overlap,
            min_chunk_words=self.settings.min_chunk_words,
        )
        vectors = self.embedder.embed(
            [chunk.text for chunk in chunks],
            batch_size=self.settings.embedding_batch_size,
        )
        self.qdrant.upsert_chunks(chunks, vectors)
        self.bm25.upsert_chunks(chunks)
        return IngestResult(
            document_id=document_id,
            filename=parsed.filename,
            content_type=parsed.content_type,
            chunk_count=len(chunks),
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            idempotent_replay=False,
        )

    def _replay_existing(
        self,
        filename: str,
        document_id: str,
        existing: list[dict[str, Any]],
    ) -> IngestResult:
        """Return the stored IDs, updating ``source`` when the filename changed."""
        existing.sort(key=lambda item: int(item.get("chunk_index", 0)))
        stored_source = str(existing[0].get("source", filename))
        if stored_source != filename:
            prior_at_name = self.qdrant.points_for_source(filename)
            if prior_at_name and str(prior_at_name[0].get("document_id")) != document_id:
                logger.info("Replacing prior chunks for source=%s before rename", filename)
                self.qdrant.delete_by_source(filename)
                self.bm25.delete_by_source(filename)
            logger.info("Updating source %s -> %s for document_id=%s", stored_source, filename, document_id)
            self.qdrant.update_source(document_id, filename)
            self.bm25.update_source(document_id, filename)
            stored_source = filename
        chunk_ids = [str(item["chunk_id"]) for item in existing]
        content_type = existing[0].get("content_type", "txt")
        logger.info("Idempotent ingest for document_id=%s source=%s", document_id, stored_source)
        return IngestResult(
            document_id=document_id,
            filename=stored_source,
            content_type=content_type,
            chunk_count=len(chunk_ids),
            chunk_ids=chunk_ids,
            idempotent_replay=True,
        )


def _validate_chunk_params(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0 or overlap < 0:
        raise InvalidChunkParamsError("chunk_size must be positive and overlap must be non-negative")
    if overlap >= chunk_size:
        raise InvalidChunkParamsError("overlap must be smaller than chunk_size")
