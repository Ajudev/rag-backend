"""Chunking wrapper around ``helpers.chunk_text`` with ingest metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.exceptions import EmptyDocumentError, InvalidChunkParamsError
from app.schemas import ContentTypeName
from app.services.ids import chunk_id_for
from app.helpers import chunk_text


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    """A chunk ready for embedding and indexing."""

    chunk_id: str
    document_id: str
    text: str
    source: str
    content_type: ContentTypeName
    page_num: int
    chunk_index: int
    char_start: int
    char_end: int
    word_start: int
    word_end: int
    chunk_size: int
    overlap: int

    def payload(self) -> dict[str, Any]:
        """Return Qdrant/BM25 payload including the passage text."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "source": self.source,
            "content_type": self.content_type,
            "page_num": self.page_num,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }


def prepare_chunks(
    pages: list[dict[str, Any]],
    *,
    document_id: str,
    content_type: ContentTypeName,
    chunk_size: int,
    overlap: int,
    min_chunk_words: int,
) -> list[PreparedChunk]:
    """Chunk pages with word windows and attach stable IDs plus source metadata.

    Args:
        pages: Parser output page dicts.
        document_id: Content-addressed parent document ID.
        content_type: Canonical type stored on every chunk.
        chunk_size: Target words per chunk.
        overlap: Overlapping words between windows.
        min_chunk_words: Minimum words required to keep a chunk.

    Returns:
        Prepared chunks sharing ``document_id``.

    Raises:
        InvalidChunkParamsError: Invalid size/overlap.
        EmptyDocumentError: No chunks survived filtering.
    """
    if chunk_size <= 0 or overlap < 0:
        raise InvalidChunkParamsError("chunk_size must be positive and overlap must be non-negative")
    if overlap >= chunk_size:
        raise InvalidChunkParamsError("overlap must be smaller than chunk_size")

    try:
        raw_chunks = chunk_text(
            pages,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_words=min_chunk_words,
        )
    except ValueError as exc:
        raise InvalidChunkParamsError(str(exc)) from exc

    prepared: list[PreparedChunk] = []
    for item in raw_chunks:
        text = item["text"]
        chunk_index = int(item["chunk_index"])
        prepared.append(
            PreparedChunk(
                chunk_id=chunk_id_for(document_id, chunk_index, text),
                document_id=document_id,
                text=text,
                source=item["source"],
                content_type=content_type,
                page_num=int(item["page_num"]),
                chunk_index=chunk_index,
                char_start=int(item["char_start"]),
                char_end=int(item["char_end"]),
                word_start=int(item["word_start"]),
                word_end=int(item["word_end"]),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    if not prepared:
        raise EmptyDocumentError("Document produced no indexable chunks")
    return prepared
