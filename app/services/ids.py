"""Stable document and chunk identifiers."""

from __future__ import annotations

import hashlib
import uuid


def document_id_from_bytes(data: bytes) -> str:
    """Return a content-addressed SHA-256 hex digest for ``data``.

    Args:
        data: Raw file bytes.

    Returns:
        64-character hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    """Collapse whitespace so identical passages share an ID.

    Args:
        text: Chunk text.

    Returns:
        Whitespace-normalized string.
    """
    return " ".join(text.split())


def chunk_id_for(document_id: str, chunk_index: int, text: str) -> str:
    """Build a deterministic 32-character chunk identifier.

    Args:
        document_id: Parent document hash.
        chunk_index: Zero-based chunk index within the document.
        text: Chunk text.

    Returns:
        First 32 hex chars of SHA-256(document_id:index:normalized_text).
    """
    material = f"{document_id}:{chunk_index}:{normalize_text(text)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def qdrant_point_id(chunk_id: str) -> str:
    """Derive a UUID5 Qdrant point ID from ``chunk_id``.

    Args:
        chunk_id: Deterministic chunk identifier.

    Returns:
        UUID string accepted by Qdrant as a point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
