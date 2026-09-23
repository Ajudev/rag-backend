from __future__ import annotations

import pytest

from app.services.ids import chunk_id_for, document_id_from_bytes, qdrant_point_id

pytestmark = pytest.mark.unit


def test_document_id_is_sha256_of_bytes() -> None:
    assert document_id_from_bytes(b"abc") == document_id_from_bytes(b"abc")
    assert document_id_from_bytes(b"abc") != document_id_from_bytes(b"abd")
    assert len(document_id_from_bytes(b"abc")) == 64


def test_chunk_id_deterministic_and_length() -> None:
    first = chunk_id_for("doc", 0, "Hello   world")
    second = chunk_id_for("doc", 0, "Hello world")
    assert first == second
    assert len(first) == 32
    assert chunk_id_for("doc", 1, "Hello world") != first


def test_qdrant_point_id_is_uuid_string() -> None:
    point_id = qdrant_point_id("a" * 32)
    assert point_id.count("-") == 4
    assert qdrant_point_id("a" * 32) == point_id
