from __future__ import annotations

import pytest

from app.core.exceptions import InvalidChunkParamsError
from app.services.chunker import prepare_chunks
from app.services.ids import chunk_id_for, document_id_from_bytes
from app.services.parser import parse_document

pytestmark = pytest.mark.unit

REQUIRED_META = {
    "document_id",
    "chunk_id",
    "source",
    "content_type",
    "page_num",
    "chunk_index",
    "char_start",
    "char_end",
    "chunk_size",
    "overlap",
}


def test_chunk_metadata_complete_and_ids_stable(txt_bytes: bytes) -> None:
    document_id = document_id_from_bytes(txt_bytes)
    parsed = parse_document("nimbus.txt", "text/plain", txt_bytes)
    first = prepare_chunks(
        parsed.pages,
        document_id=document_id,
        content_type="txt",
        chunk_size=40,
        overlap=5,
        min_chunk_words=1,
    )
    second = prepare_chunks(
        parsed.pages,
        document_id=document_id,
        content_type="txt",
        chunk_size=40,
        overlap=5,
        min_chunk_words=1,
    )
    assert first
    payload = first[0].payload()
    assert set(payload) >= REQUIRED_META
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == chunk_id_for(document_id, first[0].chunk_index, first[0].text)
    assert first[0].document_id == document_id
    assert first[0].char_end >= first[0].char_start


def test_invalid_overlap_rejected() -> None:
    pages = [{"text": "alpha beta gamma", "page_num": 1, "source": "a.txt"}]
    with pytest.raises(InvalidChunkParamsError):
        prepare_chunks(
            pages,
            document_id="abc",
            content_type="txt",
            chunk_size=10,
            overlap=10,
            min_chunk_words=1,
        )
