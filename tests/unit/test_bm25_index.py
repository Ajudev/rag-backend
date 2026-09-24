from __future__ import annotations

import pytest

from app.schemas import SearchFilters
from app.services.bm25_index import BM25Index
from app.services.chunker import PreparedChunk

pytestmark = pytest.mark.unit


def _chunk(chunk_id: str, source: str, text: str, document_id: str = "doc1") -> PreparedChunk:
    return PreparedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        source=source,
        content_type="txt",
        page_num=1,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        word_start=0,
        word_end=len(text.split()),
        chunk_size=40,
        overlap=5,
    )


def test_bm25_filters_and_ranking(tmp_path) -> None:
    index = BM25Index(tmp_path / "bm25.json")
    index.upsert_chunks(
        [
            _chunk("1" * 32, "a.txt", "compound XJ-19 fluoresces teal under ultraviolet light"),
            _chunk("2" * 32, "b.txt", "lunar-cycle irrigation opens valves at first-quarter moonrise"),
        ]
    )
    hits = index.search("XJ-19 fluoresces", top_k=5)
    assert hits
    assert hits[0]["source"] == "a.txt"
    filtered = index.search("irrigation", top_k=5, filters=SearchFilters(source="a.txt"))
    assert filtered == [] or all(item["source"] == "a.txt" for item in filtered)
    indexed = index.search("irrigation", top_k=5, filters=SearchFilters(source="b.txt"))
    assert indexed
    assert indexed[0]["chunk_id"] == "2" * 32


def test_bm25_update_source_renames_filters(tmp_path) -> None:
    index = BM25Index(tmp_path / "bm25.json")
    index.upsert_chunks([_chunk("1" * 32, "nimbus_lab.txt", "compound XJ-19 fluoresces teal under ultraviolet light")])
    index.update_source("doc1", "renamed_nimbus.txt")
    old = index.search("XJ-19", top_k=5, filters=SearchFilters(source="nimbus_lab.txt"))
    new = index.search("XJ-19", top_k=5, filters=SearchFilters(source="renamed_nimbus.txt"))
    assert old == []
    assert new
    assert new[0]["source"] == "renamed_nimbus.txt"
