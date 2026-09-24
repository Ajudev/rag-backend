from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.services.chunker import PreparedChunk
from app.services.embedder import FakeEmbedder
from app.services.qdrant_store import QdrantStore

pytestmark = pytest.mark.unit


def test_qdrant_upsert_and_search_with_fake_vectors(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        qdrant_url=":memory:",
        qdrant_collection="unit_qdrant",
        embedding_model="fake",
        bm25_index_path=tmp_path / "bm25.json",
        min_chunk_words=1,
        reranker_model="fake-reranker",
    )
    store = QdrantStore(QdrantClient(location=":memory:"), settings)
    store.ensure_collection()
    embedder = FakeEmbedder()
    chunk = PreparedChunk(
        chunk_id="c" * 32,
        document_id="d" * 64,
        text="Zephyr handshake details live here with extra words for the vector.",
        source="aurora.md",
        content_type="markdown",
        page_num=1,
        chunk_index=0,
        char_start=0,
        char_end=20,
        word_start=0,
        word_end=10,
        chunk_size=40,
        overlap=5,
    )
    vectors = embedder.embed([chunk.text], batch_size=8)
    store.upsert_chunks([chunk], vectors)
    hits = store.search(vectors[0], top_k=5)
    assert hits
    assert hits[0]["chunk_id"] == chunk.chunk_id
    assert hits[0]["document_id"] == chunk.document_id

    store.upsert_chunks([chunk], vectors)
    again = store.points_for_document(chunk.document_id)
    assert len(again) == 1

    store.update_source(chunk.document_id, "renamed.md")
    updated = store.points_for_document(chunk.document_id)
    assert len(updated) == 1
    assert updated[0]["source"] == "renamed.md"
    assert store.points_for_source("aurora.md") == []
