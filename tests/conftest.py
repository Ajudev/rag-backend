"""Shared pytest fixtures: in-memory Qdrant, fake embedder, sample files."""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.main import create_app
from app.services.embedder import FakeEmbedder

LONG_TEXT = (
    "The Zephyr handshake is a three-step nonce exchange used by the Aurora Protocol "
    "before payload data is sent between a ground station and a satellite. "
    "Operators must record every failure in the mission diary after commit."
)


def make_pdf_bytes(text: str) -> bytes:
    """Build a tiny one-page PDF with extractable Helvetica text."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1", errors="replace"))
    stream_ref = writer._add_object(stream)
    font = DictionaryObject()
    font.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    resources = DictionaryObject()
    resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font_ref})
    page[NameObject("/Resources")] = resources
    page[NameObject("/Contents")] = stream_ref
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf_bytes(LONG_TEXT)


@pytest.fixture
def markdown_bytes() -> bytes:
    return (
        b"# Aurora\n\nThe Zephyr handshake completes before any AURORA-SESSION key is derived "
        b"and this markdown file stays long enough for word chunking to keep the passage.\n"
    )


@pytest.fixture
def txt_bytes() -> bytes:
    return (
        b"Nimbus Laboratory isolated compound XJ-19 during the night shift. "
        b"XJ-19 fluoresces teal and must never be stored beside acetone according to the notebook.\n"
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        qdrant_url=":memory:",
        qdrant_collection="test_chunks",
        embedding_model="fake",
        embedding_batch_size=8,
        chunk_size=40,
        chunk_overlap=5,
        min_chunk_words=1,
        max_upload_bytes=4096,
        bm25_index_path=tmp_path / "bm25.json",
        upsert_batch_size=8,
    )


@pytest.fixture
def qdrant_client() -> QdrantClient:
    return QdrantClient(location=":memory:")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def app(settings: Settings, embedder: FakeEmbedder, qdrant_client: QdrantClient):
    return create_app(settings=settings, embedder=embedder, qdrant_client=qdrant_client)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
