from __future__ import annotations

import pytest

from app.core.exceptions import DocumentParseError, EmptyDocumentError, UnsupportedFileTypeError
from app.services.parser import detect_content_type, parse_document
from tests.conftest import LONG_TEXT, make_pdf_bytes

pytestmark = pytest.mark.unit


def test_parse_txt_as_single_page(txt_bytes: bytes) -> None:
    parsed = parse_document("notes.txt", "text/plain", txt_bytes)
    assert parsed.content_type == "txt"
    assert len(parsed.pages) == 1
    assert parsed.pages[0]["page_num"] == 1
    assert "XJ-19" in parsed.pages[0]["text"]
    assert parsed.pages[0]["source"] == "notes.txt"


def test_parse_markdown_as_single_page(markdown_bytes: bytes) -> None:
    parsed = parse_document("aurora.md", "text/markdown", markdown_bytes)
    assert parsed.content_type == "markdown"
    assert parsed.pages[0]["page_num"] == 1
    assert "Zephyr" in parsed.pages[0]["text"]


def test_parse_pdf_is_page_aware(pdf_bytes: bytes) -> None:
    parsed = parse_document("paper.pdf", "application/pdf", pdf_bytes)
    assert parsed.content_type == "pdf"
    assert parsed.pages
    assert parsed.pages[0]["page_num"] == 1
    extracted = " ".join(page["text"] for page in parsed.pages)
    assert "Zephyr" in extracted or "handshake" in extracted.lower() or LONG_TEXT.split()[0] in extracted


def test_detect_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        detect_content_type("photo.png", "image/png", b"\x89PNG")


def test_detect_rejects_empty() -> None:
    with pytest.raises(EmptyDocumentError):
        detect_content_type("a.txt", "text/plain", b"")


def test_detect_rejects_pdf_without_magic() -> None:
    with pytest.raises(DocumentParseError):
        detect_content_type("x.pdf", "application/pdf", b"not a pdf file at all")


def test_unreadable_pdf_raises() -> None:
    with pytest.raises(DocumentParseError):
        parse_document("broken.pdf", "application/pdf", b"%PDF-1.4 not-a-real-file")


def test_make_pdf_starts_with_magic() -> None:
    data = make_pdf_bytes("hello world from sample document text")
    assert data.lstrip().startswith(b"%PDF")
