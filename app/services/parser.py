"""File type detection and text parsing for PDF, Markdown, and TXT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.exceptions import DocumentParseError, EmptyDocumentError, UnsupportedFileTypeError
from app.schemas import ContentTypeName
from app.helpers import extract_pdf_bytes

PDF_MAGIC = b"%PDF"
ALLOWED_EXTENSIONS: dict[str, ContentTypeName] = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "txt",
}
ALLOWED_CONTENT_TYPES: dict[str, ContentTypeName] = {
    "application/pdf": "pdf",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    "text/plain": "txt",
}


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parsed source document as page dicts ready for chunking."""

    filename: str
    content_type: ContentTypeName
    pages: list[dict[str, Any]]


def _extension(filename: str) -> str:
    lower = filename.lower()
    for ext in sorted(ALLOWED_EXTENSIONS, key=len, reverse=True):
        if lower.endswith(ext):
            return ext
    return ""


def detect_content_type(filename: str, declared_content_type: str | None, data: bytes) -> ContentTypeName:
    """Validate extension plus bytes/content-type and return a canonical type.

    Args:
        filename: Original upload filename.
        declared_content_type: Multipart Content-Type header, if any.
        data: Raw file bytes.

    Returns:
        Canonical content type name.

    Raises:
        UnsupportedFileTypeError: If extension or payload is not supported.
        EmptyDocumentError: If ``data`` is empty.
        DocumentParseError: If a PDF signature is missing or text is not UTF-8.
    """
    if not data:
        raise EmptyDocumentError("Uploaded file is empty")

    ext = _extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            "Unsupported file type. Allowed extensions: .pdf, .md, .markdown, .txt"
        )
    from_ext = ALLOWED_EXTENSIONS[ext]

    declared = (declared_content_type or "").split(";")[0].strip().lower()
    if declared and declared not in {"application/octet-stream", "binary/octet-stream"}:
        from_header = ALLOWED_CONTENT_TYPES.get(declared)
        if from_header is None:
            raise UnsupportedFileTypeError(f"Unsupported content type: {declared_content_type}")
        if from_header != from_ext:
            raise UnsupportedFileTypeError("Filename extension does not match Content-Type")

    if from_ext == "pdf":
        if not data.lstrip().startswith(PDF_MAGIC):
            raise DocumentParseError("File does not look like a PDF")
        return "pdf"

    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentParseError("Text files must be valid UTF-8") from exc
    return from_ext


def parse_document(filename: str, declared_content_type: str | None, data: bytes) -> ParsedDocument:
    """Parse upload bytes into page-aware text units.

    Args:
        filename: Original filename used as the ``source`` metadata value.
        declared_content_type: Multipart Content-Type header, if any.
        data: Raw file bytes.

    Returns:
        Parsed pages with ``page_num``, ``text``, and ``source``.

    Raises:
        UnsupportedFileTypeError: Unsupported type.
        EmptyDocumentError: Empty bytes or no extractable text.
        DocumentParseError: Unreadable PDF or invalid UTF-8.
    """
    content_type = detect_content_type(filename, declared_content_type, data)
    if content_type == "pdf":
        try:
            pages = extract_pdf_bytes(data, filename)
        except ValueError as exc:
            raise DocumentParseError(str(exc)) from exc
        return ParsedDocument(filename=filename, content_type="pdf", pages=pages)

    text = data.decode("utf-8")
    if not text.strip():
        raise EmptyDocumentError("Uploaded file has no text content")
    pages = [{"page_num": 1, "text": text, "source": filename}]
    return ParsedDocument(filename=filename, content_type=content_type, pages=pages)
