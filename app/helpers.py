"""Prototype helpers for tokenization, PDF extraction, and word/paragraph chunking.

Production services import these functions rather than reimplementing them.
Indexing and search in ``__main__`` remain a local prototype and are not run on import.
"""

from __future__ import annotations

import io
import os
import pathlib
import re
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

MIN_CHUNK_WORDS = 20


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase alphanumeric word tokens.

    Args:
        text: Raw input string.

    Returns:
        List of tokens suitable for BM25.
    """
    return re.findall(r"\w+", text.lower())


def extract_pdf_bytes(data: bytes, source: str) -> list[dict[str, Any]]:
    """Extract page-aware text from a PDF byte payload.

    Args:
        data: Raw PDF bytes.
        source: Source filename stored on each page dict.

    Returns:
        A list of page dicts with ``page_num``, ``text``, and ``source``.

    Raises:
        ValueError: If the PDF cannot be read or has no extractable text.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, PdfStreamError, OSError) as exc:
        raise ValueError("Unable to read PDF") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("Unable to read encrypted PDF") from exc

    pages: list[dict[str, Any]] = []
    try:
        enumerated = enumerate(reader.pages)
        for page_number, page in enumerated:
            content = page.extract_text() or ""
            if content.strip():
                pages.append(
                    {
                        "page_num": page_number + 1,
                        "text": content,
                        "source": source,
                    }
                )
    except (PdfReadError, PdfStreamError, OSError) as exc:
        raise ValueError("Unable to read PDF") from exc

    if not pages:
        raise ValueError("PDF contains no extractable text")
    return pages


def extract_text(folder: str | pathlib.Path) -> list[dict[str, Any]]:
    """Extract text from every PDF in ``folder`` (prototype corpus loader).

    Args:
        folder: Directory containing PDF files.

    Returns:
        Flattened list of page dicts from all readable PDFs.
    """
    docs: list[dict[str, Any]] = []
    for file in os.listdir(folder):
        if not file.endswith(".pdf"):
            continue
        path = os.path.join(folder, file)
        with open(path, "rb") as handle:
            try:
                docs.extend(extract_pdf_bytes(handle.read(), file))
            except ValueError:
                continue
    return docs


def chunk_text(
    docs: list[dict[str, Any]],
    chunk_size: int = 400,
    overlap: int = 50,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> list[dict[str, Any]]:
    """Split page dicts into overlapping word windows.

    Args:
        docs: Page dicts with ``text``, ``page_num``, and ``source``.
        chunk_size: Target words per chunk.
        overlap: Overlapping words between consecutive windows.
        min_chunk_words: Skip trailing windows shorter than this.

    Returns:
        Chunk dicts including text, offsets, and source metadata.
    """
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size must be positive and overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for doc in docs:
        words = doc["text"].split()
        if not words:
            continue
        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if len(chunk_words) < min_chunk_words:
                continue
            text = " ".join(chunk_words)
            prefix = " ".join(words[:start])
            char_start = len(prefix) + (1 if prefix else 0)
            char_end = char_start + len(text)
            chunks.append(
                {
                    "text": text,
                    "page_num": doc["page_num"],
                    "source": doc["source"],
                    "word_start": start,
                    "word_end": start + len(chunk_words),
                    "char_start": char_start,
                    "char_end": char_end,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
    return chunks


def chunk_text_para(
    pages: list[dict[str, Any]],
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[dict[str, Any]]:
    """Paragraph-aware chunking used by the prototype indexer.

    Args:
        pages: Page dicts with ``text``, ``page_num``, and ``source``.
        chunk_size: Approximate word budget per chunk.
        overlap: Unused paragraph-overlap hint kept for caller compatibility.

    Returns:
        Chunk dicts with ``text``, ``page_num``, and ``source``.
    """
    del overlap  # overlap is approximated by retaining the last paragraph
    chunks: list[dict[str, Any]] = []
    for page in pages:
        paragraphs = [p.strip() for p in page["text"].split("\n\n") if p.strip()]
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            para_len = len(para.split())
            if current_len + para_len > chunk_size and current:
                chunks.append(
                    {
                        "text": " ".join(current),
                        "page_num": page["page_num"],
                        "source": page["source"],
                    }
                )
                current = current[-1:]
                current_len = len(current[-1].split()) if current else 0
            current.append(para)
            current_len += para_len
        if current:
            chunks.append(
                {
                    "text": " ".join(current),
                    "page_num": page["page_num"],
                    "source": page["source"],
                }
            )
    return chunks
