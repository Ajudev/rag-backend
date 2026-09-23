"""In-memory BM25 index with JSON persistence, separate from Qdrant."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.schemas import SearchFilters
from app.services.chunker import PreparedChunk
from app.helpers import tokenize


class BM25Index:
    """Sparse lexical index sharing ``document_id`` / ``chunk_id`` with Qdrant."""

    def __init__(self, persist_path: Path) -> None:
        self.persist_path = persist_path
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    def is_empty(self) -> bool:
        """Return True when no records are loaded."""
        return not self._records

    def load(self) -> None:
        """Load persisted records from disk if the file exists."""
        if not self.persist_path.exists():
            return
        with self.persist_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("records", payload if isinstance(payload, list) else [])
        with self._lock:
            self._records = list(records)
            self._rebuild_locked()

    def load_from_payloads(self, payloads: list[dict[str, Any]]) -> None:
        """Replace in-memory records from Qdrant (or other) payloads."""
        with self._lock:
            self._records = [dict(item) for item in payloads]
            self._rebuild_locked()
        self._persist()

    def upsert_chunks(self, chunks: list[PreparedChunk]) -> None:
        """Insert or replace records for the given chunk IDs."""
        incoming = {chunk.chunk_id: chunk.payload() for chunk in chunks}
        with self._lock:
            kept = [record for record in self._records if record.get("chunk_id") not in incoming]
            kept.extend(incoming.values())
            self._records = kept
            self._rebuild_locked()
        self._persist()

    def delete_by_source(self, source: str) -> None:
        """Remove all records whose source filename matches."""
        with self._lock:
            self._records = [record for record in self._records if record.get("source") != source]
            self._rebuild_locked()
        self._persist()

    def update_source(self, document_id: str, source: str) -> None:
        """Rename ``source`` on every record for ``document_id``.

        Args:
            document_id: Content-addressed parent document hash.
            source: Filename to store as the current source.
        """
        with self._lock:
            for record in self._records:
                if record.get("document_id") == document_id:
                    record["source"] = source
        self._persist()

    def search(self, query: str, top_k: int, filters: SearchFilters | None = None) -> list[dict[str, Any]]:
        """Score the (optionally filtered) corpus with BM25.

        Filters are applied before scoring so ranks are relative to the filtered set.

        Args:
            query: User query text.
            top_k: Maximum hits to return.
            filters: Optional source / document_id / content_type constraints.

        Returns:
            Payload dicts with an added ``score`` field, highest first.
        """
        with self._lock:
            corpus = [record for record in self._records if _matches(record, filters)]
            if not corpus:
                return []
            tokenized = [tokenize(record.get("text", "")) for record in corpus]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(tokenize(query))
            ranked = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)[:top_k]
            results: list[dict[str, Any]] = []
            for idx in ranked:
                payload = dict(corpus[idx])
                payload["score"] = float(scores[idx])
                results.append(payload)
            return results

    def _rebuild_locked(self) -> None:
        if not self._records:
            self._bm25 = None
            return
        tokenized = [tokenize(record.get("text", "")) for record in self._records]
        self._bm25 = BM25Okapi(tokenized)

    def _persist(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump({"records": self._records}, handle)
        tmp_path.replace(self.persist_path)


def _matches(record: dict[str, Any], filters: SearchFilters | None) -> bool:
    if filters is None:
        return True
    checks = (
        (filters.source, "source"),
        (filters.document_id, "document_id"),
        (filters.content_type, "content_type"),
    )
    return all(expected is None or record.get(key) == expected for expected, key in checks)
