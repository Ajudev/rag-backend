from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _upload(client, filename: str, data: bytes, content_type: str, extra: dict | None = None):
    files = {"file": (filename, data, content_type)}
    return client.post("/documents", files=files, data=extra or {})


def test_ingest_and_both_search_modes_share_ids(client, markdown_bytes: bytes) -> None:
    ingest = _upload(client, "aurora.md", markdown_bytes, "text/markdown")
    assert ingest.status_code == 201, ingest.text
    body = ingest.json()
    assert body["chunk_count"] >= 1
    chunk_ids = set(body["chunk_ids"])

    dense = client.post("/search", json={"query": "Zephyr handshake", "mode": "dense", "top_k": 5})
    bm25 = client.post("/search", json={"query": "Zephyr handshake", "mode": "bm25", "top_k": 5})
    assert dense.status_code == 200, dense.text
    assert bm25.status_code == 200, bm25.text
    dense_json = dense.json()
    bm25_json = bm25.json()
    assert dense_json["results"]
    assert bm25_json["results"]
    assert dense_json["results"][0]["rank"] == 1
    assert dense_json["results"][0]["chunk_id"] in chunk_ids
    assert bm25_json["results"][0]["chunk_id"] in chunk_ids
    assert dense_json["results"][0]["document_id"] == body["document_id"]
    assert "latency_ms" in dense_json
    for hit in dense_json["results"] + bm25_json["results"]:
        assert "char_start" in hit["metadata"]
        assert "char_end" in hit["metadata"]
        assert hit["metadata"]["char_end"] >= hit["metadata"]["char_start"]


def test_idempotent_same_bytes_and_replace_same_filename(client, markdown_bytes: bytes, txt_bytes: bytes) -> None:
    first = _upload(client, "same.md", markdown_bytes, "text/markdown")
    replay = _upload(client, "same.md", markdown_bytes, "text/markdown")
    assert first.status_code == 201
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["document_id"] == first.json()["document_id"]
    assert replay.json()["chunk_ids"] == first.json()["chunk_ids"]

    replacement = b"# Replacement\n\n" + txt_bytes
    replaced = _upload(client, "same.md", replacement, "text/markdown")
    assert replaced.status_code == 201
    assert replaced.json()["idempotent_replay"] is False
    assert replaced.json()["document_id"] != first.json()["document_id"]

    search = client.post(
        "/search",
        json={"query": "compound XJ-19", "mode": "bm25", "filters": {"source": "same.md"}},
    )
    assert search.status_code == 200
    hits = search.json()["results"]
    assert hits
    assert all(hit["document_id"] == replaced.json()["document_id"] for hit in hits)


def test_same_bytes_new_filename_updates_source(client, txt_bytes: bytes) -> None:
    first = _upload(client, "nimbus_lab.txt", txt_bytes, "text/plain")
    assert first.status_code == 201, first.text
    renamed = _upload(client, "renamed_nimbus.txt", txt_bytes, "text/plain")
    assert renamed.status_code == 200, renamed.text
    body = renamed.json()
    assert body["idempotent_replay"] is True
    assert body["document_id"] == first.json()["document_id"]
    assert body["chunk_ids"] == first.json()["chunk_ids"]
    assert body["filename"] == "renamed_nimbus.txt"

    by_new = client.post(
        "/search",
        json={"query": "compound XJ-19", "mode": "bm25", "filters": {"source": "renamed_nimbus.txt"}},
    )
    by_old = client.post(
        "/search",
        json={"query": "compound XJ-19", "mode": "bm25", "filters": {"source": "nimbus_lab.txt"}},
    )
    dense_new = client.post(
        "/search",
        json={"query": "compound XJ-19", "mode": "dense", "filters": {"source": "renamed_nimbus.txt"}},
    )
    assert by_new.status_code == 200
    assert by_new.json()["results"]
    assert all(hit["metadata"]["source"] == "renamed_nimbus.txt" for hit in by_new.json()["results"])
    assert all(hit["document_id"] == body["document_id"] for hit in by_new.json()["results"])
    assert by_old.json()["results"] == []
    assert dense_new.json()["results"]
    assert all(hit["metadata"]["source"] == "renamed_nimbus.txt" for hit in dense_new.json()["results"])


def test_metadata_filters(client, markdown_bytes: bytes, txt_bytes: bytes) -> None:
    _upload(client, "aurora.md", markdown_bytes, "text/markdown")
    txt = _upload(client, "nimbus.txt", txt_bytes, "text/plain")
    assert txt.status_code == 201
    document_id = txt.json()["document_id"]

    by_source = client.post(
        "/search",
        json={"query": "protocol handshake laboratory", "mode": "dense", "filters": {"source": "nimbus.txt"}},
    )
    assert by_source.status_code == 200
    assert by_source.json()["results"]
    assert all(hit["metadata"]["source"] == "nimbus.txt" for hit in by_source.json()["results"])

    by_type = client.post(
        "/search",
        json={"query": "handshake laboratory", "mode": "bm25", "filters": {"content_type": "markdown"}},
    )
    assert by_type.status_code == 200
    assert all(hit["metadata"]["content_type"] == "markdown" for hit in by_type.json()["results"])

    by_doc = client.post(
        "/search",
        json={"query": "XJ-19", "mode": "bm25", "filters": {"document_id": document_id}},
    )
    assert by_doc.status_code == 200
    assert all(hit["document_id"] == document_id for hit in by_doc.json()["results"])


def test_pdf_ingest(client, pdf_bytes: bytes) -> None:
    response = _upload(client, "paper.pdf", pdf_bytes, "application/pdf")
    assert response.status_code == 201, response.text
    assert response.json()["content_type"] == "pdf"
    assert response.json()["chunk_count"] >= 1


def test_health_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["qdrant"] is True
