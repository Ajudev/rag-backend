from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_unsupported_file_type_415(client) -> None:
    response = client.post(
        "/documents",
        files={"file": ("image.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 415


def test_empty_file_400(client) -> None:
    response = client.post(
        "/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


def test_oversized_file_413(client) -> None:
    payload = b"a" * 5000
    response = client.post(
        "/documents",
        files={"file": ("big.txt", payload, "text/plain")},
    )
    assert response.status_code == 413


def test_invalid_chunk_params_400(client, txt_bytes: bytes) -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", txt_bytes, "text/plain")},
        data={"chunk_size": "10", "overlap": "10"},
    )
    assert response.status_code == 400


def test_empty_query_422(client) -> None:
    response = client.post("/search", json={"query": "   ", "mode": "dense"})
    assert response.status_code == 422


def test_invalid_mode_422(client) -> None:
    response = client.post("/search", json={"query": "hello", "mode": "hybrid"})
    assert response.status_code == 422


def test_invalid_top_k_422(client) -> None:
    response = client.post("/search", json={"query": "hello", "mode": "dense", "top_k": 0})
    assert response.status_code == 422


def test_top_k_clamped_to_100(client, txt_bytes: bytes) -> None:
    ingest = client.post(
        "/documents",
        files={"file": ("notes.txt", txt_bytes, "text/plain")},
    )
    assert ingest.status_code == 201
    response = client.post("/search", json={"query": "XJ-19", "mode": "bm25", "top_k": 1000})
    assert response.status_code == 200
    assert response.json()["top_k"] == 100
