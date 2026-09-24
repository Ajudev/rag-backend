from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import create_app
from app.services.reranker import FakeReranker

pytestmark = pytest.mark.integration


def _upload(client, filename: str, data: bytes, content_type: str):
    return client.post("/documents", files={"file": (filename, data, content_type)})


def test_hybrid_runs_both_retrievers_and_sets_contributed_by(client, markdown_bytes: bytes) -> None:
    ingest = _upload(client, "aurora.md", markdown_bytes, "text/markdown")
    assert ingest.status_code == 201
    response = client.post(
        "/search",
        json={"query": "Zephyr handshake", "mode": "hybrid", "top_k": 5, "debug": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "hybrid"
    assert body["timing"]["dense_ms"] is not None
    assert body["timing"]["bm25_ms"] is not None
    assert body["timing"]["fusion_ms"] is not None
    assert body["timing"]["rerank_ms"] is None
    assert body["debug"]["dense_candidate_count"] == 20
    assert body["results"]
    contributors = {item for hit in body["results"] for item in hit["contributed_by"]}
    assert contributors <= {"dense", "bm25"}
    assert contributors
    for hit in body["results"]:
        assert hit["rrf_score"] is not None
        assert hit["score"] == hit["rrf_score"]
        assert hit["contributed_by"]


def test_hybrid_filters_do_not_leak_sources(client, markdown_bytes: bytes, txt_bytes: bytes) -> None:
    _upload(client, "aurora.md", markdown_bytes, "text/markdown")
    _upload(client, "nimbus.txt", txt_bytes, "text/plain")
    response = client.post(
        "/search",
        json={
            "query": "handshake laboratory XJ-19",
            "mode": "hybrid",
            "filters": {"source": "nimbus.txt"},
        },
    )
    assert response.status_code == 200
    hits = response.json()["results"]
    assert hits
    assert all(hit["metadata"]["source"] == "nimbus.txt" for hit in hits)


def test_hybrid_rerank_sets_scores_and_can_reorder(
    settings,
    embedder,
    markdown_bytes: bytes,
    txt_bytes: bytes,
) -> None:
    reranker = FakeReranker(prefer_substring="XJ-19")
    app = create_app(
        settings=settings,
        embedder=embedder,
        qdrant_client=QdrantClient(location=":memory:"),
        reranker=reranker,
    )
    with TestClient(app) as client:
        _upload(client, "aurora.md", markdown_bytes, "text/markdown")
        _upload(client, "nimbus.txt", txt_bytes, "text/plain")
        payload = {"query": "Zephyr handshake", "top_k": 5}
        hybrid = client.post("/search", json={**payload, "mode": "hybrid"})
        reranked = client.post("/search", json={**payload, "mode": "hybrid_rerank"})
        assert hybrid.status_code == 200
        assert reranked.status_code == 200
        hybrid_ids = [hit["chunk_id"] for hit in hybrid.json()["results"]]
        rerank_hits = reranked.json()["results"]
        rerank_ids = [hit["chunk_id"] for hit in rerank_hits]
        assert all(hit["rerank_score"] is not None for hit in rerank_hits)
        assert all(hit["rrf_score"] is not None for hit in rerank_hits)
        assert reranked.json()["timing"]["rerank_ms"] is not None
        assert rerank_ids[0] != hybrid_ids[0] or rerank_hits[0]["score"] == rerank_hits[0]["rerank_score"]
        assert "XJ-19" in rerank_hits[0]["text"]


def test_invalid_fusion_params_422(client) -> None:
    zero_weights = client.post(
        "/search",
        json={"query": "hello", "mode": "hybrid", "dense_weight": 0, "bm25_weight": 0},
    )
    assert zero_weights.status_code == 422
    too_small_pool = client.post(
        "/search",
        json={"query": "hello", "mode": "hybrid_rerank", "top_k": 10, "rerank_candidate_count": 3},
    )
    assert too_small_pool.status_code == 422
    unused_ok = client.post(
        "/search",
        json={"query": "hello", "mode": "dense", "dense_candidate_count": 5, "rrf_k": 10},
    )
    assert unused_ok.status_code == 200
    alias = client.post("/search", json={"query": "hello", "search_mode": "bm25"})
    assert alias.status_code == 200


def test_reranker_instance_reused(client, app, txt_bytes: bytes) -> None:
    ingest = _upload(client, "notes.txt", txt_bytes, "text/plain")
    assert ingest.status_code == 201
    reranker = app.state.reranker
    assert reranker.load_count == 1
    first = client.post("/search", json={"query": "XJ-19", "mode": "hybrid_rerank", "top_k": 3})
    second = client.post("/search", json={"query": "XJ-19", "mode": "hybrid_rerank", "top_k": 3})
    assert first.status_code == 200
    assert second.status_code == 200
    assert app.state.reranker is reranker
    assert reranker.load_count == 1
    assert reranker.predict_calls == 2
