from __future__ import annotations

import os

import pytest

from app.schemas import SemanticCitationJudgement, SemanticClaimJudgement

pytestmark = pytest.mark.integration


def _upload(client, filename: str, data: bytes, content_type: str):
    return client.post("/documents", files={"file": (filename, data, content_type)})


def test_verify_loads_from_index_not_client_body_text(client, app, markdown_bytes: bytes) -> None:
    ingest = _upload(client, "aurora.md", markdown_bytes, "text/markdown")
    assert ingest.status_code == 201
    search = client.post("/search", json={"query": "Zephyr handshake", "mode": "bm25", "top_k": 5})
    hit = search.json()["results"][0]
    chunk_id = hit["chunk_id"]
    real_quote = hit["text"][: min(40, len(hit["text"]))]
    app.state.generation_client.enqueue(
        SemanticClaimJudgement(
            status="SUPPORTED",
            citations=[
                SemanticCitationJudgement(
                    chunk_id=chunk_id,
                    contribution="supports",
                    evidence_quote=real_quote,
                )
            ],
        )
    )
    response = client.post(
        "/verify",
        json={
            "question": "What is Zephyr?",
            "answer": "The Zephyr handshake completes before key derivation.",
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": "The Zephyr handshake completes before key derivation.",
                    "citations": [{"chunk_id": chunk_id, "evidence_quote": real_quote}],
                }
            ],
            "allowed_chunk_ids": [chunk_id],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verification"]["claim_results"]
    sources = body["sources"]
    assert sources
    assert sources[0]["passage_text"] == hit["text"]
    assert "smuggled-client-only-text" not in sources[0]["passage_text"]


def test_verify_unknown_chunk_returns_404(client, markdown_bytes: bytes) -> None:
    _upload(client, "aurora.md", markdown_bytes, "text/markdown")
    response = client.post(
        "/verify",
        json={
            "answer": "A fact.",
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": "A fact.",
                    "citations": [{"chunk_id": "missing-chunk", "evidence_quote": "fact"}],
                }
            ],
            "allowed_chunk_ids": ["missing-chunk"],
        },
    )
    assert response.status_code == 404


def test_verify_all_contradicted_does_not_republish_original(
    client, app, markdown_bytes: bytes
) -> None:
    ingest = _upload(client, "aurora.md", markdown_bytes, "text/markdown")
    assert ingest.status_code == 201
    search = client.post("/search", json={"query": "Zephyr handshake", "mode": "bm25", "top_k": 5})
    hit = search.json()["results"][0]
    chunk_id = hit["chunk_id"]
    real_quote = hit["text"][: min(40, len(hit["text"]))]
    original = "The firm was founded in 2015."
    app.state.generation_client.enqueue(
        SemanticClaimJudgement(
            status="CONTRADICTED",
            citations=[
                SemanticCitationJudgement(
                    chunk_id=chunk_id,
                    contribution="contradicts",
                    evidence_quote=real_quote,
                )
            ],
        )
    )
    response = client.post(
        "/verify",
        json={
            "answer": original,
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": original,
                    "citations": [{"chunk_id": chunk_id, "evidence_quote": real_quote}],
                }
            ],
            "allowed_chunk_ids": [chunk_id],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] in {"conflicting_evidence", "insufficient_evidence"}
    assert body["claims"] == []
    assert original not in body["answer"]


def test_verify_empty_allowed_list_with_citations_returns_422(client, markdown_bytes: bytes) -> None:
    ingest = _upload(client, "aurora.md", markdown_bytes, "text/markdown")
    chunk_id = ingest.json()["chunk_ids"][0]
    response = client.post(
        "/verify",
        json={
            "answer": "A fact.",
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": "A fact.",
                    "citations": [{"chunk_id": chunk_id, "evidence_quote": "Zephyr"}],
                }
            ],
            "allowed_chunk_ids": [],
        },
    )
    assert response.status_code == 422


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="live LLM test requires RUN_LIVE_LLM=1 and OPENAI_API_KEY",
)
def test_live_verify_skipped_in_ci(client) -> None:
    response = client.post(
        "/verify",
        json={"answer": "x", "claims": [], "allowed_chunk_ids": []},
    )
    assert response.status_code in {200, 422, 503}
