from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.v1.run import STALE_BM25_PATH, run

pytestmark = pytest.mark.integration

CORPUS = {"aurora_protocol.md", "nimbus_lab.txt", "willow_garden.md"}


def test_benchmark_run_ignores_stale_bm25_and_stays_in_corpus(tmp_path: Path) -> None:
    STALE_BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    STALE_BM25_PATH.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "chunk_id": "polluted",
                        "document_id": "polluted-doc",
                        "source": "research.pdf",
                        "text": "Zephyr handshake from a leftover research PDF chunk",
                        "content_type": "pdf",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = run(
        fake_embeddings=True,
        fake_rerank=True,
        results_path=tmp_path / "results.json",
        summary_path=tmp_path / "summary.md",
    )
    assert set(payload["index_sources"]["bm25"]) == CORPUS
    assert set(payload["index_sources"]["dense"]) == CORPUS
    assert not STALE_BM25_PATH.exists()
    for block in payload["test"]["configs"].values():
        for row in block["queries"]:
            assert set(row["retrieved_sources"]) <= CORPUS
