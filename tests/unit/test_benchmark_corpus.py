from __future__ import annotations

import pytest

from benchmark.v1.run import assert_indexes_match_corpus, assert_retrieved_sources_in_corpus

pytestmark = pytest.mark.unit

CORPUS = {"aurora_protocol.md", "nimbus_lab.txt", "willow_garden.md"}


def test_assert_indexes_match_corpus_accepts_aligned_sources() -> None:
    assert_indexes_match_corpus(CORPUS, CORPUS, CORPUS)


def test_assert_indexes_match_corpus_rejects_extra_bm25_source() -> None:
    with pytest.raises(RuntimeError, match="BM25 contains sources outside the corpus"):
        assert_indexes_match_corpus(CORPUS, CORPUS | {"research.pdf"}, CORPUS)


def test_assert_indexes_match_corpus_rejects_mismatched_indexes() -> None:
    with pytest.raises(RuntimeError, match="do not match dense sources"):
        assert_indexes_match_corpus(CORPUS, {"aurora_protocol.md"}, CORPUS)


def test_assert_retrieved_sources_in_corpus_rejects_pdf_leak() -> None:
    payload = {
        "test": {
            "configs": {
                "B": {
                    "queries": [
                        {"id": "q1", "retrieved_sources": ["aurora_protocol.md", "paper.pdf"]},
                    ]
                }
            }
        }
    }
    with pytest.raises(RuntimeError, match="retrieved sources outside"):
        assert_retrieved_sources_in_corpus(payload, CORPUS)
