from __future__ import annotations

import pytest

from app.schemas import AnswerClaim, CitationRef
from app.services.atomic_claims import extract_atomic_claims_rule_based, needs_llm_decompose, split_sentences

pytestmark = pytest.mark.unit


def test_sentence_split_preserves_negation_dates_and_units() -> None:
    claims = [
        AnswerClaim(
            claim_id="c1",
            text="The lab was not founded in 2015. Storage requires 18 °C.",
            citations=[CitationRef(chunk_id="ch1", evidence_quote="not founded")],
        )
    ]
    atomics, pending = extract_atomic_claims_rule_based(claims, claims[0].text)
    assert pending == []
    assert len(atomics) == 2
    assert "not founded in 2015" in atomics[0].text
    assert "18 °C" in atomics[1].text
    assert atomics[0].parent_claim_id == "c1"
    assert atomics[1].citations[0].chunk_id == "ch1"


def test_omitted_display_facts_are_detected() -> None:
    claims = [
        AnswerClaim(
            claim_id="c1",
            text="The Zephyr handshake uses a nonce.",
            citations=[CitationRef(chunk_id="ch1", evidence_quote="nonce")],
        )
    ]
    displayed = "The Zephyr handshake uses a nonce. Operators record failures in the mission diary."
    atomics, _ = extract_atomic_claims_rule_based(claims, displayed)
    extra = [item for item in atomics if item.from_displayed_answer]
    assert extra
    assert "mission diary" in extra[0].text
    assert extra[0].citations == []


def test_needs_llm_decompose_for_multi_number_conjunction() -> None:
    text = "Revenue was $4M in 2022 and $5M in 2023."
    assert needs_llm_decompose(text)
    assert len(split_sentences(text)) == 1
