from __future__ import annotations

import pytest

from app.schemas import AnswerClaim, CitationRef, GroundedLlmOutput
from app.services.citation_validate import normalize_evidence_text, quote_in_passage, validate_grounded_output
from app.services.context_pack import PackedPassage

pytestmark = pytest.mark.unit


def _passage(chunk_id: str = "chunk-a", text: str = "The Zephyr handshake uses a nonce.") -> PackedPassage:
    return PackedPassage(
        chunk_id=chunk_id,
        document_id="doc-a",
        source="aurora.md",
        page_num=1,
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
    )


def _output(chunk_id: str, quote: str, status: str = "answered") -> GroundedLlmOutput:
    return GroundedLlmOutput(
        status=status,  # type: ignore[arg-type]
        claims=[
            AnswerClaim(
                claim_id="claim_1",
                text="The handshake uses a nonce.",
                citations=[CitationRef(chunk_id=chunk_id, evidence_quote=quote)],
            )
        ],
    )


def test_normalize_collapses_whitespace_and_case() -> None:
    assert normalize_evidence_text("  The  ZEPHYR\nhandshake ") == "the zephyr handshake"


def test_quote_in_passage_after_normalization() -> None:
    passage = "The Zephyr  handshake uses a nonce."
    assert quote_in_passage("zephyr handshake", passage)
    assert not quote_in_passage("completely unrelated sentence", passage)


def test_unknown_chunk_id_rejected() -> None:
    result = validate_grounded_output(_output("invented", "nonce"), [_passage()])
    assert result.ok is False
    assert result.claims == []
    assert any("not in the context pack" in err for err in result.errors)


def test_index_id_not_in_pack_rejected() -> None:
    packed = _passage("chunk-a")
    result = validate_grounded_output(_output("chunk-b", "nonce"), [packed])
    assert result.ok is False
    assert any("chunk-b" in err for err in result.errors)


def test_quote_missing_from_passage_rejected() -> None:
    result = validate_grounded_output(_output("chunk-a", "not in the passage"), [_passage()])
    assert result.ok is False
    assert any("evidence_quote" in err for err in result.errors)


def test_empty_citations_rejected() -> None:
    output = GroundedLlmOutput(
        status="answered",
        claims=[AnswerClaim(claim_id="claim_1", text="A fact.", citations=[])],
    )
    result = validate_grounded_output(output, [_passage()])
    assert result.ok is False
    assert any("no citations" in err for err in result.errors)


def test_semantic_unsupported_claim_still_passes_reference_validation() -> None:
    """Quote exists in the passage but does not support the claim.

    This passing result is intentional: reference validation is substring/ID
    matching only. Semantic citation verification (entailment) is out of scope.
    """
    passage = _passage(text="The greenhouse temperature is kept at 18 degrees Celsius.")
    output = GroundedLlmOutput(
        status="answered",
        claims=[
            AnswerClaim(
                claim_id="claim_1",
                text="The greenhouse temperature is kept at 90 degrees Celsius.",
                citations=[
                    CitationRef(
                        chunk_id="chunk-a",
                        evidence_quote="The greenhouse temperature is kept at 18 degrees Celsius.",
                    )
                ],
            )
        ],
    )
    result = validate_grounded_output(output, [passage])
    assert result.ok is True
    assert result.claims[0].text.startswith("The greenhouse temperature is kept at 90")


def test_insufficient_evidence_with_no_claims_is_ok() -> None:
    output = GroundedLlmOutput(status="insufficient_evidence", claims=[])
    result = validate_grounded_output(output, [_passage()])
    assert result.ok is True
    assert result.claims == []
