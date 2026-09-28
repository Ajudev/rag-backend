"""Deterministic citation *reference* validation.

This module checks that cited chunk IDs were actually supplied in the context
pack and that evidence quotes occur in those passages after conservative
normalization. It does **not** prove that a passage semantically supports a
claim. Semantic checks live in ``CitationVerificationService``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.schemas import AnswerClaim, AnswerSource, GroundedLlmOutput
from app.services.context_pack import PackedPassage

_WHITESPACE = re.compile(r"\s+")


def normalize_evidence_text(text: str) -> str:
    """NFKC, casefold, and collapse whitespace without fuzzy matching."""
    nfkc = unicodedata.normalize("NFKC", text)
    collapsed = _WHITESPACE.sub(" ", nfkc).strip()
    return collapsed.casefold()


def quote_in_passage(quote: str, passage: str) -> bool:
    """Return True if the normalized quote is a substring of the passage."""
    needle = normalize_evidence_text(quote)
    haystack = normalize_evidence_text(passage)
    if not needle:
        return False
    return needle in haystack


def locate_quote_offsets(quote: str, passage: str) -> tuple[int, int] | None:
    """Map a normalized quote to original ``[start, end)`` offsets, or None.

    Offsets are never invented: if the span cannot be aligned to the stored
    passage, this returns None.
    """
    needle = normalize_evidence_text(quote)
    if not needle:
        return None
    orig_exact = passage.find(quote)
    if orig_exact >= 0:
        return orig_exact, orig_exact + len(quote)
    folded_passage = passage.casefold()
    folded_quote = quote.casefold()
    folded_at = folded_passage.find(folded_quote)
    if folded_at >= 0:
        return folded_at, folded_at + len(quote)

    out_chars: list[str] = []
    out_orig: list[int] = []
    prev_space = False
    for orig_i, char in enumerate(passage):
        folded = unicodedata.normalize("NFKC", char).casefold()
        if not folded:
            continue
        for folded_char in folded:
            is_space = folded_char.isspace()
            if is_space:
                if prev_space or not out_chars:
                    continue
                out_chars.append(" ")
                out_orig.append(orig_i)
                prev_space = True
                continue
            out_chars.append(folded_char)
            out_orig.append(orig_i)
            prev_space = False
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_orig.pop()
    haystack = "".join(out_chars)
    idx = haystack.find(needle)
    if idx < 0 or not out_orig:
        return None
    end_idx = idx + len(needle) - 1
    if end_idx >= len(out_orig):
        return None
    return out_orig[idx], out_orig[end_idx] + 1


@dataclass
class CitationValidationResult:
    """Outcome of reference validation. Not a semantic-support score."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    claims: list[AnswerClaim] = field(default_factory=list)
    sources: list[AnswerSource] = field(default_factory=list)


def _source_for(passage: PackedPassage) -> AnswerSource:
    return AnswerSource(
        chunk_id=passage.chunk_id,
        document_id=passage.document_id,
        document_title=passage.source,
        page_number=passage.page_num,
        section=None,
        char_start=passage.char_start,
        char_end=passage.char_end,
        passage_text=passage.text,
    )


def validate_grounded_output(
    output: GroundedLlmOutput,
    passages: list[PackedPassage],
) -> CitationValidationResult:
    """Validate citations against the context pack only.

    Allowed IDs are those packed for this request, not the rest of the index.
    Invented IDs are never remapped onto nearby valid IDs.
    """
    by_id = {item.chunk_id: item for item in passages}
    allowed = set(by_id)
    errors: list[str] = []
    cited_ids: list[str] = []

    requires_claims = output.status in {"answered", "partially_answered", "conflicting_evidence"}
    if requires_claims and not output.claims:
        errors.append(f"status {output.status} requires at least one cited claim")

    for claim in output.claims:
        if not claim.citations:
            errors.append(f"claim {claim.claim_id} has no citations")
            continue
        for citation in claim.citations:
            chunk_id = citation.chunk_id
            if chunk_id not in allowed:
                errors.append(
                    f"claim {claim.claim_id} cites chunk_id {chunk_id} which was not in the context pack"
                )
                continue
            passage = by_id[chunk_id]
            if not quote_in_passage(citation.evidence_quote, passage.text):
                errors.append(
                    f"claim {claim.claim_id} evidence_quote was not found in chunk_id {chunk_id}"
                )
                continue
            cited_ids.append(chunk_id)

    if errors:
        return CitationValidationResult(ok=False, errors=errors, claims=[], sources=[])

    sources: list[AnswerSource] = []
    seen: set[str] = set()
    for chunk_id in cited_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        sources.append(_source_for(by_id[chunk_id]))
    return CitationValidationResult(
        ok=True,
        errors=[],
        claims=list(output.claims),
        sources=sources,
    )
