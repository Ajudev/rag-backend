"""Citation verification benchmark (tiny labeled set).

Fake path (default) measures deterministic invalid-citation detection and
records that programmed semantic labels are not quality metrics.

Live path (``--live``) requires OPENAI_API_KEY and classifies the held-out
test split. Do not invent metrics when live is not run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.schemas import AtomicClaim, CitationRef
from app.services.citation_validate import quote_in_passage
from app.services.citation_verification import CitationVerificationService
from app.services.context_pack import PackedPassage
from app.services.generation import OpenAIGenerationClient

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "cases.json"
RESULTS_PATH = ROOT / "results.json"
SUMMARY_PATH = ROOT / "summary.md"
STATUSES = (
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "UNCERTAIN",
)


def _load_cases() -> list[dict[str, Any]]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return list(payload["cases"])


def _passages(case: dict[str, Any]) -> list[PackedPassage]:
    packed: list[PackedPassage] = []
    for item in case["passages"]:
        text = item["text"]
        packed.append(
            PackedPassage(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                source=item["document_id"],
                page_num=1,
                chunk_index=0,
                text=text,
                char_start=0,
                char_end=len(text),
            )
        )
    return packed


def _atomic(case: dict[str, Any]) -> AtomicClaim:
    citations = [
        CitationRef(chunk_id=item["chunk_id"], evidence_quote=item["evidence_quote"])
        for item in case["citations"]
    ]
    return AtomicClaim(
        atomic_claim_id=case["id"],
        parent_claim_id=None,
        text=case["claim"],
        citations=citations,
    )


def _deterministic_invalid(case: dict[str, Any]) -> bool:
    allowed = {item["chunk_id"] for item in case["passages"]}
    by_id = {item["chunk_id"]: item["text"] for item in case["passages"]}
    for citation in case["citations"]:
        chunk_id = citation["chunk_id"]
        if chunk_id not in allowed:
            return True
        if not quote_in_passage(citation["evidence_quote"], by_id[chunk_id]):
            return True
    return False


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def _confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    matrix = {gold: dict.fromkeys(STATUSES, 0) for gold in STATUSES}
    for gold, pred in pairs:
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def _prf(pairs: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    per_class: dict[str, dict[str, float]] = {}
    for label in STATUSES:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    f1s = [item["f1"] for item in per_class.values()]
    per_class["macro"] = {
        "precision": statistics.mean([item["precision"] for item in per_class.values() if "support" in item]),
        "recall": statistics.mean([item["recall"] for item in per_class.values() if "support" in item]),
        "f1": statistics.mean(f1s),
        "support": float(len(pairs)),
    }
    return per_class


async def _live_classify(cases: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("--live requires OPENAI_API_KEY")
    from app.core.config import Settings

    settings = Settings(
        _env_file=None,
        qdrant_url="http://localhost:6333",
        qdrant_collection="bench",
        embedding_model="fake",
        reranker_model="fake",
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    )
    client = OpenAIGenerationClient(
        api_key=api_key,
        model=settings.openai_model or "gpt-4o-mini",
        timeout_seconds=30,
        max_retries=1,
    )
    service = CitationVerificationService(settings=settings, generation_client=client)
    pairs: list[tuple[str, str]] = []
    latencies: list[float] = []
    false_support = 0
    negative = 0
    try:
        for case in cases:
            started = time.perf_counter()
            result = await service.semantic_verify(_atomic(case), _passages(case))
            latencies.append((time.perf_counter() - started) * 1000.0)
            pred = result.semantic_status
            gold = case["gold_status"]
            pairs.append((gold, pred))
            if gold in {"UNSUPPORTED", "CONTRADICTED"}:
                negative += 1
                if pred == "SUPPORTED":
                    false_support += 1
    finally:
        await client.aclose()
    return {
        "pairs": pairs,
        "per_class": _prf(pairs),
        "confusion": _confusion(pairs),
        "false_support_rate": (false_support / negative) if negative else None,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "n": len(pairs),
    }


def _write_summary(payload: dict[str, Any]) -> None:
    lines = [
        "# Citation verification benchmark",
        "",
        f"Generated at {payload['generated_at']}.",
        "",
        "Tiny labeled set. Statistical power is limited; do not treat scores as production quality.",
        "Gold labels were written by a human, not by the verifier model.",
        "",
        "## Fake run",
        "",
        f"- Cases: {payload['fake']['n_cases']}",
        f"- Invalid-citation cases: {payload['fake']['n_invalid']}",
        f"- Invalid-citation detection rate: {payload['fake']['invalid_citation_detection_rate']:.3f}",
        "",
        "Semantic cases were **not** scored for quality on the fake path.",
        "Programmed/fake semantic labels are not a quality metric.",
        "",
        "## Live metrics",
        "",
    ]
    live = payload.get("live")
    if live is None:
        lines.append("Live metrics were **not measured** (no `--live` run). No scores are invented.")
        lines.append("")
        lines.append("A/B/C comparison protocol (not executed in this run):")
        lines.append("- A: grounded generation without semantic verification")
        lines.append("- B: deterministic citation reference validation only")
        lines.append("- C: full CitationVerificationService (`semantic_verify`)")
        lines.append("Compare false-support rate and macro F1 on the held-out test split only.")
    else:
        lines.append(f"- Test cases scored: {live['n']}")
        lines.append(f"- Macro F1: {live['per_class']['macro']['f1']:.3f}")
        lines.append(f"- False support rate: {live['false_support_rate']}")
        lines.append(f"- Invalid citation detection rate: {live.get('invalid_citation_detection_rate')}")
        lines.append(f"- Latency p50/p95 ms: {live['latency_p50_ms']:.1f} / {live['latency_p95_ms']:.1f}")
        lines.append("")
        lines.append("Per-class precision/recall:")
        for label in STATUSES:
            row = live["per_class"][label]
            lines.append(
                f"- {label}: P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} n={int(row['support'])}"
            )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Citation verification benchmark")
    parser.add_argument("--live", action="store_true", help="Call OpenAI on the held-out test split")
    args = parser.parse_args()
    cases = _load_cases()
    invalid_cases = [case for case in cases if case.get("invalid_citation")]
    detected = sum(1 for case in invalid_cases if _deterministic_invalid(case))
    rate = detected / len(invalid_cases) if invalid_cases else 0.0
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fake": {
            "n_cases": len(cases),
            "n_invalid": len(invalid_cases),
            "invalid_citation_detection_rate": rate,
            "semantic_quality": "not measured (fake path)",
        },
        "live": None,
    }
    if args.live:
        import asyncio

        test_cases = [case for case in cases if case["split"] == "test"]
        live = asyncio.run(_live_classify(test_cases))
        live_invalid = [case for case in test_cases if case.get("invalid_citation")]
        live_detected = sum(1 for case in live_invalid if _deterministic_invalid(case))
        live["invalid_citation_detection_rate"] = (
            live_detected / len(live_invalid) if live_invalid else None
        )
        payload["live"] = live
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_summary(payload)
    print(json.dumps(payload["fake"], indent=2))
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
