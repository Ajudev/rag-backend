"""Run the v1 retrieval benchmark across dense, BM25, hybrid, and rerank.

Ingests the sample corpus, evaluates held-out test queries for four configs,
tunes hybrid weights/candidates on validation only, then writes
``benchmark/v1/results.json`` and ``benchmark/v1/summary.md``.

Usage:
    uv run python -m benchmark.v1.run --fake-embeddings --fake-rerank
    uv run python -m benchmark.v1.run
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.main import create_app
from app.schemas import SearchResponse
from app.services.embedder import FakeEmbedder, SentenceTransformerEmbedder
from app.services.ingest import IngestService
from app.services.reranker import FakeReranker
from app.services.search import SearchService

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
SUMMARY_PATH = Path(__file__).resolve().parent / "summary.md"
STALE_BM25_PATH = Path(__file__).resolve().parent / ".bm25.json"

TEST_CONFIGS: list[dict[str, Any]] = [
    {"id": "A", "label": "dense", "mode": "dense", "kwargs": {}},
    {"id": "B", "label": "bm25", "mode": "bm25", "kwargs": {}},
    {
        "id": "C",
        "label": "hybrid",
        "mode": "hybrid",
        "kwargs": {
            "dense_candidate_count": 20,
            "bm25_candidate_count": 20,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "bm25_weight": 1.0,
        },
    },
    {
        "id": "D",
        "label": "hybrid_rerank",
        "mode": "hybrid_rerank",
        "kwargs": {
            "dense_candidate_count": 20,
            "bm25_candidate_count": 20,
            "rerank_candidate_count": 20,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "bm25_weight": 1.0,
        },
    },
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def _unique_sources(sources: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for source in sources:
        if source not in seen:
            seen.add(source)
            ordered.append(source)
    return ordered


def _recall_at_k(retrieved_sources: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = set(retrieved_sources[:k]) & gold
    return len(hit) / len(gold)


def _mrr_at_k(retrieved_sources: list[str], gold: set[str], k: int) -> float:
    for rank, source in enumerate(retrieved_sources[:k], start=1):
        if source in gold:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_sources: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = 0.0
    for rank, source in enumerate(retrieved_sources[:k], start=1):
        rel = 1.0 if source in gold else 0.0
        dcg += rel / math.log2(rank + 1)
    ideal_count = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _row_metrics(response: SearchResponse, gold: set[str], item: dict[str, Any]) -> dict[str, Any]:
    sources = _unique_sources([hit.metadata.source for hit in response.results])
    rerank_ms = response.timing.rerank_ms
    return {
        "id": item["id"],
        "query": item["query"],
        "category": item.get("category"),
        "split": item.get("split"),
        "gold_sources": item["gold_sources"],
        "retrieved_sources": sources,
        "recall_at_5": round(_recall_at_k(sources, gold, 5), 4),
        "recall_at_10": round(_recall_at_k(sources, gold, 10), 4),
        "mrr_at_10": round(_mrr_at_k(sources, gold, 10), 4),
        "ndcg_at_10": round(_ndcg_at_k(sources, gold, 10), 4),
        "latency_ms": response.latency_ms,
        "rerank_ms": rerank_ms,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "rerank_ms_p50": None,
            "rerank_ms_p95": None,
        }
    latencies = [row["latency_ms"] for row in rows]
    rerank_values = [row["rerank_ms"] for row in rows if row.get("rerank_ms") is not None]
    summary: dict[str, Any] = {
        "recall_at_5": round(statistics.fmean(row["recall_at_5"] for row in rows), 4),
        "recall_at_10": round(statistics.fmean(row["recall_at_10"] for row in rows), 4),
        "mrr_at_10": round(statistics.fmean(row["mrr_at_10"] for row in rows), 4),
        "ndcg_at_10": round(statistics.fmean(row["ndcg_at_10"] for row in rows), 4),
        "latency_ms_p50": round(_percentile(latencies, 50), 2),
        "latency_ms_p95": round(_percentile(latencies, 95), 2),
        "rerank_ms_p50": round(_percentile(rerank_values, 50), 2) if rerank_values else None,
        "rerank_ms_p95": round(_percentile(rerank_values, 95), 2) if rerank_values else None,
    }
    return summary


def _by_category(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("category") or "uncategorized", []).append(row)
    return {category: _summarize(items) for category, items in grouped.items()}


def _evaluate(
    search: SearchService,
    items: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int = 10,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        gold = set(item["gold_sources"])
        response = search.search(item["query"], mode=mode, top_k=top_k, **kwargs)
        rows.append(_row_metrics(response, gold, item))
    return rows


def assert_indexes_match_corpus(
    qdrant_sources: set[str],
    bm25_sources: set[str],
    corpus: set[str],
) -> None:
    """Fail if either index contains files outside the configured corpus.

    Args:
        qdrant_sources: Source filenames stored in the dense collection.
        bm25_sources: Source filenames stored in the BM25 index.
        corpus: Filenames listed in ``queries.json``.

    Raises:
        RuntimeError: Indexes are empty, disagree, or include extra sources.
    """
    extra_bm25 = sorted(bm25_sources - corpus)
    extra_dense = sorted(qdrant_sources - corpus)
    problems: list[str] = []
    if extra_bm25:
        problems.append(f"BM25 contains sources outside the corpus: {extra_bm25}")
    if extra_dense:
        problems.append(f"dense index contains sources outside the corpus: {extra_dense}")
    if not bm25_sources:
        problems.append("BM25 index is empty after ingest")
    if not qdrant_sources:
        problems.append("dense index is empty after ingest")
    if bm25_sources != qdrant_sources:
        problems.append(f"BM25 sources {sorted(bm25_sources)} do not match dense sources {sorted(qdrant_sources)}")
    if problems:
        raise RuntimeError("; ".join(problems))


def assert_retrieved_sources_in_corpus(payload: dict[str, Any], corpus: set[str]) -> None:
    """Fail if any evaluated hit came from a file outside the corpus."""
    extras: list[str] = []
    for config_id, block in payload.get("test", {}).get("configs", {}).items():
        for row in block.get("queries", []):
            leaked = sorted(set(row.get("retrieved_sources", [])) - corpus)
            if leaked:
                extras.append(f"{config_id}/{row.get('id')}: {leaked}")
    if extras:
        raise RuntimeError("retrieved sources outside the benchmark corpus: " + "; ".join(extras))


def _index_sources(qdrant: Any, bm25: Any) -> tuple[set[str], set[str]]:
    qdrant_sources = {str(payload.get("source", "")) for payload in qdrant.scroll_all() if payload.get("source")}
    return qdrant_sources, bm25.indexed_sources()


def _select_validation_config(grid: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(entry: dict[str, Any]) -> tuple:
        summary = entry["summary"]
        return (
            summary["mrr_at_10"],
            summary["recall_at_10"],
            summary["recall_at_5"],
            entry["params"]["dense_candidate_count"],
        )

    return max(grid, key=sort_key)


def _write_summary(payload: dict[str, Any], summary_path: Path) -> None:
    env = payload["environment"]
    flags = []
    if payload["fake_embeddings"]:
        flags.append("hash FakeEmbedder (not semantically meaningful)")
    if payload["fake_rerank"]:
        flags.append("FakeReranker term-overlap scores")
    flag_note = "; ".join(flags) if flags else "real embedding and reranker models"
    lines = [
        "# Retrieval benchmark v1",
        "",
        f"- Run date (UTC): {payload['run_date']}",
        f"- Corpus: {', '.join(payload['corpus'])}",
        f"- Isolated BM25/dense sources: {', '.join(payload.get('index_sources', {}).get('bm25', []))}",
        f"- Embedding model: {payload['models']['embedding']}",
        f"- Reranker model: {payload['models']['reranker']}",
        f"- Python: {env['python']}",
        f"- Device: {env['device']}",
        f"- Backends: {flag_note}",
        "",
        "These numbers are measured by `benchmark.v1.run`. Hybrid RRF and rerank are not assumed to win.",
        "",
        "## Test split (held-out; not used for tuning)",
        "",
        "| config | mode | R@5 | R@10 | MRR@10 | nDCG@10 | lat p50 | lat p95 | rerank p50 | rerank p95 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for config_id, block in payload["test"]["configs"].items():
        summary = block["summary"]
        rerank_p50 = summary["rerank_ms_p50"]
        rerank_p95 = summary["rerank_ms_p95"]
        lines.append(
            f"| {config_id} | {block['mode']} | {summary['recall_at_5']:.4f} | {summary['recall_at_10']:.4f} | "
            f"{summary['mrr_at_10']:.4f} | {summary['ndcg_at_10']:.4f} | {summary['latency_ms_p50']:.2f} | "
            f"{summary['latency_ms_p95']:.2f} | {rerank_p50 if rerank_p50 is not None else '—'} | "
            f"{rerank_p95 if rerank_p95 is not None else '—'} |"
        )
    selected = payload["validation"]["selected"]
    lines.extend(
        [
            "",
            "## Validation grid (hybrid only)",
            "",
            f"Selected by validation MRR@10 (ties: R@10, then R@5): "
            f"dense_weight={selected['params']['dense_weight']}, "
            f"bm25_weight={selected['params']['bm25_weight']}, "
            f"candidates={selected['params']['dense_candidate_count']}.",
            "",
            "| dense_w | bm25_w | candidates | MRR@10 | R@10 | R@5 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in payload["validation"]["grid"]:
        params = entry["params"]
        summary = entry["summary"]
        lines.append(
            f"| {params['dense_weight']} | {params['bm25_weight']} | {params['dense_candidate_count']} | "
            f"{summary['mrr_at_10']:.4f} | {summary['recall_at_10']:.4f} | {summary['recall_at_5']:.4f} |"
        )
    tuned = payload["test"]["tuned_hybrid"]
    tuned_summary = tuned["summary"]
    lines.extend(
        [
            "",
            "## Selected hybrid on test (chosen on validation only)",
            "",
            "| R@5 | R@10 | MRR@10 | nDCG@10 | lat p50 | lat p95 |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| {tuned_summary['recall_at_5']:.4f} | {tuned_summary['recall_at_10']:.4f} | "
            f"{tuned_summary['mrr_at_10']:.4f} | {tuned_summary['ndcg_at_10']:.4f} | "
            f"{tuned_summary['latency_ms_p50']:.2f} | {tuned_summary['latency_ms_p95']:.2f} |",
            "",
            "## Parameters",
            "",
            f"- Test top_k: {payload['parameters']['top_k']} (same for all four configs)",
            f"- RRF k: {payload['parameters']['rrf_k']}",
            f"- Default candidate pools: dense {payload['parameters']['dense_candidate_count']}, "
            f"BM25 {payload['parameters']['bm25_candidate_count']}, rerank {payload['parameters']['rerank_candidate_count']}",
            "",
            "## Limitations",
            "",
            "- Tiny sample corpus; metrics are not a production quality claim.",
            "- Each run uses a fresh temp BM25 index; leftover `.bm25.json` is not loaded.",
            "- Fake embeddings and fake rerank (when enabled) are for CI speed, not ranking quality.",
            "- Cross-encoder scores are uncalibrated ranking utilities, not probabilities.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    fake_embeddings: bool = False,
    fake_rerank: bool = False,
    *,
    results_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Ingest the sample corpus and evaluate four retrieval configs.

    BM25 is always a fresh temp file so a leftover ``.bm25.json`` cannot leak
    extra PDFs into lexical/hybrid modes.

    Args:
        fake_embeddings: Use hash vectors instead of BGE.
        fake_rerank: Use ``FakeReranker`` instead of MiniLM.
        results_path: Optional override for ``results.json``.
        summary_path: Optional override for ``summary.md``.

    Returns:
        Metrics payload also written to ``results.json``.
    """
    if STALE_BM25_PATH.exists():
        STALE_BM25_PATH.unlink()

    spec = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    corpus_names = spec.get("corpus") or []
    corpus = set(corpus_names)
    ingested: list[dict[str, Any]] = []
    test_items = [item for item in spec["queries"] if item.get("split") == "test"]
    validation_items = [item for item in spec["queries"] if item.get("split") == "validation"]
    if not test_items:
        test_items = list(spec["queries"])

    with tempfile.TemporaryDirectory(prefix="rag-bench-bm25-") as tmp_dir:
        settings = Settings(
            _env_file=None,
            qdrant_url=":memory:",
            qdrant_collection="benchmark_v1",
            embedding_model="fake",
            min_chunk_words=5,
            bm25_index_path=Path(tmp_dir) / "bm25.json",
            reranker_model="fake-reranker",
        )
        embedder = FakeEmbedder() if fake_embeddings else SentenceTransformerEmbedder(settings.embedding_model)
        reranker = FakeReranker() if fake_rerank else None
        client = QdrantClient(location=":memory:")
        app = create_app(settings=settings, embedder=embedder, qdrant_client=client, reranker=reranker)

        with TestClient(app) as _http:
            ingest = IngestService(settings, app.state.qdrant_store, app.state.bm25_index, embedder)
            search = SearchService(
                settings,
                app.state.qdrant_store,
                app.state.bm25_index,
                embedder,
                reranker=app.state.reranker,
            )
            paths = [DOCS_DIR / name for name in corpus_names] if corpus_names else sorted(DOCS_DIR.iterdir())
            for path in paths:
                if not path.is_file() or path.suffix.lower() not in {".md", ".markdown", ".txt", ".pdf"}:
                    continue
                result = ingest.ingest(path.name, None, path.read_bytes())
                ingested.append(
                    {
                        "filename": path.name,
                        "document_id": result.document_id,
                        "chunks": result.chunk_count,
                    }
                )
            if not corpus:
                corpus = {item["filename"] for item in ingested}
            dense_sources, bm25_sources = _index_sources(app.state.qdrant_store, app.state.bm25_index)
            assert_indexes_match_corpus(dense_sources, bm25_sources, corpus)

            test_configs: dict[str, Any] = {}
            for config in TEST_CONFIGS:
                rows = _evaluate(search, test_items, mode=config["mode"], top_k=10, **config["kwargs"])
                test_configs[config["id"]] = {
                    "mode": config["mode"],
                    "label": config["label"],
                    "params": config["kwargs"],
                    "queries": rows,
                    "summary": _summarize(rows),
                    "by_category": _by_category(rows),
                }

            grid: list[dict[str, Any]] = []
            for dense_weight, bm25_weight in ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0)):
                for candidate_count in (10, 20):
                    params = {
                        "dense_weight": dense_weight,
                        "bm25_weight": bm25_weight,
                        "dense_candidate_count": candidate_count,
                        "bm25_candidate_count": candidate_count,
                        "rrf_k": 60,
                    }
                    rows = _evaluate(search, validation_items, mode="hybrid", top_k=10, **params)
                    grid.append({"params": params, "summary": _summarize(rows), "queries": rows})
            selected = _select_validation_config(grid) if grid else {"params": TEST_CONFIGS[2]["kwargs"], "summary": {}}
            tuned_rows = _evaluate(search, test_items, mode="hybrid", top_k=10, **selected["params"])
            index_sources = {"dense": sorted(dense_sources), "bm25": sorted(bm25_sources)}

    payload: dict[str, Any] = {
        "version": spec.get("version", "v1"),
        "benchmark_version": spec.get("version", "v1"),
        "fake_embeddings": fake_embeddings,
        "fake_rerank": fake_rerank,
        "run_date": datetime.now(UTC).isoformat(),
        "corpus": spec.get("corpus", [item["filename"] for item in ingested]),
        "models": {
            "embedding": "fake-hash" if fake_embeddings else settings.embedding_model,
            "reranker": "FakeReranker" if fake_rerank else settings.reranker_model,
        },
        "parameters": {
            "top_k": 10,
            "rrf_k": 60,
            "dense_candidate_count": 20,
            "bm25_candidate_count": 20,
            "rerank_candidate_count": 20,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": settings.reranker_device,
        },
        "ingested": ingested,
        "index_sources": index_sources,
        "validation": {
            "grid": grid,
            "selected": {"params": selected["params"], "summary": selected.get("summary")},
        },
        "test": {
            "configs": test_configs,
            "tuned_hybrid": {
                "params": selected["params"],
                "queries": tuned_rows,
                "summary": _summarize(tuned_rows),
                "by_category": _by_category(tuned_rows),
            },
        },
    }
    assert_retrieved_sources_in_corpus(payload, corpus)
    written_results = results_path or RESULTS_PATH
    written_summary = summary_path or SUMMARY_PATH
    written_results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary(payload, written_summary)
    _print_table(payload)
    return payload


def _print_table(payload: dict[str, Any]) -> None:
    print(
        f"Benchmark {payload['version']}  fake_embeddings={payload['fake_embeddings']} "
        f"fake_rerank={payload['fake_rerank']}"
    )
    print(f"{'cfg':<4} {'mode':<16} {'R@5':>8} {'R@10':>8} {'MRR@10':>8} {'nDCG@10':>8} {'p50':>8} {'p95':>8}")
    for config_id, block in payload["test"]["configs"].items():
        summary = block["summary"]
        print(
            f"{config_id:<4} {block['mode']:<16} {summary['recall_at_5']:8.3f} "
            f"{summary['recall_at_10']:8.3f} {summary['mrr_at_10']:8.3f} {summary['ndcg_at_10']:8.3f} "
            f"{summary['latency_ms_p50']:8.2f} {summary['latency_ms_p95']:8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval benchmark v1")
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use hash embeddings (no model download; recall is not meaningful)",
    )
    parser.add_argument(
        "--fake-rerank",
        action="store_true",
        help="Use FakeReranker (no MiniLM download; rerank quality is not meaningful)",
    )
    args = parser.parse_args()
    run(fake_embeddings=args.fake_embeddings, fake_rerank=args.fake_rerank)


if __name__ == "__main__":
    main()
