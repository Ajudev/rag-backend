"""Run the v1 retrieval benchmark.

Ingests ``docs/``, searches each query in dense and BM25 modes, then writes
``benchmark/v1/results.json`` and prints a summary table.

Usage:
    uv run python -m benchmark.v1.run
    uv run python benchmark/v1/run.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.main import create_app
from app.services.embedder import FakeEmbedder, SentenceTransformerEmbedder
from app.services.ingest import IngestService
from app.services.search import SearchService

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


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


def _summarize(rows: list[dict]) -> dict:
    recalls5 = [row["recall_at_5"] for row in rows]
    recalls10 = [row["recall_at_10"] for row in rows]
    mrrs = [row["mrr_at_10"] for row in rows]
    latencies = [row["latency_ms"] for row in rows]
    return {
        "recall_at_5": round(statistics.fmean(recalls5), 4) if recalls5 else 0.0,
        "recall_at_10": round(statistics.fmean(recalls10), 4) if recalls10 else 0.0,
        "mrr_at_10": round(statistics.fmean(mrrs), 4) if mrrs else 0.0,
        "latency_ms_mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "latency_ms_p50": round(_percentile(latencies, 50), 2),
        "latency_ms_p95": round(_percentile(latencies, 95), 2),
    }


def run(fake_embeddings: bool = False) -> dict:
    """Ingest the sample corpus and evaluate dense vs BM25 retrieval.

    Args:
        fake_embeddings: Use hash vectors instead of BGE (fast, not semantically meaningful).

    Returns:
        Metrics payload also written to ``results.json``.
    """
    settings = Settings(
        qdrant_url=":memory:",
        qdrant_collection="benchmark_v1",
        min_chunk_words=5,
        bm25_index_path=ROOT / "benchmark" / "v1" / ".bm25.json",
    )
    embedder = FakeEmbedder() if fake_embeddings else SentenceTransformerEmbedder(settings.embedding_model)
    client = QdrantClient(location=":memory:")
    app = create_app(settings=settings, embedder=embedder, qdrant_client=client)

    spec = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    ingested: list[dict] = []
    per_mode: dict[str, list[dict]] = {"dense": [], "bm25": []}

    with TestClient(app) as _http:
        ingest = IngestService(settings, app.state.qdrant_store, app.state.bm25_index, embedder)
        search = SearchService(settings, app.state.qdrant_store, app.state.bm25_index, embedder)
        corpus_names = spec.get("corpus")
        paths = (
            [DOCS_DIR / name for name in corpus_names]
            if corpus_names
            else sorted(DOCS_DIR.iterdir())
        )
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

        for item in spec["queries"]:
            gold = set(item["gold_sources"])
            for mode in ("dense", "bm25"):
                response = search.search(item["query"], mode=mode, top_k=10)  # type: ignore[arg-type]
                sources = [hit.metadata.source for hit in response.results]
                per_mode[mode].append(
                    {
                        "id": item["id"],
                        "query": item["query"],
                        "gold_sources": item["gold_sources"],
                        "retrieved_sources": sources,
                        "recall_at_5": round(_recall_at_k(sources, gold, 5), 4),
                        "recall_at_10": round(_recall_at_k(sources, gold, 10), 4),
                        "mrr_at_10": round(_mrr_at_k(sources, gold, 10), 4),
                        "latency_ms": response.latency_ms,
                    }
                )

    payload = {
        "version": spec.get("version", "v1"),
        "fake_embeddings": fake_embeddings,
        "ingested": ingested,
        "modes": {mode: {"queries": rows, "summary": _summarize(rows)} for mode, rows in per_mode.items()},
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_table(payload)
    return payload


def _print_table(payload: dict) -> None:
    print(f"Benchmark {payload['version']}  fake_embeddings={payload['fake_embeddings']}")
    print(f"{'mode':<8} {'R@5':>8} {'R@10':>8} {'MRR@10':>8} {'lat_mean':>10} {'p50':>8} {'p95':>8}")
    for mode, block in payload["modes"].items():
        summary = block["summary"]
        print(
            f"{mode:<8} {summary['recall_at_5']:8.3f} {summary['recall_at_10']:8.3f} "
            f"{summary['mrr_at_10']:8.3f} {summary['latency_ms_mean']:10.2f} "
            f"{summary['latency_ms_p50']:8.2f} {summary['latency_ms_p95']:8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval benchmark v1")
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use hash embeddings (no model download; recall is not meaningful)",
    )
    args = parser.parse_args()
    run(fake_embeddings=args.fake_embeddings)


if __name__ == "__main__":
    main()
