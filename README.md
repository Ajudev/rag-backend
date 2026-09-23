# Hybrid Search RAG

Document ingestion and **baseline search** for a hybrid-search RAG stack. This phase indexes PDF, Markdown, and TXT files into **Qdrant** (dense, BGE) and a **separate BM25** corpus. It does **not** implement hybrid fusion, reranking, LLM answers, or citation verification.

## Architecture

```
Upload → parse (pypdf / UTF-8) → word chunk (helpers.chunk_text)
      → BAAI/bge-small-en-v1.5 embeddings (batched)
      → Qdrant cosine collection (384-d) + JSON-persisted BM25
Search → dense (Qdrant kNN + payload filter) or bm25 (rank_bm25 + same filters)
```

| Path | Role |
|------|------|
| `app/main.py` | FastAPI factory and lifespan (Qdrant, embedder, BM25 load) |
| `app/api/routes/` | `POST /documents`, `POST /search`, `GET /health` |
| `app/services/` | parse, chunk, embed, ingest, Qdrant, BM25, search |
| `app/helpers.py` | `tokenize`, PDF extract, `chunk_text` / `chunk_text_para` |
| `docs/` | Small sample corpus for the v1 benchmark |
| `main.py` | Re-exports `app` for `uvicorn main:app` |

Chunk IDs are deterministic: `document_id` is SHA-256 of the raw bytes; `chunk_id` is `sha256(document_id:chunk_index:normalized_text)[:32]`. Qdrant point IDs are UUID5 values derived from `chunk_id`.

**Idempotency:** re-uploading the **same bytes** returns HTTP **200** with `idempotent_replay: true` and the same IDs (no duplicate points). If those bytes are uploaded under a **new filename**, stored `source` in Qdrant and BM25 is rewritten to the latest name so `filters.source` matches the client upload (still no duplicate points). Re-uploading the **same filename with different bytes** deletes prior points/BM25 docs for that source, then inserts. A first-time ingest returns HTTP **201**.

## Setup

Python 3.13.5, package manager **uv**.

```bash
cp .env.example .env
uv sync
docker compose up -d qdrant
uv run uvicorn main:app --reload
```

The API listens on `http://127.0.0.1:8000`. Docs: `http://127.0.0.1:8000/docs`.

To run the API in Compose as well: `docker compose up -d` (Qdrant + `api` on port 8000). The API service uses `QDRANT_URL=http://qdrant:6333`.

## Environment variables

See `.env.example`. Defaults are suitable for local development.

| Variable | Default | Meaning |
|----------|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | Use `:memory:` only in tests |
| `QDRANT_COLLECTION` | `rag_chunks` | Cosine, 384 dimensions |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-Transformers model |
| `EMBEDDING_BATCH_SIZE` | `32` | Encode batch size |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `400` / `50` | Word windows |
| `MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Upload cap |
| `BM25_INDEX_PATH` | `data/bm25_index.json` | Persisted BM25 corpus; rebuilt from Qdrant on startup if empty |

## API examples

Health:

```bash
curl -s http://127.0.0.1:8000/health
```

Ingest (PDF, Markdown, or TXT). Optional form fields: `chunk_size`, `overlap`.

```bash
curl -s -X POST http://127.0.0.1:8000/documents \
  -F "file=@docs/aurora_protocol.md;type=text/markdown"
```

Dense search:

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the Zephyr handshake?","mode":"dense","top_k":10}'
```

BM25 search with a source filter:

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"compound XJ-19","mode":"bm25","top_k":5,"filters":{"source":"nimbus_lab.txt"}}'
```

`mode` defaults to `dense`. Invalid `mode` yields 422. `top_k` below 1 yields 422; values above 100 are clamped to 100. Both modes return ranked passages (`rank`, `chunk_id`, `document_id`, `score`, `text`, `metadata`) and `latency_ms`.

Unsupported types return **415**; empty/unreadable files and invalid chunk params return **400**; oversized uploads return **413**.

## Tests

Tests use in-memory Qdrant and a hash `FakeEmbedder` (no Docker, no model download):

```bash
uv run pytest
uv run pytest -m unit
uv run pytest -m integration
```

## Benchmark (v1)

Corpus: `docs/` (short MD/TXT files with distinctive facts). Gold labels are **source filenames**.

```bash
uv run python -m benchmark.v1.run
```

First real-model run downloads `BAAI/bge-small-en-v1.5`. For a dry run without the model:

```bash
uv run python -m benchmark.v1.run --fake-embeddings
```

The runner ingests the small files listed in `benchmark/v1/queries.json` (`corpus`), not the large research PDFs that may also live in `docs/`.
