"""FastAPI application factory and lifespan."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from app.api.routes import documents_router, health_router, search_router
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.services.bm25_index import BM25Index
from app.services.embedder import Embedder, SentenceTransformerEmbedder
from app.services.qdrant_store import QdrantStore, build_qdrant_client
from app.services.reranker import CrossEncoderReranker, Reranker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    if getattr(app.state, "qdrant_client", None) is None:
        app.state.qdrant_client = build_qdrant_client(settings)
        app.state.owns_qdrant_client = True
    else:
        app.state.owns_qdrant_client = False

    if getattr(app.state, "embedder", None) is None:
        logger.info("Loading embedding model %s", settings.embedding_model)
        app.state.embedder = SentenceTransformerEmbedder(
            settings.embedding_model,
            dim=settings.embedding_dim,
        )

    if getattr(app.state, "reranker", None) is None:
        logger.info("Loading reranker %s device=%s", settings.reranker_model, settings.reranker_device)
        app.state.reranker = CrossEncoderReranker(
            settings.reranker_model,
            device=settings.reranker_device,
        )

    store = QdrantStore(app.state.qdrant_client, settings)
    store.ensure_collection()
    app.state.qdrant_store = store

    bm25 = BM25Index(settings.bm25_index_path)
    bm25.load()
    if bm25.is_empty():
        payloads = store.scroll_all()
        if payloads:
            bm25.load_from_payloads(payloads)
    app.state.bm25_index = bm25
    yield
    if app.state.owns_qdrant_client:
        app.state.qdrant_client.close()


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    qdrant_client: QdrantClient | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Optional settings override (tests).
        embedder: Optional embedder override to avoid model downloads.
        qdrant_client: Optional client (for example ``QdrantClient(":memory:")``).
        reranker: Optional reranker override to avoid MiniLM downloads.

    Returns:
        Configured FastAPI app exposing ingest, search, and health routes.
    """
    resolved = settings or get_settings()
    app = FastAPI(
        title="Hybrid Search RAG",
        description=("Document ingestion plus dense, BM25, hybrid RRF, and cross-encoder rerank search."),
        version="0.3.0",
        lifespan=_lifespan,
    )
    app.state.settings = resolved
    if embedder is not None:
        app.state.embedder = embedder
    if qdrant_client is not None:
        app.state.qdrant_client = qdrant_client
    if reranker is not None:
        app.state.reranker = reranker

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


app = create_app()
