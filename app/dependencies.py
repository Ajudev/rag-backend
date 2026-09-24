"""FastAPI dependency aliases for request-scoped access to app state."""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.services.bm25_index import BM25Index
from app.services.embedder import Embedder
from app.services.generation import GenerationClient
from app.services.ingest import IngestService
from app.services.qdrant_store import QdrantStore
from app.services.reranker import Reranker
from app.services.search import SearchService


def get_settings_dep(request: Request) -> Settings:
    """Return settings stored on the application."""
    return request.app.state.settings


def get_qdrant_store(request: Request) -> QdrantStore:
    """Return the Qdrant store from lifespan state."""
    return request.app.state.qdrant_store


def get_bm25_index(request: Request) -> BM25Index:
    """Return the BM25 index from lifespan state."""
    return request.app.state.bm25_index


def get_embedder(request: Request) -> Embedder:
    """Return the embedding backend from lifespan state."""
    return request.app.state.embedder


def get_reranker(request: Request) -> Reranker | None:
    """Return the reranker from lifespan state, if loaded."""
    return getattr(request.app.state, "reranker", None)


def get_generation_client(request: Request) -> GenerationClient | None:
    """Return the generation client from lifespan state, if configured."""
    return getattr(request.app.state, "generation_client", None)


def get_ingest_service(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    qdrant: Annotated[QdrantStore, Depends(get_qdrant_store)],
    bm25: Annotated[BM25Index, Depends(get_bm25_index)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> IngestService:
    """Build an ingest service for the current request."""
    return IngestService(settings=settings, qdrant=qdrant, bm25=bm25, embedder=embedder)


def get_search_service(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    qdrant: Annotated[QdrantStore, Depends(get_qdrant_store)],
    bm25: Annotated[BM25Index, Depends(get_bm25_index)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    reranker: Annotated[Reranker | None, Depends(get_reranker)],
) -> SearchService:
    """Build a search service for the current request."""
    return SearchService(
        settings=settings,
        qdrant=qdrant,
        bm25=bm25,
        embedder=embedder,
        reranker=reranker,
    )


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
QdrantStoreDep = Annotated[QdrantStore, Depends(get_qdrant_store)]
