"""Baseline dense and BM25 search routes."""

from fastapi import APIRouter

from app.dependencies import SearchServiceDep
from app.schemas import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
def search_passages(body: SearchRequest, search_service: SearchServiceDep) -> SearchResponse:
    """Retrieve ranked passages using dense vectors or BM25."""
    return search_service.search(
        query=body.query,
        mode=body.mode,
        top_k=body.top_k,
        filters=body.filters,
    )
