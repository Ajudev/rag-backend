"""Dense, BM25, hybrid RRF, and hybrid-plus-rerank search routes."""

from fastapi import APIRouter

from app.dependencies import SearchServiceDep
from app.schemas import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
def search_passages(body: SearchRequest, search_service: SearchServiceDep) -> SearchResponse:
    """Retrieve ranked passages using dense, BM25, hybrid RRF, or rerank."""
    return search_service.search(
        query=body.query,
        mode=body.mode,
        top_k=body.top_k,
        filters=body.filters,
        dense_candidate_count=body.dense_candidate_count,
        bm25_candidate_count=body.bm25_candidate_count,
        rerank_candidate_count=body.rerank_candidate_count,
        rrf_k=body.rrf_k,
        dense_weight=body.dense_weight,
        bm25_weight=body.bm25_weight,
        debug=body.debug,
    )
