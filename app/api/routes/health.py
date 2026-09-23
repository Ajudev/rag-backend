"""Health check routes."""

from fastapi import APIRouter, Response, status

from app.dependencies import QdrantStoreDep
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health(qdrant_store: QdrantStoreDep, response: Response) -> HealthResponse:
    """Report process liveness and Qdrant connectivity."""
    qdrant_ok = qdrant_store.ping()
    if not qdrant_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", qdrant=False)
    return HealthResponse(status="ok", qdrant=True)
