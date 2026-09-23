from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.search import router as search_router

__all__ = ["documents_router", "health_router", "search_router"]
