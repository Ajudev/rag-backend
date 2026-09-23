"""Pydantic request and response schemas."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ContentTypeName = Literal["pdf", "markdown", "txt"]
SearchMode = Literal["dense", "bm25"]


class IngestResponse(BaseModel):
    """Result of document ingestion."""

    document_id: str
    filename: str
    content_type: ContentTypeName
    chunk_count: int
    chunk_ids: list[str]
    idempotent_replay: bool


class SearchFilters(BaseModel):
    """Optional metadata filters applied to both search modes."""

    source: str | None = None
    document_id: str | None = None
    content_type: ContentTypeName | None = None


class SearchRequest(BaseModel):
    """Dense or BM25 search request."""

    query: str = Field(min_length=1)
    mode: SearchMode = "dense"
    top_k: int = 10
    filters: SearchFilters | None = None

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped

    @field_validator("top_k")
    @classmethod
    def clamp_top_k(cls, value: int) -> int:
        if value < 1:
            raise ValueError("top_k must be a positive integer")
        return min(value, 100)


class ChunkMetadata(BaseModel):
    """Source metadata returned with a ranked passage."""

    source: str
    content_type: ContentTypeName
    page_num: int
    chunk_index: int
    char_start: int
    char_end: int


class SearchHit(BaseModel):
    """A single ranked passage."""

    rank: int
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: ChunkMetadata


class SearchResponse(BaseModel):
    """Ranked search results for one query."""

    query: str
    mode: SearchMode
    top_k: int
    latency_ms: float
    results: list[SearchHit]


class HealthResponse(BaseModel):
    """Liveness plus Qdrant connectivity."""

    status: Literal["ok", "degraded"]
    qdrant: bool
