"""Application configuration loaded from environment variables and ``.env``."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"

_REQUIRED_STRING_FIELDS = (
    "qdrant_url",
    "qdrant_collection",
    "embedding_model",
    "reranker_model",
)


class Settings(BaseSettings):
    """Runtime settings for ingestion, dense/BM25 retrieval, RRF, and rerank."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    qdrant_url: str = Field(min_length=1, alias="QDRANT_URL")
    qdrant_collection: str = Field(min_length=1, alias="QDRANT_COLLECTION")
    qdrant_timeout: float = Field(default=10.0, alias="QDRANT_TIMEOUT")
    embedding_model: str = Field(min_length=1, alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=32, ge=1, alias="EMBEDDING_BATCH_SIZE")
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    chunk_size: int = Field(default=400, ge=1, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, ge=0, alias="CHUNK_OVERLAP")
    min_chunk_words: int = Field(default=20, ge=1, alias="MIN_CHUNK_WORDS")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1, alias="MAX_UPLOAD_BYTES")
    bm25_index_path: Path = Field(default=Path("data/bm25_index.json"), alias="BM25_INDEX_PATH")
    upsert_batch_size: int = Field(default=64, ge=1, alias="UPSERT_BATCH_SIZE")
    dense_candidate_count: int = Field(default=20, ge=1, le=100, alias="DENSE_CANDIDATE_COUNT")
    bm25_candidate_count: int = Field(default=20, ge=1, le=100, alias="BM25_CANDIDATE_COUNT")
    rerank_candidate_count: int = Field(default=20, ge=1, le=100, alias="RERANK_CANDIDATE_COUNT")
    rrf_k: int = Field(default=60, ge=1, alias="RRF_K")
    dense_weight: float = Field(default=1.0, ge=0, alias="DENSE_WEIGHT")
    bm25_weight: float = Field(default=1.0, ge=0, alias="BM25_WEIGHT")
    reranker_model: str = Field(min_length=1, alias="RERANKER_MODEL")
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")
    reranker_batch_size: int = Field(default=16, ge=1, alias="RERANKER_BATCH_SIZE")

    @field_validator(*_REQUIRED_STRING_FIELDS, mode="before")
    @classmethod
    def reject_blank_strings(cls, value: object) -> object:
        """Reject blank/whitespace values so the app fails at settings load."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("must be a non-empty string")
            return stripped
        return value

    @field_validator("bm25_index_path", mode="after")
    @classmethod
    def resolve_bm25_index_path(cls, value: Path) -> Path:
        """Resolve relative BM25 paths against the project root, not process CWD."""
        if value.is_absolute():
            return value
        return (PROJECT_ROOT / value).resolve()


def get_settings() -> Settings:
    """Load settings from the project-root ``.env`` and the process environment.

    Returns:
        Validated ``Settings`` instance.

    Raises:
        ValidationError: If a required setting is missing or blank.
    """
    return Settings()
