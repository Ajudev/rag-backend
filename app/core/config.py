"""Application configuration loaded from environment variables and ``.env``."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Phase 2 ingestion and search API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="rag_chunks", alias="QDRANT_COLLECTION")
    qdrant_timeout: float = Field(default=10.0, alias="QDRANT_TIMEOUT")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=32, ge=1, alias="EMBEDDING_BATCH_SIZE")
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    chunk_size: int = Field(default=400, ge=1, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, ge=0, alias="CHUNK_OVERLAP")
    min_chunk_words: int = Field(default=20, ge=1, alias="MIN_CHUNK_WORDS")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1, alias="MAX_UPLOAD_BYTES")
    bm25_index_path: Path = Field(default=Path("data/bm25_index.json"), alias="BM25_INDEX_PATH")
    upsert_batch_size: int = Field(default=64, ge=1, alias="UPSERT_BATCH_SIZE")


def get_settings() -> Settings:
    """Load settings from the environment.

    Returns:
        Validated ``Settings`` instance.
    """
    return Settings()
