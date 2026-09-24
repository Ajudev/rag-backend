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
    """Runtime settings for ingestion, hybrid search, and grounded answers."""

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
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default=None, min_length=1, alias="OPENAI_MODEL")
    openai_timeout_seconds: float = Field(default=30.0, gt=0, alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_retries: int = Field(default=2, ge=0, alias="OPENAI_MAX_RETRIES")
    openai_input_usd_per_million: float | None = Field(default=None, ge=0, alias="OPENAI_INPUT_USD_PER_MILLION")
    openai_output_usd_per_million: float | None = Field(default=None, ge=0, alias="OPENAI_OUTPUT_USD_PER_MILLION")
    answer_top_k: int = Field(default=5, ge=1, le=100, alias="ANSWER_TOP_K")
    answer_context_max_chars: int = Field(default=12000, ge=1, alias="ANSWER_CONTEXT_MAX_CHARS")
    answer_log_prompts: bool = Field(default=False, alias="ANSWER_LOG_PROMPTS")

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

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def empty_openai_key_is_none(cls, value: object) -> object:
        """Treat blank API keys as unset so boot does not require OpenAI."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


def get_settings() -> Settings:
    """Load settings from the project-root ``.env`` and the process environment.

    Returns:
        Validated ``Settings`` instance.

    Raises:
        ValidationError: If a required setting is missing or blank.
    """
    return Settings()
