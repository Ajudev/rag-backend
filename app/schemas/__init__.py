"""Pydantic request and response schemas."""

from typing import Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

ContentTypeName = Literal["pdf", "markdown", "txt"]
SearchMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


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
    """Search request for dense, BM25, hybrid RRF, or hybrid plus rerank."""

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(min_length=1)
    mode: SearchMode = Field(default="dense", validation_alias=AliasChoices("mode", "search_mode"))
    top_k: int = 10
    filters: SearchFilters | None = None
    dense_candidate_count: int | None = Field(default=None, ge=1, le=100)
    bm25_candidate_count: int | None = Field(default=None, ge=1, le=100)
    rerank_candidate_count: int | None = Field(default=None, ge=1, le=100)
    rrf_k: int | None = Field(default=None, ge=1)
    dense_weight: float | None = Field(default=None, ge=0)
    bm25_weight: float | None = Field(default=None, ge=0)
    debug: bool = False

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

    @model_validator(mode="after")
    def validate_fusion_and_rerank(self) -> Self:
        if self.dense_weight == 0 and self.bm25_weight == 0:
            raise ValueError("dense_weight and bm25_weight cannot both be 0")
        if (
            self.mode == "hybrid_rerank"
            and self.rerank_candidate_count is not None
            and self.rerank_candidate_count < self.top_k
        ):
            raise ValueError("rerank_candidate_count must be greater than or equal to top_k")
        return self


class ChunkMetadata(BaseModel):
    """Source metadata returned with a ranked passage."""

    source: str
    content_type: ContentTypeName
    page_num: int
    chunk_index: int
    char_start: int
    char_end: int


class SearchHit(BaseModel):
    """A single ranked passage, with per-retriever fields null when unused."""

    rank: int
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: ChunkMetadata
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    contributed_by: list[str] = Field(default_factory=list)


class SearchTiming(BaseModel):
    """Stage timings in milliseconds; null when the stage did not run."""

    dense_ms: float | None = None
    bm25_ms: float | None = None
    fusion_ms: float | None = None
    rerank_ms: float | None = None
    total_ms: float | None = None


class SearchDebug(BaseModel):
    """Resolved fusion and candidate parameters echoed when ``debug`` is true."""

    dense_candidate_count: int
    bm25_candidate_count: int
    rerank_candidate_count: int
    rrf_k: int
    dense_weight: float
    bm25_weight: float
    normalized_dense_weight: float | None = None
    normalized_bm25_weight: float | None = None


class SearchResponse(BaseModel):
    """Ranked search results for one query."""

    query: str
    mode: SearchMode
    top_k: int
    latency_ms: float
    results: list[SearchHit]
    timing: SearchTiming
    debug: SearchDebug | None = None


class HealthResponse(BaseModel):
    """Liveness plus Qdrant connectivity."""

    status: Literal["ok", "degraded"]
    qdrant: bool


LlmGroundedStatus = Literal["answered", "partially_answered", "insufficient_evidence", "conflicting_evidence"]
AnswerStatus = Literal[
    "answered",
    "partially_answered",
    "insufficient_evidence",
    "conflicting_evidence",
    "citation_invalid",
]


class AnswerRequest(BaseModel):
    """Grounded-answer request: retrieve passages then generate cited claims."""

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(min_length=1)
    mode: SearchMode = Field(
        default="hybrid_rerank",
        validation_alias=AliasChoices("mode", "search_mode"),
    )
    top_k: int = 5
    filters: SearchFilters | None = None
    dense_candidate_count: int | None = Field(default=None, ge=1, le=100)
    bm25_candidate_count: int | None = Field(default=None, ge=1, le=100)
    rerank_candidate_count: int | None = Field(default=None, ge=1, le=100)
    rrf_k: int | None = Field(default=None, ge=1)
    dense_weight: float | None = Field(default=None, ge=0)
    bm25_weight: float | None = Field(default=None, ge=0)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped

    @field_validator("top_k")
    @classmethod
    def clamp_top_k(cls, value: int) -> int:
        if value < 1:
            raise ValueError("top_k must be a positive integer")
        return min(value, 100)

    @model_validator(mode="after")
    def validate_fusion_and_rerank(self) -> Self:
        if self.dense_weight == 0 and self.bm25_weight == 0:
            raise ValueError("dense_weight and bm25_weight cannot both be 0")
        if (
            self.mode == "hybrid_rerank"
            and self.rerank_candidate_count is not None
            and self.rerank_candidate_count < self.top_k
        ):
            raise ValueError("rerank_candidate_count must be greater than or equal to top_k")
        return self


class CitationRef(BaseModel):
    """A citation pointing at a supplied context-pack chunk."""

    chunk_id: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)


class AnswerClaim(BaseModel):
    """One atomic claim shown in the grounded answer."""

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citations: list[CitationRef]


class GroundedLlmOutput(BaseModel):
    """Structured generation schema parsed from the LLM (internal)."""

    claims: list[AnswerClaim] = Field(default_factory=list)
    status: LlmGroundedStatus


class AnswerSource(BaseModel):
    """Cited passage metadata. Semantic support is not verified."""

    chunk_id: str
    document_id: str
    document_title: str
    page_number: int
    section: str | None = None
    char_start: int
    char_end: int
    passage_text: str


class TokenUsage(BaseModel):
    """Token counts reported by the generation API."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CitationValidationMeta(BaseModel):
    """Deterministic reference-validation outcome, not semantic entailment."""

    ok: bool
    errors: list[str] = Field(default_factory=list)
    repair_attempted: bool = False


class AnswerMetadata(BaseModel):
    """Observability for one grounded-answer request."""

    request_id: str
    search_mode: SearchMode
    prompt_version: str
    llm_provider: str
    llm_model: str
    retrieved_chunk_ids: list[str]
    context_chunk_ids: list[str]
    retrieval_latency_ms: float
    generation_latency_ms: float
    validation_latency_ms: float
    total_latency_ms: float
    token_usage: TokenUsage
    estimated_cost_usd: float | None = None
    citation_validation: CitationValidationMeta
    dropped_chunk_ids: list[str] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    """Public grounded-answer payload. System prompts are never included."""

    answer: str
    status: AnswerStatus
    claims: list[AnswerClaim] = Field(default_factory=list)
    sources: list[AnswerSource] = Field(default_factory=list)
    metadata: AnswerMetadata
