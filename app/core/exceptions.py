"""Domain exceptions mapped to HTTP responses."""


class AppError(Exception):
    """Base application error with an HTTP status code."""

    status_code = 400

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class UnsupportedFileTypeError(AppError):
    """Uploaded file type is not supported."""

    status_code = 415


class PayloadTooLargeError(AppError):
    """Upload exceeds the configured size limit."""

    status_code = 413


class EmptyDocumentError(AppError):
    """File is empty or produced no indexable chunks."""

    status_code = 400


class InvalidChunkParamsError(AppError):
    """Chunk size / overlap combination is invalid."""

    status_code = 400


class DocumentParseError(AppError):
    """Document bytes could not be parsed."""

    status_code = 400


class QdrantUnavailableError(AppError):
    """Qdrant could not be reached."""

    status_code = 503


class InvalidSearchParamsError(AppError):
    """Search parameters failed domain validation."""

    status_code = 422


class RerankerUnavailableError(AppError):
    """Cross-encoder reranker is required but was not loaded."""

    status_code = 503


class GenerationUnavailableError(AppError):
    """Grounded generation client is required but was not configured."""

    status_code = 503


class GenerationError(AppError):
    """LLM generation failed or returned unusable structured output."""

    status_code = 502


class GenerationTimeoutError(AppError):
    """LLM generation timed out."""

    status_code = 504


class GenerationRateLimitError(AppError):
    """LLM provider rate-limited the request after retries."""

    status_code = 503
