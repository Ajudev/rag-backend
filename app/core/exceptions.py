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
