"""Document upload and ingestion routes."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile, status

from app.core.exceptions import EmptyDocumentError, PayloadTooLargeError
from app.dependencies import IngestServiceDep, SettingsDep
from app.schemas import IngestResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("")
def upload_document(
    settings: SettingsDep,
    ingest_service: IngestServiceDep,
    response: Response,
    file: Annotated[UploadFile, File()],
    chunk_size: Annotated[int | None, Form()] = None,
    overlap: Annotated[int | None, Form()] = None,
) -> IngestResponse:
    """Ingest a PDF, Markdown, or TXT file into Qdrant and BM25.

    Returns HTTP 201 for a new document and 200 when the same bytes are replayed.
    """
    data = _read_upload(file, settings.max_upload_bytes)
    filename = Path(file.filename or "upload").name
    result = ingest_service.ingest(
        filename=filename,
        declared_content_type=file.content_type,
        data=data,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    response.status_code = status.HTTP_200_OK if result.idempotent_replay else status.HTTP_201_CREATED
    return IngestResponse(
        document_id=result.document_id,
        filename=result.filename,
        content_type=result.content_type,
        chunk_count=result.chunk_count,
        chunk_ids=result.chunk_ids,
        idempotent_replay=result.idempotent_replay,
    )


def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"File exceeds max upload size of {max_bytes} bytes")
    if not data:
        raise EmptyDocumentError("Uploaded file is empty")
    return data
