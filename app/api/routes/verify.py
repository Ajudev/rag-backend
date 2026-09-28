"""Independent citation verification against indexed chunks."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.exceptions import GenerationUnavailableError
from app.dependencies import SettingsDep, get_bm25_index, get_verify_generation_client
from app.schemas import VerifyRequest, VerifyResponse
from app.services.bm25_index import BM25Index
from app.services.citation_verification import CitationVerificationService
from app.services.generation import GenerationClient

router = APIRouter(prefix="/verify", tags=["verify"])

VerifyGenerationClientDep = Annotated[GenerationClient | None, Depends(get_verify_generation_client)]
BM25IndexDep = Annotated[BM25Index, Depends(get_bm25_index)]


@router.post("")
async def verify_citations(
    body: VerifyRequest,
    settings: SettingsDep,
    generation_client: VerifyGenerationClientDep,
    bm25: BM25IndexDep,
) -> VerifyResponse:
    """Verify claims against indexed passages for allowed_chunk_ids.

    Client-supplied passage text is not used as evidence. LLM verdicts are fallible.
    """
    if generation_client is None:
        raise GenerationUnavailableError(
            "Citation verification is unavailable: no LLM client configured."
        )
    service = CitationVerificationService(
        settings=settings,
        generation_client=generation_client,
        bm25=bm25,
    )
    return await service.verify_independent(body)
