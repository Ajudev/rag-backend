"""Grounded answer generation over retrieved passages."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import (
    SearchServiceDep,
    SettingsDep,
    get_generation_client,
    get_verify_generation_client,
)
from app.schemas import AnswerRequest, AnswerResponse
from app.services.answer import AnswerService
from app.services.generation import GenerationClient

router = APIRouter(prefix="/answer", tags=["answer"])

GenerationClientDep = Annotated[GenerationClient | None, Depends(get_generation_client)]
VerifyGenerationClientDep = Annotated[GenerationClient | None, Depends(get_verify_generation_client)]


@router.post("")
async def answer_question(
    body: AnswerRequest,
    search_service: SearchServiceDep,
    settings: SettingsDep,
    generation_client: GenerationClientDep,
    verify_generation_client: VerifyGenerationClientDep,
) -> AnswerResponse:
    """Retrieve passages and generate a grounded answer with citations.

    Set ``verify`` to run ``CitationVerificationService.semantic_verify``
    (fallible LLM judgements). Default comes from ``CITATION_VERIFY_ENABLED``.
    """
    service = AnswerService(
        settings=settings,
        search_service=search_service,
        generation_client=generation_client,
        verify_generation_client=verify_generation_client,
    )
    return await service.answer(body)
