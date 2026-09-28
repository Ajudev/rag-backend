"""Grounded answer generation over retrieved passages.

Pipeline: question → SearchService → context pack → structured LLM generate →
deterministic citation reference validation → optional CitationVerificationService
semantic verification (fallible LLM judgements).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from app.core.config import Settings
from app.core.exceptions import AppError, GenerationError, GenerationUnavailableError
from app.schemas import (
    AnswerMetadata,
    AnswerRequest,
    AnswerResponse,
    CitationValidationMeta,
    GroundedLlmOutput,
    TokenUsage,
    VerificationReport,
)
from app.services.citation_validate import validate_grounded_output
from app.services.citation_verification import CitationVerificationService
from app.services.context_pack import build_messages, build_repair_messages, pack_passages
from app.services.generation import (
    CITATION_INVALID_ANSWER,
    INSUFFICIENT_EVIDENCE_ANSWER,
    PROMPT_VERSION,
    GenerationClient,
    GenerationResult,
)
from app.services.search import SearchService

logger = logging.getLogger(__name__)


def _answer_from_claims(output: GroundedLlmOutput) -> str:
    """Displayed answer is derived from claim texts so later checks can cover it."""
    if output.status == "insufficient_evidence" and not output.claims:
        return INSUFFICIENT_EVIDENCE_ANSWER
    texts = [claim.text.strip() for claim in output.claims if claim.text.strip()]
    if not texts:
        return INSUFFICIENT_EVIDENCE_ANSWER
    return " ".join(texts)


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    input_usd_per_million: float | None,
    output_usd_per_million: float | None,
) -> float | None:
    """Estimate USD cost when both rates are configured; otherwise None."""
    if input_usd_per_million is None or output_usd_per_million is None:
        return None
    return (prompt_tokens / 1_000_000.0) * input_usd_per_million + (
        completion_tokens / 1_000_000.0
    ) * output_usd_per_million


class AnswerService:
    """Retrieve, pack, generate, and reference-validate a grounded answer."""

    def __init__(
        self,
        settings: Settings,
        search_service: SearchService,
        generation_client: GenerationClient | None,
        verify_generation_client: GenerationClient | None = None,
    ) -> None:
        self.settings = settings
        self.search_service = search_service
        self.generation_client = generation_client
        self.verify_generation_client = verify_generation_client

    def _verification_client(self) -> GenerationClient | None:
        return self.verify_generation_client or self.generation_client

    async def answer(self, request: AnswerRequest) -> AnswerResponse:
        """Run the grounded-answer pipeline for one question."""
        if self.generation_client is None:
            raise GenerationUnavailableError(
                "Grounded answer generation is unavailable: no LLM client configured."
            )

        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        generation_ms = 0.0
        validation_ms = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        llm_model = self.generation_client.model
        provider = self.generation_client.provider
        do_verify = self.settings.citation_verify_enabled if request.verify is None else request.verify

        retrieval_started = time.perf_counter()
        search = await asyncio.to_thread(
            self.search_service.search,
            query=request.question,
            mode=request.mode,
            top_k=request.top_k,
            filters=request.filters,
            dense_candidate_count=request.dense_candidate_count,
            bm25_candidate_count=request.bm25_candidate_count,
            rerank_candidate_count=request.rerank_candidate_count,
            rrf_k=request.rrf_k,
            dense_weight=request.dense_weight,
            bm25_weight=request.bm25_weight,
            debug=False,
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        retrieved_ids = [hit.chunk_id for hit in search.results]
        pack = pack_passages(search.results, self.settings.answer_context_max_chars)
        context_ids = [item.chunk_id for item in pack.passages]

        if not pack.passages:
            total_ms = (time.perf_counter() - started) * 1000.0
            self._log(
                request_id=request_id,
                mode=request.mode,
                retrieved_ids=retrieved_ids,
                context_ids=context_ids,
                dropped_ids=pack.dropped_chunk_ids,
                model=llm_model,
                status="insufficient_evidence",
                validation_ok=True,
                repair=False,
                prompt_tokens=0,
                total_tokens=0,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                validation_ms=0.0,
                total_ms=total_ms,
            )
            return self._response(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                status="insufficient_evidence",
                claims=[],
                sources=[],
                request_id=request_id,
                mode=request.mode,
                retrieved_ids=retrieved_ids,
                context_ids=context_ids,
                dropped_ids=pack.dropped_chunk_ids,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                validation_ms=0.0,
                total_ms=total_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                provider=provider,
                model=llm_model,
                validation_ok=True,
                validation_errors=[],
                repair_attempted=False,
                generation_status="insufficient_evidence",
                verification=self._empty_verify_report() if do_verify else None,
            )

        messages = build_messages(request.question, pack)
        gen_started = time.perf_counter()
        first = await self._generate(messages)
        generation_ms += (time.perf_counter() - gen_started) * 1000.0
        prompt_tokens += first.prompt_tokens
        completion_tokens += first.completion_tokens
        total_tokens += first.total_tokens
        llm_model = first.model

        first_output = self._as_grounded(first)
        val_started = time.perf_counter()
        validation = validate_grounded_output(first_output, pack.passages)
        validation_ms += (time.perf_counter() - val_started) * 1000.0

        repair_attempted = False
        chosen = first_output
        if not validation.ok:
            repair_attempted = True
            repair_messages = build_repair_messages(request.question, pack, validation.errors)
            gen_started = time.perf_counter()
            repaired = await self._generate(repair_messages)
            generation_ms += (time.perf_counter() - gen_started) * 1000.0
            prompt_tokens += repaired.prompt_tokens
            completion_tokens += repaired.completion_tokens
            total_tokens += repaired.total_tokens
            llm_model = repaired.model
            val_started = time.perf_counter()
            validation = validate_grounded_output(self._as_grounded(repaired), pack.passages)
            validation_ms += (time.perf_counter() - val_started) * 1000.0
            chosen = self._as_grounded(repaired)

        total_ms = (time.perf_counter() - started) * 1000.0
        if not validation.ok:
            status = "verification_failed" if do_verify else "citation_invalid"
            self._log(
                request_id=request_id,
                mode=request.mode,
                retrieved_ids=retrieved_ids,
                context_ids=context_ids,
                dropped_ids=pack.dropped_chunk_ids,
                model=llm_model,
                status=status,
                validation_ok=False,
                repair=repair_attempted,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                validation_ms=validation_ms,
                total_ms=total_ms,
            )
            if do_verify:
                verify_client = self._verification_client()
                assert verify_client is not None
                verifier = CitationVerificationService(
                    settings=self.settings,
                    generation_client=verify_client,
                    search_service=self.search_service,
                )
                return await verifier.verify_generated_answer(
                    question=request.question,
                    grounded=chosen,
                    displayed_answer=CITATION_INVALID_ANSWER,
                    pack=pack,
                    request_id=request_id,
                    search_mode=request.mode,
                    generation_status=chosen.status,
                    citation_invalid=True,
                    citation_errors=validation.errors,
                    repair_attempted=repair_attempted,
                    retrieved_ids=retrieved_ids,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    validation_ms=validation_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    provider=provider,
                    model=llm_model,
                    started=started,
                    filters=request.filters,
                    top_k=request.top_k,
                    search_kwargs={
                        "dense_candidate_count": request.dense_candidate_count,
                        "bm25_candidate_count": request.bm25_candidate_count,
                        "rerank_candidate_count": request.rerank_candidate_count,
                        "rrf_k": request.rrf_k,
                        "dense_weight": request.dense_weight,
                        "bm25_weight": request.bm25_weight,
                    },
                )
            return self._response(
                answer=CITATION_INVALID_ANSWER,
                status="citation_invalid",
                claims=[],
                sources=[],
                request_id=request_id,
                mode=request.mode,
                retrieved_ids=retrieved_ids,
                context_ids=context_ids,
                dropped_ids=pack.dropped_chunk_ids,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                validation_ms=validation_ms,
                total_ms=total_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                provider=provider,
                model=llm_model,
                validation_ok=False,
                validation_errors=validation.errors,
                repair_attempted=repair_attempted,
                generation_status=chosen.status,
                verification=None,
            )

        public_status = chosen.status
        answer_text = _answer_from_claims(GroundedLlmOutput(claims=validation.claims, status=public_status))
        if do_verify:
            verify_client = self._verification_client()
            assert verify_client is not None
            verifier = CitationVerificationService(
                settings=self.settings,
                generation_client=verify_client,
                search_service=self.search_service,
            )
            return await verifier.verify_generated_answer(
                question=request.question,
                grounded=GroundedLlmOutput(claims=validation.claims, status=public_status),
                displayed_answer=answer_text,
                pack=pack,
                request_id=request_id,
                search_mode=request.mode,
                generation_status=public_status,
                citation_invalid=False,
                citation_errors=[],
                repair_attempted=repair_attempted,
                retrieved_ids=retrieved_ids,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                validation_ms=validation_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                provider=provider,
                model=llm_model,
                started=started,
                filters=request.filters,
                top_k=request.top_k,
                search_kwargs={
                    "dense_candidate_count": request.dense_candidate_count,
                    "bm25_candidate_count": request.bm25_candidate_count,
                    "rerank_candidate_count": request.rerank_candidate_count,
                    "rrf_k": request.rrf_k,
                    "dense_weight": request.dense_weight,
                    "bm25_weight": request.bm25_weight,
                },
            )
        self._log(
            request_id=request_id,
            mode=request.mode,
            retrieved_ids=retrieved_ids,
            context_ids=context_ids,
            dropped_ids=pack.dropped_chunk_ids,
            model=llm_model,
            status=public_status,
            validation_ok=True,
            repair=repair_attempted,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            validation_ms=validation_ms,
            total_ms=total_ms,
        )
        return self._response(
            answer=answer_text,
            status=public_status,
            claims=validation.claims,
            sources=validation.sources,
            request_id=request_id,
            mode=request.mode,
            retrieved_ids=retrieved_ids,
            context_ids=context_ids,
            dropped_ids=pack.dropped_chunk_ids,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            validation_ms=validation_ms,
            total_ms=total_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            provider=provider,
            model=llm_model,
            validation_ok=True,
            validation_errors=[],
            repair_attempted=repair_attempted,
            generation_status=public_status,
            verification=None,
        )

    async def _generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        assert self.generation_client is not None
        if self.settings.answer_log_prompts:
            logger.info("answer_prompt_messages count=%s", len(messages))
        try:
            return await self.generation_client.generate(messages, GroundedLlmOutput)
        except AppError:
            raise
        except Exception as exc:
            raise GenerationError("LLM generation returned unusable output.") from exc

    def _as_grounded(self, result: GenerationResult) -> GroundedLlmOutput:
        output = result.output
        if isinstance(output, GroundedLlmOutput):
            return output
        raise GenerationError("LLM generation returned unusable output.")

    def _empty_verify_report(self) -> VerificationReport:
        model = self.generation_client.model if self.generation_client is not None else ""
        return VerificationReport(
            prompt_version=self.settings.verify_prompt_version,
            overall_status="insufficient_evidence",
            verifier_model=model,
        )

    def _response(
        self,
        *,
        answer: str,
        status: str,
        claims: list,
        sources: list,
        request_id: str,
        mode: str,
        retrieved_ids: list[str],
        context_ids: list[str],
        dropped_ids: list[str],
        retrieval_ms: float,
        generation_ms: float,
        validation_ms: float,
        total_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        provider: str,
        model: str,
        validation_ok: bool,
        validation_errors: list[str],
        repair_attempted: bool,
        generation_status: str | None = None,
        verification: VerificationReport | None = None,
    ) -> AnswerResponse:
        cost = estimate_cost_usd(
            prompt_tokens,
            completion_tokens,
            self.settings.openai_input_usd_per_million,
            self.settings.openai_output_usd_per_million,
        )
        return AnswerResponse(
            answer=answer,
            status=status,  # type: ignore[arg-type]
            claims=claims,
            sources=sources,
            metadata=AnswerMetadata(
                request_id=request_id,
                search_mode=mode,  # type: ignore[arg-type]
                prompt_version=PROMPT_VERSION,
                llm_provider=provider,
                llm_model=model,
                retrieved_chunk_ids=retrieved_ids,
                context_chunk_ids=context_ids,
                retrieval_latency_ms=round(retrieval_ms, 3),
                generation_latency_ms=round(generation_ms, 3),
                validation_latency_ms=round(validation_ms, 3),
                total_latency_ms=round(total_ms, 3),
                token_usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                estimated_cost_usd=cost,
                citation_validation=CitationValidationMeta(
                    ok=validation_ok,
                    errors=validation_errors,
                    repair_attempted=repair_attempted,
                ),
                dropped_chunk_ids=dropped_ids,
                generation_status=generation_status,  # type: ignore[arg-type]
            ),
            verification=verification,
        )

    def _log(
        self,
        *,
        request_id: str,
        mode: str,
        retrieved_ids: list[str],
        context_ids: list[str],
        dropped_ids: list[str],
        model: str,
        status: str,
        validation_ok: bool,
        repair: bool,
        prompt_tokens: int,
        total_tokens: int,
        retrieval_ms: float,
        generation_ms: float,
        validation_ms: float,
        total_ms: float,
    ) -> None:
        logger.info(
            "grounded_answer request_id=%s mode=%s retrieved=%s context=%s dropped=%s "
            "model=%s prompt_tokens=%s total_tokens=%s retrieval_ms=%.1f generation_ms=%.1f "
            "validation_ms=%.1f total_ms=%.1f validation_ok=%s repair=%s status=%s",
            request_id,
            mode,
            retrieved_ids,
            context_ids,
            dropped_ids,
            model,
            prompt_tokens,
            total_tokens,
            retrieval_ms,
            generation_ms,
            validation_ms,
            total_ms,
            validation_ok,
            repair,
            status,
        )
