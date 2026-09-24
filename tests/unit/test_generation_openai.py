from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APITimeoutError, RateLimitError

from app.core.exceptions import GenerationError, GenerationRateLimitError, GenerationTimeoutError
from app.schemas import AnswerClaim, CitationRef, GroundedLlmOutput
from app.services.generation import OpenAIGenerationClient

pytestmark = pytest.mark.unit


def _valid_output() -> GroundedLlmOutput:
    return GroundedLlmOutput(
        status="answered",
        claims=[
            AnswerClaim(
                claim_id="claim_1",
                text="A cited fact.",
                citations=[CitationRef(chunk_id="c1", evidence_quote="cited")],
            )
        ],
    )


def _mock_completion(parsed: GroundedLlmOutput) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        model="gpt-4o-mini",
    )


def _client_with_parse(parse: AsyncMock) -> OpenAIGenerationClient:
    inner = MagicMock()
    inner.beta.chat.completions.parse = parse
    return OpenAIGenerationClient(
        api_key="sk-test",
        model="gpt-4o-mini",
        client=inner,
        owns_client=False,
    )


@pytest.mark.asyncio
async def test_openai_client_parses_structured_output_and_usage() -> None:
    parse = AsyncMock(return_value=_mock_completion(_valid_output()))
    client = _client_with_parse(parse)
    result = await client.generate(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        GroundedLlmOutput,
    )
    assert result.output.status == "answered"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.total_tokens == 18
    parse.assert_awaited_once()
    kwargs = parse.await_args.kwargs
    assert "temperature" not in kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["response_format"] is GroundedLlmOutput


@pytest.mark.asyncio
async def test_openai_client_timeout_maps_to_domain_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    parse = AsyncMock(side_effect=APITimeoutError(request=request))
    client = _client_with_parse(parse)
    with pytest.raises(GenerationTimeoutError):
        await client.generate([{"role": "user", "content": "q"}], GroundedLlmOutput)


@pytest.mark.asyncio
async def test_openai_client_rate_limit_maps_to_domain_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    parse = AsyncMock(side_effect=RateLimitError("slow down", response=response, body=None))
    client = _client_with_parse(parse)
    with pytest.raises(GenerationRateLimitError):
        await client.generate([{"role": "user", "content": "q"}], GroundedLlmOutput)


@pytest.mark.asyncio
async def test_openai_client_missing_parsed_raises_generation_error() -> None:
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))], usage=None, model="x")
    parse = AsyncMock(return_value=completion)
    client = _client_with_parse(parse)
    with pytest.raises(GenerationError):
        await client.generate([{"role": "user", "content": "q"}], GroundedLlmOutput)
