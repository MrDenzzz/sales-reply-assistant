"""Adapters for OpenAI-compatible servers: LM Studio, Ollama, OpenAI, OpenRouter, ..."""

from collections.abc import Sequence
from typing import Any

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from sales_reply_assistant.errors import ProviderUnavailableError

_STATUS_TIMEOUT_SECONDS = 5


def create_client(base_url: str, api_key: str, timeout: float) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)


class OpenAIChatModel:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort

    @property
    def name(self) -> str:
        return self._model

    async def complete_json(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        # Passed via extra_body: not every server knows the parameter, and it is optional.
        extra_body = (
            {"reasoning_effort": self._reasoning_effort} if self._reasoning_effort else None
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                extra_body=extra_body,
            )
        except openai.APIError as error:
            raise ProviderUnavailableError(f"chat model '{self._model}': {error}") from error

        if not response.choices:
            return ""
        return response.choices[0].message.content or ""


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=list(texts), encoding_format="float"
            )
        except openai.APIError as error:
            raise ProviderUnavailableError(f"embedding model '{self._model}': {error}") from error
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


async def list_model_ids(client: AsyncOpenAI) -> set[str]:
    """Models the server can serve. Used by the status indicator in the UI."""
    try:
        page = await client.with_options(timeout=_STATUS_TIMEOUT_SECONDS).models.list()
    except openai.APIError as error:
        raise ProviderUnavailableError(str(error)) from error
    return {model.id for model in page.data}
