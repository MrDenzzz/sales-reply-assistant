"""Composition root: builds the object graph from settings."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlparse

from openai import AsyncOpenAI

from sales_reply_assistant.assistant import SalesAssistant
from sales_reply_assistant.config import Settings
from sales_reply_assistant.errors import ProviderUnavailableError
from sales_reply_assistant.knowledge import KnowledgeBase, load_knowledge_base
from sales_reply_assistant.providers import (
    OpenAIChatModel,
    OpenAIEmbedder,
    create_client,
    list_model_ids,
)
from sales_reply_assistant.retrieval import Retriever
from sales_reply_assistant.schemas import ProviderStatus

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


@dataclass(frozen=True, slots=True)
class AppServices:
    """Everything the HTTP layer needs; tests pass fakes instead."""

    assistant: SalesAssistant
    knowledge_base: KnowledgeBase
    chat_model: str
    embedding_model: str
    local_inference: bool
    check_status: Callable[[], Awaitable[ProviderStatus]]


@asynccontextmanager
async def build_services(settings: Settings) -> AsyncIterator[AppServices]:
    knowledge_base = load_knowledge_base(settings.knowledge_base_path)

    chat_client = create_client(
        settings.llm_base_url, settings.llm_api_key, settings.llm_timeout_seconds
    )
    embedding_client = chat_client
    if settings.embedding_base_url or settings.embedding_api_key:
        embedding_client = create_client(
            settings.embedding_base_url or settings.llm_base_url,
            settings.embedding_api_key or settings.llm_api_key,
            settings.llm_timeout_seconds,
        )

    retriever = Retriever(
        knowledge_base,
        OpenAIEmbedder(embedding_client, settings.embedding_model),
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        max_related=settings.retrieval_max_related,
    )
    try:
        await retriever.warm_up()
        logger.info("knowledge base indexed: %d entries", len(knowledge_base.entries))
    except ProviderUnavailableError as error:
        logger.warning("knowledge base is not indexed yet, will retry on first request: %s", error)

    chat_model = OpenAIChatModel(
        chat_client,
        settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=settings.llm_reasoning_effort,
    )

    async def check_status() -> ProviderStatus:
        return await _check_status(
            chat_client, embedding_client, settings.llm_model, settings.embedding_model
        )

    try:
        yield AppServices(
            assistant=SalesAssistant(
                knowledge_base,
                retriever,
                chat_model,
                max_attempts=settings.llm_max_attempts,
                history_limit=settings.history_limit,
            ),
            knowledge_base=knowledge_base,
            chat_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            local_inference=urlparse(settings.llm_base_url).hostname in _LOCAL_HOSTS,
            check_status=check_status,
        )
    finally:
        await chat_client.close()
        if embedding_client is not chat_client:
            await embedding_client.close()


async def _check_status(
    chat_client: AsyncOpenAI,
    embedding_client: AsyncOpenAI,
    chat_model: str,
    embedding_model: str,
) -> ProviderStatus:
    try:
        chat_models = await list_model_ids(chat_client)
        embedding_models = (
            chat_models
            if embedding_client is chat_client
            else await list_model_ids(embedding_client)
        )
    except ProviderUnavailableError as error:
        return ProviderStatus(reachable=False, detail=str(error))
    return ProviderStatus(
        reachable=True,
        chat_model_available=chat_model in chat_models,
        embedding_model_available=embedding_model in embedding_models,
    )
