"""Orchestration: retrieve facts → ask the model → validate → post-process."""

import json
import logging
import re
import time
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from sales_reply_assistant.compliance import find_compliance_issues
from sales_reply_assistant.errors import InvalidModelOutputError
from sales_reply_assistant.knowledge import EntryType, KnowledgeBase, KnowledgeEntry
from sales_reply_assistant.prompts import (
    LlmDraft,
    build_draft_schema,
    build_search_query,
    build_system_prompt,
    build_user_prompt,
)
from sales_reply_assistant.retrieval import Retriever
from sales_reply_assistant.schemas import (
    ProductRef,
    SourceRef,
    Suggestion,
    SuggestRequest,
    UpsellSuggestion,
)

logger = logging.getLogger(__name__)

_SCHEMA_NAME = "sales_reply_draft"
_WORD = re.compile(r"[^\W\d_]+")
_ENGLISH_WORD = re.compile(r"\b[a-z]{3,}\b")


class ChatModel(Protocol):
    """A chat model that answers with a JSON document matching the given schema."""

    @property
    def name(self) -> str: ...

    async def complete_json(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> str: ...


class SalesAssistant:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        retriever: Retriever,
        chat_model: ChatModel,
        *,
        max_attempts: int = 2,
        history_limit: int = 12,
    ) -> None:
        self._retriever = retriever
        self._chat_model = chat_model
        self._max_attempts = max_attempts
        self._history_limit = history_limit
        self._system_prompt = build_system_prompt(knowledge_base.company)

    async def suggest(self, request: SuggestRequest) -> Suggestion:
        started = time.perf_counter()
        dialog = request.messages[-self._history_limit :]

        retrieved = await self._retriever.retrieve(build_search_query(dialog))
        context = [item.entry for item in retrieved]
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": build_user_prompt(context, dialog, request.client_name)},
        ]
        known_words = "\n".join(entry.to_prompt() for entry in context).lower()
        draft = await self._generate(messages, build_draft_schema(context), known_words)

        suggestion = self._to_suggestion(draft, context, _elapsed_ms(started))
        logger.info(
            "suggestion ready in %d ms: context=%s sources=%s attention=%s upsell=%s products=%s",
            suggestion.latency_ms,
            [f"{item.entry.id}:{item.reason}" for item in retrieved],
            [source.id for source in suggestion.sources],
            suggestion.needs_attention,
            suggestion.upsell.should_offer,
            [product.id for product in suggestion.upsell.products],
        )
        return suggestion

    async def _generate(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any],
        known_words: str,
    ) -> LlmDraft:
        """Ask the model for a draft; on a problem, show it the problem and ask again.

        A broken contract fails the request once the attempts run out. A style problem does not:
        the last valid draft is returned, the manager edits it anyway.
        """
        conversation = messages
        draft: LlmDraft | None = None
        for attempt in range(1, self._max_attempts + 1):
            raw = await self._chat_model.complete_json(
                conversation, schema=schema, schema_name=_SCHEMA_NAME
            )
            try:
                draft = LlmDraft.model_validate_json(_strip_code_fence(raw))
            except ValidationError as error:
                logger.warning("attempt %d: model output violates the contract: %s", attempt, error)
                problems = json.dumps(
                    error.errors(include_url=False), ensure_ascii=False, default=str
                )
                feedback = (
                    f"Ответ не прошёл проверку: {problems}. "
                    "Верни исправленный JSON строго по схеме, без пояснений."
                )
            else:
                english = _english_words(draft.reply, known_words)
                if not english:
                    return draft
                logger.warning("attempt %d: reply switches to English: %s", attempt, english)
                feedback = (
                    f"В reply есть английские слова: {', '.join(english)}. "
                    "Перепиши ответ только по-русски и верни JSON по той же схеме."
                )
            conversation = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": feedback},
            ]

        if draft is not None:
            return draft
        raise InvalidModelOutputError(
            f"The model returned invalid output {self._max_attempts} time(s) in a row"
        )

    def _to_suggestion(
        self, draft: LlmDraft, context: Sequence[KnowledgeEntry], latency_ms: int
    ) -> Suggestion:
        # Ids are validated again here: not every provider enforces the JSON schema.
        by_id = {entry.id: entry for entry in context}
        sources = [SourceRef.from_entry(by_id[i]) for i in _unique(draft.source_ids) if i in by_id]
        products = (
            [
                ProductRef.from_entry(by_id[i])
                for i in _unique(draft.upsell_product_ids)
                if i in by_id and by_id[i].type is EntryType.PRODUCT
            ]
            if draft.upsell_should_offer
            else []
        )
        pitch = draft.upsell_pitch if draft.upsell_should_offer else ""
        if pitch and _repeats_reply(pitch, draft.reply, [by_id[p.id] for p in products]):
            # The manager appends the pitch to the reply; a repeated offer would read as spam.
            logger.info("pitch dropped: it repeats a product already offered in the reply")
            pitch = ""
        return Suggestion(
            client_intent=draft.client_intent,
            reply=draft.reply,
            needs_attention=draft.needs_attention,
            attention_reason=(draft.attention_reason or None) if draft.needs_attention else None,
            upsell=UpsellSuggestion(
                should_offer=draft.upsell_should_offer,
                hint=draft.upsell_hint,
                pitch=pitch,
                products=products,
            ),
            compliance_warnings=find_compliance_issues(draft.reply),
            sources=sources,
            model=self._chat_model.name,
            latency_ms=latency_ms,
        )


def _strip_code_fence(raw: str) -> str:
    """Models without constrained decoding like to wrap JSON in ```json fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.removesuffix("```").strip()
    return text


def _english_words(text: str, known_words: str) -> list[str]:
    """Lowercase Latin words that did not come from the knowledge base.

    Small multilingual models sometimes slip into English mid-sentence ("meanwhile могу…").
    Capitalized names and abbreviations (Toyota, D3, EPA) are left alone.
    """
    return [word for word in _ENGLISH_WORD.findall(text) if word not in known_words]


def _repeats_reply(pitch: str, reply: str, offered: Iterable[KnowledgeEntry]) -> bool:
    """Whether the pitch offers a product the reply already mentions."""
    pitch, reply = pitch.lower(), reply.lower()
    return any((key := _name_key(product.title)) in pitch and key in reply for product in offered)


def _name_key(title: str) -> str:
    """A crude stem of the product name that survives Russian inflection: «Омега-3» → «омега»."""
    first_word = _WORD.search(title.lower())
    return first_word.group()[:5] if first_word else title.lower()


def _unique(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
