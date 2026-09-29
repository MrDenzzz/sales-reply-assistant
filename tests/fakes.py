"""Deterministic stand-ins for the model servers."""

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from openai.types.chat import ChatCompletionMessageParam

from sales_reply_assistant.errors import ProviderUnavailableError

_WORD = re.compile(r"\w+")


class KeywordEmbedder:
    """Bag-of-words vectors over 5-letter stems: texts sharing words end up close.

    Good enough to test ranking logic without a real embedding model.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in _WORD.findall(text.lower()):
            digest = hashlib.md5(word[:5].encode(), usedforsecurity=False).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dimensions] += 1.0
        return vector.tolist()


class FailingEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise ProviderUnavailableError("embedding server is down")


class ScriptedChatModel:
    """Returns prepared responses in order and records every request."""

    def __init__(self, *responses: str | dict[str, Any]) -> None:
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.requests: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "scripted-model"

    async def complete_json(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        self.requests.append({"messages": list(messages), "schema": schema})
        if not self._responses:
            raise AssertionError("ScriptedChatModel ran out of responses")
        return self._responses.pop(0)


def draft(**overrides: Any) -> dict[str, Any]:
    """A valid model draft; override fields to shape a scenario."""
    return {
        "client_intent": "Вопрос о доставке",
        "needs_attention": False,
        "attention_reason": "",
        "reply": "Здравствуйте! Доставка по России занимает 3–7 дней.",
        "source_ids": ["delivery"],
        "upsell_should_offer": True,
        "upsell_product_ids": ["shaker"],
        "upsell_hint": "Предложите шейкер за 390 ₽.",
        "upsell_pitch": "Клетчатку удобнее размешивать в шейкере.",
        **overrides,
    }
