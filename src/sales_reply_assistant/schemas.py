"""Public API contract: request and response models."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sales_reply_assistant.knowledge import CompanyProfile, DemoExample, EntryType, KnowledgeEntry


class Author(StrEnum):
    CLIENT = "client"
    MANAGER = "manager"


class DialogMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    author: Author
    text: str = Field(min_length=1, max_length=4000)


class SuggestRequest(BaseModel):
    """The chat as the manager sees it; the last message is the one to answer."""

    messages: list[DialogMessage] = Field(min_length=1, max_length=100)
    client_name: str | None = Field(default=None, max_length=80)

    @field_validator("client_name")
    @classmethod
    def _blank_name_is_none(cls, value: str | None) -> str | None:
        return (value.strip() or None) if value else None

    @model_validator(mode="after")
    def _ends_with_client_message(self) -> Self:
        if self.messages[-1].author is not Author.CLIENT:
            raise ValueError("the last message must be from the client")
        return self


class SourceRef(BaseModel):
    """A knowledge-base entry the reply is based on."""

    id: str
    title: str
    type: EntryType

    @classmethod
    def from_entry(cls, entry: KnowledgeEntry) -> Self:
        return cls(id=entry.id, title=entry.title, type=entry.type)


class ProductRef(BaseModel):
    id: str
    title: str
    price_rub: int | None

    @classmethod
    def from_entry(cls, entry: KnowledgeEntry) -> Self:
        return cls(id=entry.id, title=entry.title, price_rub=entry.price_rub)


class UpsellSuggestion(BaseModel):
    """Visible to the manager only."""

    should_offer: bool
    hint: str
    # A ready-to-send phrase for the client; empty when nothing should be offered.
    pitch: str
    products: list[ProductRef]


class Suggestion(BaseModel):
    client_intent: str
    reply: str
    needs_attention: bool
    attention_reason: str | None
    upsell: UpsellSuggestion
    compliance_warnings: list[str]
    sources: list[SourceRef]
    model: str
    latency_ms: int


class MetaResponse(BaseModel):
    company: CompanyProfile
    examples: tuple[DemoExample, ...]
    chat_model: str
    embedding_model: str
    local_inference: bool


class ProviderStatus(BaseModel):
    reachable: bool
    chat_model_available: bool | None = None
    embedding_model_available: bool | None = None
    detail: str | None = None
