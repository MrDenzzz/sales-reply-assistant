"""Knowledge base: the only source of facts the assistant may rely on."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class EntryType(StrEnum):
    PRODUCT = "product"
    POLICY = "policy"
    PROMO = "promo"


_TYPE_LABELS = {
    EntryType.PRODUCT: "Товар",
    EntryType.POLICY: "Условия",
    EntryType.PROMO: "Акция",
}


def format_price(rub: int) -> str:
    """Format a price the Russian way: ``2 240 ₽`` with non-breaking spaces."""
    return f"{rub:,}".replace(",", " ") + " ₽"


class KnowledgeEntry(BaseModel):
    """A self-contained fact sheet: a product, a store policy or a promo."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    type: EntryType
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    price_rub: int | None = Field(default=None, gt=0)
    # Products that naturally complement this one: the raw material for upsell hints.
    related: tuple[str, ...] = ()
    # Pinned entries are always in the context (e.g. current promos).
    pinned: bool = False

    @property
    def search_text(self) -> str:
        """Text that represents the entry in the vector index."""
        return f"{self.title}\n{self.content}"

    def to_prompt(self) -> str:
        """Render the entry as a compact block of the LLM context."""
        header = f"[id: {self.id}] {_TYPE_LABELS[self.type]}: {self.title}"
        if self.price_rub is not None:
            header += f" — {format_price(self.price_rub)}"
        lines = [header, self.content]
        if self.related:
            lines.append("Сочетается с: " + ", ".join(self.related))
        return "\n".join(lines)


class CompanyProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str


class DemoExample(BaseModel):
    """A ready-made client message for the demo UI."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    text: str


class KnowledgeBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    company: CompanyProfile
    examples: tuple[DemoExample, ...] = ()
    entries: tuple[KnowledgeEntry, ...] = Field(min_length=1)

    _by_id: dict[str, KnowledgeEntry] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        ids = [entry.id for entry in self.entries]
        duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate entry ids: {', '.join(duplicates)}")

        known = set(ids)
        for entry in self.entries:
            unknown = [related for related in entry.related if related not in known]
            if unknown:
                raise ValueError(f"entry '{entry.id}' relates to unknown ids: {', '.join(unknown)}")
        return self

    def model_post_init(self, context: Any, /) -> None:
        self._by_id = {entry.id: entry for entry in self.entries}

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        return self._by_id.get(entry_id)

    @property
    def pinned(self) -> tuple[KnowledgeEntry, ...]:
        return tuple(entry for entry in self.entries if entry.pinned)


def load_knowledge_base(path: Path) -> KnowledgeBase:
    """Read and validate a YAML knowledge base."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return KnowledgeBase.model_validate(raw)
