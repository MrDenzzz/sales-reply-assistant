import pytest
from pydantic import ValidationError

from sales_reply_assistant.knowledge import (
    CompanyProfile,
    EntryType,
    KnowledgeBase,
    KnowledgeEntry,
    format_price,
)

COMPANY = CompanyProfile(name="Тест", description="тест")


def entry(entry_id: str, **overrides: object) -> KnowledgeEntry:
    fields: dict[str, object] = {
        "id": entry_id,
        "type": EntryType.PRODUCT,
        "title": entry_id,
        "content": "описание",
    }
    return KnowledgeEntry.model_validate(fields | overrides)


def test_demo_knowledge_base_is_valid(demo_knowledge_base: KnowledgeBase) -> None:
    assert demo_knowledge_base.company.name
    assert len(demo_knowledge_base.entries) >= 10
    assert demo_knowledge_base.examples
    assert demo_knowledge_base.pinned


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate entry ids: a"):
        KnowledgeBase(company=COMPANY, entries=(entry("a"), entry("a")))


def test_rejects_unknown_related_ids() -> None:
    with pytest.raises(ValidationError, match="relates to unknown ids: ghost"):
        KnowledgeBase(company=COMPANY, entries=(entry("a", related=("ghost",)),))


def test_rejects_malformed_id() -> None:
    with pytest.raises(ValidationError):
        entry("Not An Id")


def test_get_by_id(knowledge_base: KnowledgeBase) -> None:
    found = knowledge_base.get("shaker")
    assert found is not None
    assert found.title == "Шейкер"
    assert knowledge_base.get("missing") is None


def test_prompt_rendering_includes_price_and_relations() -> None:
    rendered = entry("kit", title="Набор", price_rub=2240, related=("shaker",)).to_prompt()
    assert rendered.splitlines()[0] == "[id: kit] Товар: Набор — 2 240 ₽"
    assert "Сочетается с: shaker" in rendered


def test_format_price() -> None:
    assert format_price(390) == "390 ₽"
    assert format_price(12500) == "12 500 ₽"
