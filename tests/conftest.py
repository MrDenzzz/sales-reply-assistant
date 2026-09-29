from pathlib import Path

import pytest

from sales_reply_assistant.knowledge import (
    CompanyProfile,
    EntryType,
    KnowledgeBase,
    KnowledgeEntry,
    load_knowledge_base,
)

DEMO_KNOWLEDGE_BASE = Path(__file__).parents[1] / "knowledge" / "demo.yaml"


@pytest.fixture
def knowledge_base() -> KnowledgeBase:
    """A tiny knowledge base with predictable wording for ranking tests."""
    return KnowledgeBase(
        company=CompanyProfile(name="Тест", description="тестовый магазин"),
        entries=(
            KnowledgeEntry(
                id="psyllium",
                type=EntryType.PRODUCT,
                title="Клетчатка псиллиум",
                content="Клетчатка для пищеварения, банка на месяц.",
                price_rub=890,
                related=("shaker", "delivery"),
            ),
            KnowledgeEntry(
                id="shaker",
                type=EntryType.PRODUCT,
                title="Шейкер",
                content="Шейкер для размешивания порошков.",
                price_rub=390,
            ),
            KnowledgeEntry(
                id="omega",
                type=EntryType.PRODUCT,
                title="Омега-3",
                content="Рыбий жир для сердца и сосудов.",
                price_rub=1190,
            ),
            KnowledgeEntry(
                id="delivery",
                type=EntryType.POLICY,
                title="Доставка",
                content="Доставка по России СДЭК занимает 3–7 дней.",
            ),
            KnowledgeEntry(
                id="promo",
                type=EntryType.PROMO,
                title="Скидка на курс",
                content="Две упаковки со скидкой.",
                pinned=True,
            ),
        ),
    )


@pytest.fixture
def demo_knowledge_base() -> KnowledgeBase:
    return load_knowledge_base(DEMO_KNOWLEDGE_BASE)
