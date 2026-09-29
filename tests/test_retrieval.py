import asyncio

import pytest
from fakes import FailingEmbedder, KeywordEmbedder

from sales_reply_assistant.errors import ProviderUnavailableError
from sales_reply_assistant.knowledge import KnowledgeBase
from sales_reply_assistant.retrieval import MatchReason, Retriever


def make_retriever(
    knowledge_base: KnowledgeBase, *, top_k: int = 3, min_score: float = 0.4, max_related: int = 3
) -> Retriever:
    return Retriever(
        knowledge_base,
        KeywordEmbedder(),
        top_k=top_k,
        min_score=min_score,
        max_related=max_related,
    )


async def test_finds_the_matching_entry_first(knowledge_base: KnowledgeBase) -> None:
    results = await make_retriever(knowledge_base, min_score=0.2).retrieve("Омега-3 для сердца")

    assert results[0].entry.id == "omega"
    assert results[0].reason is MatchReason.SEARCH


async def test_ranks_products_and_policies_separately(knowledge_base: KnowledgeBase) -> None:
    retriever = make_retriever(knowledge_base, top_k=1, min_score=0.1)

    results = await retriever.retrieve("Доставка клетчатки псиллиум по России")

    search_hits = {r.entry.id for r in results if r.reason is MatchReason.SEARCH}
    assert search_hits == {"psyllium", "delivery"}


async def test_expands_hits_with_related_products_only(knowledge_base: KnowledgeBase) -> None:
    results = await make_retriever(knowledge_base, top_k=1, min_score=0.2).retrieve(
        "клетчатка псиллиум"
    )

    related = [r.entry.id for r in results if r.reason is MatchReason.RELATED]
    assert related == ["shaker"]  # "delivery" is related too, but it is not a product


async def test_always_includes_pinned_entries_once(knowledge_base: KnowledgeBase) -> None:
    results = await make_retriever(knowledge_base).retrieve("совсем посторонний вопрос")

    assert [(r.entry.id, r.reason) for r in results] == [("promo", MatchReason.PINNED)]


async def test_respects_related_limit(knowledge_base: KnowledgeBase) -> None:
    retriever = make_retriever(knowledge_base, top_k=1, min_score=0.2, max_related=0)

    results = await retriever.retrieve("клетчатка псиллиум")

    assert all(r.reason is not MatchReason.RELATED for r in results)


async def test_indexes_the_knowledge_base_once(knowledge_base: KnowledgeBase) -> None:
    embedder = KeywordEmbedder()
    retriever = Retriever(knowledge_base, embedder)

    await asyncio.gather(retriever.warm_up(), retriever.warm_up())
    await retriever.retrieve("шейкер")
    await retriever.retrieve("доставка")

    corpus_calls = [call for call in embedder.calls if len(call) == len(knowledge_base.entries)]
    assert len(corpus_calls) == 1


async def test_propagates_provider_errors(knowledge_base: KnowledgeBase) -> None:
    retriever = Retriever(knowledge_base, FailingEmbedder())

    with pytest.raises(ProviderUnavailableError):
        await retriever.retrieve("шейкер")
