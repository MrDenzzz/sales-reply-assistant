import json

import pytest
from fakes import KeywordEmbedder, ScriptedChatModel, draft

from sales_reply_assistant.assistant import SalesAssistant
from sales_reply_assistant.errors import InvalidModelOutputError
from sales_reply_assistant.knowledge import KnowledgeBase
from sales_reply_assistant.retrieval import Retriever
from sales_reply_assistant.schemas import Author, DialogMessage, SuggestRequest

REQUEST = SuggestRequest(
    messages=[DialogMessage(author=Author.CLIENT, text="Сколько идёт доставка клетчатки?")],
    client_name="Мария",
)


def make_assistant(knowledge_base: KnowledgeBase, model: ScriptedChatModel) -> SalesAssistant:
    retriever = Retriever(knowledge_base, KeywordEmbedder(), min_score=0.1)
    return SalesAssistant(knowledge_base, retriever, model, max_attempts=2)


async def test_builds_suggestion_from_the_draft(knowledge_base: KnowledgeBase) -> None:
    model = ScriptedChatModel(draft())

    suggestion = await make_assistant(knowledge_base, model).suggest(REQUEST)

    assert suggestion.reply.startswith("Здравствуйте")
    assert [source.title for source in suggestion.sources] == ["Доставка"]
    assert [(p.id, p.price_rub) for p in suggestion.upsell.products] == [("shaker", 390)]
    assert suggestion.upsell.pitch
    assert suggestion.model == "scripted-model"
    assert suggestion.compliance_warnings == []


async def test_sends_context_and_restricted_schema_to_the_model(
    knowledge_base: KnowledgeBase,
) -> None:
    model = ScriptedChatModel(draft())

    await make_assistant(knowledge_base, model).suggest(REQUEST)

    request = model.requests[0]
    system, user = request["messages"]
    assert "Тест" in system["content"]
    assert "Клиент: Сколько идёт доставка клетчатки?" in user["content"]
    source_enum = request["schema"]["properties"]["source_ids"]["items"]["enum"]
    assert {"delivery", "psyllium", "promo"} <= set(source_enum)


async def test_drops_ids_that_are_not_in_the_context(knowledge_base: KnowledgeBase) -> None:
    model = ScriptedChatModel(
        draft(source_ids=["delivery", "invented"], upsell_product_ids=["shaker", "delivery", "x"])
    )

    suggestion = await make_assistant(knowledge_base, model).suggest(REQUEST)

    assert [source.id for source in suggestion.sources] == ["delivery"]
    assert [product.id for product in suggestion.upsell.products] == ["shaker"]


async def test_hides_products_and_pitch_when_upsell_is_inappropriate(
    knowledge_base: KnowledgeBase,
) -> None:
    model = ScriptedChatModel(
        draft(
            needs_attention=True,
            attention_reason="Жалоба",
            upsell_should_offer=False,
            upsell_hint="Сначала решите проблему.",
        )
    )

    suggestion = await make_assistant(knowledge_base, model).suggest(REQUEST)

    assert suggestion.needs_attention
    assert suggestion.attention_reason == "Жалоба"
    assert suggestion.upsell.products == []
    assert suggestion.upsell.pitch == ""
    assert suggestion.upsell.hint == "Сначала решите проблему."


async def test_flags_medical_claims_in_the_reply(knowledge_base: KnowledgeBase) -> None:
    model = ScriptedChatModel(draft(reply="Этот набор вылечит ваш гастрит!"))

    suggestion = await make_assistant(knowledge_base, model).suggest(REQUEST)

    assert suggestion.compliance_warnings


async def test_retries_with_feedback_after_invalid_output(knowledge_base: KnowledgeBase) -> None:
    fenced = f"```json\n{json.dumps(draft(), ensure_ascii=False)}\n```"
    model = ScriptedChatModel('{"reply": "обрыв', fenced)

    suggestion = await make_assistant(knowledge_base, model).suggest(REQUEST)

    assert suggestion.reply
    retry_messages = model.requests[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": '{"reply": "обрыв'}
    assert "не прошёл проверку" in retry_messages[-1]["content"]


async def test_gives_up_after_max_attempts(knowledge_base: KnowledgeBase) -> None:
    model = ScriptedChatModel("не json", "{}")

    with pytest.raises(InvalidModelOutputError):
        await make_assistant(knowledge_base, model).suggest(REQUEST)
