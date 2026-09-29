from sales_reply_assistant.knowledge import KnowledgeBase
from sales_reply_assistant.prompts import (
    LlmDraft,
    build_draft_schema,
    build_search_query,
    build_system_prompt,
    build_user_prompt,
)
from sales_reply_assistant.schemas import Author, DialogMessage


def message(author: Author, text: str) -> DialogMessage:
    return DialogMessage(author=author, text=text)


def test_schema_matches_the_draft_model(knowledge_base: KnowledgeBase) -> None:
    schema = build_draft_schema(knowledge_base.entries)

    assert list(schema["properties"]) == list(LlmDraft.model_fields)
    assert schema["required"] == list(LlmDraft.model_fields)
    assert schema["additionalProperties"] is False


def test_schema_limits_ids_to_the_context(knowledge_base: KnowledgeBase) -> None:
    context = [knowledge_base.get("psyllium"), knowledge_base.get("delivery")]
    schema = build_draft_schema([entry for entry in context if entry])

    properties = schema["properties"]
    assert properties["source_ids"]["items"]["enum"] == ["psyllium", "delivery"]
    assert properties["upsell_product_ids"]["items"]["enum"] == ["psyllium"]


def test_schema_without_products_accepts_any_string(knowledge_base: KnowledgeBase) -> None:
    delivery = knowledge_base.get("delivery")
    assert delivery is not None

    schema = build_draft_schema([delivery])

    assert schema["properties"]["upsell_product_ids"]["items"] == {"type": "string"}


def test_system_prompt_names_the_company(knowledge_base: KnowledgeBase) -> None:
    assert "«Тест» (тестовый магазин)" in build_system_prompt(knowledge_base.company)


def test_user_prompt_contains_facts_dialog_and_name(knowledge_base: KnowledgeBase) -> None:
    dialog = [message(Author.CLIENT, "Сколько идёт доставка?")]

    prompt = build_user_prompt(knowledge_base.entries, dialog, "Мария")

    assert "[id: delivery]" in prompt
    assert "Клиент: Сколько идёт доставка?" in prompt
    assert "Имя клиента: Мария" in prompt


def test_user_prompt_forbids_inventing_an_unknown_name(knowledge_base: KnowledgeBase) -> None:
    prompt = build_user_prompt(knowledge_base.entries, [message(Author.CLIENT, "Привет")], None)

    assert "Имя клиента не указано" in prompt


def test_search_query_uses_latest_client_messages_only() -> None:
    dialog = [
        message(Author.CLIENT, "первый"),
        message(Author.CLIENT, "второй"),
        message(Author.MANAGER, "ответ менеджера"),
        message(Author.CLIENT, "третий"),
    ]

    assert build_search_query(dialog) == "второй\nтретий"
