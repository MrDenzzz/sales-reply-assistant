from collections.abc import Iterator

import pytest
from fakes import FailingEmbedder, KeywordEmbedder, ScriptedChatModel, draft
from fastapi.testclient import TestClient

from sales_reply_assistant.api import create_app
from sales_reply_assistant.assistant import SalesAssistant
from sales_reply_assistant.knowledge import KnowledgeBase
from sales_reply_assistant.retrieval import Embedder, Retriever
from sales_reply_assistant.schemas import ProviderStatus
from sales_reply_assistant.services import AppServices

CLIENT_MESSAGE = {"author": "client", "text": "Сколько идёт доставка?"}


async def _status_ok() -> ProviderStatus:
    return ProviderStatus(reachable=True, chat_model_available=True, embedding_model_available=True)


def make_client(
    knowledge_base: KnowledgeBase, model: ScriptedChatModel, embedder: Embedder
) -> TestClient:
    # Keyword vectors score lower than real embeddings, hence the low threshold.
    retriever = Retriever(knowledge_base, embedder, min_score=0.1)
    assistant = SalesAssistant(knowledge_base, retriever, model)
    services = AppServices(
        assistant=assistant,
        knowledge_base=knowledge_base,
        chat_model="scripted-model",
        embedding_model="keyword-embedder",
        local_inference=True,
        check_status=_status_ok,
    )
    return TestClient(create_app(services=services))


@pytest.fixture
def client(knowledge_base: KnowledgeBase) -> Iterator[TestClient]:
    with make_client(knowledge_base, ScriptedChatModel(draft()), KeywordEmbedder()) as client:
        yield client


def test_suggest_returns_reply_and_upsell(client: TestClient) -> None:
    response = client.post("/api/suggest", json={"messages": [CLIENT_MESSAGE]})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"]
    assert body["upsell"]["should_offer"] is True
    assert body["sources"] == [{"id": "delivery", "title": "Доставка", "type": "policy"}]


def test_suggest_requires_the_last_message_from_the_client(client: TestClient) -> None:
    messages = [CLIENT_MESSAGE, {"author": "manager", "text": "Отвечаю"}]

    response = client.post("/api/suggest", json={"messages": messages})

    assert response.status_code == 422


def test_suggest_rejects_an_empty_dialog(client: TestClient) -> None:
    assert client.post("/api/suggest", json={"messages": []}).status_code == 422


def test_suggest_reports_unavailable_model_server(knowledge_base: KnowledgeBase) -> None:
    with make_client(knowledge_base, ScriptedChatModel(), FailingEmbedder()) as client:
        response = client.post("/api/suggest", json={"messages": [CLIENT_MESSAGE]})

    assert response.status_code == 503
    assert "Нейросеть недоступна" in response.json()["detail"]


def test_suggest_reports_invalid_model_output(knowledge_base: KnowledgeBase) -> None:
    model = ScriptedChatModel("oops", "still oops")
    with make_client(knowledge_base, model, KeywordEmbedder()) as client:
        response = client.post("/api/suggest", json={"messages": [CLIENT_MESSAGE]})

    assert response.status_code == 502


def test_meta_describes_company_and_models(client: TestClient) -> None:
    body = client.get("/api/meta").json()

    assert body["company"]["name"] == "Тест"
    assert body["chat_model"] == "scripted-model"
    assert body["local_inference"] is True


def test_status(client: TestClient) -> None:
    assert client.get("/api/status").json()["reachable"] is True


def test_serves_the_demo_ui(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Sales Reply Assistant" in response.text
