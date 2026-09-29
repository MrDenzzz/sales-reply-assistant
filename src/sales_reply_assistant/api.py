"""HTTP API and the demo UI."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from sales_reply_assistant import __version__
from sales_reply_assistant.config import Settings
from sales_reply_assistant.errors import InvalidModelOutputError, ProviderUnavailableError
from sales_reply_assistant.schemas import MetaResponse, ProviderStatus, Suggestion, SuggestRequest
from sales_reply_assistant.services import AppServices, build_services

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def get_services(request: Request) -> AppServices:
    services: AppServices = request.app.state.services
    return services


Services = Annotated[AppServices, Depends(get_services)]

router = APIRouter(prefix="/api")


@router.get("/meta")
async def meta(services: Services) -> MetaResponse:
    """Company profile, demo scenarios and model names for the UI."""
    return MetaResponse(
        company=services.knowledge_base.company,
        examples=services.knowledge_base.examples,
        chat_model=services.chat_model,
        embedding_model=services.embedding_model,
        local_inference=services.local_inference,
    )


@router.get("/status")
async def status(services: Services) -> ProviderStatus:
    """Whether the model server is reachable and the configured models are available."""
    return await services.check_status()


@router.post("/suggest")
async def suggest(request: SuggestRequest, services: Services) -> Suggestion:
    """Draft a reply to the client's last message and an upsell hint for the manager."""
    return await services.assistant.suggest(request)


async def _provider_unavailable(_: Request, error: Exception) -> JSONResponse:
    logger.warning("model server unavailable: %s", error)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Нейросеть недоступна: проверьте, что сервер моделей запущен "
            "и нужные модели загружены.",
            "error": str(error),
        },
    )


async def _invalid_output(_: Request, error: Exception) -> JSONResponse:
    logger.error("model output rejected: %s", error)
    return JSONResponse(
        status_code=502,
        content={
            "detail": "Модель вернула ответ в неверном формате. Попробуйте ещё раз.",
            "error": str(error),
        },
    )


def create_app(services: AppServices | None = None, settings: Settings | None = None) -> FastAPI:
    """Create the application. Pass ready ``services`` to bypass the real model servers."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.services = services
            yield
            return
        async with build_services(settings or Settings()) as built:
            app.state.services = built
            yield

    app = FastAPI(title="Sales Reply Assistant", version=__version__, lifespan=lifespan)
    app.include_router(router)
    app.add_exception_handler(ProviderUnavailableError, _provider_unavailable)
    app.add_exception_handler(InvalidModelOutputError, _invalid_output)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app
