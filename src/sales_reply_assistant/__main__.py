"""Entry point: ``uv run sales-reply-assistant`` or ``python -m sales_reply_assistant``."""

import logging

import uvicorn

from sales_reply_assistant.api import create_app
from sales_reply_assistant.config import Settings


def main() -> None:
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
