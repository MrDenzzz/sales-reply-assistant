# syntax=docker/dockerfile:1
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first: this layer stays cached until the lock file changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY knowledge ./knowledge
RUN uv sync --locked --no-dev

# LM Studio / Ollama on the host machine are reachable via host.docker.internal.
ENV PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8000 \
    LLM_BASE_URL=http://host.docker.internal:1234/v1

EXPOSE 8000
CMD ["sales-reply-assistant"]
