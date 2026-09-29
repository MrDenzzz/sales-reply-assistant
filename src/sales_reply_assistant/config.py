"""Application settings, read from environment variables and an optional `.env` file."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be overridden by an environment variable with the same name in upper case,
    for example ``LLM_MODEL=gpt-4.1-mini``.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Chat model: any OpenAI-compatible server (LM Studio, Ollama, OpenAI, OpenRouter, ...).
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "qwen/qwen3.5-9b"
    llm_temperature: float = Field(default=0.3, ge=0, le=2)
    llm_max_tokens: int = Field(default=1200, gt=0)
    llm_timeout_seconds: float = Field(default=120, gt=0)
    # "none" turns off the thinking phase of hybrid models; empty value omits the parameter.
    llm_reasoning_effort: str | None = "none"
    llm_max_attempts: int = Field(default=2, ge=1, le=5)

    # Embeddings for the knowledge-base search. Default to the chat model's server.
    embedding_model: str = "text-embedding-bge-m3"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None

    knowledge_base_path: Path = Path("knowledge/demo.yaml")
    # Top-k is applied separately to products and to store information (policies, promos).
    retrieval_top_k: int = Field(default=3, ge=1)
    # Cosine similarity threshold; tuned for bge-m3, where unrelated texts score below ~0.4.
    retrieval_min_score: float = Field(default=0.4, ge=-1, le=1)
    retrieval_max_related: int = Field(default=3, ge=0)
    # How many latest chat messages the model sees.
    history_limit: int = Field(default=12, ge=1)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    @field_validator(
        "llm_reasoning_effort", "embedding_base_url", "embedding_api_key", mode="before"
    )
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        # `LLM_REASONING_EFFORT=` in .env should mean "not set", not an empty string.
        return None if value == "" else value
