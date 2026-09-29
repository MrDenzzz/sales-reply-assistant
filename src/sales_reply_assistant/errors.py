"""Errors that cross module boundaries and are mapped to HTTP responses."""


class AssistantError(Exception):
    """Base class for expected failures of the assistant."""


class ProviderUnavailableError(AssistantError):
    """The LLM or embedding server is unreachable or rejected the request."""


class InvalidModelOutputError(AssistantError):
    """The model kept answering with output that violates the response contract."""
