"""LLM providers: neutral types plus a name-based registry with lazy imports.

Provider modules are imported only when ``get_provider`` resolves them, so importing this
package never pulls in a vendor SDK such as ``google.genai``.
"""

import importlib

from assistant.providers.base import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
)

__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolResult",
    "available_providers",
    "get_provider",
]

_PROVIDERS: dict[str, str] = {
    "fake": "assistant.providers.fake:FakeProvider",
    "gemini": "assistant.providers.gemini:GeminiProvider",
}


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(name: str, **kwargs: object) -> LLMProvider:
    """Instantiate the provider registered as ``name`` with ``kwargs``."""
    target = _PROVIDERS.get(name) if isinstance(name, str) else None
    if target is None:
        raise LLMProviderError(
            f"Unknown provider '{name}'. Available providers: {', '.join(available_providers())}."
        )
    module_name, class_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise LLMProviderError(f"Provider '{name}' is not implemented yet.") from exc
        raise LLMProviderError(
            f"Provider '{name}' needs the missing package '{exc.name}'; "
            "install the project requirements (pip install -r requirements.txt)."
        ) from exc
    provider_class: type[LLMProvider] = getattr(module, class_name)
    return provider_class(**kwargs)
