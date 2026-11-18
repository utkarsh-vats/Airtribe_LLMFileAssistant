"""Provider-neutral conversation types and the ``LLMProvider`` interface.

The agent loop speaks only these types; each provider translates them to and from its SDK.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResult:
    """The dispatcher envelope produced for one ``ToolCall``."""

    call_id: str
    name: str
    result: dict


@dataclass(frozen=True)
class Message:
    """One conversation turn.

    Assistant messages may carry ``tool_calls``; a tool message carries the ``tool_results``
    for every call of the preceding assistant turn. ``provider_state`` is opaque per-provider
    data the agent loop stores and hands back unchanged (Gemini needs it to replay its original
    response content, including thought signatures); other providers ignore it.
    """

    role: Literal["user", "assistant", "tool"]
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    provider_state: Any = None


@dataclass(frozen=True)
class LLMResponse:
    """A model turn: final ``text``, and/or ``tool_calls`` to run before asking again."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider_state: Any = None
    usage: dict | None = None


class LLMProviderError(Exception):
    """A provider failure with a user-safe message (no API keys, no tracebacks).

    ``retryable`` marks transient failures (rate limits, timeouts, 5xx) worth retrying.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        """Store the user-safe ``message`` (also ``str(exc)``) and the ``retryable`` flag."""
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class LLMProvider(ABC):
    """Interface every LLM backend implements."""

    name: ClassVar[str]

    @abstractmethod
    def generate(
        self, system_prompt: str, messages: Sequence[Message], tools: list[dict]
    ) -> LLMResponse:
        """Send the conversation and tool schemas; return the model's next turn.

        ``tools`` is ``assistant.registry.get_tool_schemas()`` output. Raises
        ``LLMProviderError`` on failure.
        """
