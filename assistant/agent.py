"""Hand-written agent loop: model turn -> dispatch tool calls -> feed results back -> repeat.

The loop never prints. It reports progress through ``on_event`` callbacks (rendered by the CLI)
and returns an ``AgentResult``. Provider failures never raise out of ``run()``.
"""

import copy
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from assistant import registry
from assistant.prompts import DEFAULT_SYSTEM_PROMPT
from assistant.providers import LLMProvider, LLMProviderError, Message, ToolCall, ToolResult

MAX_ITERATIONS_ENV = "MAX_TOOL_ITERATIONS"
DEFAULT_MAX_ITERATIONS = 10
MAX_ITERATIONS_TEXT = (
    "I stopped before finishing because I reached the limit of model turns for one request. "
    "Try a narrower question, or raise MAX_TOOL_ITERATIONS."
)
# Result for calls the model requested on its last allowed turn; they are never executed.
NOT_EXECUTED_RESULT = {
    "success": False,
    "data": None,
    "error": {
        "code": "NOT_EXECUTED",
        "message": "Not executed: the iteration limit for this request was reached.",
    },
}

StoppedReason = Literal["completed", "max_iterations", "provider_error"]


# --- Events -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTurn:
    iteration: int
    text: str | None
    tool_calls: list[ToolCall]
    usage: dict | None


@dataclass(frozen=True)
class ToolCallStarted:
    call: ToolCall


@dataclass(frozen=True)
class ToolCallFinished:
    call: ToolCall
    result: dict
    duration_ms: float


AgentEvent = ModelTurn | ToolCallStarted | ToolCallFinished


@dataclass(frozen=True)
class AgentResult:
    text: str | None
    stopped_reason: StoppedReason
    iterations: int  # generate() calls made during this run
    tool_calls: list[tuple[ToolCall, dict]] = field(default_factory=list)  # executed calls
    usage: dict = field(default_factory=dict)  # token counts summed over the run
    error: str | None = None


def _max_iterations_from_env() -> int:
    raw = os.environ.get(MAX_ITERATIONS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_ITERATIONS
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{MAX_ITERATIONS_ENV} must be an integer, got {raw!r}.") from None


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int | None = None,
        dispatch: Callable[[str, object], dict] = registry.dispatch,
        tools: list[dict] | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_iterations = (
            _max_iterations_from_env() if max_iterations is None else max_iterations
        )
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {self.max_iterations}.")
        self.tools = registry.get_tool_schemas() if tools is None else tools
        self.history: list[Message] = []
        self._dispatch = dispatch
        self._on_event = on_event
        self._clock = clock

    def reset(self) -> None:
        self.history.clear()

    def run(self, user_text: str) -> AgentResult:
        """Answer one user message, running tools until the model replies or the cap is hit.

        On ``LLMProviderError`` the history is rolled back to its state before this call and
        the error is reported in the result. Any other exception propagates.
        """
        checkpoint = len(self.history)
        self.history.append(Message(role="user", text=user_text))
        executed: list[tuple[ToolCall, dict]] = []
        usage: Counter[str] = Counter()
        iterations = 0
        try:
            while True:
                iterations += 1
                response = self.provider.generate(self.system_prompt, self.history, self.tools)
                usage.update(_numeric(response.usage))
                self._emit(
                    ModelTurn(iterations, response.text, response.tool_calls, response.usage)
                )
                self.history.append(
                    Message(
                        role="assistant",
                        text=response.text,
                        tool_calls=response.tool_calls,
                        provider_state=response.provider_state,
                    )
                )
                if not response.tool_calls:
                    return AgentResult(
                        response.text, "completed", iterations, executed, dict(usage)
                    )
                if iterations >= self.max_iterations:
                    self._stop_at_cap(response.tool_calls)
                    return AgentResult(
                        MAX_ITERATIONS_TEXT, "max_iterations", iterations, executed, dict(usage)
                    )
                results = [self._run_tool(call, executed) for call in response.tool_calls]
                self.history.append(Message(role="tool", tool_results=results))
        except LLMProviderError as exc:
            del self.history[checkpoint:]
            return AgentResult(
                None, "provider_error", iterations, executed, dict(usage), error=exc.message
            )

    def _run_tool(self, call: ToolCall, executed: list[tuple[ToolCall, dict]]) -> ToolResult:
        self._emit(ToolCallStarted(call))
        started = self._clock()
        result = self._dispatch(call.name, call.args)
        duration_ms = (self._clock() - started) * 1000
        self._emit(ToolCallFinished(call, result, duration_ms))
        executed.append((call, result))
        return ToolResult(call_id=call.id, name=call.name, result=result)

    def _stop_at_cap(self, pending: list[ToolCall]) -> None:
        """End the turn validly: answer every pending call, then close with a fixed reply."""
        results = [
            ToolResult(call_id=call.id, name=call.name, result=copy.deepcopy(NOT_EXECUTED_RESULT))
            for call in pending
        ]
        self.history.append(Message(role="tool", tool_results=results))
        self.history.append(Message(role="assistant", text=MAX_ITERATIONS_TEXT))

    def _emit(self, event: AgentEvent) -> None:
        if self._on_event is not None:
            self._on_event(event)


def _numeric(usage: dict | None) -> dict[str, int]:
    return {k: v for k, v in (usage or {}).items() if isinstance(v, int | float)}
