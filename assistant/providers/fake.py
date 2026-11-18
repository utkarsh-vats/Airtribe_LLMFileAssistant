"""Scripted provider for tests and offline demos: replays canned responses, records every call."""

import copy
import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from assistant.providers.base import LLMProvider, LLMProviderError, LLMResponse, Message, ToolCall


@dataclass(frozen=True)
class RecordedCall:
    """Deep copies of what one ``generate`` call received."""

    system_prompt: str
    messages: list[Message]
    tools: list[dict]


class FakeProvider(LLMProvider):
    """Returns the scripted ``LLMResponse``s in order, one per ``generate`` call.

    Tool calls built with an empty id (as ``call``/``parallel`` do) get deterministic ids
    ``call_1``, ``call_2``, ... from a per-instance counter when they are returned.
    """

    name = "fake"

    def __init__(self, script: Iterable[LLMResponse] = ()) -> None:
        """Copy ``script``; each ``generate()`` call consumes its next response."""
        self._script = list(script)
        self._served = 0
        self._next_id = 0
        self.calls: list[RecordedCall] = []

    # --- Script builders ------------------------------------------------------------------

    @classmethod
    def reply(cls, text: str) -> LLMResponse:
        """A final text answer."""
        return LLMResponse(text=text)

    @classmethod
    def call(cls, name: str, args: dict, id: str | None = None) -> LLMResponse:
        """A turn with one tool call; ``id=None`` means auto-assign."""
        return LLMResponse(text=None, tool_calls=[ToolCall(id=id or "", name=name, args=args)])

    @classmethod
    def parallel(cls, *calls: tuple[str, dict]) -> LLMResponse:
        """A turn with several tool calls, given as ``(name, args)`` pairs."""
        tool_calls = [ToolCall(id="", name=name, args=args) for name, args in calls]
        return LLMResponse(text=None, tool_calls=tool_calls)

    # --- LLMProvider ----------------------------------------------------------------------

    def generate(
        self, system_prompt: str, messages: Sequence[Message], tools: list[dict]
    ) -> LLMResponse:
        """Record the call and return the next scripted response.

        Fake-specific: the inputs are ignored except for being deep-copied into ``.calls``
        (even when the script is exhausted), and tool calls with an empty id get the next
        ``call_<n>`` id. ``provider_state`` is returned exactly as scripted.

        Raises:
            LLMProviderError: If the script is exhausted (not retryable).
        """
        self.calls.append(
            RecordedCall(
                system_prompt=system_prompt,
                messages=copy.deepcopy(list(messages)),
                tools=copy.deepcopy(tools),
            )
        )
        if not self._script:
            raise LLMProviderError(
                f"FakeProvider script exhausted after {self._served} response(s); "
                "add more scripted responses."
            )
        response = self._script.pop(0)
        self._served += 1
        return dataclasses.replace(
            response, tool_calls=[self._with_id(c) for c in response.tool_calls]
        )

    def _with_id(self, tool_call: ToolCall) -> ToolCall:
        if tool_call.id:
            return tool_call
        self._next_id += 1
        return dataclasses.replace(tool_call, id=f"call_{self._next_id}")
