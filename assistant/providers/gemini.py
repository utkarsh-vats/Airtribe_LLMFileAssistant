"""Gemini provider built on ``google-genai`` with automatic function calling disabled.

The agent loop owns tool execution: this module only translates the neutral conversation types
to ``google.genai.types`` and back, and maps SDK failures to ``LLMProviderError``.
"""

import importlib
import logging
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from google import genai
from google.genai import errors, types

from assistant.providers.base import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    Message,
    ToolCall,
)

logger = logging.getLogger(__name__)

API_KEY_ENV = "GEMINI_API_KEY"
MODEL_ENV = "GEMINI_MODEL"
SYNTHETIC_ID_PREFIX = "gemini_call_"  # ids we invent when Gemini omits one; never sent back
FUNCTION_RESPONSE_ROLE = "user"  # what google-genai's own chat/AFC code uses

# Finish reasons that mean the content was withheld: retrying the same request won't help.
_BLOCKING_FINISH_REASONS = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.LANGUAGE,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.SPII,
        types.FinishReason.IMAGE_SAFETY,
        types.FinishReason.IMAGE_PROHIBITED_CONTENT,
        types.FinishReason.IMAGE_RECITATION,
        types.FinishReason.MODEL_ARMOR,
        types.FinishReason.MISSING_THOUGHT_SIGNATURE,
    }
)
# Finish reasons where the model produced something unusable but a fresh attempt may succeed.
_RETRYABLE_FINISH_REASONS = frozenset(
    {types.FinishReason.MALFORMED_FUNCTION_CALL, types.FinishReason.MALFORMED_RESPONSE}
)

_STATUS_HINTS = {
    400: "the request was rejected as invalid",
    401: "check GEMINI_API_KEY",
    403: "check GEMINI_API_KEY and that the Gemini API is enabled for it",
    404: "check GEMINI_MODEL",
    408: "the request timed out",
    429: "rate limit or quota exceeded",
}
_RETRYABLE_STATUS = frozenset({408, 429})


def _network_errors() -> tuple[type[BaseException], ...]:
    """Transport exceptions of whichever HTTP stacks are installed (google-genai uses httpx2)."""
    found: list[type[BaseException]] = [ConnectionError, TimeoutError]
    for module_name in ("httpx2", "httpx"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        found.append(module.TransportError)
    return tuple(found)


_NETWORK_ERRORS = _network_errors()


class GeminiProvider(LLMProvider):
    """Google Gemini through ``google-genai``, with automatic function calling disabled.

    The agent loop runs every tool call itself; this class only converts types, calls the API
    and maps failures to ``LLMProviderError``, retrying the transient ones.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any = None,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Resolve the key and model and build (or accept) the SDK client.

        Args:
            api_key: Gemini API key; falls back to ``GEMINI_API_KEY`` in ``os.environ``.
            model: Model id; falls back to ``GEMINI_MODEL`` in ``os.environ``.
            client: A ``google.genai.Client`` (or a test double with ``models.generate_content``);
                ``None`` builds a real client from ``api_key``.
            max_retries: Extra attempts for retryable failures (backoff 1 s, 2 s, ...).
            sleep: Used for the backoff waits; injectable so tests don't sleep.

        Raises:
            LLMProviderError: If the key or model is missing (naming the env var, never the
                key) or the client can't be created.
        """
        api_key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
        model = (model or os.environ.get(MODEL_ENV) or "").strip()
        if not api_key:
            raise LLMProviderError(f"No Gemini API key: set {API_KEY_ENV} (e.g. in .env).")
        if not model:
            raise LLMProviderError(
                f"No Gemini model: set {MODEL_ENV} (e.g. in .env) to a model id such as a "
                "current Gemini Flash model."
            )
        self._api_key = api_key
        self.model = model
        self._max_retries = max_retries
        self._sleep = sleep
        if client is None:
            try:
                client = genai.Client(api_key=api_key)
            except Exception as exc:
                raise LLMProviderError(
                    f"Could not create the Gemini client ({type(exc).__name__})."
                ) from exc
        self._client = client

    # --- LLMProvider ----------------------------------------------------------------------

    def generate(
        self, system_prompt: str, messages: Sequence[Message], tools: list[dict]
    ) -> LLMResponse:
        """Send the conversation to Gemini and map its reply to an ``LLMResponse``.

        Gemini-specific: an assistant message whose ``provider_state`` is a ``types.Content``
        is replayed as that exact object, preserving its thought signatures; the response's
        ``provider_state`` is the candidate's raw ``Content`` for the loop to store and hand
        back. Function calls without an id get positional ``gemini_call_<n>`` ids, which are
        never sent back to Gemini. Retryable failures (429, 5xx, network, malformed or empty
        responses) are retried with backoff up to ``max_retries`` times.

        Raises:
            LLMProviderError: On a blocked prompt or response, a non-retryable API error, or
                once retries are exhausted.
        """
        contents = [_to_content(message) for message in messages]
        config = _build_config(system_prompt, tools)
        for attempt in range(self._max_retries + 1):
            try:
                return self._generate_once(contents, config)
            except LLMProviderError as exc:
                if not exc.retryable or attempt == self._max_retries:
                    raise
                delay = 2**attempt  # 1s, 2s, ...
                logger.warning("Gemini call failed (%s); retrying in %ss", exc.message, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _generate_once(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> LLMResponse:
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except errors.APIError as exc:
            raise self._api_error(exc) from exc
        except _NETWORK_ERRORS as exc:
            raise LLMProviderError(
                f"Network error talking to Gemini ({type(exc).__name__}).", retryable=True
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Unexpected error calling Gemini ({type(exc).__name__})."
            ) from exc
        return _to_llm_response(response)

    def _api_error(self, exc: errors.APIError) -> LLMProviderError:
        code = exc.code if isinstance(exc.code, int) else 0
        hint = _STATUS_HINTS.get(code, "server error" if code >= 500 else "request failed")
        detail = str(exc.message or exc.status or "").replace(self._api_key, "***")[:200]
        message = f"Gemini API error {code}: {hint}."
        if detail:
            message += f" ({detail})"
        return LLMProviderError(message, retryable=code in _RETRYABLE_STATUS or code >= 500)


# --- Request building ---------------------------------------------------------------------


def _build_config(system_prompt: str, tools: list[dict]) -> types.GenerateContentConfig:
    """Config with our tools, function calling AUTO and automatic function calling disabled.

    Temperature is left at the model default, as recommended for Gemini 3.
    """
    declarations = [
        types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters_json_schema=tool["parameters"],  # raw JSON Schema, no conversion
        )
        for tool in tools
    ]
    return types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        tools=[types.Tool(function_declarations=declarations)] if declarations else None,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.AUTO
            )
        )
        if declarations
        else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def _real_id(call_id: str) -> str | None:
    """The id to send back to Gemini: ours are synthetic and must not be echoed."""
    return None if not call_id or call_id.startswith(SYNTHETIC_ID_PREFIX) else call_id


def _to_content(message: Message) -> types.Content:
    if message.role == "assistant":
        if isinstance(message.provider_state, types.Content):
            # Replay Gemini's own content untouched: it carries the thought signatures.
            return message.provider_state
        parts = [types.Part(text=message.text)] if message.text else []
        parts += [
            types.Part(
                function_call=types.FunctionCall(id=_real_id(c.id), name=c.name, args=c.args)
            )
            for c in message.tool_calls
        ]
        return types.Content(role="model", parts=parts)
    if message.role == "tool":
        return types.Content(
            role=FUNCTION_RESPONSE_ROLE,
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=_real_id(r.call_id), name=r.name, response=r.result
                    )
                )
                for r in message.tool_results
            ],
        )
    return types.Content(role="user", parts=[types.Part(text=message.text or "")])


# --- Response mapping ---------------------------------------------------------------------


def _reason_name(reason: Any) -> str:
    return getattr(reason, "name", None) or str(reason)


def _to_llm_response(response: types.GenerateContentResponse) -> LLMResponse:
    feedback = response.prompt_feedback
    block_reason = feedback.block_reason if feedback is not None else None
    if block_reason and block_reason != types.BlockedReason.BLOCKED_REASON_UNSPECIFIED:
        raise LLMProviderError(f"Gemini blocked the prompt (reason: {_reason_name(block_reason)}).")
    if not response.candidates:
        raise LLMProviderError("Gemini returned no candidates.")

    candidate = response.candidates[0]
    finish = candidate.finish_reason
    if finish in _BLOCKING_FINISH_REASONS:
        raise LLMProviderError(
            f"Gemini stopped the response (finish reason: {_reason_name(finish)})."
        )
    if finish in _RETRYABLE_FINISH_REASONS:
        raise LLMProviderError(
            f"Gemini produced an unusable response (finish reason: {_reason_name(finish)}).",
            retryable=True,
        )

    content = candidate.content
    parts = (content.parts if content else None) or []
    tool_calls = []
    for part in parts:
        if part.function_call is None:
            continue
        call = part.function_call
        call_id = call.id or f"{SYNTHETIC_ID_PREFIX}{len(tool_calls) + 1}"
        tool_calls.append(ToolCall(id=call_id, name=call.name or "", args=dict(call.args or {})))
    texts = [part.text for part in parts if part.text and not part.thought]
    if not tool_calls and not texts:
        raise LLMProviderError(
            f"Gemini returned an empty response (finish reason: {_reason_name(finish)}).",
            retryable=True,
        )
    return LLMResponse(
        text="".join(texts) if texts else None,
        tool_calls=tool_calls,
        provider_state=content,
        usage=_usage(response.usage_metadata),
    )


def _usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> dict | None:
    if metadata is None:
        return None
    usage = {
        "prompt_tokens": metadata.prompt_token_count,
        "candidates_tokens": metadata.candidates_token_count,
        "thoughts_tokens": metadata.thoughts_token_count,
        "total_tokens": metadata.total_token_count,
    }
    usage = {key: value for key, value in usage.items() if value is not None}
    return usage or None
