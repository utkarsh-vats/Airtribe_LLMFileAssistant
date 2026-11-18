"""GeminiProvider tests against a fake client returning real google-genai types; no network."""

import httpx2
import pytest
from google.genai import errors, types

from assistant.providers import LLMProviderError, Message, ToolCall, ToolResult, get_provider
from assistant.providers.gemini import GeminiProvider
from assistant.registry import TOOL_NAMES, get_tool_schemas

API_KEY = "test-secret-key-123"
MODEL = "test-model"
USER = Message(role="user", text="List the PDF files in resumes")


class FakeModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []

    def generate_content(self, **kwargs: object) -> types.GenerateContentResponse:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]


class FakeClient:
    def __init__(self, *outcomes: object) -> None:
        self.models = FakeModels(list(outcomes))


def _provider(*outcomes: object, sleeps: list[float] | None = None, **kwargs: object):
    client = FakeClient(*outcomes)
    sleep = sleeps.append if sleeps is not None else lambda seconds: None
    provider = GeminiProvider(api_key=API_KEY, model=MODEL, client=client, sleep=sleep, **kwargs)
    return provider, client.models


def _response(
    *parts: types.Part,
    finish: types.FinishReason = types.FinishReason.STOP,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)), finish_reason=finish
            )
        ],
        usage_metadata=usage,
    )


def _text(text: str, thought: bool | None = None) -> types.Part:
    return types.Part(text=text, thought=thought)


def _call(name: str, args: dict, call_id: str | None = None) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(id=call_id, name=name, args=args),
        thought_signature=b"sig-" + name.encode(),
    )


def _api_error(cls: type[errors.APIError], code: int, status: str, message: str) -> errors.APIError:
    return cls(code, {"error": {"code": code, "message": message, "status": status}})


def _generate(provider: GeminiProvider, messages: list[Message] | None = None):
    return provider.generate("You are a resume assistant.", messages or [USER], get_tool_schemas())


# --- Construction -------------------------------------------------------------------------


def test_missing_api_key_names_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(LLMProviderError) as excinfo:
        GeminiProvider(model=MODEL, client=FakeClient())

    assert "GEMINI_API_KEY" in excinfo.value.message
    assert excinfo.value.retryable is False


def test_missing_model_names_env_var_without_echoing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    with pytest.raises(LLMProviderError) as excinfo:
        GeminiProvider(api_key=API_KEY, client=FakeClient())

    assert "GEMINI_MODEL" in excinfo.value.message
    assert API_KEY not in excinfo.value.message


def test_key_and_model_fall_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", API_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "env-model")

    provider = GeminiProvider(client=FakeClient())

    assert provider.model == "env-model"


def test_get_provider_returns_gemini_provider() -> None:
    provider = get_provider("gemini", api_key="test", model="test-model", client=FakeClient())

    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"


# --- Request config -----------------------------------------------------------------------


def test_config_disables_afc_and_declares_all_tools() -> None:
    provider, models = _provider(_response(_text("ok")))

    _generate(provider)

    request = models.requests[0]
    assert request["model"] == MODEL
    config = request["config"]
    assert config.automatic_function_calling.disable is True
    assert config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.AUTO
    assert config.system_instruction == "You are a resume assistant."
    assert config.temperature is None
    (tool,) = config.tools
    declared = {d.name: d for d in tool.function_declarations}
    assert tuple(declared) == TOOL_NAMES
    for schema in get_tool_schemas():
        declaration = declared[schema["name"]]
        assert declaration.description == schema["description"]
        assert declaration.parameters_json_schema == schema["parameters"]
        assert declaration.parameters is None


def test_no_tools_means_no_tool_config() -> None:
    provider, models = _provider(_response(_text("ok")))

    provider.generate("sys", [USER], [])

    config = models.requests[0]["config"]
    assert config.tools is None and config.tool_config is None
    assert config.automatic_function_calling.disable is True


# --- History conversion -------------------------------------------------------------------


def test_history_conversion() -> None:
    provider, models = _provider(_response(_text("ok")))
    history = [
        USER,
        Message(
            role="assistant",
            text="Let me look.",
            tool_calls=[ToolCall("abc", "list_files", {"directory": "resumes"})],
        ),
        Message(
            role="tool",
            tool_results=[ToolResult("abc", "list_files", {"success": True, "data": {"count": 0}})],
        ),
    ]

    _generate(provider, history)

    user, model, tool = models.requests[0]["contents"]
    assert user.role == "user" and user.parts[0].text == USER.text
    assert model.role == "model"
    assert model.parts[0].text == "Let me look."
    assert model.parts[1].function_call.id == "abc"
    assert model.parts[1].function_call.name == "list_files"
    assert model.parts[1].function_call.args == {"directory": "resumes"}
    assert tool.role == "user"
    response = tool.parts[0].function_response
    assert (response.id, response.name) == ("abc", "list_files")
    assert response.response == {"success": True, "data": {"count": 0}}


def test_assistant_provider_state_is_passed_through_as_same_object() -> None:
    provider, models = _provider(_response(_text("ok")))
    raw = types.Content(role="model", parts=[_call("list_files", {"directory": "resumes"}, "abc")])
    history = [USER, Message(role="assistant", tool_calls=[], provider_state=raw)]

    _generate(provider, history)

    assert models.requests[0]["contents"][1] is raw


def test_parallel_tool_results_land_in_one_content() -> None:
    provider, models = _provider(_response(_text("ok")))
    results = [
        ToolResult("c1", "search_in_file", {"success": True, "data": {"total_matches": 1}}),
        ToolResult("c2", "search_in_file", {"success": True, "data": {"total_matches": 0}}),
    ]

    _generate(provider, [USER, Message(role="tool", tool_results=results)])

    tool_content = models.requests[0]["contents"][1]
    assert [p.function_response.id for p in tool_content.parts] == ["c1", "c2"]


def test_synthetic_ids_are_not_sent_back() -> None:
    provider, models = _provider(_response(_text("ok")))
    history = [
        USER,
        Message(role="assistant", tool_calls=[ToolCall("gemini_call_1", "list_files", {})]),
        Message(role="tool", tool_results=[ToolResult("gemini_call_1", "list_files", {})]),
    ]

    _generate(provider, history)

    _, model, tool = models.requests[0]["contents"]
    assert model.parts[0].function_call.id is None
    assert tool.parts[0].function_response.id is None


# --- Response mapping ---------------------------------------------------------------------


def test_text_only_response() -> None:
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=12,
        candidates_token_count=5,
        thoughts_token_count=3,
        total_token_count=20,
    )
    response = _response(_text("Three PDFs."), usage=usage)
    provider, _ = _provider(response)

    result = _generate(provider)

    assert result.text == "Three PDFs."
    assert result.tool_calls == []
    assert result.usage == {
        "prompt_tokens": 12,
        "candidates_tokens": 5,
        "thoughts_tokens": 3,
        "total_tokens": 20,
    }
    assert result.provider_state is response.candidates[0].content


def test_single_tool_call() -> None:
    provider, _ = _provider(_response(_call("list_files", {"directory": "resumes"}, "abc")))

    result = _generate(provider)

    assert result.text is None
    assert result.tool_calls == [ToolCall("abc", "list_files", {"directory": "resumes"})]
    assert result.usage is None


def test_parallel_calls_with_missing_ids_get_positional_ids() -> None:
    provider, _ = _provider(
        _response(
            _call("search_in_file", {"filepath": "a.pdf", "keyword": "python"}),
            _call("search_in_file", {"filepath": "b.txt", "keyword": "python"}, "given"),
            _call("search_in_file", {"filepath": "c.docx", "keyword": "python"}),
        )
    )

    result = _generate(provider)

    assert [c.id for c in result.tool_calls] == ["gemini_call_1", "given", "gemini_call_3"]
    assert [c.args["filepath"] for c in result.tool_calls] == ["a.pdf", "b.txt", "c.docx"]


def test_thought_parts_are_excluded_from_text() -> None:
    provider, _ = _provider(
        _response(_text("Thinking about files...", thought=True), _text("Here "), _text("you go."))
    )

    assert _generate(provider).text == "Here you go."


def test_call_with_no_args_maps_to_empty_dict() -> None:
    provider, _ = _provider(
        _response(types.Part(function_call=types.FunctionCall(id="x", name="list_files")))
    )

    assert _generate(provider).tool_calls[0].args == {}


# --- Failures -----------------------------------------------------------------------------


def test_blocked_prompt() -> None:
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY
        )
    )
    provider, models = _provider(blocked)

    with pytest.raises(LLMProviderError, match="SAFETY") as excinfo:
        _generate(provider)

    assert excinfo.value.retryable is False
    assert len(models.requests) == 1


def test_no_candidates() -> None:
    provider, _ = _provider(types.GenerateContentResponse(candidates=[]))

    with pytest.raises(LLMProviderError, match="no candidates") as excinfo:
        _generate(provider)

    assert excinfo.value.retryable is False


@pytest.mark.parametrize("finish", [types.FinishReason.SAFETY, types.FinishReason.RECITATION])
def test_blocking_finish_reason(finish: types.FinishReason) -> None:
    provider, models = _provider(_response(_text("partial"), finish=finish))

    with pytest.raises(LLMProviderError, match=finish.name) as excinfo:
        _generate(provider)

    assert excinfo.value.retryable is False
    assert len(models.requests) == 1


def test_malformed_function_call_is_retried() -> None:
    sleeps: list[float] = []
    provider, models = _provider(
        _response(finish=types.FinishReason.MALFORMED_FUNCTION_CALL),
        _response(_text("recovered")),
        sleeps=sleeps,
    )

    assert _generate(provider).text == "recovered"
    assert sleeps == [1]
    assert len(models.requests) == 2


def test_rate_limit_then_success() -> None:
    sleeps: list[float] = []
    provider, _ = _provider(
        _api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", "Quota exceeded."),
        _response(_text("ok")),
        sleeps=sleeps,
    )

    assert _generate(provider).text == "ok"
    assert sleeps == [1]


def test_server_errors_exhaust_retries() -> None:
    sleeps: list[float] = []
    overloaded = _api_error(errors.ServerError, 503, "UNAVAILABLE", "The model is overloaded.")
    provider, models = _provider(overloaded, overloaded, overloaded, sleeps=sleeps)

    with pytest.raises(LLMProviderError) as excinfo:
        _generate(provider)

    assert sleeps == [1, 2]
    assert len(models.requests) == 3
    assert excinfo.value.retryable is True
    assert "503" in excinfo.value.message


@pytest.mark.parametrize(
    ("code", "status", "hint"),
    [
        (400, "INVALID_ARGUMENT", "invalid"),
        (401, "UNAUTHENTICATED", "GEMINI_API_KEY"),
        (403, "PERMISSION_DENIED", "GEMINI_API_KEY"),
        (404, "NOT_FOUND", "GEMINI_MODEL"),
    ],
)
def test_client_errors_are_not_retried(code: int, status: str, hint: str) -> None:
    sleeps: list[float] = []
    error = _api_error(errors.ClientError, code, status, f"API key {API_KEY} rejected.")
    provider, models = _provider(error, sleeps=sleeps)

    with pytest.raises(LLMProviderError) as excinfo:
        _generate(provider)

    assert excinfo.value.retryable is False
    assert str(code) in excinfo.value.message
    assert hint in excinfo.value.message
    assert API_KEY not in excinfo.value.message
    assert sleeps == [] and len(models.requests) == 1


@pytest.mark.parametrize(
    "error",
    [httpx2.ConnectError("connection refused"), httpx2.ReadTimeout("timed out"), TimeoutError()],
)
def test_network_errors_are_retried(error: Exception) -> None:
    sleeps: list[float] = []
    provider, _ = _provider(error, _response(_text("ok")), sleeps=sleeps)

    assert _generate(provider).text == "ok"
    assert sleeps == [1]


def test_unexpected_sdk_exception_is_wrapped() -> None:
    provider, _ = _provider(ValueError("internal detail"))

    with pytest.raises(LLMProviderError) as excinfo:
        _generate(provider)

    assert excinfo.value.retryable is False
    assert "internal detail" not in excinfo.value.message


def test_empty_response_is_retryable() -> None:
    sleeps: list[float] = []
    provider, _ = _provider(_response(), _response(_text("ok")), sleeps=sleeps)

    assert _generate(provider).text == "ok"
    assert sleeps == [1]


def test_max_retries_zero_does_not_retry() -> None:
    sleeps: list[float] = []
    error = _api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", "Quota exceeded.")
    provider, _ = _provider(error, sleeps=sleeps, max_retries=0)

    with pytest.raises(LLMProviderError):
        _generate(provider)

    assert sleeps == []
