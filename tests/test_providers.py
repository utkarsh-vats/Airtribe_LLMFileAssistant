import subprocess
import sys
from pathlib import Path

import pytest

from assistant.providers import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    available_providers,
    get_provider,
)
from assistant.providers.fake import FakeProvider

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = [{"name": "read_file", "description": "Read a file.", "parameters": {"type": "object"}}]


# --- Registry -----------------------------------------------------------------------------


def test_get_provider_fake() -> None:
    provider = get_provider("fake", script=[FakeProvider.reply("hi")])

    assert isinstance(provider, FakeProvider)
    assert isinstance(provider, LLMProvider)
    assert provider.name == "fake"
    assert provider.generate("sys", [], TOOLS).text == "hi"


def test_available_providers() -> None:
    providers = available_providers()

    assert {"fake", "gemini"} <= set(providers)
    assert providers == sorted(providers)


@pytest.mark.parametrize("name", ["openai", "", "FAKE", None])
def test_unknown_provider_lists_valid_names(name: object) -> None:
    with pytest.raises(LLMProviderError) as excinfo:
        get_provider(name)  # type: ignore[arg-type]

    for valid in available_providers():
        assert valid in excinfo.value.message


def test_importing_providers_does_not_import_google_genai() -> None:
    code = (
        "import sys\n"
        "import assistant.providers\n"
        "from assistant.providers import get_provider\n"
        "get_provider('fake')\n"
        "print('google.genai' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"


def test_unimplemented_gemini_raises_provider_error_not_import_error() -> None:
    # Placeholder until step 8b adds assistant/providers/gemini.py.
    with pytest.raises(LLMProviderError) as excinfo:
        get_provider("gemini")

    assert not isinstance(excinfo.value, ImportError)
    assert "gemini" in excinfo.value.message


# --- FakeProvider -------------------------------------------------------------------------


def test_fake_returns_responses_in_order() -> None:
    provider = FakeProvider(
        [FakeProvider.call("list_files", {"directory": "resumes"}), FakeProvider.reply("done")]
    )

    first = provider.generate("sys", [Message(role="user", text="hi")], TOOLS)
    second = provider.generate("sys", [Message(role="user", text="hi")], TOOLS)

    assert first.text is None
    assert first.tool_calls == [
        ToolCall(id="call_1", name="list_files", args={"directory": "resumes"})
    ]
    assert second.text == "done"
    assert second.tool_calls == []


def test_fake_records_deep_copied_calls() -> None:
    provider = FakeProvider([FakeProvider.reply("ok")])
    args = {"filepath": "a.txt"}
    messages = [Message(role="assistant", tool_calls=[ToolCall("call_1", "read_file", args)])]
    tools = [dict(TOOLS[0])]

    provider.generate("system prompt", messages, tools)
    messages.append(Message(role="user", text="later"))
    args["filepath"] = "changed.txt"
    tools[0]["name"] = "changed"

    (recorded,) = provider.calls
    assert recorded.system_prompt == "system prompt"
    assert len(recorded.messages) == 1
    assert recorded.messages[0].tool_calls[0].args == {"filepath": "a.txt"}
    assert recorded.tools[0]["name"] == "read_file"


def test_fake_exhausted_script_raises() -> None:
    provider = FakeProvider([FakeProvider.reply("only one")])
    provider.generate("sys", [], TOOLS)

    with pytest.raises(LLMProviderError, match="FakeProvider script exhausted"):
        provider.generate("sys", [], TOOLS)

    assert len(provider.calls) == 2  # the failing call is still recorded


def test_fake_ids_are_deterministic_per_instance() -> None:
    def ids(provider: FakeProvider) -> list[str]:
        return [c.id for _ in range(3) for c in provider.generate("s", [], TOOLS).tool_calls]

    script = [
        FakeProvider.call("list_files", {"directory": ""}),
        FakeProvider.call("read_file", {"filepath": "a.txt"}, id="explicit"),
        FakeProvider.call("read_file", {"filepath": "b.txt"}),
    ]

    assert ids(FakeProvider(script)) == ["call_1", "explicit", "call_2"]
    assert ids(FakeProvider(script)) == ["call_1", "explicit", "call_2"]  # fresh counter


def test_fake_parallel_calls_in_one_response() -> None:
    provider = FakeProvider(
        [
            FakeProvider.parallel(
                ("search_in_file", {"filepath": "a.txt", "keyword": "python"}),
                ("search_in_file", {"filepath": "b.pdf", "keyword": "python"}),
            )
        ]
    )

    response = provider.generate("sys", [], TOOLS)

    assert [(c.id, c.args["filepath"]) for c in response.tool_calls] == [
        ("call_1", "a.txt"),
        ("call_2", "b.pdf"),
    ]


def test_fake_does_not_mutate_scripted_responses() -> None:
    scripted = FakeProvider.call("list_files", {"directory": ""})

    FakeProvider([scripted]).generate("sys", [], TOOLS)

    assert scripted.tool_calls[0].id == ""


# --- Types --------------------------------------------------------------------------------


def test_dataclass_list_defaults_are_not_shared() -> None:
    first, second = Message(role="assistant"), Message(role="assistant")
    first.tool_calls.append(ToolCall("call_1", "read_file", {}))
    first.tool_results.append(ToolResult("call_1", "read_file", {"success": True}))

    assert second.tool_calls == []
    assert second.tool_results == []
    assert LLMResponse(text="a").tool_calls is not LLMResponse(text="b").tool_calls


def test_message_defaults() -> None:
    message = Message(role="user", text="hello")

    assert message.tool_calls == [] and message.tool_results == []
    assert message.provider_state is None


def test_provider_error_carries_message_and_retryable() -> None:
    error = LLMProviderError("Rate limited, try again shortly.", retryable=True)

    assert error.message == str(error) == "Rate limited, try again shortly."
    assert error.retryable is True
    assert LLMProviderError("Bad API key.").retryable is False


def test_llm_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]
