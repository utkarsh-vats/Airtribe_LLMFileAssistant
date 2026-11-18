"""Agent loop tests: FakeProvider only, no network, fake clock."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from helpers import assert_ok

from assistant.agent import (
    MAX_ITERATIONS_TEXT,
    Agent,
    AgentEvent,
    ModelTurn,
    ToolCallFinished,
    ToolCallStarted,
)
from assistant.prompts import DEFAULT_SYSTEM_PROMPT
from assistant.providers import LLMProviderError, LLMResponse, Message, ToolCall
from assistant.providers.fake import FakeProvider
from assistant.registry import get_tool_schemas

reply, call, parallel = FakeProvider.reply, FakeProvider.call, FakeProvider.parallel


class StubDispatch:
    """Records calls and returns a success envelope echoing the call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __call__(self, name: str, args: object) -> dict:
        self.calls.append((name, args))
        return {"success": True, "data": {"tool": name, "args": args}, "error": None}


class FakeClock:
    """Each reading advances by 0.25 s, so every tool call measures exactly 250 ms."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.25
        return self.now


class FlakyProvider(FakeProvider):
    """FakeProvider that raises a provider error on the given (1-based) generate calls."""

    def __init__(self, script: Sequence[LLMResponse], fail_on: set[int]) -> None:
        super().__init__(script)
        self.fail_on = fail_on
        self.attempts = 0

    def generate(self, system_prompt: str, messages: Sequence[Message], tools: list[dict]):
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise LLMProviderError("Gemini API error 503: server error.", retryable=True)
        return super().generate(system_prompt, messages, tools)


def _agent(
    script: Sequence[LLMResponse], **kwargs: object
) -> tuple[Agent, FakeProvider, StubDispatch]:
    provider = FakeProvider(script)
    dispatch = StubDispatch()
    kwargs.setdefault("dispatch", dispatch)
    return Agent(provider, clock=FakeClock(), **kwargs), provider, dispatch  # type: ignore[arg-type]


def _assert_history_valid(history: list[Message]) -> None:
    """Every assistant tool call is answered by the immediately following tool message."""
    for index, message in enumerate(history):
        if message.role == "assistant" and message.tool_calls:
            follower = history[index + 1]
            assert follower.role == "tool"
            assert [r.call_id for r in follower.tool_results] == [c.id for c in message.tool_calls]
        if message.role == "tool":
            assert history[index - 1].role == "assistant" and history[index - 1].tool_calls
    assert history[-1].role == "assistant" and not history[-1].tool_calls


# --- Basic flow ---------------------------------------------------------------------------


def test_plain_reply_completes_in_one_call() -> None:
    agent, provider, dispatch = _agent([reply("Hello!")])

    result = agent.run("hi")

    assert result.text == "Hello!"
    assert result.stopped_reason == "completed"
    assert result.iterations == 1
    assert result.tool_calls == [] and result.error is None
    assert len(provider.calls) == 1 and dispatch.calls == []
    assert [m.role for m in agent.history] == ["user", "assistant"]


def test_one_tool_call_then_reply() -> None:
    agent, provider, dispatch = _agent(
        [call("list_files", {"directory": "resumes"}), reply("Found 2 files.")]
    )

    result = agent.run("What files are there?")

    assert result.text == "Found 2 files."
    assert result.iterations == 2
    assert dispatch.calls == [("list_files", {"directory": "resumes"})]
    _user, assistant, tool, final = agent.history
    assert assistant.tool_calls[0].id == "call_1"
    assert tool.role == "tool"
    (tool_result,) = tool.tool_results
    assert (tool_result.call_id, tool_result.name) == ("call_1", "list_files")
    assert tool_result.result == dispatch("list_files", {"directory": "resumes"})
    # The envelope went back to the model on the next call.
    sent = provider.calls[1].messages
    assert sent[-1].tool_results[0].result["data"]["tool"] == "list_files"
    assert final.text == "Found 2 files."
    (executed,) = result.tool_calls
    assert executed[0].name == "list_files" and executed[1]["success"] is True


def test_parallel_calls_produce_one_tool_message_in_call_order() -> None:
    agent, _, dispatch = _agent(
        [
            parallel(
                ("search_in_file", {"filepath": "a.pdf", "keyword": "python"}),
                ("search_in_file", {"filepath": "b.txt", "keyword": "python"}),
                ("search_in_file", {"filepath": "c.docx", "keyword": "python"}),
            ),
            reply("done"),
        ]
    )

    agent.run("Who knows Python?")

    tool_messages = [m for m in agent.history if m.role == "tool"]
    assert len(tool_messages) == 1
    assert [r.call_id for r in tool_messages[0].tool_results] == ["call_1", "call_2", "call_3"]
    assert [args["filepath"] for _, args in dispatch.calls] == ["a.pdf", "b.txt", "c.docx"]
    _assert_history_valid(agent.history)


def test_multi_step_chain() -> None:
    agent, provider, dispatch = _agent(
        [
            call("list_files", {"directory": "resumes"}),
            parallel(
                (
                    "search_in_file",
                    {"filepath": "resumes/a.pdf", "keyword": "go", "whole_word": True},
                ),
                (
                    "search_in_file",
                    {"filepath": "resumes/b.txt", "keyword": "go", "whole_word": True},
                ),
            ),
            reply("Only a.pdf mentions Go."),
        ]
    )

    result = agent.run("Who knows Go?")

    assert result.iterations == 3 and len(provider.calls) == 3
    assert [name for name, _ in dispatch.calls] == [
        "list_files",
        "search_in_file",
        "search_in_file",
    ]
    assert [m.role for m in agent.history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    _assert_history_valid(agent.history)


def test_provider_state_is_copied_through_and_sent_back() -> None:
    state_1, state_2 = {"gemini": "content-1"}, {"gemini": "content-2"}
    first = LLMResponse(
        text=None,
        tool_calls=[ToolCall("x1", "list_files", {"directory": ""})],
        provider_state=state_1,
    )
    second = LLMResponse(text="ok", provider_state=state_2)
    agent, provider, _ = _agent([first, second])

    agent.run("list")

    assistants = [m for m in agent.history if m.role == "assistant"]
    assert assistants[0].provider_state is state_1
    assert assistants[1].provider_state is state_2
    sent_assistant = provider.calls[1].messages[1]
    assert sent_assistant.role == "assistant" and sent_assistant.provider_state == state_1


def test_system_prompt_and_tools_sent_on_every_call() -> None:
    agent, provider, _ = _agent([call("list_files", {"directory": ""}), reply("ok")])

    agent.run("list")

    assert agent.system_prompt == DEFAULT_SYSTEM_PROMPT
    for recorded in provider.calls:
        assert recorded.system_prompt == DEFAULT_SYSTEM_PROMPT
        assert recorded.tools == get_tool_schemas()


def test_custom_system_prompt_and_tools() -> None:
    tools = [{"name": "only", "description": "d", "parameters": {"type": "object"}}]
    agent, provider, _ = _agent([reply("ok")], system_prompt="Be brief.", tools=tools)

    agent.run("hi")

    assert provider.calls[0].system_prompt == "Be brief."
    assert provider.calls[0].tools == tools


# --- Model mistakes, cap, errors ----------------------------------------------------------


def test_model_mistakes_are_fed_back_and_loop_continues(workspace: Path) -> None:
    (workspace / "cv.txt").write_text("Jane Roe, Python", encoding="utf-8")
    agent = Agent(
        FakeProvider(
            [
                call("delete_file", {"filepath": "cv.txt"}),
                call("read_file", {"path": "cv.txt"}),
                call("read_file", {"filepath": "cv.txt"}),
                reply("Jane Roe knows Python (cv.txt)."),
            ]
        ),
        clock=FakeClock(),
    )

    result = agent.run("Read cv.txt")

    assert result.stopped_reason == "completed"
    assert result.text == "Jane Roe knows Python (cv.txt)."
    codes = [r["error"]["code"] if not r["success"] else "OK" for _, r in result.tool_calls]
    assert codes == ["UNKNOWN_TOOL", "INVALID_ARGUMENT", "OK"]
    _assert_history_valid(agent.history)


def test_max_iterations_stops_with_valid_history() -> None:
    always_calls = [call("list_files", {"directory": ""}) for _ in range(10)]
    agent, provider, dispatch = _agent(always_calls, max_iterations=3)

    result = agent.run("loop forever")

    assert result.stopped_reason == "max_iterations"
    assert result.text == MAX_ITERATIONS_TEXT
    assert result.iterations == 3 and len(provider.calls) == 3
    assert len(dispatch.calls) == 2  # calls from the last allowed turn are not executed
    assert len(result.tool_calls) == 2
    last_tool = [m for m in agent.history if m.role == "tool"][-1]
    assert last_tool.tool_results[0].result["error"]["code"] == "NOT_EXECUTED"
    assert agent.history[-1].text == MAX_ITERATIONS_TEXT
    _assert_history_valid(agent.history)


def test_max_iterations_one_allows_a_plain_reply() -> None:
    agent, _, _ = _agent([reply("ok")], max_iterations=1)

    assert agent.run("hi").stopped_reason == "completed"


def test_max_iterations_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_TOOL_ITERATIONS", "4")
    assert _agent([])[0].max_iterations == 4

    monkeypatch.delenv("MAX_TOOL_ITERATIONS")
    assert _agent([])[0].max_iterations == 10

    assert _agent([], max_iterations=2)[0].max_iterations == 2


@pytest.mark.parametrize("raw", ["0", "-1", "ten"])
def test_invalid_max_iterations_env(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("MAX_TOOL_ITERATIONS", raw)

    with pytest.raises(ValueError, match=r"(?i)max_tool_iterations|max_iterations"):
        _agent([])


def test_provider_error_rolls_back_history_and_reports() -> None:
    provider = FlakyProvider(
        [reply("first answer"), call("list_files", {"directory": ""}), reply("third answer")],
        fail_on={3},
    )
    dispatch = StubDispatch()
    agent = Agent(provider, dispatch=dispatch, clock=FakeClock())
    agent.run("first question")
    before = list(agent.history)

    result = agent.run("second question")  # tool call succeeds, then the 2nd generate fails

    assert result.stopped_reason == "provider_error"
    assert result.error == "Gemini API error 503: server error."
    assert result.text is None
    assert result.iterations == 2
    assert len(result.tool_calls) == 1  # the tool did run before the failure
    assert agent.history == before

    later = agent.run("third question")
    assert later.stopped_reason == "completed" and later.text == "third answer"
    assert [m.text for m in agent.history if m.role == "user"] == [
        "first question",
        "third question",
    ]


def test_provider_error_on_first_call() -> None:
    agent = Agent(FlakyProvider([], fail_on={1}), dispatch=StubDispatch(), clock=FakeClock())

    result = agent.run("hi")

    assert result.stopped_reason == "provider_error"
    assert result.iterations == 1
    assert agent.history == []


def test_other_exceptions_propagate() -> None:
    def broken_dispatch(name: str, args: object) -> dict:
        raise RuntimeError("bug in dispatch")

    agent, _, _ = _agent([call("list_files", {"directory": ""})], dispatch=broken_dispatch)

    with pytest.raises(RuntimeError, match="bug in dispatch"):
        agent.run("list")


# --- History, events, usage ---------------------------------------------------------------


def test_history_persists_across_runs_and_reset_clears_it() -> None:
    agent, provider, _ = _agent([reply("one"), reply("two")])

    agent.run("first")
    agent.run("second")

    assert [m.text for m in agent.history] == ["first", "one", "second", "two"]
    assert [m.text for m in provider.calls[1].messages] == ["first", "one", "second"]
    agent.reset()
    assert agent.history == []


def test_events_in_order_with_fake_clock_durations() -> None:
    events: list[AgentEvent] = []
    usage = {"prompt_tokens": 10, "total_tokens": 12}
    first = LLMResponse(
        text="Checking.", tool_calls=[ToolCall("x1", "list_files", {"directory": ""})], usage=usage
    )
    agent, _, dispatch = _agent([first, reply("done")], on_event=events.append)

    agent.run("list")

    call_1 = first.tool_calls[0]
    assert events == [
        ModelTurn(1, "Checking.", [call_1], usage),
        ToolCallStarted(call_1),
        ToolCallFinished(call_1, dispatch("list_files", {"directory": ""}), 250.0),
        ModelTurn(2, "done", [], None),
    ]


def test_parallel_call_events_and_durations() -> None:
    events: list[AgentEvent] = []
    agent, _, _ = _agent(
        [parallel(("read_file", {"filepath": "a"}), ("read_file", {"filepath": "b"})), reply("ok")],
        on_event=events.append,
    )

    agent.run("read")

    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "ModelTurn",
        "ToolCallStarted",
        "ToolCallFinished",
        "ToolCallStarted",
        "ToolCallFinished",
        "ModelTurn",
    ]
    assert [e.duration_ms for e in events if isinstance(e, ToolCallFinished)] == [250.0, 250.0]


def test_usage_is_summed_and_missing_usage_is_handled() -> None:
    agent, _, _ = _agent(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("x1", "list_files", {"directory": ""})],
                usage={"prompt_tokens": 100, "candidates_tokens": 10, "total_tokens": 110},
            ),
            LLMResponse(text=None, tool_calls=[ToolCall("x2", "list_files", {"directory": ""})]),
            LLMResponse(
                text="ok", usage={"prompt_tokens": 150, "total_tokens": 160, "note": "ignored"}
            ),
        ]
    )

    result = agent.run("list")

    assert result.usage == {"prompt_tokens": 250, "candidates_tokens": 10, "total_tokens": 270}


def test_no_usage_at_all_is_empty_dict() -> None:
    agent, _, _ = _agent([reply("ok")])

    assert agent.run("hi").usage == {}


# --- End to end with the real dispatcher --------------------------------------------------


@pytest.fixture
def resumes(
    workspace: Path, make_pdf: Callable[..., Path], make_docx: Callable[..., Path]
) -> list[str]:
    make_pdf(
        "resumes/resume_john_doe.pdf",
        [["John Doe", "Senior Backend Python Developer"], ["Projects: Python packages"]],
    )
    make_docx(
        "resumes/resume_elena_rodriguez.docx",
        before=["Elena Rodriguez", "Java / Spring Developer"],
        table=[["Languages", "Java 17, Kotlin"]],
    )
    (workspace / "resumes" / "resume_kenji_tanaka.txt").write_text(
        "Kenji Tanaka\nML Engineer\nSkills: python, PyTorch\n", encoding="utf-8"
    )
    return [
        "resumes/resume_elena_rodriguez.docx",
        "resumes/resume_john_doe.pdf",
        "resumes/resume_kenji_tanaka.txt",
    ]


def _e2e_agent(script: Sequence[LLMResponse]) -> Agent:
    return Agent(FakeProvider(script), clock=FakeClock())


def test_e2e_read_all_resumes(resumes: list[str]) -> None:
    agent = _e2e_agent(
        [
            call("list_files", {"directory": "resumes"}),
            parallel(*[("read_file", {"filepath": path}) for path in resumes]),
            reply("Read 3 resumes: Elena Rodriguez, John Doe, Kenji Tanaka."),
        ]
    )

    result = agent.run("Read all resumes in the resumes folder")

    assert result.stopped_reason == "completed"
    (_listing, listed), *reads = result.tool_calls
    assert [f["path"] for f in assert_ok(listed)["files"]] == resumes
    contents = {c.args["filepath"]: assert_ok(r)["content"] for c, r in reads}
    assert "Elena Rodriguez" in contents["resumes/resume_elena_rodriguez.docx"]
    assert "Languages | Java 17, Kotlin" in contents["resumes/resume_elena_rodriguez.docx"]
    assert "John Doe" in contents["resumes/resume_john_doe.pdf"]
    assert "Kenji Tanaka" in contents["resumes/resume_kenji_tanaka.txt"]
    _assert_history_valid(agent.history)


def test_e2e_find_resumes_mentioning_python(resumes: list[str]) -> None:
    agent = _e2e_agent(
        [
            call("list_files", {"directory": "resumes"}),
            parallel(*[("search_in_file", {"filepath": p, "keyword": "Python"}) for p in resumes]),
            reply("Python: resume_john_doe.pdf, resume_kenji_tanaka.txt."),
        ]
    )

    result = agent.run("Find resumes mentioning Python")

    searches = [r for c, r in result.tool_calls if c.name == "search_in_file"]
    hits = {assert_ok(r)["path"]: assert_ok(r)["total_matches"] for r in searches}
    assert hits == {
        "resumes/resume_elena_rodriguez.docx": 0,
        "resumes/resume_john_doe.pdf": 2,
        "resumes/resume_kenji_tanaka.txt": 1,
    }
    assert result.text == "Python: resume_john_doe.pdf, resume_kenji_tanaka.txt."


def test_e2e_create_summary_file(resumes: list[str], workspace: Path) -> None:
    summary = "# John Doe\n\n- Senior Backend Python Developer\n"
    agent = _e2e_agent(
        [
            call("read_file", {"filepath": "resumes/resume_john_doe.pdf"}),
            call(
                "write_file",
                {"filepath": "summaries/resume_john_doe_summary.md", "content": summary},
            ),
            reply("Saved summaries/resume_john_doe_summary.md."),
        ]
    )

    result = agent.run("Create a summary file for resume_john_doe.pdf")

    assert [c.name for c, _ in result.tool_calls] == ["read_file", "write_file"]
    written = assert_ok(result.tool_calls[1][1])
    assert written["path"] == "summaries/resume_john_doe_summary.md"
    assert written["created_dirs"] == ["summaries"]
    target = workspace / "summaries" / "resume_john_doe_summary.md"
    assert target.read_text(encoding="utf-8") == summary
    assert result.stopped_reason == "completed"
