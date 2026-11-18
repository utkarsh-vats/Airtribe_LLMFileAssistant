"""CLI tests: FakeProvider via provider_factory, tmp workspace, recorded rich console, no .env."""

import io
import logging
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from rich.console import Console

import llm_file_assistant
from assistant.providers import LLMProviderError, LLMResponse
from assistant.providers.fake import FakeProvider
from assistant.registry import TOOL_NAMES
from assistant.render import format_args, summarize_result

REPO_ROOT = Path(__file__).resolve().parent.parent
reply, call = FakeProvider.reply, FakeProvider.call


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root, pypdf = logging.getLogger(), logging.getLogger("pypdf")
    levels = root.level, pypdf.level
    yield
    root.setLevel(levels[0])
    pypdf.setLevel(levels[1])


@pytest.fixture
def resumes(workspace: Path) -> Path:
    (workspace / "resumes").mkdir()
    (workspace / "resumes" / "jane.txt").write_text("Jane Roe\nPython\n", encoding="utf-8")
    (workspace / "resumes" / "john.md").write_text("John Doe\nJava\n", encoding="utf-8")
    return workspace


def _console() -> Console:
    return Console(record=True, file=io.StringIO(), width=120, color_system=None)


def _run(
    argv: Sequence[str],
    script: Sequence[LLMResponse] = (),
    inputs: Sequence[object] = (),
    provider: FakeProvider | None = None,
) -> tuple[int, str, FakeProvider]:
    provider = provider or FakeProvider(script)
    console = _console()
    pending = list(inputs)

    def input_fn(prompt: str) -> str:
        item = pending.pop(0) if pending else EOFError()
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

    code = llm_file_assistant.main(
        list(argv),
        provider_factory=lambda name, **kwargs: provider,
        input_fn=input_fn,
        console=console,
        load_env=False,
    )
    return code, console.export_text(), provider


# --- summarize_result / format_args -------------------------------------------------------


def _ok(data: dict) -> dict:
    return {"success": True, "data": data, "error": None}


def _read(chars: int, pages: int | None = None, warnings: Sequence[str] = ()) -> dict:
    meta: dict = {"char_count": chars, "warnings": list(warnings)}
    if pages is not None:
        meta["page_count"] = pages
    return _ok({"content": "x", "metadata": meta})


@pytest.mark.parametrize(
    ("name", "result", "expected"),
    [
        ("read_file", _read(1234, pages=2), "\N{CHECK MARK} 1,234 chars \N{MIDDLE DOT} 2 pages"),
        ("read_file", _read(1, pages=1), "\N{CHECK MARK} 1 char \N{MIDDLE DOT} 1 page"),
        ("read_file", _read(980), "\N{CHECK MARK} 980 chars"),
        (
            "read_file",
            _read(0, pages=1, warnings=["No extractable text."]),
            "\N{CHECK MARK} 0 chars \N{MIDDLE DOT} 1 page "
            "\N{MIDDLE DOT} warning: No extractable text.",
        ),
        ("list_files", _ok({"files": [], "count": 8}), "\N{CHECK MARK} 8 files"),
        ("list_files", _ok({"files": [], "count": 1}), "\N{CHECK MARK} 1 file"),
        (
            "search_in_file",
            _ok({"total_matches": 9, "truncated": False}),
            "\N{CHECK MARK} 9 matches",
        ),
        ("search_in_file", _ok({"total_matches": 1, "truncated": False}), "\N{CHECK MARK} 1 match"),
        (
            "search_in_file",
            _ok({"total_matches": 0, "truncated": False}),
            "\N{CHECK MARK} no matches",
        ),
        (
            "search_in_file",
            _ok({"total_matches": 120, "truncated": True, "returned_matches": 50}),
            "\N{CHECK MARK} 120 matches (showing 50)",
        ),
        (
            "write_file",
            _ok({"path": "summaries/x.md", "size_bytes": 131, "overwritten": True}),
            "\N{CHECK MARK} wrote summaries/x.md (131 B, overwritten)",
        ),
        (
            "write_file",
            _ok({"path": "summaries/x.md", "size_bytes": 2048, "overwritten": False}),
            "\N{CHECK MARK} wrote summaries/x.md (2,048 B)",
        ),
        ("list_files", _ok({"unexpected": True}), "\N{CHECK MARK} ok"),
    ],
)
def test_summarize_result_success(name: str, result: dict, expected: str) -> None:
    assert summarize_result(name, result) == expected


def test_summarize_result_error() -> None:
    result = {
        "success": False,
        "data": None,
        "error": {"code": "FILE_NOT_FOUND", "message": "File 'x.pdf' does not exist."},
    }

    assert summarize_result("read_file", result) == (
        "\N{BALLOT X} FILE_NOT_FOUND: File 'x.pdf' does not exist."
    )


def test_format_args_shortens_long_values() -> None:
    formatted = format_args({"filepath": "summaries/a.md", "content": "x" * 200})

    assert formatted.startswith("filepath='summaries/a.md', content='xxx")
    assert formatted.endswith("\N{HORIZONTAL ELLIPSIS}")
    assert len(formatted) < 100


# --- One-shot mode ------------------------------------------------------------------------


def test_one_shot_prints_tool_line_summary_and_answer(resumes: Path) -> None:
    code, out, provider = _run(
        ["--workspace", str(resumes), "-q", "What files are in resumes?"],
        [call("list_files", {"directory": "resumes"}), reply("There are **2** resumes.")],
    )

    assert code == 0
    assert "Step 1" in out
    assert "\N{RIGHTWARDS ARROW} list_files(directory='resumes')" in out
    assert "\N{CHECK MARK} 2 files" in out
    assert "There are 2 resumes." in out  # Markdown rendered, ** removed
    assert "2 iterations \N{MIDDLE DOT} 1 tool call \N{MIDDLE DOT} tokens n/a" in out
    assert str(resumes.resolve()) in out  # banner shows the absolute workspace
    assert provider.calls[0].messages[0].text == "What files are in resumes?"


def test_one_shot_shows_tool_errors(resumes: Path) -> None:
    code, out, _ = _run(
        ["--workspace", str(resumes), "-q", "read it"],
        [call("read_file", {"filepath": "missing.pdf"}), reply("It does not exist.")],
    )

    assert code == 0
    assert "\N{BALLOT X} FILE_NOT_FOUND" in out


def test_one_shot_provider_error_exits_1(resumes: Path) -> None:
    code, out, _ = _run(["--workspace", str(resumes), "-q", "hi"], [])  # script exhausted

    assert code == 1
    assert "Provider error" in out
    assert "FakeProvider script exhausted" in out


def test_one_shot_max_iterations_exits_1(resumes: Path) -> None:
    script = [call("list_files", {"directory": ""}) for _ in range(3)]

    code, out, _ = _run(
        ["--workspace", str(resumes), "--max-iterations", "2", "-q", "loop"], script
    )

    assert code == 1
    assert "limit of model turns" in out


def test_verbose_prints_json_result(resumes: Path) -> None:
    _, out, _ = _run(
        ["--workspace", str(resumes), "--verbose", "-q", "list"],
        [call("list_files", {"directory": "resumes"}), reply("ok")],
    )

    assert '"count": 2' in out
    assert '"name": "jane.txt"' in out


def test_file_content_is_not_parsed_as_markup(resumes: Path) -> None:
    _, out, _ = _run(
        ["--workspace", str(resumes), "-q", "x"],
        [call("read_file", {"filepath": "[bold]x[/bold].txt"}), reply("done")],
    )

    assert "[bold]x[/bold].txt" in out


# --- Startup errors -----------------------------------------------------------------------


def test_missing_workspace_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)

    code, out, provider = _run(["--workspace", str(tmp_path / "nope"), "-q", "hi"])

    assert code == 2
    assert "Workspace directory not found" in out
    assert provider.calls == []
    assert "WORKSPACE_ROOT" not in os.environ


def test_provider_factory_error_exits_2_with_env_hint(resumes: Path) -> None:
    def failing_factory(name: str, **kwargs: object) -> FakeProvider:
        raise LLMProviderError("No Gemini API key: set GEMINI_API_KEY (e.g. in .env).")

    console = _console()
    code = llm_file_assistant.main(
        ["--workspace", str(resumes), "-q", "hi"],
        provider_factory=failing_factory,
        console=console,
        load_env=False,
    )

    out = console.export_text()
    assert code == 2
    assert "GEMINI_API_KEY" in out
    assert ".env.example" in out


def test_model_flag_is_passed_to_provider(resumes: Path) -> None:
    seen: dict = {}

    def factory(name: str, **kwargs: object) -> FakeProvider:
        seen.update(name=name, **kwargs)
        return FakeProvider([reply("ok")])

    llm_file_assistant.main(
        ["--workspace", str(resumes), "--model", "my-model", "-q", "hi"],
        provider_factory=factory,
        console=_console(),
        load_env=False,
    )

    assert seen == {"name": "gemini", "model": "my-model"}


def test_invalid_max_iterations_exits_2(resumes: Path) -> None:
    code, out, _ = _run(["--workspace", str(resumes), "--max-iterations", "0", "-q", "hi"])

    assert code == 2
    assert "max_iterations" in out


def test_pypdf_logger_is_error_after_startup(resumes: Path) -> None:
    _run(["--workspace", str(resumes), "-q", "hi"], [reply("ok")])

    assert logging.getLogger("pypdf").level == logging.ERROR
    assert logging.getLogger().level == logging.WARNING


def test_debug_flag_sets_debug_level(resumes: Path) -> None:
    _run(["--workspace", str(resumes), "--debug", "-q", "hi"], [reply("ok")])

    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("pypdf").level == logging.ERROR


def test_help_runs_as_a_script() -> None:
    result = subprocess.run(
        [sys.executable, "llm_file_assistant.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for flag in ("--query", "--provider", "--model", "--workspace", "--max-iterations"):
        assert flag in result.stdout


def test_help_via_main_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        llm_file_assistant.main(["--help"], load_env=False)

    assert excinfo.value.code == 0
    assert "--verbose" in capsys.readouterr().out


def test_fake_provider_is_not_a_cli_choice() -> None:
    with pytest.raises(SystemExit) as excinfo:
        llm_file_assistant.build_parser().parse_args(["--provider", "fake"])

    assert excinfo.value.code == 2


# --- REPL ---------------------------------------------------------------------------------


def test_repl_two_turns_share_history(resumes: Path) -> None:
    code, out, provider = _run(
        ["--workspace", str(resumes)],
        [reply("First answer."), reply("Second answer.")],
        inputs=["first question", "follow-up", "/exit"],
    )

    assert code == 0
    assert "First answer." in out and "Second answer." in out
    second_call = provider.calls[1].messages
    assert [m.text for m in second_call] == ["first question", "First answer.", "follow-up"]
    assert "Goodbye!" in out


def test_repl_reset_clears_history(resumes: Path) -> None:
    _, out, provider = _run(
        ["--workspace", str(resumes)],
        [reply("one"), reply("two")],
        inputs=["first", "/reset", "second", "/quit"],
    )

    assert "Conversation history cleared." in out
    assert [m.text for m in provider.calls[1].messages] == ["second"]


def test_repl_tools_lists_all_tools(resumes: Path) -> None:
    _, out, _ = _run(["--workspace", str(resumes)], inputs=["/tools", "/exit"])

    tool_lines = [line for line in out.splitlines() if line.split(" ")[0] in TOOL_NAMES]
    assert [line.split(" ")[0] for line in tool_lines] == list(TOOL_NAMES)
    flat = " ".join(out.split())  # rich wraps long lines at the console width
    assert "read_file Read one file and return its full extracted text" in flat
    assert "write_file Write UTF-8 text to a .txt or .md file in the workspace;" in flat
    assert "List the files in one workspace directory (not recursive" in flat


def test_repl_help_and_unknown_command(resumes: Path) -> None:
    _, out, _ = _run(["--workspace", str(resumes)], inputs=["/help", "/nope", "/exit"])

    assert "/reset" in out and "/tools" in out
    assert "Unknown command '/nope'" in out


@pytest.mark.parametrize("interrupt", [EOFError(), KeyboardInterrupt()])
def test_repl_eof_and_ctrl_c_exit_cleanly(resumes: Path, interrupt: BaseException) -> None:
    code, out, provider = _run(["--workspace", str(resumes)], inputs=[interrupt])

    assert code == 0
    assert "Goodbye!" in out
    assert "Traceback" not in out
    assert provider.calls == []


def test_repl_ignores_empty_input(resumes: Path) -> None:
    _, _, provider = _run(
        ["--workspace", str(resumes)], [reply("ok")], inputs=["", "   ", "hello", "/exit"]
    )

    assert len(provider.calls) == 1
    assert provider.calls[0].messages[0].text == "hello"


def test_repl_ctrl_c_during_a_question_discards_it(resumes: Path) -> None:
    class InterruptingProvider(FakeProvider):
        """Simulates Ctrl-C while the first question is waiting on the model."""

        interrupted = False

        def generate(self, *args: object, **kwargs: object) -> LLMResponse:
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return super().generate(*args, **kwargs)  # type: ignore[arg-type]

    provider = InterruptingProvider([reply("answer")])
    code, out, _ = _run(
        ["--workspace", str(resumes)], inputs=["slow question", "next", "/exit"], provider=provider
    )

    assert code == 0
    assert "Interrupted" in out
    assert [m.text for m in provider.calls[0].messages] == ["next"]  # first question discarded


def test_repl_provider_error_keeps_session_alive(resumes: Path) -> None:
    code, out, _ = _run(["--workspace", str(resumes)], [], inputs=["hi", "/exit"])

    assert code == 0
    assert "Provider error" in out and "Goodbye!" in out
