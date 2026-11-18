"""Terminal rendering for the CLI, driven only by agent events (rich).

All user/file-derived strings are wrapped in ``rich.text.Text`` so content such as "[bold]" in a
resume is shown literally instead of being parsed as console markup.
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from assistant.agent import AgentEvent, AgentResult, ModelTurn, ToolCallFinished, ToolCallStarted

CHECK = "\N{CHECK MARK}"
CROSS = "\N{BALLOT X}"
ARROW = "\N{RIGHTWARDS ARROW}"
DOT = "\N{MIDDLE DOT}"
ELLIPSIS = "\N{HORIZONTAL ELLIPSIS}"
MAX_ARG_CHARS = 40


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def shorten(text: str, limit: int = MAX_ARG_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + ELLIPSIS


def format_args(args: dict) -> str:
    """``key='value', ...`` on one line, long values shortened."""
    return ", ".join(f"{key}={shorten(repr(value))}" for key, value in args.items())


def summarize_result(name: str, result: dict) -> str:
    """One-line, human-readable summary of a dispatcher envelope."""
    if not result.get("success"):
        error = result.get("error") or {}
        return f"{CROSS} {error.get('code', 'ERROR')}: {error.get('message', 'unknown error')}"
    data = result.get("data") or {}
    try:
        if name == "read_file":
            meta = data["metadata"]
            parts = [_plural(meta["char_count"], "char")]
            if "page_count" in meta:
                parts.append(_plural(meta["page_count"], "page"))
            parts += [f"warning: {w}" for w in meta.get("warnings", [])]
            return f"{CHECK} " + f" {DOT} ".join(parts)
        if name == "list_files":
            return f"{CHECK} {_plural(data['count'], 'file')}"
        if name == "search_in_file":
            total = data["total_matches"]
            summary = {0: "no matches", 1: "1 match"}.get(total, f"{total:,} matches")
            if data.get("truncated"):
                summary += f" (showing {data['returned_matches']})"
            return f"{CHECK} {summary}"
        if name == "write_file":
            details = f"{data['size_bytes']:,} B" + (", overwritten" if data["overwritten"] else "")
            return f"{CHECK} wrote {data['path']} ({details})"
    except KeyError, TypeError:
        pass
    return f"{CHECK} ok"


class Renderer:
    """Turns agent events and results into console output."""

    def __init__(self, console: Console, *, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self._status: Status | None = None
        self._pending_calls = 0

    # --- Spinner --------------------------------------------------------------------------

    def start_thinking(self) -> None:
        if self._status is None and self.console.is_terminal:
            self._status = self.console.status(Text(f"Thinking{ELLIPSIS}"))
            self._status.start()

    def stop_thinking(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    # --- Output ---------------------------------------------------------------------------

    def banner(self, provider: str, model: str, workspace: str, tools: list[str]) -> None:
        body = Text()
        for label, value in (
            ("provider", provider),
            ("model", model),
            ("workspace", workspace),
            ("tools", ", ".join(tools)),
        ):
            body.append(f"{label:<10}", style="bold")
            body.append(f"{value}\n")
        body.rstrip()
        self.console.print(Panel(body, title="LLM File Assistant", border_style="cyan"))

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, ModelTurn):
            self.stop_thinking()
            if event.tool_calls:
                self._pending_calls = len(event.tool_calls)
                self.console.print(Text(f"Step {event.iteration}", style="bold cyan"))
                if event.text:
                    self.console.print(Text(event.text.strip(), style="dim"))
        elif isinstance(event, ToolCallStarted):
            line = Text(f"  {ARROW} ", style="cyan")
            line.append(event.call.name, style="bold")
            line.append(f"({format_args(event.call.args)})")
            self.console.print(line)
        elif isinstance(event, ToolCallFinished):
            summary = summarize_result(event.call.name, event.result)
            style = "green" if event.result.get("success") else "red"
            line = Text(f"    {summary}", style=style)
            line.append(f"  {event.duration_ms:,.0f} ms", style="dim")
            self.console.print(line)
            if self.verbose:
                self.console.print_json(data=event.result)
            self._pending_calls -= 1
            if self._pending_calls <= 0:
                self.start_thinking()

    def result(self, result: AgentResult) -> None:
        self.stop_thinking()
        if result.stopped_reason == "provider_error":
            self.console.print(
                Panel(
                    Text(result.error or "Unknown provider error."),
                    title="Provider error",
                    border_style="red",
                )
            )
            return
        if result.stopped_reason == "max_iterations":
            self.console.print(
                Text(result.text or "Stopped at the iteration limit.", style="yellow")
            )
        else:
            answer = Markdown(result.text) if result.text else Text("(no answer)", style="dim")
            self.console.print(Panel(answer, title="Answer", border_style="green"))
        self.console.print(Text(self._footer(result), style="dim"))

    @staticmethod
    def _footer(result: AgentResult) -> str:
        tokens = result.usage.get("total_tokens")
        token_text = f"{tokens:,} tokens" if tokens is not None else "tokens n/a"
        return f" {DOT} ".join(
            [
                _plural(result.iterations, "iteration"),
                _plural(len(result.tool_calls), "tool call"),
                token_text,
            ]
        )
