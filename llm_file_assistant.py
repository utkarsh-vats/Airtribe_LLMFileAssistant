"""Part B: CLI entry point for the LLM file assistant.

Runs an interactive REPL (or a one-shot ``--query``) in which an LLM answers
questions about resumes in the workspace by calling the tools in ``fs_tools``.
Every tool call is rendered to the terminal for visibility. See DESIGN.md for
the agent loop and provider architecture.

    python llm_file_assistant.py                       # interactive REPL
    python llm_file_assistant.py -q "Find resumes mentioning Python"
"""

import argparse
import contextlib
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.text import Text

import fs_tools
from assistant.agent import Agent, AgentResult
from assistant.providers import LLMProvider, LLMProviderError, available_providers, get_provider
from assistant.registry import TOOL_NAMES, TOOL_SPECS
from assistant.render import Renderer

PROMPT = "you> "
ENV_HINT = "Copy .env.example to .env and set GEMINI_API_KEY and GEMINI_MODEL."
HELP_TEXT = """\
Ask a question about the files in the workspace, e.g. "Find resumes mentioning Python".
Commands:
  /help         show this help
  /tools        list the available tools
  /reset        forget the conversation so far
  /exit, /quit  leave (Ctrl-D and Ctrl-C work too)"""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; ``--provider`` choices exclude ``fake``."""
    parser = argparse.ArgumentParser(
        prog="llm_file_assistant.py",
        description="Ask an LLM questions about resume files; it answers by calling file tools.",
    )
    parser.add_argument("-q", "--query", metavar="TEXT", help="ask one question and exit")
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=[name for name in available_providers() if name != "fake"],
        help="LLM provider (default: gemini)",
    )
    parser.add_argument("--model", help="model id; overrides GEMINI_MODEL")
    parser.add_argument(
        "--workspace", metavar="PATH", help="workspace root; overrides WORKSPACE_ROOT"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        metavar="N",
        help="max model turns per question; overrides MAX_TOOL_ITERATIONS (default 10)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="also print each tool result as JSON"
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    return parser


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.WARNING)
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # hide "EOF marker not found" etc.


def _use_utf8_streams() -> None:
    """Let the Windows console print check marks and arrows instead of raising."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _error(console: Console, message: str, hint: str | None = None) -> None:
    console.print(Text(message, style="bold red"))
    if hint:
        console.print(Text(hint, style="yellow"))


def _first_sentence(text: str) -> str:
    head = text.split("\n", 1)[0]
    return head.split(". ", 1)[0].rstrip(".") + "."


def run_query(agent: Agent, renderer: Renderer, text: str) -> AgentResult:
    """Ask the agent one question with the spinner showing, render the outcome, return it."""
    renderer.start_thinking()
    try:
        result = agent.run(text)
    finally:
        renderer.stop_thinking()
    renderer.result(result)
    return result


def repl(agent: Agent, renderer: Renderer, input_fn: Callable[[str], str]) -> int:
    """Run the interactive loop until /exit, /quit, EOF or Ctrl-C at the prompt.

    One ``agent`` serves the whole session, so questions share history. Ctrl-C while a
    question is running discards that turn and returns to the prompt.

    Returns:
        Always 0.
    """
    console = renderer.console
    console.print(Text("Type a question, or /help for commands.", style="dim"))
    while True:
        try:
            line = input_fn(PROMPT).strip()
        except EOFError, KeyboardInterrupt:
            console.print("\nGoodbye!")
            return 0
        if not line:
            continue
        command = line.lower()
        if command in ("/exit", "/quit"):
            console.print("Goodbye!")
            return 0
        if command == "/help":
            console.print(Text(HELP_TEXT))
        elif command == "/tools":
            for spec in TOOL_SPECS:
                console.print(
                    Text.assemble((spec.name, "bold"), f"  {_first_sentence(spec.description)}")
                )
        elif command == "/reset":
            agent.reset()
            console.print(Text("Conversation history cleared.", style="green"))
        elif command.startswith("/"):
            console.print(Text(f"Unknown command {line!r}; type /help.", style="yellow"))
        else:
            checkpoint = len(agent.history)
            try:
                run_query(agent, renderer, line)
            except KeyboardInterrupt:
                del agent.history[checkpoint:]  # drop the half-finished turn
                console.print(Text("\nInterrupted; that question was discarded.", style="yellow"))


def main(
    argv: Sequence[str] | None = None,
    *,
    provider_factory: Callable[..., LLMProvider] = get_provider,
    input_fn: Callable[[str], str] = input,
    console: Console | None = None,
    load_env: bool = True,
) -> int:
    """Parse arguments, set up the environment and run one-shot mode or the REPL.

    Args:
        argv: Command-line arguments without the program name; ``None`` means ``sys.argv[1:]``.
        provider_factory: Builds the provider from its name and keyword arguments; tests pass
            one that returns a ``FakeProvider``.
        input_fn: Reads one REPL line given the prompt.
        console: Rich console for all output; ``None`` creates one (and, on Windows, switches
            stdout/stderr to UTF-8).
        load_env: Whether to load ``.env``; tests pass ``False``.

    Returns:
        The exit code: 0 for a completed one-shot answer or a REPL session, 1 for a one-shot
        run that didn't complete, 2 for a startup error.
    """
    args = build_parser().parse_args(argv)
    if load_env:
        load_dotenv(override=False)  # real environment variables win over .env
    _configure_logging(args.debug)
    if console is None:
        _use_utf8_streams()
        console = Console()

    workspace = Path(
        args.workspace
        if args.workspace is not None
        else os.environ.get("WORKSPACE_ROOT") or fs_tools.DEFAULT_WORKSPACE_ROOT
    )
    if not workspace.is_dir():
        _error(
            console,
            f"Workspace directory not found: {workspace}",
            "Pass --workspace PATH or set WORKSPACE_ROOT to an existing directory.",
        )
        return 2
    os.environ["WORKSPACE_ROOT"] = str(workspace.resolve())

    try:
        provider = provider_factory(args.provider, **({"model": args.model} if args.model else {}))
    except LLMProviderError as exc:
        _error(console, f"Could not start the {args.provider} provider: {exc.message}", ENV_HINT)
        return 2

    renderer = Renderer(console, verbose=args.verbose)
    try:
        agent = Agent(provider, max_iterations=args.max_iterations, on_event=renderer.on_event)
    except ValueError as exc:
        _error(console, str(exc))
        return 2

    renderer.banner(
        provider=getattr(provider, "name", args.provider),
        model=getattr(provider, "model", None) or "-",
        workspace=str(workspace.resolve()),
        tools=list(TOOL_NAMES),
    )
    if args.query is not None:
        return 0 if run_query(agent, renderer, args.query).stopped_reason == "completed" else 1
    return repl(agent, renderer, input_fn)


if __name__ == "__main__":
    sys.exit(main())
