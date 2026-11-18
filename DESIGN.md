# Design: Airtribe LLM File Assistant

An LLM-driven assistant that answers questions about a folder of resumes (PDF, TXT, DOCX) by
calling a small set of sandboxed file-system tools through function calling.

- **Part A** – `fs_tools.py`: the four file tools, usable as plain Python functions.
- **Part B** – `llm_file_assistant.py` + `assistant/`: tool registry, provider abstraction,
  hand-written agent loop, and a CLI.

```
┌──────────────────────┐   user query    ┌──────────────┐  generate(msgs, tools)  ┌─────────────┐
│ llm_file_assistant.py│ ──────────────▶ │  Agent loop  │ ──────────────────────▶ │ LLMProvider │
│  (REPL / --query)    │ ◀────────────── │ (assistant/) │ ◀────────────────────── │  (Gemini…)  │
└──────────────────────┘  answer + trace └──────┬───────┘   text | tool calls     └─────────────┘
                                                │ dispatch(name, args)
                                         ┌──────▼───────┐
                                         │ Tool registry│  JSON schemas + exception → envelope
                                         └──────┬───────┘
                                         ┌──────▼───────┐
                                         │  fs_tools.py │  sandbox → extract → result
                                         └──────────────┘
```

Planned layout:

| Path | Role |
|---|---|
| `fs_tools.py` | Part A tools, sandbox, extraction, `FsToolError` |
| `assistant/registry.py` | Tool JSON schemas + dispatcher |
| `assistant/agent.py` | Agent loop, `AgentResult`, events |
| `assistant/prompts.py` | `DEFAULT_SYSTEM_PROMPT` |
| `assistant/render.py` | Rich rendering of agent events and results, `summarize_result` |
| `assistant/providers/__init__.py` | Provider registry (`get_provider`, lazy imports) |
| `assistant/providers/base.py` | `LLMProvider` ABC, neutral message/tool-call types, `LLMProviderError` |
| `assistant/providers/fake.py` | `FakeProvider`: scripted responses for tests and offline demos |
| `assistant/providers/gemini.py` | Gemini implementation (`google-genai`) |
| `llm_file_assistant.py` | CLI |
| `scripts/generate_samples.py` | Generates sample resumes + edge cases |
| `tests/` | pytest suite (includes `FakeProvider`) |

---

## 1. Tool signatures

Exactly as the assignment brief specifies:

```python
def read_file(filepath: str) -> dict: ...
def list_files(directory: str, extension: str | None = None) -> list: ...
def write_file(filepath: str, content: str) -> dict: ...
def search_in_file(filepath: str, keyword: str, whole_word: bool = False) -> dict: ...
```

`search_in_file` adds one optional `whole_word` flag; calling it with the brief's two arguments
behaves exactly as specified.

Paths are strings relative to `WORKSPACE_ROOT` (see §3). `extension` accepts `"pdf"` or `".pdf"`,
case-insensitive.

## 2. Result envelope and errors

The three dict-returning tools always return the same shape:

```python
{
    "success": bool,
    "data": Any,  # None on failure
    "error": {"code": str, "message": str} | None,  # None on success
}
```

Representative `data` payloads:

| Tool | `data` |
|---|---|
| `read_file` | `{"content", "metadata": {"path", "name", "extension", "size_bytes", "modified", "char_count", "word_count", "page_count"?, "encoding"?, "warnings"}}` – `page_count` for PDF only, `encoding` for TXT/MD only; `modified` is ISO 8601 UTC |
| `write_file` | `{"path", "name", "size_bytes", "overwritten": bool, "created_dirs": [str], "modified"}` – `created_dirs` lists the workspace-relative directories it created |
| `search_in_file` | `{"path", "keyword", "whole_word", "found", "total_matches", "returned_matches", "truncated", "matches": [{"start", "end", "matched_text", "line_number", "page", "context"}], "warnings"}` – `page` is `null` except for PDFs |

**`list_files`** returns a `list` per spec – one dict per file:
`{"name", "path", "extension", "size_bytes", "modified", "supported"}` (`extension` lowercase with
dot; `supported` = readable by `read_file`), sorted by name case-insensitively for determinism. It
is non-recursive and lists files only; `""` and `"."` mean the workspace root. Dotfiles are skipped,
as are entries that are symlinks resolving outside the workspace (their metadata is never
returned). The extension filter is case-insensitive and accepts `pdf`, `.pdf`, `.PDF` or `*.pdf`;
`None` or blank means no filter, and anything that isn't a plain extension after normalisation is
`INVALID_ARGUMENT`. An empty directory or no matches is `[]`, not an error. Because a bare list has
no room for an error, it **raises** `FsToolError` on failure (`PATH_OUTSIDE_WORKSPACE`,
`NOT_FOUND`, `NOT_A_DIRECTORY`, `PERMISSION_DENIED`, `INVALID_ARGUMENT`).

**`FsToolError(code, message)`** is the single typed exception used internally by all tools. The
dict-returning tools catch it and build the error envelope; `list_files` lets it propagate.

Error codes (stable strings, safe to show to the LLM):

| Code | Meaning |
|---|---|
| `INVALID_ARGUMENT` | Empty/ill-typed argument (e.g. empty keyword) |
| `PATH_OUTSIDE_WORKSPACE` | Path escapes the sandbox |
| `FILE_NOT_FOUND` | File does not exist |
| `NOT_FOUND` | Directory does not exist (`list_files`) |
| `NOT_A_FILE` / `NOT_A_DIRECTORY` | Wrong kind of path |
| `UNSUPPORTED_FORMAT` | Extension not supported for this operation |
| `FILE_TOO_LARGE` | File exceeds the read cap (10 MB), checked before reading |
| `CONTENT_TOO_LARGE` | `write_file` content exceeds the write cap (1 MB, UTF-8 encoded) |
| `EXTRACTION_FAILED` | Corrupt/unreadable PDF or DOCX |
| `PERMISSION_DENIED` | OS-level permission error |
| `IO_ERROR` | Other `OSError` |
| `INTERNAL_ERROR` | Unexpected exception; generic message, details only in the log |
| `UNKNOWN_TOOL` | Dispatcher: the LLM called a tool name that isn't registered (message lists valid names) |

**Dispatcher contract** (`assistant/registry.py`). `dispatch(name, args)` never raises and always
returns the envelope, so the LLM sees one uniform shape for every tool and **never sees a
traceback**:

- Unknown tool name → `UNKNOWN_TOOL`. Arguments are checked against the tool's schema first:
  non-object args, a missing required arg, an unknown arg name or a wrong JSON type →
  `INVALID_ARGUMENT` naming the argument. `null` args are treated as omitted (`None` args as `{}`).
- An `FsToolError` becomes `{"success": False, "data": None, "error": {...}}`; any other
  unexpected exception becomes `INTERNAL_ERROR` with a generic message (details logged at debug
  level, not sent).
- `list_files`' list is wrapped as `{"success": True, "data": {"files": [...], "count": n}, ...}`.
- Context guard: `read_file` content longer than 20,000 characters is truncated, with a warning
  and `metadata.content_truncated: true` (otherwise `false`).
- Every result is checked to be JSON-serialisable before it is returned.

Tool schemas are hand-written in the JSON Schema subset Gemini, Anthropic and OpenAI all accept
(`type`, `properties`, `required`, per-property `type`/`description`/`enum`); a test keeps them in
sync with the function signatures. Their descriptions carry the guidance the LLM needs (relative
forward-slash paths, call `list_files` first, `whole_word` for ambiguous terms, one search per file
in parallel).

## 3. Path sandbox

LLM-supplied paths are **untrusted input**. Every tool resolves its path through one helper:

1. Reject empty paths and paths containing NUL bytes.
2. `root = Path(WORKSPACE_ROOT).resolve()` (read from env on every call, default `./sample_data`,
   so tests can monkeypatch it).
3. If the given path is absolute, use it as-is; otherwise join it onto `root`.
4. `candidate = path.resolve()` – this collapses `..` **and follows symlinks**.
5. Require `candidate.is_relative_to(root)`; otherwise raise `PATH_OUTSIDE_WORKSPACE`.

This single post-resolution check covers `../` traversal, absolute paths outside the root, and
symlinks (or Windows junctions) that point outside. An absolute path that lands *inside* the root is
allowed. Paths returned to callers are workspace-relative with forward slashes (`as_posix()`), so
output is identical on Windows and Linux. Error messages report the path as given and never reveal
the absolute host path. For
`write_file`, the check runs on the target before any parent directories are created.

## 4. Text extraction and search

One internal `extract_text(path) -> (text, metadata)` dispatches on extension:

| Format | Method |
|---|---|
| `.txt` (also `.md` for read-back) | Decode UTF-8 (a leading BOM is stripped); on `UnicodeDecodeError`, fall back to `cp1252`, then `latin-1`. The encoding used is reported in metadata. |
| `.pdf` | `pypdf.PdfReader`; page texts joined with `"\n\f\n"` (form feed, as `pdftotext` does); `page_count` in metadata. Reader construction *and* page access share one error mapping, since pypdf may only log on open and raise later. Encrypted or malformed files → `EXTRACTION_FAILED`. |
| `.docx` | `python-docx`: paragraphs and tables in document order (`iter_inner_content()`); each table row's cells joined by ` \| ` (merged cells once), so skills listed in tables are searchable. Invalid zip/XML → `EXTRACTION_FAILED`. |

For every format the extracted text is then sanitised: `\r\n` and lone `\r` become `\n`, and every
other Unicode control character (Cc) except `\n`, `\t` and `\f` becomes one space (1:1, before any
offsets are computed – e.g. the DEL pypdf extracts for bullet glyphs).

Extensions match case-insensitively. Anything else → `UNSUPPORTED_FORMAT`. Files over 10 MB →
`FILE_TOO_LARGE`, checked via `stat` before reading. Extraction error messages name only the
exception type – never a traceback or host path. An empty file, or a PDF/DOCX with no extractable
text, is a **success** with empty `content` and a `warnings` entry (e.g. "possibly a scanned/image
PDF"), not an error.

**`search_in_file`** calls the same `_load_document` helper as `read_file`, so it works identically
across all formats (including DOCX table text) and fails with exactly the envelope `read_file` would
return (`FILE_NOT_FOUND`, `UNSUPPORTED_FORMAT`, `EXTRACTION_FAILED`, ...). Extraction warnings (e.g.
a no-text PDF) are passed through; an empty document or no hits is a success with `found: false`.

- **Keyword**: a non-empty `str` after stripping, at most 200 characters, else `INVALID_ARGUMENT`.
  It is matched literally (`re.escape`, so `C++`, `node.js` and `(a)` are not patterns – the LLM
  can't inject regexes) and case-insensitively with `re.IGNORECASE` on the original text. Lowering
  or casefolding the text first is avoided because it can change string length and shift offsets.
- **`whole_word=True`** adds `\b` only on a side where the keyword starts/ends with a word
  character, so `Java` no longer matches `JavaScript` while `C++` and `.NET` still match.
- **Each match**: `start`/`end` character offsets into the extracted content (so
  `content[start:end] == matched_text`, with the document's original casing), 1-based
  `line_number`, 1-based `page` for PDFs (counted from the form-feed page separators; `null`
  otherwise), and `context` – ±60 characters around the match with non-printable characters (form
  feeds, the DEL pypdf extracts for some bullet glyphs) turned into spaces, all whitespace
  collapsed to single spaces, and `…` where the snippet was cut.
- **Cap**: at most 50 matches are returned, but every match is counted: `total_matches`,
  `returned_matches` and `truncated` tell the LLM when it is seeing a subset.

## 5. `write_file`

- **Atomic**: write to a hidden temp file in the target's directory (`tempfile.mkstemp`, name
  starting with `.` so `list_files` never shows it), flush + `fsync`, then `os.replace(tmp, target)`.
  On any failure the temp file is removed and an existing target is left untouched, so a crash never
  leaves a half-written target. The target keeps its previous permission bits (new files get
  `0o644`, since `mkstemp` would otherwise leave `0o600`).
- Creates missing parent directories only after the sandbox check passes, and reports them in
  `created_dirs`. A path component that is an existing file → `NOT_A_DIRECTORY`; a target that is
  a directory → `NOT_A_FILE`; a missing workspace root → `NOT_FOUND` (the root is never created).
- **Text only**: allowed extensions `.txt` and `.md` (case-insensitive); anything else →
  `UNSUPPORTED_FORMAT`. This prevents the LLM from clobbering source PDFs/DOCX files with plain text.
- `content` must be a `str` (empty is allowed) → otherwise `INVALID_ARGUMENT`, as is text that
  can't be UTF-8 encoded (lone surrogates). Encoded size over 1 MB → `CONTENT_TOO_LARGE`.
- Writes UTF-8 with `newline=""`, so `"\n"` is never translated to `"\r\n"` on Windows.

## 6. LLM layer

**Provider abstraction** (`assistant/providers/base.py`). `LLMProvider` is an ABC with one core
method:

```python
class LLMProvider(ABC):
    name: ClassVar[str]

    @abstractmethod
    def generate(
        self, system_prompt: str, messages: Sequence[Message], tools: list[dict]
    ) -> LLMResponse: ...
```

`tools` is `registry.get_tool_schemas()` output. The agent loop speaks only these frozen,
provider-neutral dataclasses; each provider translates them to and from its SDK types:

| Type | Fields |
|---|---|
| `ToolCall` | `id`, `name`, `args` |
| `ToolResult` | `call_id`, `name`, `result` (the dispatcher envelope) |
| `Message` | `role` (`user` / `assistant` / `tool`), `text`, `tool_calls`, `tool_results`, `provider_state` |
| `LLMResponse` | `text`, `tool_calls`, `provider_state`, `usage` |

An assistant message may carry `tool_calls`; one `tool` message carries the results of every call
from that turn. Provider failures raise `LLMProviderError(message, retryable=...)`: a user-safe
message (no keys, no tracebacks) plus a flag marking transient failures worth retrying.

**`provider_state`** is opaque per-provider data that the loop copies from each `LLMResponse` onto
the assistant `Message` it stores, and hands back unchanged. Gemini 3 requires its original
response content – including *thought signatures* attached to function-call parts – to be replayed
on the next turn, and rebuilding it from the neutral fields would drop them. Providers that don't
need it leave it `None`.

**Registry** (`assistant/providers/__init__.py`). Names map to `"module:Class"` strings (`"gemini"`
→ `assistant.providers.gemini:GeminiProvider`, `"fake"` → `assistant.providers.fake:FakeProvider`)
and `get_provider(name, **kwargs)` imports the module only when it is resolved, so importing the
package never imports `google.genai`. An unknown name, or a provider whose module or SDK is
missing, raises `LLMProviderError` (never `ImportError`). Adding a provider is one module plus one
mapping entry; the CLI picks it with `--provider`.

**`FakeProvider`** (`"fake"`) replays a scripted list of `LLMResponse`s, one per `generate` call,
and records deep copies of every call in `.calls`. Builders `reply(text)`, `call(name, args)` and
`parallel((name, args), ...)` keep scripts readable; calls without an explicit id get
deterministic ids `call_1`, `call_2`, ... per instance. An exhausted script raises
`LLMProviderError`. It drives every agent-loop test (no network, no API key) and can run an
offline demo of the CLI.

**Gemini** (`assistant/providers/gemini.py`, `google-genai` 3.5). `GeminiProvider(api_key=None,
model=None, client=None, max_retries=2, sleep=time.sleep)` falls back to `GEMINI_API_KEY` /
`GEMINI_MODEL` from `os.environ` (loading `.env` is the CLI's job); a missing value raises
`LLMProviderError` naming the variable, never echoing the key. `client` and `sleep` are injectable
so tests need no network and no real waiting.

- **Config per call**: `system_instruction`, one `types.Tool` whose `FunctionDeclaration`s take the
  registry schemas verbatim via `parameters_json_schema` (no `types.Schema` conversion),
  `ToolConfig(function_calling_config=FunctionCallingConfig(mode=AUTO))`, and
  **automatic function calling disabled** (`AutomaticFunctionCallingConfig(disable=True)`), so the
  SDK never runs Python functions itself – the loop below is ours, which keeps the sandbox, logging
  and iteration cap under our control and makes the flow visible. Temperature is left at the
  model default (Gemini 3 guidance).
- **History → `types.Content`**: `user` → role `"user"` text part. `assistant` with a
  `types.Content` in `provider_state` → that object, **unchanged** (it carries the thought
  signatures); without it → role `"model"` with text and `function_call` parts. `tool` → one
  role-`"user"` Content (the role google-genai's own chat code uses) with one `function_response`
  part per result: `name`, `response` = the envelope, and `id`.
- **Response → `LLMResponse`**: `function_call` parts of the first candidate become `ToolCall`s
  (`id` from Gemini, else a positional `gemini_call_<n>` – such synthetic ids are never sent back);
  `text` joins the non-thought text parts; `provider_state` is the candidate's raw `Content`;
  `usage` has `prompt_tokens`, `candidates_tokens`, `thoughts_tokens`, `total_tokens` when present.

| Failure | `LLMProviderError` |
|---|---|
| `prompt_feedback.block_reason`, no candidates, finish `SAFETY` / `RECITATION` / `BLOCKLIST` / `PROHIBITED_CONTENT` / `SPII` / `MODEL_ARMOR` / `MISSING_THOUGHT_SIGNATURE` (and image variants) | not retryable, names the reason |
| finish `MALFORMED_FUNCTION_CALL` / `MALFORMED_RESPONSE`, or an empty response | retryable |
| `errors.APIError` 408, 429, 5xx | retryable |
| `errors.APIError` 400 / 401 / 403 / 404 (other 4xx) | not retryable; status code + hint (401/403 → check `GEMINI_API_KEY`, 404 → check `GEMINI_MODEL`); the API message is included only with the key redacted |
| `httpx2.TransportError` (google-genai's HTTP stack), `ConnectionError`, `TimeoutError` | retryable |
| anything else from the SDK | not retryable, exception type only |

Retryable errors are retried up to `max_retries` times with exponential backoff (1 s, 2 s) through
the injected `sleep`; then the last error is raised. The SDK's own retries stay off (no
`HttpRetryOptions`, so it makes a single attempt), so the two never multiply.

**Agent loop** (`assistant/agent.py`, system prompt in `assistant/prompts.py`).
`Agent(provider, *, system_prompt, max_iterations, dispatch, tools, on_event, clock)` keeps
`agent.history: list[Message]` across `run()` calls (multi-turn REPL; `reset()` clears it).
`max_iterations` defaults to the `MAX_TOOL_ITERATIONS` env var, else 10, and caps `generate()`
calls per `run()`.

```
run(user_text):
    checkpoint = len(history); history += [user(user_text)]
    for iteration in 1, 2, ...:
        resp = provider.generate(system_prompt, history, tools)      # tools sent every time
        emit ModelTurn(iteration, resp.text, resp.tool_calls, resp.usage)
        history += [assistant(resp.text, resp.tool_calls, provider_state=resp.provider_state)]
        if not resp.tool_calls:
            return AgentResult(resp.text, "completed", ...)
        if iteration == max_iterations:                              # never exceed the cap
            history += [tool(NOT_EXECUTED result per pending call), assistant(fixed text)]
            return AgentResult(fixed text, "max_iterations", ...)
        results = []
        for call in resp.tool_calls:                                 # parallel calls, in order
            emit ToolCallStarted(call)
            result = dispatch(call.name, call.args)                  # always an envelope
            emit ToolCallFinished(call, result, duration_ms)         # via the injected clock
            results += [ToolResult(call.id, call.name, result)]
        history += [tool(results)]                                   # one message per turn
on LLMProviderError:
    del history[checkpoint:]                                         # as if run() never started
    return AgentResult(None, "provider_error", error=exc.message)
```

- **Model mistakes are data, not exceptions**: unknown tools and bad arguments come back from
  `dispatch` as `UNKNOWN_TOOL` / `INVALID_ARGUMENT` envelopes, which go to the model so it can
  correct itself.
- **Iteration cap**: calls requested on the last allowed turn are *not executed* (no side effects
  the model never sees); each gets a `NOT_EXECUTED` result so every function call still has a
  response, and a fixed assistant message closes the turn, so the history stays valid for the
  next `run()`.
- **Provider errors** roll the history back to its pre-`run()` state so a half-finished turn can't
  corrupt the next request; they are reported, not raised. Tools that already ran are not undone
  (e.g. a file already written), and they are still listed in `AgentResult.tool_calls`. Any other
  exception propagates.
- `AgentResult(text, stopped_reason, iterations, tool_calls, usage, error)`: `tool_calls` is the
  executed `(ToolCall, envelope)` pairs and `usage` the numeric token counts summed across turns.
- The agent never prints; the CLI renders the `ModelTurn` / `ToolCallStarted` /
  `ToolCallFinished` events.

**System prompt** (`DEFAULT_SYSTEM_PROMPT`): workspace-relative forward-slash paths; discover with
`list_files`, never guess; to find mentions, `list_files` then one `search_in_file` per file (in
parallel), `whole_word=true` for short or ambiguous terms; summaries go to
`summaries/<file-stem>_summary.md` after a `read_file`, never into `resumes/`; answer only from tool
results, report tool errors plainly and carry on; concise answers that name their source files.

Config (`GEMINI_API_KEY`, `GEMINI_MODEL`, `WORKSPACE_ROOT`, `MAX_TOOL_ITERATIONS`) is loaded from
`.env` via `python-dotenv` by the CLI.

## 7. CLI

`llm_file_assistant.py` (argument parsing, startup, REPL) plus `assistant/render.py` (all
output). `main(argv=None, *, provider_factory=get_provider, input_fn=input, console=None,
load_env=True) -> int`, so tests inject a `FakeProvider`, scripted input and a recording console.

| Option | Effect |
|---|---|
| `-q/--query TEXT` | one-shot mode; without it, the REPL starts |
| `--provider` | default `gemini`; choices are the registered providers minus `fake` (it needs a script) |
| `--model` | overrides `GEMINI_MODEL` |
| `--workspace PATH` | overrides `WORKSPACE_ROOT`; must be an existing directory |
| `--max-iterations N` | overrides `MAX_TOOL_ITERATIONS` |
| `--verbose` | also prints every tool result as pretty JSON |
| `--debug` | root logging at DEBUG |

**Startup**: load `.env` (`override=False`, so real environment variables win); root logging at
WARNING and the `pypdf` logger at ERROR (even with `--debug`), so "EOF marker not found" never
reaches the demo; on Windows, stdout/stderr are reconfigured to UTF-8 (`errors="replace"`). The
effective workspace (flag, else env, else `./sample_data`) must exist, else a friendly error and
exit 2; it is then exported as an absolute `WORKSPACE_ROOT`. A provider that can't be built (missing
key/model) prints its message plus a pointer to `.env.example` and exits 2, as does an invalid
iteration cap.

**Rendering** (`Renderer`, driven only by agent events; strings are wrapped in `rich.text.Text` so
file content can't inject console markup):

- Banner panel: provider, model, absolute workspace, tool names.
- A "Thinking…" status spinner while waiting for the model (terminals only; it pauses while tools
  run and resumes after the last call of a turn).
- `ModelTurn` with tool calls → "Step i" plus any interim model text, dimmed.
- `ToolCallStarted` → `→ tool_name(arg='value', …)`, values over 40 characters shortened.
- `ToolCallFinished` → one line from the pure `summarize_result(name, result)` plus the duration:
  `✓ 1,234 chars · 2 pages` (+ warnings), `✓ 8 files`, `✓ 9 matches` / `✓ no matches`,
  `✓ wrote summaries/x.md (131 B, overwritten)`, or `✗ CODE: message` for any error envelope.
- Final answer as Markdown in a panel, with a dim footer: iterations · tool calls · total tokens.
  `max_iterations` → a yellow note; `provider_error` → a red panel with the safe error message.

**REPL**: prompt `you> `, one `Agent` for the whole session (multi-turn history). Commands `/help`,
`/tools` (each tool with the first sentence of its description), `/reset` (clears history and
confirms), `/exit`, `/quit`. Empty input is ignored; Ctrl-D/EOF or Ctrl-C at the prompt exits with
"Goodbye!" and no traceback; Ctrl-C while a question is running discards that turn from the history
and returns to the prompt.

**Exit codes**: one-shot 0 if `completed`, else 1; the REPL exits 0; startup errors exit 2.

## 8. Testing

- **`fs_tools`**: tests build files in pytest's `tmp_path` and point `WORKSPACE_ROOT` at it via
  `monkeypatch`. PDFs are generated with `reportlab`, DOCX with `python-docx`, so no binary fixtures
  are committed. Coverage includes each format, encoding fallback, corrupt/empty files, sandbox
  escapes (`..`, outside-absolute, symlink – skipped where the OS disallows symlinks), atomic
  overwrite, and search positions/context.
- **Agent loop**: a `FakeProvider` replays a scripted list of `LLMResponse`s (tool calls then a final
  answer) and records the messages it received, so tests assert on dispatch order, multi-call turns,
  error recovery and the iteration cap.
- **No real LLM calls** in tests or CI; no network.
- **Deterministic**: no hardcoded dates or wall-clock/random values in assertions – timestamps are
  checked for type/format or compared against values read from the file itself.
- CI (GitHub Actions, Python 3.14) runs `ruff check`, `ruff format --check`, and – from step 3 –
  `pytest`.
