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
| `assistant/agent.py` | Agent loop |
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

**Gemini first** via `google-genai`. Tools are passed as `FunctionDeclaration`s built from the
registry's JSON schemas. **Automatic function calling is disabled**
(`AutomaticFunctionCallingConfig(disable=True)`) so the SDK never executes Python functions on its
own – the loop below is ours, which keeps the sandbox, logging and iteration cap under our control
and makes the flow visible for the assignment.

**Agent loop** (`assistant/agent.py`):

```
messages = [system prompt, user query]
for i in range(MAX_TOOL_ITERATIONS):
    resp = provider.generate(messages, tools)
    if not resp.tool_calls:
        return resp.text                     # final answer
    messages.append(assistant turn with resp.tool_calls)
    for call in resp.tool_calls:            # supports parallel calls in one turn
        result = registry.dispatch(call.name, call.args)   # always an envelope
        on_tool_call(call, result)          # CLI trace hook
        messages.append(tool response for call.id)
return "stopped: reached MAX_TOOL_ITERATIONS"   # graceful, not an exception
```

Unknown tool names and invalid arguments produce error envelopes (`UNKNOWN_TOOL`,
`INVALID_ARGUMENT`) fed back to the model so it can recover. Config (`GEMINI_API_KEY`,
`GEMINI_MODEL`, `WORKSPACE_ROOT`, `MAX_TOOL_ITERATIONS`) is loaded from `.env` via `python-dotenv`.

## 7. CLI

`python llm_file_assistant.py` starts an interactive REPL (`exit`/`quit`/Ctrl-D to leave; history kept
across turns). `python llm_file_assistant.py --query "..."` runs one question and exits (non-zero on
failure). Options: `--provider`, `--model`, `--workspace`, `--max-iterations`, overriding env vars.

Every tool call is rendered with `rich` as it happens – tool name, arguments, and a short result
(success + key fields, or error code/message, truncated) – so a demo shows exactly what the model
did. The final answer is rendered as Markdown. A missing API key gives a clear message, not a
traceback.

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
