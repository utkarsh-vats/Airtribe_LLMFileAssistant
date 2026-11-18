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
| `assistant/providers/base.py` | `LLMProvider` ABC, neutral message/tool-call types, provider registry |
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
def search_in_file(filepath: str, keyword: str) -> dict: ...
```

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
| `read_file` | `{"path", "format", "content", "metadata": {"size_bytes", "modified", "pages"?, "encoding"?, "char_count"}}` |
| `write_file` | `{"path", "bytes_written", "overwritten": bool}` |
| `search_in_file` | `{"path", "keyword", "match_count", "matches": [{"position", "line", "context"}]}` |

**`list_files`** returns a `list` per spec – one dict per file:
`{"name", "path", "extension", "size_bytes", "modified"}`, sorted by path for determinism. Because a
bare list has no room for an error, it **raises** `FsToolError` on failure.

**`FsToolError(code, message)`** is the single typed exception used internally by all tools. The
dict-returning tools catch it and build the error envelope; `list_files` lets it propagate.

Error codes (stable strings, safe to show to the LLM):

| Code | Meaning |
|---|---|
| `INVALID_ARGUMENT` | Empty/ill-typed argument (e.g. empty keyword) |
| `PATH_OUTSIDE_WORKSPACE` | Path escapes the sandbox |
| `NOT_FOUND` | File or directory does not exist |
| `NOT_A_FILE` / `NOT_A_DIRECTORY` | Wrong kind of path |
| `UNSUPPORTED_FORMAT` | Extension not supported for this operation |
| `EXTRACTION_FAILED` | Corrupt/unreadable PDF or DOCX |
| `PERMISSION_DENIED` | OS-level permission error |
| `IO_ERROR` | Other `OSError` |

**Dispatcher contract.** The tool registry wraps every tool call: an `FsToolError` becomes
`{"success": False, "data": None, "error": {...}}`; any other unexpected exception becomes
`INTERNAL_ERROR` with a generic message (details are logged, not sent). `list_files`' list is wrapped
as `{"success": True, "data": [...], "error": None}` so the LLM sees one uniform shape for every
tool and **never sees a traceback**.

## 3. Path sandbox

LLM-supplied paths are **untrusted input**. Every tool resolves its path through one helper:

1. Reject empty paths and paths containing NUL bytes.
2. `root = Path(WORKSPACE_ROOT).resolve()` (read from env, default `./sample_data`).
3. If the given path is absolute, use it as-is; otherwise join it onto `root`.
4. `candidate = path.resolve()` – this collapses `..` **and follows symlinks**.
5. Require `candidate.is_relative_to(root)`; otherwise raise `PATH_OUTSIDE_WORKSPACE`.

This single post-resolution check covers `../` traversal, absolute paths outside the root, and
symlinks (or Windows junctions) that point outside. An absolute path that lands *inside* the root is
allowed. Error messages report the path as given and never reveal the absolute host path. For
`write_file`, the check runs on the target before any parent directories are created.

## 4. Text extraction and search

One internal `extract_text(path) -> (text, metadata)` dispatches on extension:

| Format | Method |
|---|---|
| `.txt` (also `.md` for read-back) | Decode UTF-8 (`utf-8-sig` to drop a BOM); on `UnicodeDecodeError`, fall back to `cp1252`, then `latin-1`. The encoding used is reported in metadata. |
| `.pdf` | `pypdf.PdfReader`; text of each page joined with newlines; `pages` in metadata. Encrypted or malformed files → `EXTRACTION_FAILED`. |
| `.docx` | `python-docx`: paragraphs in order, then table cells (row cells joined by ` \| `), so skills listed in tables are searchable. Invalid zip/XML → `EXTRACTION_FAILED`. |

Anything else → `UNSUPPORTED_FORMAT`. An empty file is a success with empty `content` (not an error).

**`search_in_file`** calls the same extraction as `read_file`, so it works identically across all
three formats. Matching is case-insensitive (`casefold`) on a literal keyword (no regex, so the LLM
can't inject patterns). Each match returns its character `position` in the extracted text, the
1-based `line`, and a `context` snippet of roughly ±60 characters, whitespace-collapsed. Matches are
capped (e.g. 50) with a `truncated` flag to keep tool responses small.

## 5. `write_file`

- **Atomic**: write to a temp file in the target's directory (`tempfile.NamedTemporaryFile(delete=False,
  dir=parent)`), flush + `fsync`, then `os.replace(tmp, target)`. The temp file is removed on failure,
  so a crash never leaves a half-written target.
- Creates missing parent directories (after the sandbox check).
- **Text only**: allowed extensions `.txt` and `.md`; anything else → `UNSUPPORTED_FORMAT`. This
  prevents the LLM from clobbering source PDFs/DOCX files with plain text.
- Writes UTF-8. Metadata reports `bytes_written` and `overwritten` (whether the target existed).

## 6. LLM layer

**Provider abstraction.** `LLMProvider` is an ABC with one core method:

```python
class LLMProvider(ABC):
    name: ClassVar[str]

    @abstractmethod
    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse: ...
```

`Message`, `ToolSpec`, `ToolCall(id, name, args)` and `LLMResponse(text, tool_calls)` are
provider-neutral dataclasses; each provider translates to/from its SDK types. A registry maps names to
classes (`register_provider("gemini", GeminiProvider)`, `get_provider(name)`), so a new provider is
one module + one registration, and the CLI picks it with `--provider`.

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
