"""Tool registry: provider-neutral JSON schemas for the fs_tools functions, plus a dispatcher.

``dispatch(name, args)`` is the only way the agent loop runs a tool. It validates the LLM's
arguments against the schema, calls the function, and always returns the standard envelope
``{"success", "data", "error"}`` - it never raises, so the LLM never sees a traceback.

Schemas use only the JSON Schema subset that Gemini, Anthropic and OpenAI all accept:
``type``, ``properties``, ``required``, and per-property ``type``/``description``/``enum``.
"""

import copy
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import fs_tools
from fs_tools import ErrorCode, FsToolError, _err, _ok

logger = logging.getLogger(__name__)

UNKNOWN_TOOL = "UNKNOWN_TOOL"  # registry-level code; fs_tools.ErrorCode covers the rest
MAX_TOOL_RESULT_CHARS = 20_000  # read_file content beyond this is truncated to protect context

_PYTHON_TYPES = {"string": str, "boolean": bool}


@dataclass(frozen=True)
class ToolSpec:
    """One tool as exposed to the LLM.

    Attributes:
        name: Tool name the model calls; equals the ``fs_tools`` function name.
        description: Guidance for the model on when and how to use the tool.
        parameters: JSON Schema for the arguments, in the portable subset (see module docstring).
        func: The ``fs_tools`` function that ``dispatch`` calls with the validated arguments.
    """

    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]


_PATH_HINT = (
    "Paths are relative to the workspace root and use forward slashes, "
    "e.g. 'resumes/resume_john_doe.pdf'. Call list_files first to discover file names "
    "instead of guessing them."
)

TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_files",
        description=(
            "List the files in one workspace directory (not recursive; subdirectories and hidden "
            "files are not listed). Returns name, path, extension, size_bytes, modified and "
            "supported for each file, plus a count. 'supported' means the format can be read by "
            "read_file and search_in_file (txt, md, pdf, docx); it does not guarantee the file "
            "reads successfully - it may still be corrupt or empty. Use this first to discover "
            "which files exist before reading or searching them."
        ),
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": (
                        "Workspace-relative directory with forward slashes, e.g. 'resumes'. "
                        "Use '' or '.' for the workspace root."
                    ),
                },
                "extension": {
                    "type": "string",
                    "description": (
                        "Optional file-type filter such as 'pdf' or '.docx' (case-insensitive). "
                        "Omit it to list every file."
                    ),
                },
            },
            "required": ["directory"],
        },
        func=fs_tools.list_files,
    ),
    ToolSpec(
        name="read_file",
        description=(
            "Read one file and return its full extracted text plus metadata (size, modified time, "
            "word count, page_count for PDFs, and any warnings such as a PDF with no text layer). "
            "Supported formats: .txt, .md, .pdf and .docx (DOCX table text is included). "
            f"{_PATH_HINT} Very long text is truncated (metadata.content_truncated is then true); "
            "use search_in_file to locate specific terms in long documents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Workspace-relative path of the file to read.",
                },
            },
            "required": ["filepath"],
        },
        func=fs_tools.read_file,
    ),
    ToolSpec(
        name="search_in_file",
        description=(
            "Search one file for a keyword and return every match with its line number, page "
            "(PDFs only), character offsets and surrounding context. Matching is case-insensitive "
            "and literal: no regular expressions or wildcards. Works on .txt, .md, .pdf and "
            ".docx. To search many files, call this once per file - the calls can be made in "
            "parallel in a single turn. At most 50 matches are returned; total_matches counts "
            f"all of them. {_PATH_HINT}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Workspace-relative path of the file to search.",
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "Literal text to find, e.g. 'Python' or 'machine learning' "
                        "(max 200 characters)."
                    ),
                },
                "whole_word": {
                    "type": "boolean",
                    "description": (
                        "Match whole words only. Set true for short or ambiguous terms such as "
                        "'Java' (which otherwise also matches 'JavaScript'), 'Go' or 'R'. "
                        "Defaults to false."
                    ),
                },
            },
            "required": ["filepath", "keyword"],
        },
        func=fs_tools.search_in_file,
    ),
    ToolSpec(
        name="write_file",
        description=(
            "Write UTF-8 text to a .txt or .md file in the workspace; no other formats can be "
            "written. Missing parent directories are created. If the file already exists it is "
            "overwritten - check with list_files first if that matters. Content is limited to "
            f"1 MB. {_PATH_HINT}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": (
                        "Workspace-relative path ending in .txt or .md, "
                        "e.g. 'summaries/python_candidates.md'."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Full text to write (Markdown is fine); replaces any existing content."
                    ),
                },
            },
            "required": ["filepath", "content"],
        },
        func=fs_tools.write_file,
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in TOOL_SPECS)
_TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def get_tool_schemas() -> list[dict]:
    """``[{name, description, parameters}]`` for providers; deep copies, safe to mutate."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": copy.deepcopy(spec.parameters),
        }
        for spec in TOOL_SPECS
    ]


def _validate_args(spec: ToolSpec, args: Any) -> dict:
    """Check ``args`` against the spec's schema and return the kwargs to call it with."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise FsToolError(
            ErrorCode.INVALID_ARGUMENT,
            f"Arguments for {spec.name} must be a JSON object, not {type(args).__name__}.",
        )
    properties = spec.parameters["properties"]
    unknown = sorted(str(key) for key in args if key not in properties)
    if unknown:
        raise FsToolError(
            ErrorCode.INVALID_ARGUMENT,
            f"Unknown argument(s) for {spec.name}: {', '.join(unknown)}. "
            f"Valid arguments: {', '.join(properties)}.",
        )
    kwargs = {key: value for key, value in args.items() if value is not None}  # null = omitted
    for key in spec.parameters["required"]:
        if key not in kwargs:
            raise FsToolError(
                ErrorCode.INVALID_ARGUMENT, f"Missing required argument '{key}' for {spec.name}."
            )
    for key, value in kwargs.items():
        expected = properties[key]["type"]
        if not isinstance(value, _PYTHON_TYPES[expected]):
            raise FsToolError(
                ErrorCode.INVALID_ARGUMENT,
                f"Argument '{key}' for {spec.name} must be a {expected}, "
                f"not {type(value).__name__}.",
            )
    return kwargs


def _wrap_list(files: list) -> dict:
    return _ok({"files": files, "count": len(files)})


def _guard_read_result(result: dict) -> dict:
    """Truncate oversized read_file content so one file can't flood the LLM's context."""
    if not result["success"]:
        return result
    content = result["data"]["content"]
    metadata = result["data"]["metadata"]
    limit = MAX_TOOL_RESULT_CHARS
    metadata["content_truncated"] = len(content) > limit
    if metadata["content_truncated"]:
        result["data"]["content"] = content[:limit]
        metadata["warnings"].append(
            f"Content truncated to the first {limit} of {len(content)} characters; "
            "use search_in_file to find specific text in the rest of the file."
        )
    return result


_POSTPROCESS: dict[str, Callable[[Any], dict]] = {
    "list_files": _wrap_list,
    "read_file": _guard_read_result,
}


def dispatch(name: str, args: Any) -> dict:
    """Run tool ``name`` with ``args`` (the LLM's decoded JSON arguments). Never raises."""
    try:
        spec = _TOOLS.get(name) if isinstance(name, str) else None
        if spec is None:
            return _err(
                UNKNOWN_TOOL,
                f"Unknown tool '{str(name)[:100]}'. Valid tools: {', '.join(TOOL_NAMES)}.",
            )
        result = spec.func(**_validate_args(spec, args))
        result = _POSTPROCESS.get(spec.name, lambda r: r)(result)
        json.dumps(result)  # contract: every result is JSON-serialisable
        return result
    except FsToolError as exc:
        return _err(exc.code, exc.message)
    except Exception:
        logger.debug("Tool %r failed unexpectedly", name, exc_info=True)
        return _err(ErrorCode.INTERNAL_ERROR, "Unexpected internal error while running the tool.")
