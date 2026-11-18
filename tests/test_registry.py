import dataclasses
import inspect
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from helpers import assert_err, assert_ok

from assistant import registry
from assistant.registry import TOOL_NAMES, TOOL_SPECS, dispatch, get_tool_schemas

SCHEMA_KEYS = {"type", "properties", "required", "description", "enum"}


def _dispatch(name: object, args: object) -> dict:
    result = dispatch(name, args)  # type: ignore[arg-type]
    json.dumps(result)  # every result must be JSON-serialisable
    return result


@pytest.fixture
def files(workspace: Path) -> Path:
    (workspace / "resumes").mkdir()
    (workspace / "resumes" / "jane.txt").write_text(
        "Jane Roe\nSkills: Python, JavaScript\n", encoding="utf-8"
    )
    (workspace / "resumes" / "john.md").write_text("John Doe\nSkills: Java\n", encoding="utf-8")
    return workspace


# --- Schemas ------------------------------------------------------------------------------


def test_tool_names_and_order() -> None:
    assert TOOL_NAMES == ("list_files", "read_file", "search_in_file", "write_file")
    assert [s["name"] for s in get_tool_schemas()] == list(TOOL_NAMES)


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
def test_schema_matches_function_signature(spec: registry.ToolSpec) -> None:
    params = inspect.signature(spec.func).parameters

    assert list(spec.parameters["properties"]) == list(params)
    assert spec.parameters["required"] == [
        name for name, p in params.items() if p.default is inspect.Parameter.empty
    ]


def _walk_schema(node: dict, where: str) -> None:
    assert set(node) <= SCHEMA_KEYS, f"{where}: disallowed keys {set(node) - SCHEMA_KEYS}"
    for name, child in node.get("properties", {}).items():
        _walk_schema(child, f"{where}.{name}")


@pytest.mark.parametrize("schema", get_tool_schemas(), ids=lambda s: s["name"])
def test_schemas_use_only_portable_keys(schema: dict) -> None:
    assert set(schema) == {"name", "description", "parameters"}
    params = schema["parameters"]
    assert params["type"] == "object"
    _walk_schema(params, schema["name"])
    for name, prop in params["properties"].items():
        assert prop["type"] in {"string", "boolean"}, name
        assert prop["description"].strip(), name
    assert set(params["required"]) <= set(params["properties"])
    json.dumps(schema)


def test_get_tool_schemas_returns_copies() -> None:
    schemas = get_tool_schemas()
    schemas[0]["parameters"]["properties"].clear()

    assert get_tool_schemas()[0]["parameters"]["properties"]


@pytest.mark.parametrize(
    ("tool", "phrases"),
    [
        ("list_files", ["supported", "does not guarantee", "first"]),
        ("read_file", ["forward slashes", "list_files first", ".pdf", ".docx"]),
        ("search_in_file", ["case-insensitive", "literal", "once per file", "parallel"]),
        ("write_file", [".txt or .md", "overwritten"]),
    ],
)
def test_descriptions_carry_llm_guidance(tool: str, phrases: list[str]) -> None:
    description = next(s["description"] for s in get_tool_schemas() if s["name"] == tool)

    for phrase in phrases:
        assert phrase in description


def test_whole_word_description_mentions_ambiguous_terms() -> None:
    search = next(s for s in get_tool_schemas() if s["name"] == "search_in_file")

    assert "JavaScript" in search["parameters"]["properties"]["whole_word"]["description"]


# --- Dispatch: success --------------------------------------------------------------------


def test_dispatch_list_files_wraps_list_with_count(files: Path) -> None:
    data = assert_ok(_dispatch("list_files", {"directory": "resumes"}))

    assert set(data) == {"files", "count"}
    assert data["count"] == 2
    assert [f["name"] for f in data["files"]] == ["jane.txt", "john.md"]


def test_dispatch_read_file(files: Path) -> None:
    data = assert_ok(_dispatch("read_file", {"filepath": "resumes/jane.txt"}))

    assert data["content"].startswith("Jane Roe")
    assert data["metadata"]["content_truncated"] is False


def test_dispatch_search_in_file(files: Path) -> None:
    args = {"filepath": "resumes/jane.txt", "keyword": "java", "whole_word": True}

    data = assert_ok(_dispatch("search_in_file", args))

    assert data["total_matches"] == 0
    assert data["whole_word"] is True


def test_dispatch_write_file(files: Path) -> None:
    data = assert_ok(_dispatch("write_file", {"filepath": "out/notes.md", "content": "# Hi\n"}))

    assert data["created_dirs"] == ["out"]
    assert (files / "out" / "notes.md").read_text(encoding="utf-8") == "# Hi\n"


def test_null_optional_argument_is_treated_as_omitted(files: Path) -> None:
    listed = assert_ok(_dispatch("list_files", {"directory": "resumes", "extension": None}))
    searched = assert_ok(
        _dispatch(
            "search_in_file",
            {"filepath": "resumes/jane.txt", "keyword": "java", "whole_word": None},
        )
    )

    assert listed["count"] == 2
    assert searched["whole_word"] is False
    assert searched["total_matches"] == 1  # matches "JavaScript" without whole_word


# --- Dispatch: tool errors ----------------------------------------------------------------


def test_list_files_fs_tool_error_becomes_envelope(files: Path) -> None:
    assert_err(_dispatch("list_files", {"directory": "missing"}), "NOT_FOUND")
    assert_err(_dispatch("list_files", {"directory": "../"}), "PATH_OUTSIDE_WORKSPACE")
    assert_err(_dispatch("list_files", {"directory": "", "extension": "p df"}), "INVALID_ARGUMENT")


def test_envelope_tools_pass_their_errors_through(files: Path) -> None:
    assert_err(_dispatch("read_file", {"filepath": "missing.txt"}), "FILE_NOT_FOUND")
    assert_err(
        _dispatch("write_file", {"filepath": "cv.pdf", "content": "x"}), "UNSUPPORTED_FORMAT"
    )
    assert_err(
        _dispatch("search_in_file", {"filepath": "resumes/jane.txt", "keyword": " "}),
        "INVALID_ARGUMENT",
    )


# --- Dispatch: argument validation --------------------------------------------------------


@pytest.mark.parametrize("name", ["delete_file", "", "READ_FILE", None, 3])
def test_unknown_tool(name: object) -> None:
    message = assert_err(_dispatch(name, {}), "UNKNOWN_TOOL")

    for tool in TOOL_NAMES:
        assert tool in message


def test_args_none_for_tool_with_required_params(files: Path) -> None:
    message = assert_err(_dispatch("read_file", None), "INVALID_ARGUMENT")

    assert "filepath" in message


@pytest.mark.parametrize("args", [[], "resumes/jane.txt", 5, ("a",)])
def test_non_dict_args(files: Path, args: object) -> None:
    assert_err(_dispatch("read_file", args), "INVALID_ARGUMENT")


def test_missing_required_argument(files: Path) -> None:
    message = assert_err(
        _dispatch("search_in_file", {"filepath": "resumes/jane.txt"}), "INVALID_ARGUMENT"
    )

    assert "keyword" in message


def test_required_argument_passed_as_null_is_missing(files: Path) -> None:
    message = assert_err(_dispatch("read_file", {"filepath": None}), "INVALID_ARGUMENT")

    assert "filepath" in message


def test_unknown_argument(files: Path) -> None:
    message = assert_err(
        _dispatch("read_file", {"filepath": "resumes/jane.txt", "encoding": "utf-8"}),
        "INVALID_ARGUMENT",
    )

    assert "encoding" in message


@pytest.mark.parametrize(
    ("tool", "args", "bad"),
    [
        ("read_file", {"filepath": 5}, "filepath"),
        ("list_files", {"directory": ["resumes"]}, "directory"),
        (
            "search_in_file",
            {"filepath": "a.txt", "keyword": "x", "whole_word": "true"},
            "whole_word",
        ),
        ("search_in_file", {"filepath": "a.txt", "keyword": "x", "whole_word": 1}, "whole_word"),
        ("search_in_file", {"filepath": "a.txt", "keyword": True}, "keyword"),
        ("write_file", {"filepath": "a.txt", "content": {"text": "x"}}, "content"),
    ],
)
def test_wrong_argument_type(files: Path, tool: str, args: dict, bad: str) -> None:
    message = assert_err(_dispatch(tool, args), "INVALID_ARGUMENT")

    assert f"'{bad}'" in message


# --- Dispatch: unexpected failures --------------------------------------------------------


def _replace_func(monkeypatch: pytest.MonkeyPatch, tool: str, func: Callable[..., object]) -> None:
    spec = registry._TOOLS[tool]
    monkeypatch.setitem(registry._TOOLS, tool, dataclasses.replace(spec, func=func))


def test_unexpected_exception_is_internal_error(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**kwargs: object) -> dict:
        raise RuntimeError("sensitive internal detail")

    _replace_func(monkeypatch, "read_file", boom)

    message = assert_err(_dispatch("read_file", {"filepath": "resumes/jane.txt"}), "INTERNAL_ERROR")
    assert "sensitive internal detail" not in message


def test_non_serialisable_result_is_internal_error(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _replace_func(monkeypatch, "read_file", lambda **kwargs: {"success": True, "data": {1, 2}})

    assert_err(_dispatch("read_file", {"filepath": "resumes/jane.txt"}), "INTERNAL_ERROR")


# --- Context guard ------------------------------------------------------------------------


def test_read_file_content_over_cap_is_truncated(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "MAX_TOOL_RESULT_CHARS", 10)

    data = assert_ok(_dispatch("read_file", {"filepath": "resumes/jane.txt"}))

    assert data["content"] == "Jane Roe\nS"
    assert data["metadata"]["content_truncated"] is True
    assert any("truncated" in w for w in data["metadata"]["warnings"])
    assert data["metadata"]["char_count"] > 10  # metadata still describes the whole file


def test_read_file_content_at_cap_is_not_truncated(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = (files / "resumes" / "jane.txt").read_text(encoding="utf-8")
    monkeypatch.setattr(registry, "MAX_TOOL_RESULT_CHARS", len(content))

    data = assert_ok(_dispatch("read_file", {"filepath": "resumes/jane.txt"}))

    assert data["content"] == content
    assert data["metadata"]["content_truncated"] is False
    assert data["metadata"]["warnings"] == []


def test_default_result_cap() -> None:
    assert registry.MAX_TOOL_RESULT_CHARS == 20_000
