"""Part A: sandboxed file-system tools for resume files (PDF, TXT, DOCX).

Exposes four tools intended to be called by an LLM via function calling:

- ``read_file(filepath) -> dict``
- ``list_files(directory, extension=None) -> list``
- ``write_file(filepath, content) -> dict``
- ``search_in_file(filepath, keyword) -> dict``

All paths are resolved inside ``WORKSPACE_ROOT``. See DESIGN.md for the
result envelope, error codes and sandbox rules. Implementations land in
later steps of project-plan.md.
"""
