"""Part B: CLI entry point for the LLM file assistant.

Runs an interactive REPL (or a one-shot ``--query``) in which an LLM answers
questions about resumes in the workspace by calling the tools in ``fs_tools``.
Every tool call is rendered to the terminal for visibility. See DESIGN.md for
the agent loop and provider architecture.
"""
