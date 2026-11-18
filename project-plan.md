# Project plan

Vertical slices; each one leaves `ruff check .`, `ruff format --check .` (and, from step 3,
`pytest`) green. See [DESIGN.md](DESIGN.md) for the decisions behind each step.

- [x] **1. Scaffold + docs + CI (lint)** – layout, requirements, `pyproject.toml`, `.env.example`,
      `.gitignore`, DESIGN.md, this plan, GitHub Actions running ruff.
- [x] **2. Sample data** – `scripts/generate_samples.py` producing 8 resumes in
      `sample_data/resumes/` (3 PDF, 3 DOCX, 2 TXT) with varied skills, plus
      `sample_data/edge_cases/` (corrupt PDF, empty file).
- [x] **3. Sandbox + envelope + read_file** – path sandbox, result envelope, `FsToolError`,
      `read_file` for TXT/PDF/DOCX + tests; add `pytest` to CI.
- [x] **4. list_files** + tests.
- [x] **5. write_file** (atomic, text-only, parent dirs, `overwritten` flag) + tests.
- [ ] **6. search_in_file** (shared extraction, case-insensitive, positions + context) + tests.
- [ ] **7. Tool registry** – JSON schemas + dispatcher (exceptions → envelope) + tests.
- [ ] **8. Providers** – `LLMProvider` ABC + provider registry + Gemini provider (AFC disabled) +
      `FakeProvider`.
- [ ] **9. Agent loop** + tests with `FakeProvider` (multi-call turns, errors, iteration cap).
- [ ] **10. CLI** – REPL, `--query`, rich tool-call trace.
- [ ] **11. README + demo** – setup, usage, architecture, example queries, demo script.
