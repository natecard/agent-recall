# AGENTS.md

## Cursor Cloud specific instructions

**Project**: `agent-recall` — a Python CLI tool providing persistent memory for AI coding agents. Single-package project (not a monorepo).

### Development commands

All commands are run via `uv run` from the workspace root. See `pyproject.toml` `[project.optional-dependencies]` for extras and the README `## Development` section for the canonical list.

| Task | Command |
|---|---|
| Install deps | `uv sync --extra dev --extra all` |
| Lint | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Type check | `uv run ty check` |
| Tests | `uv run pytest` |
| CLI help | `uv run agent-recall --help` |

### Non-obvious caveats

- **No external services required**: All tests are self-contained — SQLite is embedded and LLM calls are mocked with `respx`. No databases, Docker, or API keys needed to run the full test suite.
- **`uv` must be on PATH**: After installation via `curl -LsSf https://astral.sh/uv/install.sh | sh`, the binary lands in `~/.local/bin`. Ensure `PATH` includes that directory.
- **Python 3.11+ required**: The project targets `>=3.11`. The system Python 3.12 works fine.
- **Pre-commit hooks**: The repo uses `pre-commit` with ruff + ty + pytest (pre-push). Install with `uv run pre-commit install && uv run pre-commit install --hook-type pre-push` if you want local hooks, but CI covers the same checks.
- **CLI requires a git repo with `.agent/` init**: To exercise the CLI (e.g. `agent-recall init`, `start`, `log`, `end`), run inside a git-initialized directory. Use a temp directory like `/tmp/test-recall` to avoid polluting the workspace.
