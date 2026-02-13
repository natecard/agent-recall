# agent-recall

Persistent memory for coding agents in your repository.

`agent-recall` helps you capture what worked, what failed, and team preferences so future sessions start with context instead of guesswork.

## Install

```bash
pip install agent-recall
```

Optional provider extras:

```bash
pip install "agent-recall[anthropic]"  # Anthropic Claude API
pip install "agent-recall[google]"     # Google Gemini API
```

Python 3.11+ is supported.

## Quick Start (Open-First)

```bash
# 1) Initialize in your repository
agent-recall init

# 2) Open the TUI (recommended entrypoint)
#    Runs onboarding when needed.
agent-recall open

# 3) Start a focused session
agent-recall start "implementing user authentication"

# 4) Log important learnings while you work
agent-recall log "JWT worked better than server sessions" --label pattern --tags auth,jwt
agent-recall log "bcrypt < 4.0 caused issues here" --label hard_failure --tags auth,security

# 5) End the session with a summary
agent-recall end "Implemented JWT auth with refresh tokens"

# 6) Retrieve context for your next task
agent-recall context --task "add password reset"
```

## Core Workflow

Use these commands most often:

- `agent-recall open` - launches the TUI dashboard and onboarding flow.
- `agent-recall start` / `agent-recall log` / `agent-recall end` - session lifecycle.
- `agent-recall sync --verbose` - ingest conversations from configured sources.
- `agent-recall compact` - synthesize logs into knowledge files.
- `agent-recall context` and `agent-recall refresh-context` - retrieve and persist task context.

### Sync examples

```bash
# Sync enabled sources from config
agent-recall sync --verbose

# Sync only one source
agent-recall sync --source codex --no-compact --verbose

# Cursor testing override
agent-recall sync --source cursor --cursor-db-path "/path/to/state.vscdb" --no-compact --verbose
```

## TUI Keybindings

- `Ctrl+P` command palette
- `Ctrl+G` settings/preferences
- `Ctrl+K` run knowledge update
- `Ctrl+Y` sync conversations
- `Ctrl+T` theme picker
- `Ctrl+Q` quit

## Configuration

Repository config lives in `.agent/config.yaml`.

```yaml
llm:
  provider: openai  # anthropic, openai, google, ollama, vllm, lmstudio, openai-compatible
  model: gpt-5-mini
  base_url: null    # e.g. http://localhost:11434/v1 for ollama
```

You can manage provider/model settings from CLI:

```bash
agent-recall providers
agent-recall config model --provider ollama --model llama3.1
agent-recall config model --temperature 0.2 --max-tokens 8192
agent-recall test-llm
```

## Storage Modes

Default mode is local SQLite. For shared/team memory, switch to shared storage:

```yaml
storage:
  backend: shared
  shared:
    base_url: "file:///path/to/shared/agent-recall.db"  # file://, sqlite://, or https://
    tenant_id: "org-123"
    project_id: "repo-abc"
    api_key_env: AGENT_RECALL_SHARED_API_KEY
```

Migration path from local to shared:

1. Configure `storage.backend: shared` and your `shared` block.
2. Run `agent-recall sync` and `agent-recall compact` once to seed shared state.
3. Point additional machines to the same `base_url` + tenant/project namespace.

## Onboarding and Secrets

- `agent-recall open` runs onboarding automatically if repo setup is incomplete.
- `agent-recall config setup` reruns onboarding manually (`--quick` applies defaults).
- Interactive onboarding fetches provider models and lets you tune `temperature` and `max_tokens`.
- API keys are stored in `secrets.yaml` under your agent-recall home.
- Set `AGENT_RECALL_HOME` to override where onboarding state and secrets are stored.

## Labels

- `hard_failure` - never do this; it broke
- `gotcha` - non-obvious issue to watch for
- `correction` - do Y instead of X
- `preference` - team preference
- `pattern` - reusable pattern
- `decision` - reasoning behind a choice
- `exploration` - experiment and outcome
- `narrative` - general notes

## Compatibility

### Agent sources

- Cursor
- Claude Code
- OpenCode
- OpenAI Codex

### LLM providers

- OpenAI (built-in)
- Anthropic (`agent-recall[anthropic]`)
- Google (`agent-recall[google]`)
- Ollama (OpenAI-compatible endpoint)
- vLLM (OpenAI-compatible endpoint)
- LM Studio (OpenAI-compatible endpoint)

## Advanced Capabilities

Advanced command families (background sync, curation queue, tier maintenance, command inventory, Ralph loop automation, write-* helpers) are available, but intentionally kept out of the quick-start path.

- Getting started guide: `docs/getting-started.md`
- Full command reference: `docs/cli-reference.md`

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ty check
```

Pre-commit hooks (ruff format → ruff check --fix → ruff format → ty) catch most CI failures before commit. A pre-push hook runs pytest before pushing:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```
