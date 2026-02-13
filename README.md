# agent-recall

Persistent memory for AI coding agents. Learn from past sessions and share knowledge across your team.

## Installation

```bash
pip install agent-recall
```

## Compatibility

### Python Versions

| Python | Status |
|--------|--------|
| 3.11   | Supported |
| 3.12   | Supported |
| 3.13   | Supported |

### Agent Sources

| Source | macOS | Linux | Windows | Status |
|--------|-------|-------|---------|--------|
| Cursor | ✓ | ✓ | ✓ | Stable |
| Claude Code | ✓ | ✓ | ✓ | Stable |
| OpenCode | ✓ | ✓ | ✓ | Stable |
| OpenAI Codex | ✓ | ✓ | ✓ | Stable |

### LLM Providers

| Provider | Required Package | Notes |
|----------|-----------------|-------|
| Anthropic | `pip install agent-recall[anthropic]` | Claude models |
| OpenAI | Built-in | GPT models |
| Google | `pip install agent-recall[google]` | Gemini models |
| Ollama | Built-in | Local models via OpenAI-compatible API |
| vLLM | Built-in | Self-hosted models |
| LM Studio | Built-in | Local models |

## Quick Start

```bash
# Initialize in your repo
agent-recall init

# First run: launch the TUI to complete onboarding
# (provider selection, API key setup, repo + agent source confirmation)
agent-recall tui
# TUI keybindings:
#   Ctrl+P  command palette
#   Ctrl+,  settings dialog

# Start a session
agent-recall start "implementing user authentication"

# Log observations during work
agent-recall log "JWT tokens work better than sessions here" --label pattern --tags auth,jwt
agent-recall log "Don't use bcrypt < 4.0, has vulnerability" --label hard_failure --tags auth,security

# End session
agent-recall end "Implemented JWT auth with refresh tokens"

# Get context for a new session
agent-recall context --task "add password reset"

# Refresh persisted context bundles for active task/repo state
agent-recall refresh-context

# Generate adapter-ready context payloads for supported agents
agent-recall refresh-context --adapter-payloads

# Run compaction to update knowledge files
agent-recall compact

# Auto-ingest Cursor / Claude Code / OpenCode / Codex sessions
agent-recall sync --verbose

# Force Cursor DB during testing
agent-recall sync --source cursor --cursor-db-path "/path/to/state.vscdb" --no-compact --verbose

# Force OpenCode storage root during testing
agent-recall sync --source opencode --no-compact --verbose

# Sync only OpenAI Codex sessions
agent-recall sync --source codex --no-compact --verbose
```

## How It Works

1. Start sessions with a task description.
2. Log observations with semantic labels (`hard_failure`, `gotcha`, `pattern`, `preference`, etc.).
3. End sessions with a summary.
4. Compaction uses an LLM to synthesize logs into `GUARDRAILS.md`, `STYLE.md`, and `RECENT.md`.
5. Context retrieves relevant knowledge for new sessions.

## Configuration

Edit `.agent/config.yaml`:

```yaml
llm:
  provider: anthropic  # anthropic, openai, google, ollama, vllm, lmstudio, openai-compatible
  model: claude-sonnet-4-20250514
  base_url: null       # for ollama: http://localhost:11434/v1
```

## Shared storage (team mode)

Opt-in to shared memory by switching storage backend to shared and configuring the shared connection.

```yaml
storage:
  backend: shared
  shared:
    base_url: "file:///path/to/shared/agent-recall.db"  # file://, sqlite://, https://
    tenant_id: "org-123"
    project_id: "repo-abc"
    api_key_env: AGENT_RECALL_SHARED_API_KEY
```

Migration path for existing local memory:

1. Pick a shared backend URL (`file://`, `sqlite://`, or `https://`).
2. Run `agent-recall compact` and `agent-recall sync` once with `storage.backend: shared` to seed shared storage.
3. Point additional machines at the same `base_url` and namespace (tenant/project).

You can also configure providers via CLI:

```bash
agent-recall providers
agent-recall config model --provider ollama --model llama3.1
agent-recall config model --temperature 0.2 --max-tokens 8192
agent-recall test-llm
```

## Onboarding and Secrets

- `agent-recall tui` runs onboarding automatically the first time a repository is opened.
- `agent-recall config setup` reruns onboarding manually (`--quick` applies saved defaults).
- The TUI uses a command palette (`Ctrl+P`) with searchable actions and keybinding hints.
- Type any command in the palette search and run it directly from there.
- Setup and model configuration are available as command-palette actions.
- Use the settings dialog (`Ctrl+,`) to switch views and runtime TUI preferences.
- On interactive onboarding, models are fetched live from the selected provider and shown in a picker.
- On interactive onboarding, you can tune `temperature` and `max_tokens` (helpful for local models).
- API keys are stored locally in `secrets.yaml` under your agent-recall home directory.
- Set `AGENT_RECALL_HOME` to override where onboarding settings and secrets are stored.

## Labels

- `hard_failure` - Never do this, it broke
- `gotcha` - Non-obvious issue to watch for
- `correction` - Do Y instead of X
- `preference` - Team prefers X over Y
- `pattern` - Useful pattern to follow
- `decision` - Why we chose X (kept long-form)
- `exploration` - Tried X, result was Y
- `narrative` - General session notes

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ty check
```
