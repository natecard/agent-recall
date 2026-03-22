# CLI Reference

- `agent-recall --version` / `agent-recall -v`
- `agent-recall init [--force]`
- `agent-recall open` (recommended dashboard entrypoint, runs onboarding)
- `agent-recall tui [--no-onboarding] [--all-cursor-workspaces]`
- `agent-recall config setup [--force] [--quick]`
- `agent-recall status`
- `agent-recall sync [--source S] [--since-days N] [--compact/--no-compact] [--verbose] [--cursor-db-path PATH]`
- `agent-recall sync background [--interval-minutes N]`
- `agent-recall start <task>`
- `agent-recall log <content> --label <semantic_label> [--tags x,y]`
- `agent-recall end <summary>`
- `agent-recall compact [--force]`
- `agent-recall context [--task <task>] [--format md|json] [--top-k N] [--backend fts5|hybrid] [--rerank/--no-rerank]`
- `agent-recall context refresh [--task <task>] [--adapter-payloads/--no-adapter-payloads]`
- `agent-recall retrieve <query> [--top-k N] [--backend fts5|hybrid]`
- `agent-recall sources [--all-cursor-workspaces] [--max-sessions N]`
- `agent-recall sessions [--source S] [--since-days N] [--format table|json]`
- `agent-recall sync reset [--source S] [--session-id ID]`
- `agent-recall ingest <path> [--source-session-id ID]`
- `agent-recall command-inventory`
- `agent-recall metrics report [--limit N] [--format table|json]`
- `agent-recall providers`
- `agent-recall config model [--provider P] [--model M] [--base-url URL] [--temperature T] [--max-tokens N]`
- `agent-recall config adapters [--enabled/--disabled] [--token-budget N] [--per-adapter-token-budget name=N]`
- `agent-recall test-llm`
- `agent-recall theme list|set|show`
- `agent-recall curation list|approve|reject`
- `agent-recall tiers compact|lint|stats`
- `agent-recall tiers write guardrails|guardrails-failure|style|recent`
- `agent-recall external-compaction list|export|apply|patch-preview|apply-approved|mcp-server|cleanup-state`
- `agent-recall external-compaction queue add|list|approve|reject`
- `agent-recall ralph status|enable|disable [--max-iterations N] [--sleep-seconds N]`
- `agent-recall ralph run --agent-cmd <cmd> [--max-iterations N] [--compact-mode always|on-failure|off] [--agent-transport pipe|pty|auto]`

Breaking changes:
- Removed deprecated commands: `agent-recall onboard`, `agent-recall config-llm`.
- Renamed config keys:
  - `onboarding.selected_agents` -> `onboarding.selected_sources`
  - `adapters.token_budget` -> `adapters.default_token_budget`
  - `retrieval.embedding_enabled` -> `retrieval.semantic_index_enabled`

Inside `agent-recall open` / `tui`, use keybindings:
- `Ctrl+P` opens a searchable command palette.
- `Ctrl+G` opens settings/preferences.
- `Ctrl+K` runs a full knowledge update (sync + compact).
- `Ctrl+Y` runs sync + compact.
- `Ctrl+R` refreshes the dashboard immediately.
- `Ctrl+T` opens the theme picker.
- `Ctrl+Q` quits the TUI.

In the command palette, search both actions and CLI commands; Enter runs the selection. Typing a full command and pressing Enter runs it directly. Setup and model configuration are also available as palette actions.

Ralph transport troubleshooting:
- Default transport is `pipe` to avoid macOS PTY `script` write-master false failures.
- Use `--agent-transport pty` (or `RALPH_AGENT_TRANSPORT=pty`) only when PTY rendering is needed.
- `--agent-output-mode stream-json` always uses `pipe` transport to preserve JSON marker parsing.

Telemetry:
- `agent-recall metrics report` summarizes local pipeline telemetry (`ingest`, `extract`, `compact`, `apply`).
- Telemetry event schema and interpretation guide: `docs/telemetry.md`.

External compaction flow (safe defaults):
1. Run `agent-recall sync --compact`.
2. Export pending conversations: `agent-recall external-compaction export --pending-only > notes.json`.
3. Apply generated notes safely:
`agent-recall external-compaction apply --input notes.json` (dry-run default)
`agent-recall external-compaction apply --input notes.json --commit` (writes enabled)
4. Optional review queue flow:
`agent-recall external-compaction queue add --input notes.json`
`agent-recall external-compaction queue list --state pending`
`agent-recall external-compaction queue approve --id <id>`
`agent-recall external-compaction patch-preview --state approved`
`agent-recall external-compaction apply-approved --commit`

Template writes are disabled by default. Enable `compaction.external.allow_template_writes: true`
before using `--write-target templates`.
