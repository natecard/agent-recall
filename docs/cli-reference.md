# CLI Reference

- `agent-recall --version` / `agent-recall -v`
- `agent-recall init [--force]`
- `agent-recall open` (recommended dashboard entrypoint, runs onboarding)
- `agent-recall tui [--no-onboarding] [--refresh-seconds N] [--all-cursor-workspaces]`
- `agent-recall config setup [--force] [--quick]`
- `agent-recall status`
- `agent-recall sync [--source S] [--since-days N] [--compact/--no-compact] [--verbose] [--cursor-db-path PATH]`
- `agent-recall sync-background [--interval-minutes N]`
- `agent-recall start <task>`
- `agent-recall log <content> --label <semantic_label> [--tags x,y]`
- `agent-recall end <summary>`
- `agent-recall compact [--force]`
- `agent-recall context [--task <task>] [--format md|json] [--top-k N] [--backend fts5|hybrid] [--rerank/--no-rerank]`
- `agent-recall refresh-context [--task <task>] [--adapter-payloads/--no-adapter-payloads]`
- `agent-recall retrieve <query> [--top-k N] [--backend fts5|hybrid]`
- `agent-recall sources [--all-cursor-workspaces] [--max-sessions N]`
- `agent-recall sessions [--source S] [--since-days N] [--format table|json]`
- `agent-recall reset-sync [--source S] [--session-id ID]`
- `agent-recall ingest <path> [--source-session-id ID]`
- `agent-recall command-inventory`
- `agent-recall providers`
- `agent-recall config model [--provider P] [--model M] [--base-url URL] [--temperature T] [--max-tokens N]`
- `agent-recall config adapters [--enabled/--disabled] [--token-budget N] [--per-adapter-token-budget name=N]`
- `agent-recall test-llm`
- `agent-recall theme list|set|show`
- `agent-recall curation list|approve|reject`
- `agent-recall compact-tiers / lint-tiers / tier-stats`
- `agent-recall write-guardrails / write-style / write-recent`
- `agent-recall ralph status|enable|disable [--max-iterations N] [--sleep-seconds N]`

Inside `agent-recall open` / `tui`, use keybindings:
- `Ctrl+P` opens a searchable command palette.
- `Ctrl+G` opens settings/preferences.
- `Ctrl+K` runs a full knowledge update (sync + compact).
- `Ctrl+Y` runs sync only.
- `Ctrl+R` refreshes the dashboard immediately.
- `Ctrl+T` opens the theme picker.
- `Ctrl+Q` quits the TUI.

In the command palette, search both actions and CLI commands; Enter runs the selection. Typing a full command and pressing Enter runs it directly. Setup and model configuration are also available as palette actions.
