# CLI Reference

- `agent-recall init [--force]`
- `agent-recall config setup [--force] [--quick]`
- `agent-recall start <task>`
- `agent-recall log <content> --label <semantic_label> [--tags x,y]`
- `agent-recall end <summary>`
- `agent-recall context [--task <task>] [--format md|json]`
- `agent-recall refresh-context [--task <task>] [--output-dir PATH]`
- `agent-recall compact [--force]`
- `agent-recall sync [--source cursor|claude-code|opencode|codex] [--since-days N] [--session-id ID --session-id ID2] [--max-sessions N] [--compact/--no-compact] [--cursor-db-path PATH] [--cursor-storage-dir PATH] [--all-cursor-workspaces] [--verbose]`
- `agent-recall sessions [--source cursor|claude-code|opencode|codex] [--since-days N] [--session-id ID --session-id ID2] [--max-sessions N] [--format table|json] [--cursor-db-path PATH] [--cursor-storage-dir PATH] [--all-cursor-workspaces]`
- `agent-recall tui [--onboarding/--no-onboarding] [--force-onboarding] [--refresh-seconds N] [--all-cursor-workspaces]`
- `agent-recall reset-sync [--source cursor|claude-code|opencode|codex] [--session-id ID]`
- `agent-recall sources [--all-cursor-workspaces] [--max-sessions N]`
- `agent-recall status`
- `agent-recall retrieve <query> [--top-k N]`
- `agent-recall ingest <path> [--source-session-id ID]`
- `agent-recall providers`
- `agent-recall config model [--provider P] [--model M] [--base-url URL] [--temperature T] [--max-tokens N] [--validate/--no-validate]`
- `agent-recall test-llm`
- `agent-recall theme list|set|show`

Inside `agent-recall tui`, use keybindings:
- `Ctrl+P` opens a searchable command palette.
- `Ctrl+,` opens settings.
- In command palette, search both actions and CLI commands; Enter runs the selection.
- In command palette, typing a full command and pressing Enter can run it directly.
- Setup and model configuration are available as command-palette actions.
