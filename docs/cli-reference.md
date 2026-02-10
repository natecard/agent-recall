# CLI Reference

- `agent-recall init [--force]`
- `agent-recall start <task>`
- `agent-recall log <content> --label <semantic_label> [--tags x,y]`
- `agent-recall end <summary>`
- `agent-recall context [--task <task>] [--format md|json]`
- `agent-recall compact [--force]`
- `agent-recall sync [--source cursor|claude-code] [--since-days N] [--compact/--no-compact] [--cursor-db-path PATH] [--cursor-storage-dir PATH] [--all-cursor-workspaces] [--verbose]`
- `agent-recall reset-sync [--source cursor|claude-code] [--session-id ID]`
- `agent-recall sources [--all-cursor-workspaces]`
- `agent-recall status`
- `agent-recall retrieve <query> [--top-k N]`
- `agent-recall ingest <path> [--source-session-id ID]`
- `agent-recall providers`
- `agent-recall config-llm [--provider P] [--model M] [--base-url URL] [--validate/--no-validate]`
- `agent-recall test-llm`
- `agent-recall theme list|set|show`
