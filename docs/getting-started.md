# Getting Started

1. Install `agent-recall`.
2. Run `agent-recall init` in your repository.
3. Run `agent-recall open` to launch the dashboard and complete onboarding (provider selection, API keys).
4. While in the TUI, use `Ctrl+P` for the command palette and `Ctrl+G` for settings.
5. Start a session: `agent-recall start "task description"`.
6. Log learnings: `agent-recall log "content" --label pattern`.
7. Sync and compact: Run `agent-recall sync` to ingest conversations and then `agent-recall compact` to synthesize knowledge into `GUARDRAILS.md`, `STYLE.md`, and `RECENT.md`.
8. Retrieve context: `agent-recall context --task "new task"` to get relevant context for your next agent session.
