# Agent Recall Super Ralph

This is a dedicated Ralph setup for the Agent Recall project.

It differs from the generic loop by enforcing per-iteration memory evolution in:

- `.agent/RULES.md` (user-authored policy)
- `.agent/GUARDRAILS.md`
- `.agent/STYLE.md`
- `.agent/RECENT.md`

`RULES.md` is user-maintained and injected into every loop prompt.
`GUARDRAILS.md`, `STYLE.md`, and `RECENT.md` are system-updated from iteration reports every iteration.

## Structure

- `agent_recall/scripts/ralph-agent-recall-loop.sh`: main loop
- `agent_recall/scripts/ralph-agent-recall-once.sh`: one-iteration HITL wrapper
- `agent_recall/ralph/agent-prompt.md`: base prompt template
- `agent_recall/ralph/prd.json`: Agent Recall PRD template
- `agent_recall/ralph/progress.txt`: append-only iteration log

## Quick Start (HITL)

```bash
./agent_recall/scripts/ralph-agent-recall-once.sh \
  --agent-cmd 'codex exec --prompt-file {prompt_file}' \
  --validate-cmd 'uv run pytest && uv run ruff check . && uv run ty check'
```

## Quick Start (AFK)

```bash
./agent_recall/scripts/ralph-agent-recall-loop.sh \
  --agent-cmd 'codex exec --prompt-file {prompt_file}' \
  --validate-cmd 'uv run pytest && uv run ruff check . && uv run ty check' \
  --max-iterations 20
```

## Compaction Integration

Default compaction command per iteration:

```bash
uv run agent-recall compact
```

Behavior is controlled by:

- `--compact-mode always` (default)
- `--compact-mode on-failure`
- `--compact-mode off`
- `--compact-cmd 'uv run agent-recall compact --force'`

## Memory Path Options

Defaults target the current repo's `.agent` directory. Override if needed:

```bash
--memory-dir .agent \
--rules-file .agent/RULES.md \
--guardrails-file .agent/GUARDRAILS.md \
--style-file .agent/STYLE.md \
--recent-file .agent/RECENT.md
```

## Notes

- Prompt files are generated under `agent_recall/ralph/.runtime/`.
- `{prompt_file}` in `--agent-cmd` is replaced by the generated prompt path.
- Completion markers still gate exit: `<promise>COMPLETE</promise>` / `<promise>NO MORE TASKS</promise>`.
- Abort marker exits with failure: `<promise>ABORT</promise>`.
