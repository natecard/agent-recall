# Agent Recall Ralph Task

You are running an autonomous Ralph loop for Agent Recall.

The loop injects current PRD data, recent progress, memory files, recent Ralph commits, and the CI gate command.

## Context Provided

- Read-only tier files: `.agent/GUARDRAILS.md`, `.agent/STYLE.md`, `.agent/RECENT.md`
- PRD state with unpassed items
- Iteration report path: `{current_report_path}`
- Current task: `{item_id} - {item_title}`
- Task description: `{description}`
- Validation command: `{validation_command}`

Tier files are read-only context. Do not write to them. The system updates them from iteration reports.

## Task Selection

Choose the next task using this order:

1. Critical bugfixes
2. Tracer-bullet slices for new capabilities
3. Polish and quick wins
4. Refactors

Tracer-bullet means: deliver a tiny end-to-end slice first to get feedback early.

After selecting the task:

- Re-rank unpassed PRD items by setting numeric `priority` values.
- Keep only one selected item in active implementation scope.

## Execution

Explore the codebase before editing. Pull in only the context needed to complete the selected task safely.

Complete the selected task.

If the task is larger than expected, print `HANG ON A SECOND`, then reduce scope and complete the smallest viable chunk for this iteration.

Do not start a second feature in the same iteration.

## Required Output

Update `{current_report_path}` with this JSON schema (agent-writable fields only):

```json
{
  "outcome": "COMPLETED|VALIDATION_FAILED|SCOPE_REDUCED|BLOCKED|TIMEOUT",
  "summary": "string",
  "failure_reason": "string|null",
  "gotcha_discovered": "string|null",
  "pattern_that_worked": "string|null",
  "scope_change": "string|null"
}
```

## What NOT To Do

❌ Do NOT write to `.agent/GUARDRAILS.md`, `.agent/STYLE.md`, or `.agent/RECENT.md`.
❌ Do NOT append to `agent_recall/ralph/progress.txt`.
❌ Do NOT start a second feature.

## Feedback Loops

Before commit, run the validation command provided in loop context and keep it green.

If validation fails, fix or reduce scope until passing.

## Commit

Create one commit for this iteration.

Commit message requirements:

1. Start with `RALPH:`
2. Include completed task and PRD reference
3. Include key decisions
4. Include key files changed
5. Include blockers/notes for next iteration (if any)

## Completion / Abort

If no further PRD work remains and validation is green, output the exact completion marker from loop context.

If blocked and unable to proceed safely, output the exact abort marker from loop context.

If a tool (for example, TodoWrite) is unavailable in your environment, skip it and proceed without it.

ONLY WORK ON A SINGLE TASK.
