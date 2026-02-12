# Agent Recall Ralph Task

You are running an autonomous Ralph loop for Agent Recall.

The loop injects current PRD data, recent progress, memory files, recent Ralph commits, and the CI gate command.

## Task Breakdown

Parse unpassed PRD items and break candidate work into the smallest practical tasks.

A PRD item can contain multiple steps. You should still complete only one tightly scoped task this iteration.

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

## Exploration

Explore the codebase before editing. Pull in only the context needed to complete the selected task safely.

## Execution

Complete the selected task.

If the task is larger than expected, print `HANG ON A SECOND`, then reduce scope and complete the smallest viable chunk for this iteration.

Do not start a second feature in the same iteration.

## Feedback Loops

Before commit, run the validation command provided in loop context and keep it green.

If validation fails, fix or reduce scope until passing.

## Required State Updates

- Append a timestamped, concise entry to `agent_recall/ralph/progress.txt` (append-only).
- Update `agent_recall/ralph/prd.json`:
  - Adjust priorities to reflect current ordering.
  - Set `passes: true` only when acceptance criteria are truly complete.
- Update all memory files every iteration:
  - `.agent/GUARDRAILS.md`: specific do/don't rules from failures, risks, or near-misses.
  - `.agent/STYLE.md`: concrete coding patterns/preferences that worked.
  - `.agent/RECENT.md`: what changed, why, and what the next iteration should know.

Keep entries short, specific, and actionable.

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

ONLY WORK ON A SINGLE TASK.
