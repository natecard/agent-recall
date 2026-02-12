# Agent Recall Ralph Task

You are running an autonomous Ralph loop for Agent Recall.

Your goal each iteration is to complete one focused PRD item while improving project memory quality.

## Core Rules

- Work ONLY on the single target PRD item for this iteration.
- Keep changes small and reviewable.
- Run validation before committing.
- Commit with `RALPH` in the message.

## Required State Updates

- Append a concise entry to `agent_recall/ralph/progress.txt`.
- Update PRD item `passes` to `true` only when complete.

## Memory Update Requirement

Every iteration must improve memory quality by updating these files:

- `.agent/GUARDRAILS.md`: add concrete "do/don't" rules from failures or near-misses.
- `.agent/STYLE.md`: add coding patterns/preferences that worked well.
- `.agent/RECENT.md`: append a timestamped summary of what changed and why.

Keep entries short, specific, and actionable.

## Completion Protocol

When there is no further PRD work left, output the exact completion marker provided in loop context.
If blocked and unable to proceed safely, output the exact abort marker provided in loop context.
