#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./agent_recall/scripts/ralph-agent-recall-loop.sh --agent-cmd '<command>' [options]

Required:
  --agent-cmd CMD              Agent command. If CMD contains {prompt_file}, it is replaced
                               with the generated prompt path. Otherwise prompt is piped via stdin.

Core options:
  --validate-cmd CMD           Validation command to keep CI green
  --agent-output-mode MODE     Agent output format: plain|stream-json (default: plain)
  --prd-file PATH              PRD JSON path (default: agent_recall/ralph/prd.json)
  --progress-file PATH         Progress log path (default: agent_recall/ralph/progress.txt)
  --prompt-template PATH       Base prompt template (default: agent_recall/ralph/agent-prompt.md)
  --max-iterations N           Max loop iterations (default: 10)
  --sleep-seconds N            Sleep between iterations (default: 2)

Markers:
  --complete-marker TEXT       Primary completion marker (default: <promise>COMPLETE</promise>)
  --alt-complete-marker TEXT   Secondary completion marker (default: <promise>NO MORE TASKS</promise>)
  --abort-marker TEXT          Abort marker (default: <promise>ABORT</promise>)

Context options:
  --recent-commit-count N      Number of prior Ralph commits to include (default: 10)
  --recent-commit-grep TEXT    grep pattern for prior Ralph commits (default: RALPH)
  --progress-tail-lines N      Progress lines included in prompt (default: 200)
  --memory-tail-lines N        Memory file lines included in prompt (default: 120)

Memory options:
  --memory-dir PATH            Memory directory (default: .agent)
  --guardrails-file PATH       Guardrails file path (default: .agent/GUARDRAILS.md)
  --style-file PATH            Style file path (default: .agent/STYLE.md)
  --recent-file PATH           Recent file path (default: .agent/RECENT.md)

Agent Recall options:
  --compact-cmd CMD            Compaction command (default: 'uv run agent-recall compact')
  --compact-mode MODE          Compact frequency: always|on-failure|off (default: always)

Automation options:
  --pre-iteration-cmd CMD      Command run before each iteration (cleanup/hooks)
  --notify-cmd CMD             Optional command on exit; supports {iterations} and {status}

Misc:
  -h, --help                   Show help

Environment fallbacks:
  RALPH_AGENT_CMD
  RALPH_VALIDATE_CMD
USAGE
}

AGENT_CMD="${RALPH_AGENT_CMD:-}"
VALIDATE_CMD="${RALPH_VALIDATE_CMD:-}"
AGENT_OUTPUT_MODE="plain"
PRD_FILE="agent_recall/ralph/prd.json"
PROGRESS_FILE="agent_recall/ralph/progress.txt"
PROMPT_TEMPLATE="agent_recall/ralph/agent-prompt.md"
MAX_ITERATIONS=10
SLEEP_SECONDS=2
COMPLETE_MARKER="<promise>COMPLETE</promise>"
ALT_COMPLETE_MARKER="<promise>NO MORE TASKS</promise>"
ABORT_MARKER="<promise>ABORT</promise>"
RECENT_COMMIT_COUNT=10
RECENT_COMMIT_GREP="RALPH"
PROGRESS_TAIL_LINES=200
MEMORY_TAIL_LINES=120
PRE_ITERATION_CMD=""
NOTIFY_CMD=""
RUNTIME_DIR="agent_recall/ralph/.runtime"
MEMORY_DIR=".agent"
GUARDRAILS_FILE="$MEMORY_DIR/GUARDRAILS.md"
STYLE_FILE="$MEMORY_DIR/STYLE.md"
RECENT_FILE="$MEMORY_DIR/RECENT.md"
COMPACT_CMD="uv run agent-recall compact"
COMPACT_MODE="always"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-cmd)
      AGENT_CMD="$2"
      shift 2
      ;;
    --validate-cmd)
      VALIDATE_CMD="$2"
      shift 2
      ;;
    --agent-output-mode)
      AGENT_OUTPUT_MODE="$2"
      shift 2
      ;;
    --prd-file)
      PRD_FILE="$2"
      shift 2
      ;;
    --progress-file)
      PROGRESS_FILE="$2"
      shift 2
      ;;
    --prompt-template)
      PROMPT_TEMPLATE="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --complete-marker)
      COMPLETE_MARKER="$2"
      shift 2
      ;;
    --alt-complete-marker)
      ALT_COMPLETE_MARKER="$2"
      shift 2
      ;;
    --abort-marker)
      ABORT_MARKER="$2"
      shift 2
      ;;
    --recent-commit-count)
      RECENT_COMMIT_COUNT="$2"
      shift 2
      ;;
    --recent-commit-grep)
      RECENT_COMMIT_GREP="$2"
      shift 2
      ;;
    --progress-tail-lines)
      PROGRESS_TAIL_LINES="$2"
      shift 2
      ;;
    --memory-tail-lines)
      MEMORY_TAIL_LINES="$2"
      shift 2
      ;;
    --memory-dir)
      MEMORY_DIR="$2"
      GUARDRAILS_FILE="$MEMORY_DIR/GUARDRAILS.md"
      STYLE_FILE="$MEMORY_DIR/STYLE.md"
      RECENT_FILE="$MEMORY_DIR/RECENT.md"
      shift 2
      ;;
    --guardrails-file)
      GUARDRAILS_FILE="$2"
      shift 2
      ;;
    --style-file)
      STYLE_FILE="$2"
      shift 2
      ;;
    --recent-file)
      RECENT_FILE="$2"
      shift 2
      ;;
    --compact-cmd)
      COMPACT_CMD="$2"
      shift 2
      ;;
    --compact-mode)
      COMPACT_MODE="$2"
      shift 2
      ;;
    --pre-iteration-cmd)
      PRE_ITERATION_CMD="$2"
      shift 2
      ;;
    --notify-cmd)
      NOTIFY_CMD="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

file_hash() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    printf 'missing\n'
    return 0
  fi

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
    return 0
  fi

  cksum "$file" | awk '{print $1 ":" $2}'
}

ensure_memory_files() {
  mkdir -p "$(dirname "$GUARDRAILS_FILE")"
  mkdir -p "$(dirname "$STYLE_FILE")"
  mkdir -p "$(dirname "$RECENT_FILE")"

  if [[ ! -f "$GUARDRAILS_FILE" ]]; then
    cat > "$GUARDRAILS_FILE" <<'EOF_G'
# Guardrails

Rules and warnings learned during Ralph iterations.
EOF_G
  fi

  if [[ ! -f "$STYLE_FILE" ]]; then
    cat > "$STYLE_FILE" <<'EOF_S'
# Style

Patterns and preferences learned during Ralph iterations.
EOF_S
  fi

  if [[ ! -f "$RECENT_FILE" ]]; then
    cat > "$RECENT_FILE" <<'EOF_R'
# Recent

Recent Ralph iteration summaries.
EOF_R
  fi
}

if [[ -z "$AGENT_CMD" ]]; then
  echo "Error: --agent-cmd (or RALPH_AGENT_CMD) is required." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required to read/write PRD JSON." >&2
  exit 1
fi

if [[ ! -f "$PRD_FILE" ]]; then
  echo "Error: PRD file not found: $PRD_FILE" >&2
  exit 1
fi

if ! jq -e '.items | type == "array"' "$PRD_FILE" >/dev/null 2>&1; then
  echo "Error: PRD JSON must contain an .items array." >&2
  exit 1
fi

if [[ ! -f "$PROMPT_TEMPLATE" ]]; then
  echo "Error: prompt template not found: $PROMPT_TEMPLATE" >&2
  exit 1
fi

if ! is_positive_int "$MAX_ITERATIONS"; then
  echo "Error: --max-iterations must be a positive integer." >&2
  exit 1
fi

if ! is_non_negative_int "$SLEEP_SECONDS"; then
  echo "Error: --sleep-seconds must be a non-negative integer." >&2
  exit 1
fi

if ! is_positive_int "$RECENT_COMMIT_COUNT"; then
  echo "Error: --recent-commit-count must be a positive integer." >&2
  exit 1
fi

if ! is_positive_int "$PROGRESS_TAIL_LINES"; then
  echo "Error: --progress-tail-lines must be a positive integer." >&2
  exit 1
fi

if ! is_positive_int "$MEMORY_TAIL_LINES"; then
  echo "Error: --memory-tail-lines must be a positive integer." >&2
  exit 1
fi

if [[ "$AGENT_OUTPUT_MODE" != "plain" && "$AGENT_OUTPUT_MODE" != "stream-json" ]]; then
  echo "Error: --agent-output-mode must be plain or stream-json." >&2
  exit 1
fi

if [[ "$COMPACT_MODE" != "always" && "$COMPACT_MODE" != "on-failure" && "$COMPACT_MODE" != "off" ]]; then
  echo "Error: --compact-mode must be always, on-failure, or off." >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
mkdir -p "$(dirname "$PROGRESS_FILE")"
touch "$PROGRESS_FILE"
ensure_memory_files

all_done() {
  jq -e '.items | all(.passes == true)' "$PRD_FILE" >/dev/null
}

next_item_json() {
  jq -c '.items
    | map(select(.passes != true))
    | sort_by(.priority // 999999)
    | .[0] // empty' "$PRD_FILE"
}

remaining_count() {
  jq -r '.items | map(select(.passes != true)) | length' "$PRD_FILE"
}

recent_commits() {
  local commits
  commits="$(git log --grep="$RECENT_COMMIT_GREP" -n "$RECENT_COMMIT_COUNT" --format='%H%n%ad%n%B---' --date=short 2>/dev/null || true)"
  if [[ -z "$commits" ]]; then
    printf 'No commits matching pattern "%s".\n' "$RECENT_COMMIT_GREP"
  else
    printf '%s\n' "$commits"
  fi
}

run_agent() {
  local prompt_file="$1"
  local log_file="$2"
  local cmd="$AGENT_CMD"

  if [[ "$cmd" == *"{prompt_file}"* ]]; then
    cmd="${cmd//\{prompt_file\}/$prompt_file}"
    bash -lc "$cmd" 2>&1 | tee "$log_file"
  else
    bash -lc "$cmd" < "$prompt_file" 2>&1 | tee "$log_file"
  fi
}

extract_from_stream_json() {
  local log_file="$1"
  local jq_filter="$2"
  local json_lines
  local parsed

  json_lines="$(sed -n '/^{/p' "$log_file")"
  if [[ -z "$json_lines" ]]; then
    return 0
  fi

  set +e
  parsed="$(printf '%s\n' "$json_lines" | jq -r "$jq_filter" 2>/dev/null)"
  local status=$?
  set -e

  if [[ $status -eq 0 && -n "$parsed" ]]; then
    printf '%s\n' "$parsed"
  fi
}

LAST_VALIDATE_EXIT=0
LAST_VALIDATE_OUTPUT=""

run_validation() {
  local iteration_label="$1"

  if [[ -z "$VALIDATE_CMD" ]]; then
    LAST_VALIDATE_EXIT=0
    LAST_VALIDATE_OUTPUT=""
    return 0
  fi

  set +e
  local output
  output="$(bash -lc "$VALIDATE_CMD" 2>&1)"
  local exit_code=$?
  set -e

  printf '%s\n' "$output" > "$RUNTIME_DIR/validate-${iteration_label}.log"
  LAST_VALIDATE_EXIT=$exit_code
  LAST_VALIDATE_OUTPUT="$output"

  if [[ $exit_code -eq 0 ]]; then
    LAST_VALIDATE_OUTPUT=""
    echo "Validation passed."
  else
    echo "Validation failed; output saved to $RUNTIME_DIR/validate-${iteration_label}.log"
  fi

  return $exit_code
}

log_has_marker() {
  local log_file="$1"
  local marker="$2"

  if [[ -z "$marker" || ! -f "$log_file" ]]; then
    return 1
  fi

  grep -Fq "$marker" "$log_file"
}

run_notify() {
  local status="$1"
  local iteration="$2"

  if [[ -z "$NOTIFY_CMD" ]]; then
    return 0
  fi

  local cmd="$NOTIFY_CMD"
  cmd="${cmd//\{iterations\}/$iteration}"
  cmd="${cmd//\{status\}/$status}"

  set +e
  bash -lc "$cmd" >/dev/null 2>&1
  set -e
}

append_guardrail_note() {
  local iteration="$1"
  local item_id="$2"
  local item_title="$3"
  local reason="$4"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  local failure_hint=""
  if [[ -n "$LAST_VALIDATE_OUTPUT" ]]; then
    failure_hint="$(printf '%s\n' "$LAST_VALIDATE_OUTPUT" | sed -n '1,3p' | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
  fi

  {
    echo ""
    echo "## ${timestamp} Iteration ${iteration} (${item_id})"
    echo "- Scope item: ${item_title}"
    if [[ "$reason" == "validation_failed" ]]; then
      echo "- Do not move to a new PRD item while validation is red."
      if [[ -n "$failure_hint" ]]; then
        echo "- Failure clue: ${failure_hint}"
      fi
    elif [[ "$reason" == "abort" ]]; then
      echo "- Abort means scope exceeded safety; reduce change size next iteration."
    else
      echo "- Keep changes isolated and verifiable before commit."
    fi
  } >> "$GUARDRAILS_FILE"
}

append_style_note() {
  local iteration="$1"
  local item_id="$2"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  {
    echo ""
    echo "## ${timestamp} Iteration ${iteration} (${item_id})"
    echo "- Prefer one logical change per commit."
    if [[ -n "$VALIDATE_CMD" ]]; then
      echo "- Keep validation command green before committing: ${VALIDATE_CMD}"
    else
      echo "- Run local checks before commit when available."
    fi
  } >> "$STYLE_FILE"
}

append_recent_note() {
  local iteration="$1"
  local item_id="$2"
  local item_title="$3"
  local work_mode="$4"
  local agent_exit="$5"
  local validate_exit="$6"
  local reason="$7"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  local validate_status="passed"
  if [[ "$validate_exit" -ne 0 ]]; then
    validate_status="failed"
  fi

  {
    echo ""
    echo "## ${timestamp} Iteration ${iteration}"
    echo "- Item: ${item_id} - ${item_title}"
    echo "- Mode: ${work_mode}"
    echo "- Agent exit code: ${agent_exit}"
    echo "- Validation: ${validate_status}"
    echo "- Outcome: ${reason}"
  } >> "$RECENT_FILE"
}

enforce_memory_updates() {
  local iteration="$1"
  local next_item_json="$2"
  local work_mode="$3"
  local agent_exit="$4"
  local abort_seen="$5"
  local pre_guard_hash="$6"
  local pre_style_hash="$7"
  local pre_recent_hash="$8"

  local item_id
  local item_title
  item_id="$(printf '%s\n' "$next_item_json" | jq -r '.id // "unknown"')"
  item_title="$(printf '%s\n' "$next_item_json" | jq -r '.title // "untitled"')"

  local reason="progressed"
  if [[ "$abort_seen" -eq 1 ]]; then
    reason="abort"
  elif [[ "$LAST_VALIDATE_EXIT" -ne 0 ]]; then
    reason="validation_failed"
  elif [[ "$work_mode" == "stabilize" ]]; then
    reason="stabilized_validation"
  fi

  local post_guard_hash
  local post_style_hash
  local post_recent_hash

  post_guard_hash="$(file_hash "$GUARDRAILS_FILE")"
  post_style_hash="$(file_hash "$STYLE_FILE")"
  post_recent_hash="$(file_hash "$RECENT_FILE")"

  if [[ "$post_guard_hash" == "$pre_guard_hash" ]]; then
    append_guardrail_note "$iteration" "$item_id" "$item_title" "$reason"
  fi

  if [[ "$post_style_hash" == "$pre_style_hash" ]]; then
    append_style_note "$iteration" "$item_id"
  fi

  if [[ "$post_recent_hash" == "$pre_recent_hash" ]]; then
    append_recent_note "$iteration" "$item_id" "$item_title" "$work_mode" "$agent_exit" "$LAST_VALIDATE_EXIT" "$reason"
  fi
}

run_compaction() {
  local iteration="$1"
  local should_run="$2"

  if [[ "$should_run" -ne 1 || "$COMPACT_MODE" == "off" || -z "$COMPACT_CMD" ]]; then
    return 0
  fi

  set +e
  local compact_output
  compact_output="$(bash -lc "$COMPACT_CMD" 2>&1)"
  local compact_exit=$?
  set -e

  printf '%s\n' "$compact_output" > "$RUNTIME_DIR/compact-${iteration}.log"

  if [[ $compact_exit -eq 0 ]]; then
    echo "Compaction succeeded."
  else
    echo "Compaction failed (exit $compact_exit); output saved to $RUNTIME_DIR/compact-${iteration}.log"
  fi
}

if all_done; then
  echo "PRD already complete."
  run_validation "initial" || true
  if [[ $LAST_VALIDATE_EXIT -eq 0 ]]; then
    echo "Validation is green. Nothing to do."
    run_notify "complete" "0"
    exit 0
  fi
  echo "Validation is failing; continuing so the agent can stabilize CI."
fi

for ((i = 1; i <= MAX_ITERATIONS; i++)); do
  echo ""
  echo "=== Agent Recall Ralph iteration $i/$MAX_ITERATIONS ==="

  if [[ -n "$PRE_ITERATION_CMD" ]]; then
    echo "Running pre-iteration command"
    set +e
    PRE_ITERATION_OUTPUT="$(bash -lc "$PRE_ITERATION_CMD" 2>&1)"
    PRE_ITERATION_EXIT=$?
    set -e
    printf '%s\n' "$PRE_ITERATION_OUTPUT" > "$RUNTIME_DIR/pre-${i}.log"
    if [[ $PRE_ITERATION_EXIT -ne 0 ]]; then
      echo "Pre-iteration command exited non-zero ($PRE_ITERATION_EXIT); continuing."
    fi
  fi

  REMAINING="$(remaining_count)"
  WORK_MODE="feature"
  NEXT_ITEM=""

  if [[ "$REMAINING" == "0" ]]; then
    echo "No remaining PRD items."
    run_validation "$i" || true
    if [[ $LAST_VALIDATE_EXIT -eq 0 ]]; then
      echo "All PRD items passed and validation is green. Exiting."
      run_notify "complete" "$i"
      exit 0
    fi

    WORK_MODE="stabilize"
    NEXT_ITEM='{"id":"AR-STABILIZE","priority":0,"title":"Stabilize failing validation and memory quality","user_story":"As a maintainer, I need CI and memory files to remain reliable for autonomous runs.","passes":false}'
    echo "Validation is failing; running stabilization iteration."
  else
    NEXT_ITEM="$(next_item_json)"
    if [[ -z "$NEXT_ITEM" ]]; then
      echo "Unable to find next PRD item while work remains." >&2
      exit 1
    fi
  fi

  PRE_GUARD_HASH="$(file_hash "$GUARDRAILS_FILE")"
  PRE_STYLE_HASH="$(file_hash "$STYLE_FILE")"
  PRE_RECENT_HASH="$(file_hash "$RECENT_FILE")"

  PROMPT_FILE="$RUNTIME_DIR/prompt-${i}.md"
  AGENT_LOG="$RUNTIME_DIR/agent-${i}.log"
  ASSISTANT_LOG="$RUNTIME_DIR/assistant-${i}.log"
  RESULT_LOG="$RUNTIME_DIR/result-${i}.log"
  RECENT_COMMITS="$(recent_commits)"
  : > "$ASSISTANT_LOG"
  : > "$RESULT_LOG"

  {
    cat "$PROMPT_TEMPLATE"
    echo ""
    echo "## Ralph Loop Context"
    echo "- Iteration: $i of $MAX_ITERATIONS"
    echo "- Work mode: $WORK_MODE"
    echo "- Primary completion marker: $COMPLETE_MARKER"
    echo "- Alternate completion marker: $ALT_COMPLETE_MARKER"
    echo "- Abort marker: $ABORT_MARKER"
    echo "- PRD file: $PRD_FILE"
    echo "- Progress file: $PROGRESS_FILE"
    echo "- Remaining PRD items: $REMAINING"

    echo ""
    echo "## Current PRD"
    echo '```json'
    jq . "$PRD_FILE"
    echo '```'

    echo ""
    echo "## Target Item For This Iteration"
    echo '```json'
    printf '%s\n' "$NEXT_ITEM" | jq .
    echo '```'

    if [[ -s "$PROGRESS_FILE" ]]; then
      echo ""
      echo "## Recent Progress Context"
      echo '```text'
      tail -n "$PROGRESS_TAIL_LINES" "$PROGRESS_FILE"
      echo '```'
    fi

    echo ""
    echo "## Agent Recall Memory Context"
    echo "### GUARDRAILS.md"
    echo '```md'
    tail -n "$MEMORY_TAIL_LINES" "$GUARDRAILS_FILE"
    echo '```'

    echo "### STYLE.md"
    echo '```md'
    tail -n "$MEMORY_TAIL_LINES" "$STYLE_FILE"
    echo '```'

    echo "### RECENT.md"
    echo '```md'
    tail -n "$MEMORY_TAIL_LINES" "$RECENT_FILE"
    echo '```'

    echo ""
    echo "## Previous Ralph Commits"
    echo '```text'
    printf '%s\n' "$RECENT_COMMITS"
    echo '```'

    if [[ -n "$VALIDATE_CMD" ]]; then
      echo ""
      echo "## CI Gate"
      echo "Validation command to run before committing:"
      echo '```bash'
      printf '%s\n' "$VALIDATE_CMD"
      echo '```'
    fi

    if [[ -n "$LAST_VALIDATE_OUTPUT" ]]; then
      echo ""
      echo "## Last Validation Failure (Fix This)"
      echo '```text'
      printf '%s\n' "$LAST_VALIDATE_OUTPUT"
      echo '```'
    fi

    echo ""
    echo "## Agent Recall Directives"
    echo "1. Work ONLY on the target item for this iteration."
    echo "2. Keep changes scoped; do not start a second feature."
    echo "3. Append a timestamped entry to $PROGRESS_FILE (append-only)."
    echo "4. Update $PRD_FILE for completed work only."
    echo "5. Update ALL memory files this iteration:"
    echo "   - $GUARDRAILS_FILE"
    echo "   - $STYLE_FILE"
    echo "   - $RECENT_FILE"
    if [[ -n "$VALIDATE_CMD" ]]; then
      echo "6. Run validation and ensure it passes before committing."
    else
      echo "6. Run available local checks before committing."
    fi
    echo "7. Make one commit for this iteration and include 'RALPH' in the message."
    echo "8. If all PRD work is complete and no further work remains, print exactly:"
    printf '   %s\n' "$COMPLETE_MARKER"
    if [[ -n "$ALT_COMPLETE_MARKER" ]]; then
      echo "9. Alternate completion marker accepted by the loop:"
      printf '   %s\n' "$ALT_COMPLETE_MARKER"
    fi
    echo "10. If blocked and cannot proceed safely, print exactly:"
    printf '   %s\n' "$ABORT_MARKER"
  } > "$PROMPT_FILE"

  set +e
  run_agent "$PROMPT_FILE" "$AGENT_LOG"
  AGENT_EXIT=$?
  set -e

  if [[ "$AGENT_OUTPUT_MODE" == "stream-json" ]]; then
    STREAM_ASSISTANT_TEXT="$(extract_from_stream_json "$AGENT_LOG" 'select(.type == "assistant").message.content[]? | select(.type == "text").text // empty')"
    STREAM_FINAL_RESULT="$(extract_from_stream_json "$AGENT_LOG" 'select(.type == "result").result // empty')"

    if [[ -n "$STREAM_ASSISTANT_TEXT" ]]; then
      printf '%s\n' "$STREAM_ASSISTANT_TEXT" > "$ASSISTANT_LOG"
    fi
    if [[ -n "$STREAM_FINAL_RESULT" ]]; then
      printf '%s\n' "$STREAM_FINAL_RESULT" > "$RESULT_LOG"
    fi
  fi

  if [[ $AGENT_EXIT -ne 0 ]]; then
    echo "Agent command exited non-zero ($AGENT_EXIT). Continuing loop."
  fi

  ABORT_SEEN=0
  if log_has_marker "$AGENT_LOG" "$ABORT_MARKER" || log_has_marker "$ASSISTANT_LOG" "$ABORT_MARKER" || log_has_marker "$RESULT_LOG" "$ABORT_MARKER"; then
    ABORT_SEEN=1
    echo "Abort marker seen."
  fi

  run_validation "$i" || true

  HAS_COMPLETE=0
  if log_has_marker "$AGENT_LOG" "$COMPLETE_MARKER" || log_has_marker "$ASSISTANT_LOG" "$COMPLETE_MARKER" || log_has_marker "$RESULT_LOG" "$COMPLETE_MARKER"; then
    HAS_COMPLETE=1
  elif log_has_marker "$AGENT_LOG" "$ALT_COMPLETE_MARKER" || log_has_marker "$ASSISTANT_LOG" "$ALT_COMPLETE_MARKER" || log_has_marker "$RESULT_LOG" "$ALT_COMPLETE_MARKER"; then
    HAS_COMPLETE=1
  fi

  enforce_memory_updates "$i" "$NEXT_ITEM" "$WORK_MODE" "$AGENT_EXIT" "$ABORT_SEEN" "$PRE_GUARD_HASH" "$PRE_STYLE_HASH" "$PRE_RECENT_HASH"

  SHOULD_COMPACT=0
  if [[ "$COMPACT_MODE" == "always" ]]; then
    SHOULD_COMPACT=1
  elif [[ "$COMPACT_MODE" == "on-failure" ]]; then
    if [[ $LAST_VALIDATE_EXIT -ne 0 || $ABORT_SEEN -eq 1 ]]; then
      SHOULD_COMPACT=1
    fi
  fi
  run_compaction "$i" "$SHOULD_COMPACT"

  if [[ $ABORT_SEEN -eq 1 ]]; then
    echo "Abort marker seen. Exiting with failure."
    run_notify "aborted" "$i"
    exit 1
  fi

  if all_done; then
    if [[ $LAST_VALIDATE_EXIT -eq 0 ]]; then
      echo "All PRD items passed. Exiting."
      run_notify "complete" "$i"
      exit 0
    fi
    echo "PRD complete but validation failing. Continuing for self-fix."
  fi

  if [[ $HAS_COMPLETE -eq 1 ]]; then
    if [[ $LAST_VALIDATE_EXIT -eq 0 ]]; then
      echo "Completion marker seen and validation green. Exiting early."
      run_notify "complete" "$i"
      exit 0
    fi
    echo "Completion marker seen, but validation failed. Continuing for self-fix."
  fi

  sleep "$SLEEP_SECONDS"
done

echo "Reached max iterations without meeting stop condition." >&2
run_notify "max-iterations" "$MAX_ITERATIONS"
exit 2
