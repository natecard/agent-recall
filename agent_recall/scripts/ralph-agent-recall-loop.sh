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
  --agent-timeout-seconds N    Timeout per agent iteration in seconds (default: 0, disabled)
  --prd-file PATH              PRD JSON path (default: agent_recall/ralph/prd.json)
  --prd-ids ID1,ID2,...        Comma-separated PRD item IDs to process (omit for all)
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
  --archive-only               Run archive-completed (archive + prune) and exit; no agent loop
  -h, --help                   Show help

Environment fallbacks:
  RALPH_AGENT_CMD
  RALPH_VALIDATE_CMD
  RALPH_PRD_IDS              Comma-separated PRD IDs (or use agent-recall ralph get-selected-prds)
USAGE
}

AGENT_CMD="${RALPH_AGENT_CMD:-}"
VALIDATE_CMD="${RALPH_VALIDATE_CMD:-}"
PRD_IDS="${RALPH_PRD_IDS:-}"
AGENT_OUTPUT_MODE="plain"
AGENT_TIMEOUT_SECONDS=0
AGENT_TIMEOUT_BACKEND="none"
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
LOCK_FILE="$RUNTIME_DIR/loop.lock"
LOCK_ACQUIRED=0
ARCHIVE_ONLY=0
SELECTED_IDS_JSON="null"

ensure_memory_files() {
  mkdir -p "$(dirname "$GUARDRAILS_FILE")"
  mkdir -p "$(dirname "$STYLE_FILE")"
  mkdir -p "$(dirname "$RECENT_FILE")"

  if [[ ! -f $GUARDRAILS_FILE ]]; then
    cat >"$GUARDRAILS_FILE" <<'EOF_G'
# Guardrails

Rules and warnings learned during Ralph iterations.
EOF_G
  fi

  if [[ ! -f $STYLE_FILE ]]; then
    cat >"$STYLE_FILE" <<'EOF_S'
# Style

Patterns and preferences learned during Ralph iterations.
EOF_S
  fi

  if [[ ! -f $RECENT_FILE ]]; then
    cat >"$RECENT_FILE" <<'EOF_R'
# Recent

Recent Ralph iteration summaries.
EOF_R
  fi
}

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
  --agent-timeout-seconds)
    AGENT_TIMEOUT_SECONDS="$2"
    shift 2
    ;;
  --prd-file)
    PRD_FILE="$2"
    shift 2
    ;;
  --prd-ids)
    PRD_IDS="$2"
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
  --archive-only)
    ARCHIVE_ONLY=1
    shift
    ;;
  -h | --help)
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

if [[ ${ARCHIVE_ONLY:-0} -eq 1 ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required for --archive-only." >&2
    exit 1
  fi
  if [[ ! -f $PRD_FILE ]]; then
    echo "Error: PRD file not found: $PRD_FILE" >&2
    exit 1
  fi
  if ! jq -e '.items | type == "array"' "$PRD_FILE" >/dev/null 2>&1; then
    echo "Error: PRD JSON must contain an .items array." >&2
    exit 1
  fi
  uv run agent-recall ralph archive-completed --prd-file "$PRD_FILE" --iteration 0
  exit 0
fi

is_positive_int() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

is_non_negative_int() {
  [[ $1 =~ ^[0-9]+$ ]]
}

file_hash() {
  local file="$1"
  if [[ ! -f $file ]]; then
    printf 'missing\n'
    return 0
  fi

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
    return 0
  fi

  cksum "$file" | awk '{print $1 ":" $2}'
}

if [[ -z $AGENT_CMD ]]; then
  echo "Error: --agent-cmd (or RALPH_AGENT_CMD) is required." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required to read/write PRD JSON." >&2
  exit 1
fi

if [[ ! -f $PRD_FILE ]]; then
  echo "Error: PRD file not found: $PRD_FILE" >&2
  exit 1
fi

if ! jq -e '.items | type == "array"' "$PRD_FILE" >/dev/null 2>&1; then
  echo "Error: PRD JSON must contain an .items array." >&2
  exit 1
fi

if [[ ! -f $PROMPT_TEMPLATE ]]; then
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

if ! is_non_negative_int "$AGENT_TIMEOUT_SECONDS"; then
  echo "Error: --agent-timeout-seconds must be a non-negative integer." >&2
  exit 1
fi

if [[ $AGENT_OUTPUT_MODE != "plain" && $AGENT_OUTPUT_MODE != "stream-json" ]]; then
  echo "Error: --agent-output-mode must be plain or stream-json." >&2
  exit 1
fi

if [[ $COMPACT_MODE != "always" && $COMPACT_MODE != "on-failure" && $COMPACT_MODE != "off" ]]; then
  echo "Error: --compact-mode must be always, on-failure, or off." >&2
  exit 1
fi

detect_timeout_backend() {
  if [[ $AGENT_TIMEOUT_SECONDS -eq 0 ]]; then
    AGENT_TIMEOUT_BACKEND="none"
    return 0
  fi

  if command -v timeout >/dev/null 2>&1; then
    AGENT_TIMEOUT_BACKEND="timeout"
    return 0
  fi

  if command -v gtimeout >/dev/null 2>&1; then
    AGENT_TIMEOUT_BACKEND="gtimeout"
    return 0
  fi

  if command -v perl >/dev/null 2>&1; then
    AGENT_TIMEOUT_BACKEND="perl"
    return 0
  fi

  echo "Error: timeout requested but no timeout backend found (need timeout, gtimeout, or perl)." >&2
  exit 1
}

release_lock() {
  if [[ ${LOCK_ACQUIRED:-0} -ne 1 ]]; then
    return 0
  fi

  if [[ ! -f $LOCK_FILE ]]; then
    return 0
  fi

  local holder
  holder="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [[ $holder == "$$" ]]; then
    rm -f "$LOCK_FILE"
  fi
}

acquire_lock() {
  mkdir -p "$(dirname "$LOCK_FILE")"

  if [[ -f $LOCK_FILE ]]; then
    local holder
    holder="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [[ -n $holder && $holder =~ ^[0-9]+$ ]] && kill -0 "$holder" 2>/dev/null; then
      echo "Another Agent Recall Ralph loop is already running (pid $holder)." >&2
      echo "Lock file: $LOCK_FILE" >&2
      exit 1
    fi
    rm -f "$LOCK_FILE"
  fi

  if (
    set -o noclobber
    echo "$$" >"$LOCK_FILE"
  ) 2>/dev/null; then
    LOCK_ACQUIRED=1
    trap release_lock EXIT INT TERM
    return 0
  fi

  echo "Unable to acquire lock file: $LOCK_FILE" >&2
  exit 1
}

mkdir -p "$RUNTIME_DIR"
mkdir -p "$(dirname "$PROGRESS_FILE")"
touch "$PROGRESS_FILE"
ensure_memory_files
detect_timeout_backend
acquire_lock

# When --prd-ids (or RALPH_PRD_IDS) is set, scope candidate selection to only those items.
# Keep PRD_FILE pointing at the original PRD so agent edits and archive/prune operate on source.
if [[ -n $PRD_IDS ]]; then
  SELECTED_IDS_JSON="$(echo "$PRD_IDS" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$' | jq -R . | jq -s .)"
  filtered_count="$(
    jq -r --argjson ids "$SELECTED_IDS_JSON" \
      '[.items[]? | select(.id as $id | $ids | index($id))] | length' \
      "$PRD_FILE"
  )"
  if [[ $filtered_count -eq 0 ]]; then
    echo "Error: No PRD items match --prd-ids ($PRD_IDS). Check IDs exist in $PRD_FILE." >&2
    exit 1
  fi
fi

all_done() {
  if [[ $SELECTED_IDS_JSON == "null" ]]; then
    jq -e '.items | all(.passes == true)' "$PRD_FILE" >/dev/null
    return
  fi
  jq -e --argjson ids "$SELECTED_IDS_JSON" \
    '[.items[]? | select(.id as $id | $ids | index($id))] | all(.passes == true)' \
    "$PRD_FILE" >/dev/null
}

next_item_json() {
  if [[ $SELECTED_IDS_JSON == "null" ]]; then
    jq -c '.items
      | map(select(.passes != true))
      | sort_by(.priority // 999999)
      | .[0] // empty' "$PRD_FILE"
    return
  fi
  jq -c --argjson ids "$SELECTED_IDS_JSON" '.items
    | map(select(.id as $id | $ids | index($id)))
    | map(select(.passes != true))
    | sort_by(.priority // 999999)
    | .[0] // empty' "$PRD_FILE"
}

unpassed_items_json() {
  if [[ $SELECTED_IDS_JSON == "null" ]]; then
    jq -c '.items | map(select(.passes != true))' "$PRD_FILE"
    return
  fi
  jq -c --argjson ids "$SELECTED_IDS_JSON" \
    '.items | map(select(.id as $id | $ids | index($id))) | map(select(.passes != true))' \
    "$PRD_FILE"
}

remaining_count() {
  if [[ $SELECTED_IDS_JSON == "null" ]]; then
    jq -r '.items | map(select(.passes != true)) | length' "$PRD_FILE"
    return
  fi
  jq -r --argjson ids "$SELECTED_IDS_JSON" \
    '.items | map(select(.id as $id | $ids | index($id))) | map(select(.passes != true)) | length' \
    "$PRD_FILE"
}

recent_commits() {
  local commits
  commits="$(git log --grep="$RECENT_COMMIT_GREP" -n "$RECENT_COMMIT_COUNT" --format='%H%n%ad%n%B---' --date=short 2>/dev/null || true)"
  if [[ -z $commits ]]; then
    printf 'No commits matching pattern "%s".\n' "$RECENT_COMMIT_GREP"
  else
    printf '%s\n' "$commits"
  fi
}

can_use_script_pty() {
  if [[ ${OSTYPE:-} != darwin* ]]; then
    return 1
  fi
  if ! command -v script >/dev/null 2>&1; then
    return 1
  fi
  script -q /dev/null true >/dev/null 2>&1
}

run_agent() {
  local prompt_file="$1"
  local log_file="$2"
  local cmd="$AGENT_CMD"
  local -a timeout_prefix=()
  local transport="legacy(pipe)"
  local command_has_prompt_file=0

  if [[ $AGENT_TIMEOUT_SECONDS -gt 0 ]]; then
    case "$AGENT_TIMEOUT_BACKEND" in
    timeout)
      timeout_prefix=(timeout "$AGENT_TIMEOUT_SECONDS")
      ;;
    gtimeout)
      timeout_prefix=(gtimeout "$AGENT_TIMEOUT_SECONDS")
      ;;
    perl)
      timeout_prefix=(perl -e 'alarm shift; exec @ARGV' "$AGENT_TIMEOUT_SECONDS")
      ;;
    esac
  fi

  if [[ $cmd == *"{prompt_file}"* ]]; then
    cmd="${cmd//\{prompt_file\}/$prompt_file}"
    command_has_prompt_file=1
  fi

  if [[ $AGENT_OUTPUT_MODE != "stream-json" ]] && can_use_script_pty; then
    transport="pty(script)"
    if [[ $command_has_prompt_file -eq 1 ]]; then
      if [[ $AGENT_TIMEOUT_SECONDS -gt 0 ]]; then
        {
          echo "Agent transport: $transport"
          "${timeout_prefix[@]}" script -q /dev/null bash -lc "$cmd"
        } 2>&1 | tee "$log_file"
      else
        {
          echo "Agent transport: $transport"
          script -q /dev/null bash -lc "$cmd"
        } 2>&1 | tee "$log_file"
      fi
    else
      if [[ $AGENT_TIMEOUT_SECONDS -gt 0 ]]; then
        {
          echo "Agent transport: $transport"
          "${timeout_prefix[@]}" script -q /dev/null bash -lc "$cmd" <"$prompt_file"
        } 2>&1 | tee "$log_file"
      else
        {
          echo "Agent transport: $transport"
          script -q /dev/null bash -lc "$cmd" <"$prompt_file"
        } 2>&1 | tee "$log_file"
      fi
    fi
    return
  fi

  if [[ $command_has_prompt_file -eq 1 ]]; then
    if [[ $AGENT_TIMEOUT_SECONDS -gt 0 ]]; then
      {
        echo "Agent transport: $transport"
        "${timeout_prefix[@]}" bash -lc "$cmd"
      } 2>&1 | tee "$log_file"
    else
      {
        echo "Agent transport: $transport"
        bash -lc "$cmd"
      } 2>&1 | tee "$log_file"
    fi
  else
    if [[ $AGENT_TIMEOUT_SECONDS -gt 0 ]]; then
      {
        echo "Agent transport: $transport"
        "${timeout_prefix[@]}" bash -lc "$cmd" <"$prompt_file"
      } 2>&1 | tee "$log_file"
    else
      {
        echo "Agent transport: $transport"
        bash -lc "$cmd" <"$prompt_file"
      } 2>&1 | tee "$log_file"
    fi
  fi
}

agent_timed_out_exit() {
  local exit_code="$1"

  if [[ $AGENT_TIMEOUT_SECONDS -eq 0 ]]; then
    return 1
  fi

  case "$AGENT_TIMEOUT_BACKEND" in
  timeout | gtimeout)
    [[ $exit_code -eq 124 ]]
    ;;
  perl)
    [[ $exit_code -eq 142 || $exit_code -eq 124 ]]
    ;;
  *)
    return 1
    ;;
  esac
}

extract_from_stream_json() {
  local log_file="$1"
  local jq_filter="$2"
  local json_lines
  local parsed

  json_lines="$(sed -n '/^{/p' "$log_file")"
  if [[ -z $json_lines ]]; then
    return 0
  fi

  set +e
  parsed="$(printf '%s\n' "$json_lines" | jq -r "$jq_filter" 2>/dev/null)"
  local status=$?
  set -e

  if [[ $status -eq 0 && -n $parsed ]]; then
    printf '%s\n' "$parsed"
  fi
}

LAST_VALIDATE_EXIT=0
LAST_VALIDATE_OUTPUT=""

run_validation() {
  local iteration_label="$1"

  if [[ -z $VALIDATE_CMD ]]; then
    LAST_VALIDATE_EXIT=0
    LAST_VALIDATE_OUTPUT=""
    return 0
  fi

  set +e
  local output
  output="$(bash -lc "$VALIDATE_CMD" 2>&1)"
  local exit_code=$?
  set -e

  printf '%s\n' "$output" >"$RUNTIME_DIR/validate-${iteration_label}.log"
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

  if [[ -z $marker || ! -f $log_file ]]; then
    return 1
  fi

  grep -Fq "$marker" "$log_file"
}

log_has_exact_marker_line() {
  local log_file="$1"
  local marker="$2"

  if [[ -z $marker || ! -f $log_file ]]; then
    return 1
  fi

  grep -Fxq "$marker" "$log_file"
}

run_notify() {
  local status="$1"
  local iteration="$2"

  if [[ -z $NOTIFY_CMD ]]; then
    return 0
  fi

  local cmd="$NOTIFY_CMD"
  cmd="${cmd//\{iterations\}/$iteration}"
  cmd="${cmd//\{status\}/$status}"

  set +e
  bash -lc "$cmd" >/dev/null 2>&1
  set -e
}

validation_log_path() {
  local iteration="$1"
  printf '%s/validate-%s.log' "$RUNTIME_DIR" "$iteration"
}

agent_log_path() {
  local iteration="$1"
  printf '%s/agent-%s.log' "$RUNTIME_DIR" "$iteration"
}

normalize_runtime_line() {
  local value="$1"
  printf '%s\n' "$value" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//'
}

extract_actionable_validation_lines() {
  local iteration="$1"
  local max_lines="${2:-6}"
  local log_file
  log_file="$(validation_log_path "$iteration")"

  if [[ ! -s $log_file ]]; then
    return 0
  fi

  local filtered
  filtered="$(
    sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' "$log_file" \
      | sed '/^$/d' \
      | grep -Eiv '^(=+|-+|=+.*=+)$' \
      | grep -Eiv '(test session starts|test session ends|^platform |^rootdir:|^configfile:|^plugins:|^collected [0-9]+ items?)' \
      | head -n "$max_lines" || true
  )"

  if [[ -n $filtered ]]; then
    printf '%s\n' "$filtered"
    return 0
  fi

  sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' "$log_file" | sed '/^$/d' | head -n "$max_lines"
}

extract_validation_hint() {
  local iteration="$1"
  extract_actionable_validation_lines "$iteration" 1 | head -n 1
}

run_compaction() {
  local iteration="$1"
  local should_run="$2"

  if [[ $should_run -ne 1 || $COMPACT_MODE == "off" || -z $COMPACT_CMD ]]; then
    return 0
  fi

  set +e
  local compact_output
  compact_output="$(bash -lc "$COMPACT_CMD" 2>&1)"
  local compact_exit=$?
  set -e

  printf '%s\n' "$compact_output" >"$RUNTIME_DIR/compact-${iteration}.log"

  if [[ $compact_exit -eq 0 ]]; then
    echo "Compaction succeeded."
  else
    echo "Compaction failed (exit $compact_exit); output saved to $RUNTIME_DIR/compact-${iteration}.log"
  fi
}

setup_iteration() {
  local iteration="$1"
  local item_id="$2"
  local item_title="$3"

  if ! command -v uv >/dev/null 2>&1; then
    echo "Warning: uv not available; skipping create-report." >&2
    return 0
  fi

  uv run agent-recall ralph create-report --iteration "$iteration" --item-id "${item_id:-UNKNOWN}" --item-title "${item_title:-Untitled}" \
    || {
      echo "Warning: create-report failed; continuing." >&2
      true
    }
}

render_prompt_template() {
  local template_path="$1"
  local report_path="$2"
  local item_id="$3"
  local item_title="$4"
  local description="$5"
  local validation_command="$6"

  local template_content
  template_content="$(<"$template_path")"
  template_content="${template_content//\{current_report_path\}/$report_path}"
  template_content="${template_content//\{item_id\}/$item_id}"
  template_content="${template_content//\{item_title\}/$item_title}"
  template_content="${template_content//\{description\}/$description}"
  template_content="${template_content//\{validation_command\}/$validation_command}"
  printf '%s\n' "$template_content"
}

finalize_iteration() {
  local iteration="$1"
  local validation_exit="$2"
  local validation_hint="$3"

  if ! command -v uv >/dev/null 2>&1; then
    echo "Warning: uv not available; skipping finalize/report/forecast." >&2
    return 0
  fi

  uv run agent-recall ralph extract-iteration --iteration "$iteration" --runtime-dir "$RUNTIME_DIR" \
    || {
      echo "Warning: extract-iteration failed; continuing." >&2
      true
    }
  uv run agent-recall ralph finalize-report --validation-exit "$validation_exit" --validation-hint "${validation_hint:-}" \
    || {
      echo "Warning: finalize-report failed; continuing." >&2
      true
    }
  uv run agent-recall ralph rebuild-forecast \
    || {
      echo "Warning: rebuild-forecast failed; continuing." >&2
      true
    }
}

refresh_context_after_compaction() {
  local iteration="$1"
  local item_id="$2"
  local item_title="$3"

  if ! command -v uv >/dev/null 2>&1; then
    echo "Warning: uv not available; skipping refresh-context." >&2
    return 0
  fi

  uv run agent-recall ralph refresh-context --task "${item_title:-}" --item "${item_id:-}" --iteration "$iteration" \
    || {
      echo "Warning: refresh-context failed; continuing." >&2
      true
    }
}

run_archive_completed() {
  local iteration="$1"

  if ! command -v uv >/dev/null 2>&1; then
    return 0
  fi

  set +e
  local archive_output
  archive_output="$(uv run agent-recall ralph archive-completed --prd-file "$PRD_FILE" --iteration "$iteration" 2>&1)"
  local archive_exit=$?
  set -e

  printf '%s\n' "$archive_output" >"$RUNTIME_DIR/archive-${iteration}.log"
  if [[ $archive_exit -ne 0 ]]; then
    echo "Warning: archive-completed failed; continuing."
  fi
}

run_synthesize_climate() {
  local iteration="$1"

  if ! command -v uv >/dev/null 2>&1; then
    echo "Warning: uv not available; skipping synthesize-climate." >&2
    return 0
  fi

  uv run agent-recall ralph synthesize-climate \
    || {
      echo "Warning: synthesize-climate failed; continuing." >&2
      true
    }
}

if all_done; then
  echo "PRD already complete."
  run_validation "initial" || true
  if [[ $LAST_VALIDATE_EXIT -eq 0 ]]; then
    echo "Validation is green. Nothing to do."
    run_archive_completed "0"
    run_synthesize_climate "0"
    run_notify "complete" "0"
    exit 0
  fi
  echo "Validation is failing; continuing so the agent can stabilize CI."
fi

for ((i = 1; i <= MAX_ITERATIONS; i++)); do
  echo ""
  echo "=== Agent Recall Ralph iteration $i/$MAX_ITERATIONS ==="

  if [[ -n $PRE_ITERATION_CMD ]]; then
    echo "Running pre-iteration command"
    set +e
    PRE_ITERATION_OUTPUT="$(bash -lc "$PRE_ITERATION_CMD" 2>&1)"
    PRE_ITERATION_EXIT=$?
    set -e
    printf '%s\n' "$PRE_ITERATION_OUTPUT" >"$RUNTIME_DIR/pre-${i}.log"
    if [[ $PRE_ITERATION_EXIT -ne 0 ]]; then
      echo "Pre-iteration command exited non-zero ($PRE_ITERATION_EXIT); continuing."
    fi
  fi

  REMAINING="$(remaining_count)"
  WORK_MODE="feature"
  NEXT_ITEM=""
  UNPASSED_ITEMS="[]"

  if [[ $REMAINING == "0" ]]; then
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
    UNPASSED_ITEMS="$(unpassed_items_json)"
    if [[ $UNPASSED_ITEMS == "[]" ]]; then
      echo "Unable to enumerate unpassed PRD items while work remains." >&2
      exit 1
    fi
    NEXT_ITEM='{"id":"AGENT-SELECTED","priority":null,"title":"Agent-selected highest-priority item for this iteration","passes":false}'
  fi

  ITERATION_ITEM_ID=""
  ITERATION_ITEM_TITLE=""
  if [[ $WORK_MODE == "feature" ]]; then
    SELECTED_ITEM="$(next_item_json)"
    if [[ -n $SELECTED_ITEM ]]; then
      ITERATION_ITEM_ID="$(printf '%s\n' "$SELECTED_ITEM" | jq -r '.id // empty')"
      ITERATION_ITEM_TITLE="$(printf '%s\n' "$SELECTED_ITEM" | jq -r '.title // .description // empty')"
    fi
  else
    ITERATION_ITEM_ID="$(printf '%s\n' "$NEXT_ITEM" | jq -r '.id // empty')"
    ITERATION_ITEM_TITLE="$(printf '%s\n' "$NEXT_ITEM" | jq -r '.title // .description // empty')"
  fi

  PROMPT_FILE="$RUNTIME_DIR/prompt-${i}.md"
  AGENT_LOG="$RUNTIME_DIR/agent-${i}.log"
  ASSISTANT_LOG="$RUNTIME_DIR/assistant-${i}.log"
  RESULT_LOG="$RUNTIME_DIR/result-${i}.log"
  RECENT_COMMITS="$(recent_commits)"
  : >"$ASSISTANT_LOG"
  : >"$RESULT_LOG"

  {
    CURRENT_REPORT_PATH=".agent/ralph/iterations/current.json"
    TEMPLATE_ITEM_ID="$(printf '%s\n' "$NEXT_ITEM" | jq -r '.id // "AGENT-SELECTED"')"
    TEMPLATE_ITEM_TITLE="$(printf '%s\n' "$NEXT_ITEM" | jq -r '.title // .description // "Agent-selected highest-priority item for this iteration"')"
    TEMPLATE_DESCRIPTION="$(printf '%s\n' "$NEXT_ITEM" | jq -r '.description // .user_story // "Select one unpassed PRD item and complete the smallest viable slice."')"
    TEMPLATE_VALIDATION_COMMAND="$VALIDATE_CMD"
    if [[ -z $TEMPLATE_VALIDATION_COMMAND ]]; then
      TEMPLATE_VALIDATION_COMMAND="(none provided; loop runs post-iteration validation if configured)"
    fi
    render_prompt_template "$PROMPT_TEMPLATE" "$CURRENT_REPORT_PATH" "$TEMPLATE_ITEM_ID" "$TEMPLATE_ITEM_TITLE" "$TEMPLATE_DESCRIPTION" "$TEMPLATE_VALIDATION_COMMAND"
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
    if [[ $SELECTED_IDS_JSON == "null" ]]; then
      jq . "$PRD_FILE"
    else
      jq --argjson ids "$SELECTED_IDS_JSON" \
        '.items = [.items[]? | select(.id as $id | $ids | index($id))]' \
        "$PRD_FILE"
    fi
    echo '```'

    if [[ $WORK_MODE == "feature" ]]; then
      echo ""
      echo "## Unpassed PRD Items (Agent Must Select One)"
      echo '```json'
      printf '%s\n' "$UNPASSED_ITEMS" | jq .
      echo '```'
    else
      echo ""
      echo "## Target Item For This Iteration"
      echo '```json'
      printf '%s\n' "$NEXT_ITEM" | jq .
      echo '```'
    fi

    if [[ -s $PROGRESS_FILE ]]; then
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
    if [[ -f $GUARDRAILS_FILE ]]; then
      tail -n "$MEMORY_TAIL_LINES" "$GUARDRAILS_FILE"
    else
      echo "(missing)"
    fi
    echo '```'

    echo "### STYLE.md"
    echo '```md'
    if [[ -f $STYLE_FILE ]]; then
      tail -n "$MEMORY_TAIL_LINES" "$STYLE_FILE"
    else
      echo "(missing)"
    fi
    echo '```'

    echo "### RECENT.md"
    echo '```md'
    if [[ -f $RECENT_FILE ]]; then
      tail -n "$MEMORY_TAIL_LINES" "$RECENT_FILE"
    else
      echo "(missing)"
    fi
    echo '```'

    echo ""
    echo "## Previous Ralph Commits"
    echo '```text'
    printf '%s\n' "$RECENT_COMMITS"
    echo '```'

    if [[ -n $VALIDATE_CMD ]]; then
      echo ""
      echo "## CI Gate"
      echo "Validation command to run before committing:"
      echo '```bash'
      printf '%s\n' "$VALIDATE_CMD"
      echo '```'
    fi

    if [[ -n $LAST_VALIDATE_OUTPUT ]]; then
      echo ""
      echo "## Last Validation Failure (Fix This)"
      echo '```text'
      printf '%s\n' "$LAST_VALIDATE_OUTPUT"
      echo '```'
    fi

    echo ""
    echo "## Agent Recall Directives"
    echo "The directives in this section override any conflicting instructions above."
    if [[ $WORK_MODE == "feature" ]]; then
      echo "1. Select what you deem the highest-priority unpassed PRD item."
      echo "2. Assign or adjust priority values across unpassed items to reflect your ordering."
      echo "3. Work ONLY on your selected item; do not start a second feature."
    else
      echo "1. Work ONLY on the stabilization item for this iteration."
      echo "2. Keep changes scoped; do not start a second feature."
      echo "3. Keep priority assignments coherent for any remaining PRD items."
    fi
    echo "4. Do not request missing placeholders; use the provided report path: $CURRENT_REPORT_PATH."
    echo "5. Update $PRD_FILE for completed work only."
    echo "6. Treat tier files as read-only; do NOT edit $GUARDRAILS_FILE, $STYLE_FILE, or $RECENT_FILE (system updates them from iteration reports)."
    if [[ -n $VALIDATE_CMD ]]; then
      echo "7. Run validation and ensure it passes before committing."
    else
      echo "7. Run available local checks before committing."
    fi
    echo "8. Make one commit for this iteration and include 'RALPH' in the message."
    echo "9. If all PRD work is complete and no further work remains, print exactly:"
    printf '   %s\n' "$COMPLETE_MARKER"
    if [[ -n $ALT_COMPLETE_MARKER ]]; then
      echo "10. Alternate completion marker accepted by the loop:"
      printf '   %s\n' "$ALT_COMPLETE_MARKER"
    fi
    echo "11. If blocked and cannot proceed safely, print exactly:"
    printf '   %s\n' "$ABORT_MARKER"
  } >"$PROMPT_FILE"

  setup_iteration "$i" "$ITERATION_ITEM_ID" "$ITERATION_ITEM_TITLE"

  set +e
  run_agent "$PROMPT_FILE" "$AGENT_LOG"
  AGENT_EXIT=$?
  set -e
  AGENT_TIMED_OUT=0
  if agent_timed_out_exit "$AGENT_EXIT"; then
    AGENT_TIMED_OUT=1
    echo "Agent timed out after ${AGENT_TIMEOUT_SECONDS}s (backend: ${AGENT_TIMEOUT_BACKEND})."
  fi

  if [[ $AGENT_OUTPUT_MODE == "stream-json" ]]; then
    STREAM_ASSISTANT_TEXT="$(extract_from_stream_json "$AGENT_LOG" 'select(.type == "assistant").message.content[]? | select(.type == "text").text // empty')"
    STREAM_FINAL_RESULT="$(extract_from_stream_json "$AGENT_LOG" 'select(.type == "result").result // empty')"

    if [[ -n $STREAM_ASSISTANT_TEXT ]]; then
      printf '%s\n' "$STREAM_ASSISTANT_TEXT" >"$ASSISTANT_LOG"
    fi
    if [[ -n $STREAM_FINAL_RESULT ]]; then
      printf '%s\n' "$STREAM_FINAL_RESULT" >"$RESULT_LOG"
    fi
  fi

  if [[ $AGENT_EXIT -ne 0 ]]; then
    if [[ $AGENT_TIMED_OUT -eq 0 ]]; then
      echo "Agent command exited non-zero ($AGENT_EXIT). Continuing loop."
    fi
  fi

  ABORT_SEEN=0
  if [[ $AGENT_OUTPUT_MODE == "stream-json" ]]; then
    if log_has_marker "$RESULT_LOG" "$ABORT_MARKER"; then
      ABORT_SEEN=1
      echo "Abort marker seen in result payload."
    fi
  else
    if log_has_exact_marker_line "$AGENT_LOG" "$ABORT_MARKER"; then
      ABORT_SEEN=1
      echo "Abort marker seen."
    fi
  fi

  run_validation "$i" || true
  VALIDATION_HINT="$(extract_validation_hint "$i")"

  finalize_iteration "$i" "$LAST_VALIDATE_EXIT" "$VALIDATION_HINT"

  HAS_COMPLETE=0
  if [[ $AGENT_OUTPUT_MODE == "stream-json" ]]; then
    if log_has_marker "$RESULT_LOG" "$COMPLETE_MARKER"; then
      HAS_COMPLETE=1
    elif log_has_marker "$RESULT_LOG" "$ALT_COMPLETE_MARKER"; then
      HAS_COMPLETE=1
    fi
  else
    if log_has_exact_marker_line "$AGENT_LOG" "$COMPLETE_MARKER"; then
      HAS_COMPLETE=1
    elif log_has_exact_marker_line "$AGENT_LOG" "$ALT_COMPLETE_MARKER"; then
      HAS_COMPLETE=1
    fi
  fi

  SHOULD_COMPACT=0
  if [[ $COMPACT_MODE == "always" ]]; then
    SHOULD_COMPACT=1
  elif [[ $COMPACT_MODE == "on-failure" ]]; then
    if [[ $LAST_VALIDATE_EXIT -ne 0 || $ABORT_SEEN -eq 1 ]]; then
      SHOULD_COMPACT=1
    fi
  fi
  run_compaction "$i" "$SHOULD_COMPACT"
  if [[ $SHOULD_COMPACT -eq 1 ]]; then
    run_synthesize_climate "$i"
  fi
  refresh_context_after_compaction "$i" "$ITERATION_ITEM_ID" "$ITERATION_ITEM_TITLE"

  if [[ $ABORT_SEEN -eq 1 ]]; then
    echo "Abort marker seen. Exiting with failure."
    run_synthesize_climate "$i"
    run_notify "aborted" "$i"
    exit 1
  fi

  if all_done; then
    if [[ $LAST_VALIDATE_EXIT -eq 0 ]]; then
      echo "All PRD items passed. Exiting."
      run_archive_completed "$i"
      run_synthesize_climate "$i"
      run_notify "complete" "$i"
      exit 0
    fi
    echo "PRD complete but validation failing. Continuing for self-fix."
  fi

  if [[ $HAS_COMPLETE -eq 1 ]]; then
    if [[ $LAST_VALIDATE_EXIT -eq 0 ]]; then
      echo "Completion marker seen and validation green. Exiting early."
      run_archive_completed "$i"
      run_synthesize_climate "$i"
      run_notify "complete" "$i"
      exit 0
    fi
    echo "Completion marker seen, but validation failed. Continuing for self-fix."
  fi

  sleep "$SLEEP_SECONDS"
done

echo "Reached max iterations without meeting stop condition." >&2
run_synthesize_climate "$MAX_ITERATIONS"
run_notify "max-iterations" "$MAX_ITERATIONS"
exit 2
