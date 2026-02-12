#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./agent_recall/scripts/copy-to-self-docs.sh [options]

Syncs ONLY scripts:
  /Users/natecard/OnHere/Repos/ralph/agent_recall/scripts
to:
  /Users/natecard/OnHere/Repos/self-docs/agent_recall/scripts

Options:
  --dry-run           Show what would change without writing files
  --quiet             Reduce output
  --install-hook      Install a post-commit hook in this repo to auto-run sync
  --uninstall-hook    Remove the managed post-commit hook
  -h, --help          Show this help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RALPH_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_DIR="$RALPH_REPO/agent_recall"
TARGET_REPO="/Users/natecard/OnHere/Repos/self-docs"
TARGET_DIR="$TARGET_REPO/agent_recall"
SOURCE_SCRIPTS_DIR="$SOURCE_DIR/scripts"
TARGET_SCRIPTS_DIR="$TARGET_DIR/scripts"
HOOK_FILE="$RALPH_REPO/.git/hooks/post-commit"
HOOK_MARKER="# ralph-agent-recall-copy-hook"

DRY_RUN=0
QUIET=0
INSTALL_HOOK=0
UNINSTALL_HOOK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --install-hook)
      INSTALL_HOOK=1
      shift
      ;;
    --uninstall-hook)
      UNINSTALL_HOOK=1
      shift
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

if [[ $INSTALL_HOOK -eq 1 && $UNINSTALL_HOOK -eq 1 ]]; then
  echo "Use only one of --install-hook or --uninstall-hook." >&2
  exit 1
fi

install_hook() {
  mkdir -p "$(dirname "$HOOK_FILE")"

  if [[ -f "$HOOK_FILE" ]] && ! grep -Fq "$HOOK_MARKER" "$HOOK_FILE"; then
    echo "Refusing to overwrite existing hook: $HOOK_FILE" >&2
    echo "Add this command manually to your existing post-commit hook:" >&2
    echo "  \"$SCRIPT_DIR/copy-to-self-docs.sh\" --quiet" >&2
    exit 1
  fi

  cat > "$HOOK_FILE" <<EOF_HOOK
#!/usr/bin/env bash
set -euo pipefail
$HOOK_MARKER
"$SCRIPT_DIR/copy-to-self-docs.sh" --quiet || true
EOF_HOOK
  chmod +x "$HOOK_FILE"
  echo "Installed post-commit hook: $HOOK_FILE"
}

uninstall_hook() {
  if [[ ! -f "$HOOK_FILE" ]]; then
    echo "No post-commit hook found at $HOOK_FILE"
    return 0
  fi

  if ! grep -Fq "$HOOK_MARKER" "$HOOK_FILE"; then
    echo "Hook exists but is not managed by copy-to-self-docs.sh: $HOOK_FILE" >&2
    return 1
  fi

  rm -f "$HOOK_FILE"
  echo "Removed managed post-commit hook: $HOOK_FILE"
}

if [[ $INSTALL_HOOK -eq 1 ]]; then
  install_hook
  exit 0
fi

if [[ $UNINSTALL_HOOK -eq 1 ]]; then
  uninstall_hook
  exit 0
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_SCRIPTS_DIR" ]]; then
  echo "Source scripts directory not found: $SOURCE_SCRIPTS_DIR" >&2
  exit 1
fi

if [[ ! -d "$TARGET_REPO/.git" ]]; then
  echo "Target repo not found or missing .git: $TARGET_REPO" >&2
  exit 1
fi

mkdir -p "$TARGET_SCRIPTS_DIR"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required for copy-to-self-docs.sh" >&2
  exit 1
fi

RSYNC_ARGS=(
  -a
  --delete
  --exclude '.DS_Store'
)

if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

if [[ $QUIET -eq 0 ]]; then
  echo "Syncing:"
  echo "  $SOURCE_SCRIPTS_DIR"
  echo "-> $TARGET_SCRIPTS_DIR"
fi

rsync "${RSYNC_ARGS[@]}" "$SOURCE_SCRIPTS_DIR"/ "$TARGET_SCRIPTS_DIR"/

if [[ $QUIET -eq 0 ]]; then
  echo "Done."
  echo ""
  echo "Target repo status (agent_recall/scripts only):"
  git -C "$TARGET_REPO" status --short -- agent_recall/scripts || true
fi
