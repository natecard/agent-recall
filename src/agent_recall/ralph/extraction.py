from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

from agent_recall.ralph.iteration_store import IterationOutcome

_ERROR_MARKERS = ("error", "failed", "exception", "assert")
_SEPARATOR_CHARS = set("=-_*#")


def extract_outcome(
    validation_exit_code: int,
    agent_exit_code: int,
    elapsed_seconds: float,
    timeout_seconds: float,
) -> IterationOutcome:
    if validation_exit_code == 0:
        return IterationOutcome.COMPLETED
    if timeout_seconds > 0 and elapsed_seconds >= timeout_seconds:
        return IterationOutcome.TIMEOUT
    return IterationOutcome.VALIDATION_FAILED


def extract_failure_reason(validation_output: list[str]) -> str | None:
    for line in validation_output:
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker in lowered for marker in _ERROR_MARKERS):
            return stripped[:200]
    return None


def extract_token_usage(output_lines: Iterable[str]) -> tuple[dict[str, int] | None, str | None]:
    usage: dict[str, int] = {}
    model: str | None = None

    for line in output_lines:
        if not line:
            continue
        payload = _parse_token_json_line(line)
        if payload:
            parsed_usage, parsed_model = payload
            usage.update(parsed_usage)
            if parsed_model:
                model = parsed_model
            continue
        parsed_usage, parsed_model = _parse_token_line(line)
        usage.update(parsed_usage)
        if parsed_model:
            model = parsed_model

    return (usage or None, model)


def extract_files_changed(repo_dir: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def extract_git_diff(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def get_current_commit_hash(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def get_last_commit_message(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%B"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def extract_commit_diff(repo_dir: Path, commit_hash: str) -> str:
    if not commit_hash:
        return ""
    try:
        result = subprocess.run(
            ["git", "show", commit_hash, "--format=", "--patch"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def extract_smart_diff(repo_dir: Path) -> dict[str, str | None]:
    """
    Intelligently extract diff after agent run.

    Agents are instructed to include 'RALPH' in commit messages.
    This function checks if the last commit is a RALPH commit:
    - If yes: return the commit's diff and commit hash
    - If no: return unstaged diff (git diff) with no commit hash

    Returns dict with keys: 'diff', 'commit_hash'
    """
    last_message = get_last_commit_message(repo_dir)
    commit_hash = None
    diff = ""

    if last_message and "RALPH" in last_message:
        commit_hash = get_current_commit_hash(repo_dir)
        if commit_hash:
            diff = extract_commit_diff(repo_dir, commit_hash)

    if not diff:
        diff = extract_git_diff(repo_dir)
        if not diff and commit_hash:
            commit_hash = None

    return {
        "diff": diff,
        "commit_hash": commit_hash,
    }


def extract_validation_hint(validation_output: list[str]) -> str | None:
    for line in validation_output:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_separator_line(stripped):
            continue
        return stripped
    return None


def extract_from_artifacts(
    validation_exit: int,
    validation_output: list[str],
    agent_exit: int,
    elapsed: float,
    timeout: float,
    repo_dir: Path,
) -> dict[str, object]:
    smart = extract_smart_diff(repo_dir)
    return {
        "outcome": extract_outcome(validation_exit, agent_exit, elapsed, timeout),
        "failure_reason": extract_failure_reason(validation_output),
        "validation_hint": extract_validation_hint(validation_output),
        "files_changed": extract_files_changed(repo_dir),
        "git_diff": smart.get("diff") or "",
        "commit_hash": smart.get("commit_hash"),
    }


def _is_separator_line(value: str) -> bool:
    if len(value) < 3:
        return False
    return all(char in _SEPARATOR_CHARS for char in value)


def _parse_token_json_line(line: str) -> tuple[dict[str, int], str | None] | None:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    usage_raw = payload.get("usage")
    if not isinstance(usage_raw, dict):
        return None
    usage = _normalize_usage_dict(usage_raw)
    model = _parse_model_value(payload.get("model"))
    return (usage, model)


def _parse_token_line(line: str) -> tuple[dict[str, int], str | None]:
    usage: dict[str, int] = {}
    model: str | None = None
    lowered = line.lower()
    if "token" not in lowered:
        return usage, model

    match = re.search(r"model\s*[:=]\s*([\w.:-]+)", line, re.IGNORECASE)
    if match:
        model = match.group(1)

    total_match = re.search(r"total\s*tokens?\s*[:=]\s*(\d+)", line, re.IGNORECASE)
    if total_match:
        usage["total_tokens"] = int(total_match.group(1))

    prompt_match = re.search(r"prompt\s*tokens?\s*[:=]\s*(\d+)", line, re.IGNORECASE)
    if prompt_match:
        usage["prompt_tokens"] = int(prompt_match.group(1))

    completion_match = re.search(r"completion\s*tokens?\s*[:=]\s*(\d+)", line, re.IGNORECASE)
    if completion_match:
        usage["completion_tokens"] = int(completion_match.group(1))

    return usage, model


def _normalize_usage_dict(usage: dict[str, object]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for key, value in usage.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, int | float | str):
            continue
        try:
            parsed[key] = int(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _parse_model_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None
