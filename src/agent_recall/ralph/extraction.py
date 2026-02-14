from __future__ import annotations

import subprocess
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
    return {
        "outcome": extract_outcome(validation_exit, agent_exit, elapsed, timeout),
        "failure_reason": extract_failure_reason(validation_output),
        "validation_hint": extract_validation_hint(validation_output),
        "files_changed": extract_files_changed(repo_dir),
    }


def _is_separator_line(value: str) -> bool:
    if len(value) < 3:
        return False
    return all(char in _SEPARATOR_CHARS for char in value)
