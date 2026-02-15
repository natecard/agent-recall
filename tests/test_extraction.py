from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from agent_recall.ralph.extraction import (
    extract_failure_reason,
    extract_files_changed,
    extract_from_artifacts,
    extract_git_diff,
    extract_outcome,
    extract_validation_hint,
)
from agent_recall.ralph.iteration_store import IterationOutcome


def test_extract_outcome_completed() -> None:
    assert (
        extract_outcome(0, agent_exit_code=1, elapsed_seconds=12.0, timeout_seconds=60.0)
        == IterationOutcome.COMPLETED
    )


def test_extract_outcome_timeout() -> None:
    assert (
        extract_outcome(2, agent_exit_code=0, elapsed_seconds=60.0, timeout_seconds=60.0)
        == IterationOutcome.TIMEOUT
    )


def test_extract_outcome_validation_failed() -> None:
    assert (
        extract_outcome(1, agent_exit_code=0, elapsed_seconds=10.0, timeout_seconds=60.0)
        == IterationOutcome.VALIDATION_FAILED
    )


def test_extract_failure_reason_picks_first_error_line() -> None:
    output = [
        "",
        "Running tests",
        "AssertionError: boom",
        "FAILED other test",
    ]

    assert extract_failure_reason(output) == "AssertionError: boom"


def test_extract_failure_reason_truncates() -> None:
    line = "ERROR: " + ("x" * 300)
    assert extract_failure_reason([line]) == line[:200]


def test_extract_failure_reason_none_when_no_errors() -> None:
    assert extract_failure_reason(["all good", "done"]) is None


def test_extract_files_changed_returns_repo_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file.txt").write_text("hello", encoding="utf-8")
    (repo / "changed.txt").write_text("world", encoding="utf-8")

    def _fake_run(*_args: object, **_kwargs: object):
        class Result:
            returncode = 0
            stdout = "changed.txt\nfile.txt\n"

        return Result()

    with patch("agent_recall.ralph.extraction.subprocess.run", _fake_run):
        assert extract_files_changed(repo) == ["changed.txt", "file.txt"]


def test_extract_git_diff_returns_diff_text(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def _fake_run(*_args: object, **_kwargs: object):
        class Result:
            returncode = 0
            stdout = "diff --git a/foo b/foo\n+add\n"

        return Result()

    with patch("agent_recall.ralph.extraction.subprocess.run", _fake_run):
        assert extract_git_diff(repo) == "diff --git a/foo b/foo\n+add\n"


def test_extract_validation_hint_skips_separators_and_blanks() -> None:
    output = ["", "-----", "====", " first actionable ", "later"]
    assert extract_validation_hint(output) == "first actionable"


def test_extract_from_artifacts_builds_deterministic_dict(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = ["ERROR: boom"]

    def _fake_run(*_args: object, **_kwargs: object):
        class Result:
            returncode = 0

            def __init__(self, stdout: str):
                self.stdout = stdout

        command = _args[0] if _args else []
        if isinstance(command, list):
            if len(command) >= 2 and command[1] == "diff" and "--name-only" in command:
                return Result("changed.txt\n")
            if len(command) >= 2 and command[1] == "diff":
                return Result("diff --git a/foo b/foo\n+add\n")
        return Result("")

    with patch("agent_recall.ralph.extraction.subprocess.run", _fake_run):
        result = extract_from_artifacts(1, output, 0, 5.0, 60.0, repo)

    assert result == {
        "outcome": IterationOutcome.VALIDATION_FAILED,
        "failure_reason": "ERROR: boom",
        "validation_hint": "ERROR: boom",
        "files_changed": ["changed.txt"],
        "git_diff": "diff --git a/foo b/foo\n+add\n",
    }
