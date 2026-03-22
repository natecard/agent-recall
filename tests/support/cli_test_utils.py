from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from agent_recall.storage.files import FileStorage


def initialize_agent_repo(runner: CliRunner, app) -> FileStorage:
    """Initialize .agent in the active test workspace and return FileStorage."""
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    return FileStorage(Path(".agent"))


def load_agent_config() -> dict[str, Any]:
    """Read the active test repo config from .agent/config.yaml."""
    return FileStorage(Path(".agent")).read_config()
