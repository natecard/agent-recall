from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_recall.storage.files import FileStorage
from agent_recall.storage.sqlite import SQLiteStorage


@pytest.fixture
def temp_agent_dir() -> Iterator[Path]:
    """Create a temporary .agent directory for testing."""
    temp_dir = Path(tempfile.mkdtemp()) / ".agent"
    temp_dir.mkdir()
    (temp_dir / "logs").mkdir()
    (temp_dir / "archive").mkdir()
    (temp_dir / "GUARDRAILS.md").write_text("# Guardrails\n")
    (temp_dir / "STYLE.md").write_text("# Style\n")
    (temp_dir / "RECENT.md").write_text("# Recent\n")
    (temp_dir / "config.yaml").write_text("llm:\n  provider: openai\n  model: gpt-4o-mini\n")

    yield temp_dir

    shutil.rmtree(temp_dir.parent)


@pytest.fixture
def storage(temp_agent_dir: Path) -> SQLiteStorage:
    """Create a SQLiteStorage instance for testing."""
    return SQLiteStorage(temp_agent_dir / "state.db")


@pytest.fixture
def files(temp_agent_dir: Path) -> FileStorage:
    """Create a FileStorage instance for testing."""
    return FileStorage(temp_agent_dir)
