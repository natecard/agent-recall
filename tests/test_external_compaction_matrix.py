from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_recall.external_compaction.mcp_server import run_external_compaction_mcp
from agent_recall.external_compaction.service import ExternalCompactionService
from agent_recall.external_compaction.write_guard import WriteTarget
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import (
    LogEntry,
    LogSource,
    SemanticLabel,
    SharedStorageConfig,
)
from agent_recall.storage.remote import RemoteStorage
from agent_recall.storage.sqlite import SQLiteStorage

_MATRIX_CASES = [
    ("local", "runtime", False),
    ("local", "runtime", True),
    ("local", "templates", False),
    ("local", "templates", True),
    ("shared_file", "runtime", False),
    ("shared_file", "runtime", True),
    ("shared_file", "templates", False),
    ("shared_file", "templates", True),
]


class _FakeFastMCP:
    last_instance: _FakeFastMCP | None = None

    def __init__(self, _name: str) -> None:
        self.tools: dict[str, Any] = {}
        _FakeFastMCP.last_instance = self

    def tool(self):  # noqa: ANN201
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator

    def run(self) -> None:
        return


def _build_storage(tmp_path: Path, backend: str):
    if backend == "local":
        return SQLiteStorage(tmp_path / ".agent" / "state.db")
    if backend == "shared_file":
        shared_path = tmp_path / "shared-memory"
        config = SharedStorageConfig(
            base_url=f"sqlite://{shared_path}",
            tenant_id="matrix-tenant",
            project_id="matrix-project",
        )
        return RemoteStorage(config)
    raise AssertionError(f"Unsupported backend in test matrix: {backend}")


def _build_files(tmp_path: Path, write_target: str) -> FileStorage:
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "GUARDRAILS.md").write_text("# Guardrails\n", encoding="utf-8")
    (agent_dir / "STYLE.md").write_text("# Style Guide\n", encoding="utf-8")
    (agent_dir / "RECENT.md").write_text("# Recent Sessions\n", encoding="utf-8")
    (agent_dir / "config.yaml").write_text(
        (
            "compaction:\n"
            "  external:\n"
            f"    write_target: {write_target}\n"
            "    allow_template_writes: true\n"
            "    conflict_policy: prefer_newest\n"
        ),
        encoding="utf-8",
    )
    return FileStorage(agent_dir)


@pytest.mark.parametrize(
    ("backend", "write_target", "use_mcp"),
    _MATRIX_CASES,
    ids=[
        f"{backend}-{write_target}-mcp_{use_mcp}"
        for backend, write_target, use_mcp in _MATRIX_CASES
    ],
)
def test_external_compaction_matrix_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    write_target: str,
    use_mcp: bool,
) -> None:
    files = _build_files(tmp_path, write_target)
    storage = _build_storage(tmp_path, backend)
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=tmp_path,
    )

    source_session_id = f"{backend}-{write_target}-{int(use_mcp)}"
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id=source_session_id,
            content="Prefer deterministic migration ordering and dry-run checkpoints",
            label=SemanticLabel.GOTCHA,
        )
    )

    line = "- [GOTCHA] Validate migration ordering before apply."
    if use_mcp:
        _FakeFastMCP.last_instance = None
        fake_module = SimpleNamespace(FastMCP=_FakeFastMCP)
        monkeypatch.setattr(
            "agent_recall.external_compaction.mcp_server.importlib.import_module",
            lambda _name: fake_module,
        )
        run_external_compaction_mcp(service, write_target=cast(WriteTarget, write_target))
        assert _FakeFastMCP.last_instance is not None
        tools = _FakeFastMCP.last_instance.tools
        payload = tools["build_compaction_payload"](
            limit=10,
            pending_only=True,
            entry_limit=50,
            target=write_target,
        )
        assert any(
            item["source_session_id"] == source_session_id for item in payload["conversations"]
        )
        result = tools["apply_compaction_notes"](
            notes=[
                {
                    "tier": "GUARDRAILS",
                    "line": line,
                    "source_session_ids": [source_session_id],
                }
            ],
            target=write_target,
            dry_run=False,
            mark_processed=True,
        )
    else:
        payload = service.build_payload(
            limit=10,
            pending_only=True,
            entry_limit=50,
            write_target=cast(WriteTarget, write_target),
        )
        assert any(
            item["source_session_id"] == source_session_id for item in payload["conversations"]
        )
        result = service.apply_notes_payload(
            {
                "notes": [
                    {
                        "tier": "GUARDRAILS",
                        "line": line,
                        "source_session_ids": [source_session_id],
                    }
                ]
            },
            write_target=cast(WriteTarget, write_target),
            dry_run=False,
            mark_processed=True,
        )

    assert result["notes_applied"] == 1
    if write_target == "runtime":
        guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
        assert "Validate migration ordering before apply." in guardrails
    else:
        template_path = tmp_path / "src" / "agent_recall" / "templates" / "GUARDRAILS.md"
        assert template_path.exists()
        assert "Validate migration ordering before apply." in template_path.read_text(
            encoding="utf-8"
        )
