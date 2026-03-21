from __future__ import annotations

from pathlib import Path

import pytest

from agent_recall.external_compaction.write_guard import (
    ExternalWritePolicyError,
    ExternalWriteScopeGuard,
)
from agent_recall.storage.files import KnowledgeTier


def test_write_guard_blocks_template_target_by_default() -> None:
    guard = ExternalWriteScopeGuard.from_config(
        repo_root=Path.cwd(),
        config={"compaction": {"external": {"write_target": "templates"}}},
    )

    with pytest.raises(ExternalWritePolicyError, match="allow_template_writes"):
        guard.resolve_target()


def test_write_guard_allows_template_target_when_config_enabled(tmp_path: Path) -> None:
    guard = ExternalWriteScopeGuard.from_config(
        repo_root=tmp_path,
        config={
            "compaction": {
                "external": {
                    "write_target": "templates",
                    "allow_template_writes": True,
                }
            }
        },
    )

    assert guard.resolve_target() == "templates"
    assert guard.template_path_for_tier(KnowledgeTier.GUARDRAILS) == (
        tmp_path / "src" / "agent_recall" / "templates" / "GUARDRAILS.md"
    )


def test_write_guard_rejects_non_allowlisted_template_path(tmp_path: Path) -> None:
    guard = ExternalWriteScopeGuard(
        repo_root=tmp_path,
        allow_template_writes=True,
    )

    with pytest.raises(ExternalWritePolicyError, match="allowlist"):
        guard.resolve_template_relative_path("src/agent_recall/templates/EXTRA.md")


def test_write_guard_rejects_parent_traversal(tmp_path: Path) -> None:
    guard = ExternalWriteScopeGuard(
        repo_root=tmp_path,
        allow_template_writes=True,
    )

    with pytest.raises(ExternalWritePolicyError, match="parent traversal"):
        guard.resolve_template_relative_path("../outside.md")


def test_write_guard_rejects_symlink_escape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    templates_dir = repo_root / "src" / "agent_recall" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    link = templates_dir / "GUARDRAILS.md"
    try:
        link.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - platform-specific fallback
        pytest.skip(f"symlink creation unavailable: {exc}")

    guard = ExternalWriteScopeGuard(
        repo_root=repo_root,
        allow_template_writes=True,
    )
    with pytest.raises(ExternalWritePolicyError, match="symlink"):
        guard.template_path_for_tier(KnowledgeTier.GUARDRAILS)
