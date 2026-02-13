from __future__ import annotations

import pytest

from agent_recall.storage import create_storage_backend
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import (
    AgentRecallConfig,
    Chunk,
    ChunkSource,
    SemanticLabel,
    Session,
)
from agent_recall.storage.remote import RemoteStorage
from agent_recall.storage.sqlite import SQLiteStorage


def test_create_storage_backend_uses_sqlite_for_local(tmp_path) -> None:
    config = AgentRecallConfig.model_validate({"storage": {"backend": "local"}})

    storage = create_storage_backend(config, tmp_path / "state.db")

    assert isinstance(storage, SQLiteStorage)


def test_create_storage_backend_uses_remote_storage_for_shared_file_url(tmp_path) -> None:
    shared_dir = tmp_path / "shared-memory"
    config = AgentRecallConfig.model_validate(
        {
            "storage": {
                "backend": "shared",
                "shared": {"base_url": f"file://{shared_dir}"},
            }
        }
    )

    storage = create_storage_backend(config, tmp_path / "state.db")

    assert isinstance(storage, RemoteStorage)


def test_create_storage_backend_shared_http_still_not_implemented(tmp_path) -> None:
    config = AgentRecallConfig.model_validate(
        {
            "storage": {
                "backend": "shared",
                "shared": {"base_url": "https://memory.example.com"},
            }
        }
    )

    with pytest.raises(NotImplementedError, match="HTTP shared storage service mode"):
        create_storage_backend(config, tmp_path / "state.db")


def test_remote_storage_shares_state_across_instances(tmp_path) -> None:
    shared_dir = tmp_path / "team-shared"
    shared_url = f"sqlite://{shared_dir}"
    config = AgentRecallConfig.model_validate(
        {
            "storage": {
                "backend": "shared",
                "shared": {"base_url": shared_url},
            }
        }
    )

    storage_a = create_storage_backend(config, tmp_path / "local-a.db")
    storage_b = create_storage_backend(config, tmp_path / "local-b.db")

    session = Session(task="shared session")
    storage_a.create_session(session)
    storage_a.store_chunk(
        Chunk(
            source=ChunkSource.MANUAL,
            content="Use shared backend URL for team memory.",
            label=SemanticLabel.PATTERN,
        )
    )

    shared_session = storage_b.get_session(session.id)

    assert shared_session is not None
    assert shared_session.task == "shared session"
    assert storage_b.count_chunks() == 1


def test_file_storage_syncs_tiers_to_shared_directory(tmp_path) -> None:
    repo_agent_dir = tmp_path / "repo-a" / ".agent"
    shared_dir = tmp_path / "team-shared"
    repo_agent_dir.mkdir(parents=True)

    files = FileStorage(repo_agent_dir, shared_tiers_dir=shared_dir)
    files.write_tier(KnowledgeTier.GUARDRAILS, "# Shared guardrails\n- Rule one\n")

    local_guardrails = repo_agent_dir / "GUARDRAILS.md"
    shared_guardrails = shared_dir / "GUARDRAILS.md"

    assert local_guardrails.read_text() == "# Shared guardrails\n- Rule one\n"
    assert shared_guardrails.read_text() == "# Shared guardrails\n- Rule one\n"


def test_file_storage_reads_shared_tier_before_local(tmp_path) -> None:
    repo_agent_dir = tmp_path / "repo-b" / ".agent"
    shared_dir = tmp_path / "team-shared"
    repo_agent_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    (repo_agent_dir / "STYLE.md").write_text("# Local style\n- Local only\n")
    (shared_dir / "STYLE.md").write_text("# Shared style\n- Team rule\n")

    files = FileStorage(repo_agent_dir, shared_tiers_dir=shared_dir)

    assert files.read_tier(KnowledgeTier.STYLE) == "# Shared style\n- Team rule\n"
