"""Unit and integration tests for PRD archive data model and persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_recall.ralph.prd_archive import ArchivedPRDItem, PRDArchive
from agent_recall.storage.models import ChunkSource, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage

# --- ArchivedPRDItem ---


def test_archived_prd_item_to_dict_from_dict_round_trip() -> None:
    item = ArchivedPRDItem(
        id="AR-001",
        title="Test Item",
        description="A test",
        user_story="As a dev, I want tests.",
        steps=["Step 1", "Step 2"],
        acceptance_criteria=["Accept 1", "Accept 2"],
        validation_commands=["pytest", "ruff check"],
        completed_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        completion_iteration=3,
        key_decisions=["Use pytest"],
        lessons_learned=["Mock storage"],
    )
    data = item.to_dict()
    restored = ArchivedPRDItem.from_dict(data)
    assert restored.id == item.id
    assert restored.title == item.title
    assert restored.description == item.description
    assert restored.user_story == item.user_story
    assert restored.steps == item.steps
    assert restored.acceptance_criteria == item.acceptance_criteria
    assert restored.validation_commands == item.validation_commands
    assert restored.completion_iteration == item.completion_iteration
    assert restored.key_decisions == item.key_decisions
    assert restored.lessons_learned == item.lessons_learned


def test_archived_prd_item_to_searchable_text_includes_key_fields() -> None:
    item = ArchivedPRDItem(
        id="AR-002",
        title="Auth Feature",
        description="JWT auth",
        user_story="As a user, I want auth.",
        steps=["Implement JWT"],
        acceptance_criteria=["Tokens work"],
        validation_commands=[],
        completed_at=datetime.now(UTC),
        completion_iteration=1,
    )
    text = item.to_searchable_text()
    assert "AR-002" in text
    assert "Auth Feature" in text
    assert "JWT auth" in text
    assert "Implement JWT" in text
    assert "Tokens work" in text


def test_archived_prd_item_from_dict_acceptance_alias() -> None:
    data = {
        "id": "AR-003",
        "title": "Alias Test",
        "description": "",
        "user_story": "",
        "steps": [],
        "acceptance": ["Use acceptance alias"],
        "validation_commands": [],
        "completed_at": "2024-01-15T10:00:00Z",
        "completion_iteration": 0,
    }
    item = ArchivedPRDItem.from_dict(data)
    assert item.acceptance_criteria == ["Use acceptance alias"]


def test_archived_prd_item_from_dict_acceptance_criteria_preferred() -> None:
    data = {
        "id": "AR-004",
        "title": "Both Fields",
        "description": "",
        "user_story": "",
        "steps": [],
        "acceptance_criteria": ["Preferred"],
        "acceptance": ["Ignored"],
        "validation_commands": [],
        "completed_at": "2024-01-15T10:00:00Z",
        "completion_iteration": 0,
    }
    item = ArchivedPRDItem.from_dict(data)
    assert item.acceptance_criteria == ["Preferred"]


def test_archived_prd_item_from_dict_default_values_missing_fields() -> None:
    data = {"id": "AR-005", "title": "Minimal"}
    item = ArchivedPRDItem.from_dict(data)
    assert item.id == "AR-005"
    assert item.title == "Minimal"
    assert item.description == ""
    assert item.user_story == ""
    assert item.steps == []
    assert item.acceptance_criteria == []
    assert item.validation_commands == []
    assert item.completion_iteration == 0
    assert item.key_decisions == []
    assert item.lessons_learned == []


# --- PRDArchive persistence ---


def test_prd_archive_load_archive_missing_file_returns_empty(tmp_path: Path) -> None:
    archive = PRDArchive(tmp_path)
    items = archive._load_archive()
    assert items == []


def test_prd_archive_load_archive_corrupt_json_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    (tmp_path / "ralph" / "prd_archive.json").write_text("not valid json {{{")
    archive = PRDArchive(tmp_path)
    items = archive._load_archive()
    assert items == []


def test_prd_archive_save_archive_includes_version_metadata(tmp_path: Path) -> None:
    archive = PRDArchive(tmp_path)
    item = ArchivedPRDItem(
        id="AR-001",
        title="Test",
        description="",
        user_story="",
        steps=[],
        acceptance_criteria=[],
        validation_commands=[],
        completed_at=datetime.now(UTC),
        completion_iteration=0,
    )
    archive._save_archive([item])
    payload = json.loads((tmp_path / "ralph" / "prd_archive.json").read_text())
    assert payload.get("version") == 1
    assert "updated_at" in payload
    assert len(payload.get("items", [])) == 1


def test_prd_archive_archive_item_deduplication_by_id(tmp_path: Path) -> None:
    archive = PRDArchive(tmp_path)
    prd_item = {"id": "AR-001", "title": "First", "passes": True}
    first = archive.archive_item(prd_item)
    prd_item["title"] = "Second"
    second = archive.archive_item(prd_item)
    assert second.id == first.id
    assert second.title == "Second"
    items = archive.list_all()
    assert len(items) == 1


# --- PRDArchive.archive_completed_from_prd ---


def test_archive_completed_from_prd_mixed_pass_fail(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    prd_path = tmp_path / "prd.json"
    prd_path.write_text(
        json.dumps(
            {
                "project": "Test",
                "items": [
                    {"id": "AR-001", "title": "Pass", "passes": True},
                    {"id": "AR-002", "title": "Fail", "passes": False},
                    {"id": "AR-003", "title": "Pass2", "passes": True},
                ],
            }
        )
    )
    archive = PRDArchive(tmp_path)
    archived = archive.archive_completed_from_prd(prd_path)
    assert len(archived) == 2
    ids = {a.id for a in archived}
    assert "AR-001" in ids
    assert "AR-003" in ids
    assert "AR-002" not in ids


def test_archive_completed_from_prd_empty_prd(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    prd_path = tmp_path / "prd.json"
    prd_path.write_text(json.dumps({"project": "Empty", "items": []}))
    archive = PRDArchive(tmp_path)
    archived = archive.archive_completed_from_prd(prd_path)
    assert archived == []


def test_archive_completed_from_prd_no_passing_items(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    prd_path = tmp_path / "prd.json"
    prd_path.write_text(
        json.dumps(
            {
                "project": "All Fail",
                "items": [
                    {"id": "AR-001", "passes": False},
                    {"id": "AR-002", "passes": False},
                ],
            }
        )
    )
    archive = PRDArchive(tmp_path)
    archived = archive.archive_completed_from_prd(prd_path)
    assert archived == []


def test_archive_completed_from_prd_skips_already_archived(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    prd_path = tmp_path / "prd.json"
    prd_path.write_text(
        json.dumps(
            {
                "project": "Test",
                "items": [
                    {"id": "AR-001", "title": "Done", "passes": True},
                    {"id": "AR-002", "title": "New", "passes": True},
                ],
            }
        )
    )
    archive = PRDArchive(tmp_path)
    first_run = archive.archive_completed_from_prd(prd_path)
    assert len(first_run) == 2
    second_run = archive.archive_completed_from_prd(prd_path)
    assert len(second_run) == 0


def test_archive_completed_from_prd_prunes_archived_items_from_prd(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    prd_path = tmp_path / "prd.json"
    prd_path.write_text(
        json.dumps(
            {
                "project": "Test",
                "items": [
                    {"id": "AR-001", "title": "Done", "passes": True},
                    {"id": "AR-002", "title": "Pending", "passes": False},
                ],
            }
        )
    )
    archive = PRDArchive(tmp_path)
    archived = archive.archive_completed_from_prd(prd_path)

    assert len(archived) == 1
    updated_payload = json.loads(prd_path.read_text())
    updated_ids = [item["id"] for item in updated_payload["items"]]
    assert updated_ids == ["AR-002"]


def test_archive_completed_from_prd_prunes_previously_archived_items_from_prd(
    tmp_path: Path,
) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    prd_path = tmp_path / "prd.json"
    prd_path.write_text(
        json.dumps(
            {
                "project": "Test",
                "items": [
                    {"id": "AR-001", "title": "Done", "passes": True},
                    {"id": "AR-002", "title": "Pending", "passes": False},
                ],
            }
        )
    )
    archive = PRDArchive(tmp_path)
    assert len(archive.archive_completed_from_prd(prd_path)) == 1

    prd_path.write_text(
        json.dumps(
            {
                "project": "Test",
                "items": [
                    {"id": "AR-001", "title": "Done", "passes": True},
                    {"id": "AR-002", "title": "Pending", "passes": False},
                ],
            }
        )
    )

    archived = archive.archive_completed_from_prd(prd_path)
    assert archived == []

    updated_payload = json.loads(prd_path.read_text())
    updated_ids = [item["id"] for item in updated_payload["items"]]
    assert updated_ids == ["AR-002"]


def test_archive_completed_from_prd_does_not_prune_on_partial_archive_failure(
    tmp_path: Path,
) -> None:
    class PartiallyFailingArchive(PRDArchive):
        def __init__(self, agent_dir: Path):
            super().__init__(agent_dir)
            self._calls = 0

        def archive_item(
            self, prd_item: dict[str, object], *, iteration: int = 0
        ) -> ArchivedPRDItem:
            self._calls += 1
            if self._calls == 2:
                raise RuntimeError("simulated archive failure")
            return super().archive_item(prd_item, iteration=iteration)

    (tmp_path / "ralph").mkdir(parents=True)
    prd_path = tmp_path / "prd.json"
    original_payload = {
        "project": "Test",
        "items": [
            {"id": "AR-001", "title": "Done1", "passes": True},
            {"id": "AR-002", "title": "Done2", "passes": True},
            {"id": "AR-003", "title": "Pending", "passes": False},
        ],
    }
    prd_path.write_text(json.dumps(original_payload))

    archive = PartiallyFailingArchive(tmp_path)
    with pytest.raises(RuntimeError, match="simulated archive failure"):
        archive.archive_completed_from_prd(prd_path)

    assert json.loads(prd_path.read_text()) == original_payload


# --- PRDArchive.search ---


def test_prd_archive_search_returns_ordered_by_similarity(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    archive = PRDArchive(tmp_path)
    archive.archive_item({"id": "AR-001", "title": "Authentication", "description": "JWT tokens"})
    archive.archive_item({"id": "AR-002", "title": "Database", "description": "SQL"})
    results = archive.search("authentication tokens", top_k=2)
    assert len(results) == 2
    assert results[0][0].id == "AR-001"
    assert results[0][1] >= results[1][1]


def test_prd_archive_search_respects_top_k(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    archive = PRDArchive(tmp_path)
    for i in range(5):
        archive.archive_item({"id": f"AR-00{i}", "title": f"Item {i}"})
    results = archive.search("item", top_k=2)
    assert len(results) == 2


def test_prd_archive_search_item_ids_filter(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    archive = PRDArchive(tmp_path)
    archive.archive_item({"id": "AR-001", "title": "Auth"})
    archive.archive_item({"id": "AR-002", "title": "Database"})
    results = archive.search("auth", top_k=5, item_ids=["AR-001"])
    assert len(results) == 1
    assert results[0][0].id == "AR-001"


def test_prd_archive_search_empty_archive_returns_empty(tmp_path: Path) -> None:
    archive = PRDArchive(tmp_path)
    results = archive.search("anything", top_k=5)
    assert results == []


# --- PRDArchive._index_archived_item ---


def test_prd_archive_index_creates_chunk_with_correct_metadata(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    storage = SQLiteStorage(tmp_path / "state.db")
    archive = PRDArchive(tmp_path, storage=storage)
    archive.archive_item({"id": "AR-001", "title": "Test", "description": "Chunk content"})
    chunks = storage.list_chunks_with_embeddings()
    prd_chunks = [c for c in chunks if "prd" in c.tags]
    assert len(prd_chunks) == 1
    chunk = prd_chunks[0]
    assert chunk.source == ChunkSource.IMPORT
    assert chunk.label == SemanticLabel.DECISION_RATIONALE
    assert "prd" in chunk.tags
    assert "archived" in chunk.tags
    assert "ar-001" in chunk.tags


def test_prd_archive_index_skipped_when_storage_none(tmp_path: Path) -> None:
    archive = PRDArchive(tmp_path, storage=None)
    item = ArchivedPRDItem(
        id="AR-001",
        title="Test",
        description="",
        user_story="",
        steps=[],
        acceptance_criteria=[],
        validation_commands=[],
        completed_at=datetime.now(UTC),
        completion_iteration=0,
    )
    archive._index_archived_item(item)
    assert not (tmp_path / "state.db").exists()


# --- PRDArchive.match_knowledge_to_prd ---


def test_prd_archive_match_knowledge_to_prd_returns_related_items(tmp_path: Path) -> None:
    (tmp_path / "ralph").mkdir(parents=True)
    archive = PRDArchive(tmp_path)
    archive.archive_item({"id": "AR-001", "title": "Auth", "description": "JWT implementation"})
    results = archive.match_knowledge_to_prd("JWT authentication", top_k=3)
    assert len(results) >= 1
    assert results[0][0].id == "AR-001"
