from __future__ import annotations

from pathlib import Path

from agent_recall.core.memory_pack import (
    build_memory_pack,
    import_memory_pack,
    read_memory_pack,
    validate_memory_pack,
    write_memory_pack,
)
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage


def test_memory_pack_export_round_trip(storage, files, tmp_path: Path) -> None:
    files.write_tier(KnowledgeTier.GUARDRAILS, "# Guardrails\n\n- Keep tests deterministic\n")
    storage.store_chunk(
        Chunk(
            source=ChunkSource.MANUAL,
            source_ids=[],
            content="bounded retries for outbound sync",
            label=SemanticLabel.PATTERN,
            tags=["sync"],
        )
    )

    pack = build_memory_pack(storage, files)
    output = tmp_path / "memory-pack.json"
    write_memory_pack(output, pack)
    loaded = read_memory_pack(output)
    validation = validate_memory_pack(loaded)
    assert validation["valid"] is True
    assert loaded.tiers["GUARDRAILS"].startswith("# Guardrails")
    assert len(loaded.chunks) == 1


def test_memory_pack_import_append_strategy(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / ".agent"
    source_dir.mkdir(parents=True)
    (source_dir / "GUARDRAILS.md").write_text("# Guardrails\n\n- Source rule\n", encoding="utf-8")
    (source_dir / "STYLE.md").write_text("# Style\n\n- Source style\n", encoding="utf-8")
    (source_dir / "RECENT.md").write_text("# Recent\n", encoding="utf-8")
    source_files = FileStorage(source_dir)
    source_storage = SQLiteStorage(source_dir / "state.db")
    source_storage.store_chunk(
        Chunk(
            source=ChunkSource.MANUAL,
            source_ids=[],
            content="source memory chunk",
            label=SemanticLabel.PATTERN,
            tags=["source"],
        )
    )
    pack = build_memory_pack(source_storage, source_files)

    target_dir = tmp_path / "target" / ".agent"
    target_dir.mkdir(parents=True)
    (target_dir / "GUARDRAILS.md").write_text("# Guardrails\n\n- Existing rule\n", encoding="utf-8")
    (target_dir / "STYLE.md").write_text("# Style\n", encoding="utf-8")
    (target_dir / "RECENT.md").write_text("# Recent\n", encoding="utf-8")
    target_files = FileStorage(target_dir)
    target_storage = SQLiteStorage(target_dir / "state.db")

    report = import_memory_pack(target_storage, target_files, pack, strategy="append")
    assert report["chunks_written"] == 1
    assert report["tier_updates"] >= 1
    merged_guardrails = target_files.read_tier(KnowledgeTier.GUARDRAILS)
    assert "Existing rule" in merged_guardrails
    assert "Source rule" in merged_guardrails
