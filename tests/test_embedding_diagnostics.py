from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from agent_recall.cli import app_commands as cli_main
from agent_recall.core.embedding_diagnostics import EmbeddingDiagnostics
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage


def _chunk(
    content: str,
    embedding: list[float] | None = None,
    created_at: datetime | None = None,
) -> Chunk:
    return Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content=content,
        label=SemanticLabel.PATTERN,
        embedding=embedding,
        created_at=created_at or datetime.now(UTC),
    )


def test_embedding_diagnostics_coverage_and_size(storage) -> None:
    storage.store_chunk(_chunk("a"))
    storage.store_chunk(_chunk("b", embedding=[0.1, 0.2, 0.3, 0.4]))

    diagnostics = EmbeddingDiagnostics(storage)
    coverage = diagnostics.get_coverage_stats()
    size = diagnostics.estimate_embedding_size()

    assert coverage["total_chunks"] == 2
    assert coverage["embedded_chunks"] == 1
    assert coverage["pending"] == 1
    assert float(coverage["coverage_percent"]) == 50.0

    assert size["total_bytes"] == 16
    assert float(size["per_chunk_kb"]) > 0.0


def test_embedding_diagnostics_similarity_distribution(storage) -> None:
    storage.store_chunk(_chunk("x", embedding=[1.0, 0.0]))
    storage.store_chunk(_chunk("y", embedding=[1.0, 0.0]))
    storage.store_chunk(_chunk("z", embedding=[0.0, 1.0]))

    diagnostics = EmbeddingDiagnostics(storage)
    distribution = diagnostics.get_similarity_distribution()

    assert distribution["max"] >= distribution["median"] >= distribution["min"]
    assert distribution["std"] >= 0.0


def test_embedding_diagnostics_stale_check(storage) -> None:
    old = datetime.now(UTC) - timedelta(days=120)
    recent = datetime.now(UTC) - timedelta(days=5)
    storage.store_chunk(_chunk("old", embedding=[1.0], created_at=old))
    storage.store_chunk(_chunk("recent", embedding=[1.0], created_at=recent))

    diagnostics = EmbeddingDiagnostics(storage)
    stale = diagnostics.check_stale_embeddings(threshold_days=90)

    assert stale["embedded_chunks"] == 2
    assert stale["stale_chunks"] == 1


def test_cli_embedding_stats_command() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        cli_main.get_storage.cache_clear()
        result = runner.invoke(cli_main.app, ["embedding", "stats"])
        assert result.exit_code == 0
        assert "Embedding Stats" in result.output
        assert "Coverage:" in result.output

        storage = SQLiteStorage(Path(".agent") / "state.db")
        storage.store_chunk(
            Chunk(
                source=ChunkSource.MANUAL,
                source_ids=[],
                content="diagnostics",
                label=SemanticLabel.PATTERN,
                embedding=[0.1, 0.2, 0.3],
            )
        )

        result_after = runner.invoke(cli_main.app, ["embedding", "stats", "--stale-days", "0"])
        assert result_after.exit_code == 0
        assert "Embedded chunks:" in result_after.output
