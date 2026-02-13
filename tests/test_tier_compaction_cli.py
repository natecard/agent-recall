"""CLI tests for tier compaction command."""

import pytest
from typer.testing import CliRunner

from agent_recall.cli.main import app, get_storage

runner = CliRunner()


class TestCompactTiersCommand:
    """Test compact-tiers CLI command."""

    @pytest.fixture(autouse=True)
    def clear_storage_cache(self):
        """Clear storage cache before each test."""
        get_storage.cache_clear()
        yield

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with initialized agent."""
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Initialize the repo
            result = runner.invoke(app, ["init", "--no-splash"])
            assert result.exit_code == 0

            # Add some test entries to GUARDRAILS
            agent_dir = tmp_path / ".agent"
            guardrails_content = """# Guardrails

Rules and warnings learned during development.

## 2026-02-12T10:00:00Z Iteration 1 (AR-001)
- First entry

## 2026-02-12T11:00:00Z Iteration 2 (AR-002)
- Second entry

## 2026-02-12T12:00:00Z Iteration 1 (AR-001)
- Duplicate entry

## 2026-02-12T13:00:00Z Iteration 3 (AR-003)
- Third entry
"""
            (agent_dir / "GUARDRAILS.md").write_text(guardrails_content)

            yield tmp_path
        finally:
            os.chdir(original_cwd)

    def test_compact_tiers_basic(self, temp_repo):
        """Test basic compact-tiers command."""
        result = runner.invoke(app, ["compact-tiers"])

        assert result.exit_code == 0
        assert "Tier compaction complete" in result.output
        assert "GUARDRAILS" in result.output
        assert "4 → 3" in result.output  # 4 entries, 1 duplicate removed
        assert "Duplicates removed: 1" in result.output

    def test_compact_tiers_dry_run(self, temp_repo):
        """Test compact-tiers --dry-run."""
        result = runner.invoke(app, ["compact-tiers", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run mode" in result.output
        assert "Tier compaction complete" in result.output

    def test_compact_tiers_with_max_entries(self, temp_repo):
        """Test compact-tiers with custom max-entries."""
        # First add more entries
        agent_dir = temp_repo / ".agent"
        content = (agent_dir / "GUARDRAILS.md").read_text()
        for i in range(10):
            ts = f"2026-02-12T{i + 20:02d}:00:00Z"
            item = f"AR-{i + 100:03d}"
            content += f"\n## {ts} Iteration {i + 10} ({item})\n- Entry {i}\n"
        (agent_dir / "GUARDRAILS.md").write_text(content)

        result = runner.invoke(app, ["compact-tiers", "--max-entries", "10"])

        assert result.exit_code == 0
        # Should have around 14 entries (original 4 + 10 new, minus 1 duplicate = 13)
        # With max-entries 10, should end up with 10
        assert "Entries:" in result.output

    def test_compact_tiers_strict_mode(self, temp_repo):
        """Test compact-tiers with --strict."""
        result = runner.invoke(app, ["compact-tiers", "--strict"])

        assert result.exit_code == 0
        assert "Tier compaction complete" in result.output

    def test_compact_tiers_multiple_runs_idempotent(self, temp_repo):
        """Test that running compact-tiers twice is idempotent."""
        # First run
        result1 = runner.invoke(app, ["compact-tiers"])
        assert result1.exit_code == 0
        assert "Duplicates removed: 1" in result1.output

        # Second run should have no duplicates to remove
        result2 = runner.invoke(app, ["compact-tiers"])
        assert result2.exit_code == 0
        # After first run, no more duplicates
        assert (
            "Duplicates removed: 0" in result2.output or "Duplicates removed" not in result2.output
        )

    def test_compact_tiers_shows_all_tiers(self, temp_repo):
        """Test that all three tiers are shown in output."""
        result = runner.invoke(app, ["compact-tiers"])

        assert result.exit_code == 0
        assert "GUARDRAILS:" in result.output
        assert "STYLE:" in result.output
        assert "RECENT:" in result.output
        assert "Total:" in result.output

    def test_compact_tiers_not_initialized(self):
        """Test compact-tiers fails when not initialized."""
        with runner.isolated_filesystem():
            result = runner.invoke(app, ["compact-tiers"])

            assert result.exit_code == 1
            assert "Not initialized" in result.output or "init" in result.output.lower()
