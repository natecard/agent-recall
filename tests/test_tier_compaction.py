"""Tests for tier compaction hook."""


import pytest

from agent_recall.core.tier_compaction import (
    TierCompactionConfig,
    TierCompactionHook,
    TierCompactionResult,
    TierCompactionSummary,
    format_compaction_summary,
)
from agent_recall.storage.files import FileStorage, KnowledgeTier


class TestTierCompactionConfig:
    """Test TierCompactionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TierCompactionConfig()
        assert config.auto_run is True
        assert config.max_entries_per_tier == 50
        assert config.strict_deduplication is False
        assert config.summary_threshold_entries == 40
        assert config.summary_max_entries == 20

    def test_from_config_empty(self):
        """Test loading config from empty dict."""
        config = TierCompactionConfig.from_config({})
        assert config.auto_run is True
        assert config.max_entries_per_tier == 50

    def test_from_config_custom(self):
        """Test loading config with custom values."""
        config_dict = {
            "tier_compaction": {
                "auto_run": False,
                "max_entries_per_tier": 30,
                "strict_deduplication": True,
                "summary_threshold_entries": 35,
                "summary_max_entries": 15,
            }
        }
        config = TierCompactionConfig.from_config(config_dict)
        assert config.auto_run is False
        assert config.max_entries_per_tier == 30
        assert config.strict_deduplication is True
        assert config.summary_threshold_entries == 35
        assert config.summary_max_entries == 15


class TestTierCompactionHookBasic:
    """Test basic tier compaction hook functionality."""

    @pytest.fixture
    def temp_agent_dir(self, tmp_path):
        """Create a temporary agent directory."""
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        (agent_dir / "logs").mkdir()
        (agent_dir / "archive").mkdir()

        # Create initial tier files
        (agent_dir / "GUARDRAILS.md").write_text(
            "# Guardrails\n\nRules and warnings learned during development.\n"
        )
        (agent_dir / "STYLE.md").write_text(
            "# Style\n\nPatterns and preferences learned during development.\n"
        )
        (agent_dir / "RECENT.md").write_text("# Recent\n\nRecent development activity summaries.\n")
        (agent_dir / "config.yaml").write_text("llm:\n  provider: anthropic\n")

        return agent_dir

    def test_compact_empty_files(self, temp_agent_dir):
        """Test compacting empty tier files."""
        files = FileStorage(temp_agent_dir)
        config = TierCompactionConfig()
        hook = TierCompactionHook(files, config)

        summary = hook.compact_all()

        assert len(summary.results) == 3
        for result in summary.results:
            assert result.entries_before == 0
            assert result.entries_after == 0
            assert result.duplicates_removed == 0
            assert result.entries_summarized == 0

    def test_compact_no_changes_needed(self, temp_agent_dir):
        """Test compacting files that don't need changes."""
        files = FileStorage(temp_agent_dir)

        # Add a single entry
        content = files.read_tier(KnowledgeTier.GUARDRAILS)
        content += "\n## 2026-02-12T10:00:00Z Iteration 1 (AR-001)\n- Test entry\n"
        files.write_tier(KnowledgeTier.GUARDRAILS, content)

        config = TierCompactionConfig()
        hook = TierCompactionHook(files, config)
        summary = hook.compact_all()

        guardrails_result = [r for r in summary.results if r.tier == KnowledgeTier.GUARDRAILS][0]
        assert guardrails_result.entries_before == 1
        assert guardrails_result.entries_after == 1
        assert guardrails_result.duplicates_removed == 0


class TestTierCompactionDeduplication:
    """Test deduplication functionality."""

    @pytest.fixture
    def temp_agent_dir(self, tmp_path):
        """Create a temporary agent directory with entries."""
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        (agent_dir / "logs").mkdir()
        (agent_dir / "archive").mkdir()

        content = """# Guardrails

Rules and warnings learned during development.

## 2026-02-12T10:00:00Z Iteration 1 (AR-001)
- First entry

## 2026-02-12T11:00:00Z Iteration 2 (AR-002)
- Second entry

## 2026-02-12T12:00:00Z Iteration 1 (AR-001)
- Duplicate iteration/item_id

## 2026-02-12T13:00:00Z Iteration 3 (AR-003)
- Third entry
"""

        (agent_dir / "GUARDRAILS.md").write_text(content)
        (agent_dir / "STYLE.md").write_text(
            "# Style\n\nPatterns and preferences learned during development.\n"
        )
        (agent_dir / "RECENT.md").write_text("# Recent\n\nRecent development activity summaries.\n")
        (agent_dir / "config.yaml").write_text("llm:\n  provider: anthropic\n")

        return agent_dir

    def test_deduplication_by_iteration_item_id(self, temp_agent_dir):
        """Test that duplicates by iteration+item_id are removed."""
        files = FileStorage(temp_agent_dir)
        config = TierCompactionConfig()
        hook = TierCompactionHook(files, config)

        summary = hook.compact_all()

        guardrails_result = [r for r in summary.results if r.tier == KnowledgeTier.GUARDRAILS][0]
        assert guardrails_result.entries_before == 4
        assert guardrails_result.entries_after == 3
        assert guardrails_result.duplicates_removed == 1

    def test_strict_deduplication_with_same_content(self, temp_agent_dir):
        """Test that strict mode considers content hash when iteration+item_id are same."""
        files = FileStorage(temp_agent_dir)

        # Add an entry with same iteration+item_id but DIFFERENT content
        content = files.read_tier(KnowledgeTier.GUARDRAILS)
        content += "\n## 2026-02-12T14:00:00Z Iteration 1 (AR-001)\n- Different content here\n"
        files.write_tier(KnowledgeTier.GUARDRAILS, content)

        config = TierCompactionConfig(strict_deduplication=True)
        hook = TierCompactionHook(files, config)

        summary = hook.compact_all()

        guardrails_result = [r for r in summary.results if r.tier == KnowledgeTier.GUARDRAILS][0]
        # With strict mode, same iteration+item_id but different content should NOT be deduped
        # So we should still have the original 4 entries (no duplicates removed)
        assert guardrails_result.entries_before == 5  # Added one more
        assert guardrails_result.entries_after == 5  # All kept because content differs


class TestTierCompactionSizeBudget:
    """Test size budget enforcement."""

    @pytest.fixture
    def temp_agent_dir_many_entries(self, tmp_path):
        """Create a temporary agent directory with many entries."""
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        (agent_dir / "logs").mkdir()
        (agent_dir / "archive").mkdir()

        # Create 60 entries (over default 50 limit)
        lines = ["# Guardrails", "", "Rules and warnings learned during development."]
        for i in range(60):
            lines.append(
                f"\n## 2026-02-{(i % 28) + 1:02d}T{i:02d}:00:00Z Iteration {i} (AR-{i:03d})"
            )
            lines.append(f"- Entry number {i}")

        (agent_dir / "GUARDRAILS.md").write_text("\n".join(lines))
        (agent_dir / "STYLE.md").write_text(
            "# Style\n\nPatterns and preferences learned during development.\n"
        )
        (agent_dir / "RECENT.md").write_text("# Recent\n\nRecent development activity summaries.\n")
        (agent_dir / "config.yaml").write_text("llm:\n  provider: anthropic\n")

        return agent_dir

    def test_size_budget_enforced(self, temp_agent_dir_many_entries):
        """Test that size budget removes oldest entries."""
        files = FileStorage(temp_agent_dir_many_entries)
        config = TierCompactionConfig(max_entries_per_tier=50)
        hook = TierCompactionHook(files, config)

        summary = hook.compact_all()

        guardrails_result = [r for r in summary.results if r.tier == KnowledgeTier.GUARDRAILS][0]
        assert guardrails_result.entries_before == 60
        assert guardrails_result.entries_after == 50

    def test_custom_size_budget(self, temp_agent_dir_many_entries):
        """Test custom size budget."""
        files = FileStorage(temp_agent_dir_many_entries)
        config = TierCompactionConfig(max_entries_per_tier=30)
        hook = TierCompactionHook(files, config)

        summary = hook.compact_all()

        guardrails_result = [r for r in summary.results if r.tier == KnowledgeTier.GUARDRAILS][0]
        assert guardrails_result.entries_after == 30


class TestTierCompactionSummary:
    """Test summary functionality."""

    def test_summary_totals(self):
        """Test that summary totals are calculated correctly."""
        results = [
            TierCompactionResult(
                tier=KnowledgeTier.GUARDRAILS,
                entries_before=10,
                entries_after=8,
                bytes_before=1000,
                bytes_after=800,
                duplicates_removed=2,
                entries_summarized=0,
            ),
            TierCompactionResult(
                tier=KnowledgeTier.STYLE,
                entries_before=20,
                entries_after=15,
                bytes_before=2000,
                bytes_after=1500,
                duplicates_removed=3,
                entries_summarized=2,
            ),
            TierCompactionResult(
                tier=KnowledgeTier.RECENT,
                entries_before=5,
                entries_after=5,
                bytes_before=500,
                bytes_after=500,
                duplicates_removed=0,
                entries_summarized=0,
            ),
        ]

        summary = TierCompactionSummary(results=results)

        assert summary.total_entries_before == 35
        assert summary.total_entries_after == 28
        assert summary.total_bytes_before == 3500
        assert summary.total_bytes_after == 2800
        assert summary.total_duplicates_removed == 5
        assert summary.total_entries_summarized == 2


class TestFormatCompactionSummary:
    """Test summary formatting."""

    def test_format_summary(self):
        """Test formatting a compaction summary."""
        results = [
            TierCompactionResult(
                tier=KnowledgeTier.GUARDRAILS,
                entries_before=10,
                entries_after=8,
                bytes_before=1000,
                bytes_after=800,
                duplicates_removed=2,
                entries_summarized=0,
            ),
        ]
        summary = TierCompactionSummary(results=results)

        formatted = format_compaction_summary(summary)

        assert "GUARDRAILS" in formatted
        assert "10 → 8" in formatted
        assert "1000 → 800 bytes" in formatted
        assert "Duplicates removed: 2" in formatted

    def test_format_summary_no_changes(self):
        """Test formatting a summary with no changes."""
        results = [
            TierCompactionResult(
                tier=KnowledgeTier.GUARDRAILS,
                entries_before=5,
                entries_after=5,
                bytes_before=500,
                bytes_after=500,
                duplicates_removed=0,
                entries_summarized=0,
            ),
        ]
        summary = TierCompactionSummary(results=results)

        formatted = format_compaction_summary(summary)

        assert "GUARDRAILS" in formatted
        assert "5 → 5" in formatted
        assert "Duplicates removed" not in formatted
        assert "summarized" not in formatted


class TestTierCompactionHookParity:
    """Test parity between manual hook and CLI/TUI paths."""

    @pytest.fixture
    def temp_agent_dir(self, tmp_path):
        """Create a temporary agent directory with sample entries."""
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        (agent_dir / "logs").mkdir()
        (agent_dir / "archive").mkdir()

        # Create GUARDRAILS with duplicates and many entries
        lines = ["# Guardrails", "", "Rules and warnings learned during development."]
        for i in range(55):
            # Create some duplicates
            item_id = f"AR-{i % 10:03d}"  # Will create duplicates
            lines.append(
                f"\n## 2026-02-{(i % 28) + 1:02d}T{i:02d}:00:00Z Iteration {i % 5} ({item_id})"
            )
            lines.append(f"- Entry number {i}")

        (agent_dir / "GUARDRAILS.md").write_text("\n".join(lines))
        (agent_dir / "STYLE.md").write_text(
            "# Style\n\nPatterns and preferences learned during development.\n"
        )
        (agent_dir / "RECENT.md").write_text("# Recent\n\nRecent development activity summaries.\n")
        (agent_dir / "config.yaml").write_text("llm:\n  provider: anthropic\n")

        return agent_dir

    def test_manual_hook_produces_same_result_as_repeated_calls(self, temp_agent_dir):
        """Test that running the hook twice produces stable results."""
        files = FileStorage(temp_agent_dir)
        config = TierCompactionConfig()
        hook = TierCompactionHook(files, config)

        # First compaction
        summary1 = hook.compact_all()

        # Second compaction should be idempotent
        summary2 = hook.compact_all()

        # Second run should have no changes
        for r1, r2 in zip(summary1.results, summary2.results):
            assert r2.entries_before == r2.entries_after
            assert r2.duplicates_removed == 0

    def test_config_preserved_across_runs(self, temp_agent_dir):
        """Test that configuration is respected consistently."""
        files = FileStorage(temp_agent_dir)

        # Test with different configs
        for max_entries in [30, 40, 50]:
            config = TierCompactionConfig(max_entries_per_tier=max_entries)
            hook = TierCompactionHook(files, config)
            summary = hook.compact_all()

            guardrails_result = [r for r in summary.results if r.tier == KnowledgeTier.GUARDRAILS][
                0
            ]
            assert guardrails_result.entries_after <= max_entries
