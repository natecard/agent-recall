"""Tests for tier_writer module with structured tier file writing policies."""

from pathlib import Path

from agent_recall.core.tier_writer import (
    GUARDRAILS_SCHEMA,
    RECENT_SCHEMA,
    STYLE_SCHEMA,
    TierWriter,
    WriteMode,
    WritePolicy,
    _compute_content_hash,
    _normalize_for_dedup,
    get_tier_statistics,
    lint_tier_file,
)
from agent_recall.storage.files import FileStorage, KnowledgeTier


class TestContentNormalization:
    """Test content normalization for deduplication."""

    def test_normalize_for_dedup_collapses_whitespace(self):
        text = "  hello    world  \n\n  test  "
        result = _normalize_for_dedup(text)
        assert result == "hello world test"

    def test_normalize_for_dedup_lowercases(self):
        text = "Hello World TEST"
        result = _normalize_for_dedup(text)
        assert result == "hello world test"

    def test_compute_content_hash_consistency(self):
        text = "Test content"
        hash1 = _compute_content_hash(text)
        hash2 = _compute_content_hash(text)
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_content_hash_differentiates(self):
        hash1 = _compute_content_hash("Content A")
        hash2 = _compute_content_hash("Content B")
        assert hash1 != hash2


class TestWritePolicy:
    """Test WritePolicy configuration."""

    def test_default_policy(self):
        policy = WritePolicy()
        assert policy.mode == WriteMode.APPEND
        assert policy.deduplicate is True
        assert policy.max_entries is None
        assert policy.section_target is None

    def test_custom_policy(self):
        policy = WritePolicy(
            mode=WriteMode.REPLACE_SECTION,
            deduplicate=False,
            max_entries=50,
            section_target="general",
        )
        assert policy.mode == WriteMode.REPLACE_SECTION
        assert policy.deduplicate is False
        assert policy.max_entries == 50
        assert policy.section_target == "general"


class TestTierSchemas:
    """Test tier file schemas."""

    def test_guardrails_schema_has_required_header(self):
        header_section = next(s for s in GUARDRAILS_SCHEMA if s.name == "header")
        assert header_section.required is True

    def test_guardrails_schema_has_hard_failure_section(self):
        failure_section = next(s for s in GUARDRAILS_SCHEMA if s.name == "hard_failure")
        assert failure_section.max_entries == 50

    def test_style_schema_has_iteration_section(self):
        iteration_section = next(s for s in STYLE_SCHEMA if s.name == "iteration")
        assert iteration_section.max_entries == 100

    def test_recent_schema_has_iteration_section(self):
        iteration_section = next(s for s in RECENT_SCHEMA if s.name == "iteration")
        assert iteration_section.max_entries == 50


class TestTierWriterGuardrails:
    """Test TierWriter guardrails entry writing."""

    def test_write_guardrails_entry_creates_content(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        writer = TierWriter(files)

        written = writer.write_guardrails_entry(
            iteration=1,
            item_id="TEST-001",
            item_title="Test Item",
            reason="validation_failed",
            validation_hint="pytest failed",
        )

        assert written is True
        content = files.read_tier(KnowledgeTier.GUARDRAILS)
        assert "Iteration 1 (TEST-001)" in content
        assert "Test Item" in content
        assert "pytest failed" in content
        assert "Do not move to a new PRD item" in content

    def test_write_guardrails_entry_skips_duplicate(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        writer = TierWriter(files)

        # Write first entry
        writer.write_guardrails_entry(
            iteration=1,
            item_id="TEST-001",
            item_title="Test Item",
            reason="validation_failed",
        )

        # Try to write similar entry (should be skipped)
        written = writer.write_guardrails_entry(
            iteration=1,
            item_id="TEST-001",
            item_title="Test Item",
            reason="validation_failed",
        )

        assert written is False

    def test_write_guardrails_entry_different_iterations_not_duplicate(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        writer = TierWriter(files)

        writer.write_guardrails_entry(
            iteration=1,
            item_id="TEST-001",
            item_title="Test Item",
            reason="validation_failed",
        )

        written = writer.write_guardrails_entry(
            iteration=2,
            item_id="TEST-001",
            item_title="Test Item",
            reason="validation_failed",
        )

        assert written is True

    def test_write_guardrails_hard_failure(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        writer = TierWriter(files)

        written = writer.write_guardrails_hard_failure(
            iteration=1,
            item_id="TEST-001",
            item_title="Test Item",
            validation_errors=["Error 1", "Error 2"],
            validation_hint="pytest failed",
        )

        assert written is True
        content = files.read_tier(KnowledgeTier.GUARDRAILS)
        assert "HARD FAILURE" in content
        assert "Error 1" in content
        assert "Error 2" in content


class TestTierWriterStyle:
    """Test TierWriter style entry writing."""

    def test_write_style_entry_creates_content(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        writer = TierWriter(files)

        written = writer.write_style_entry(
            iteration=1,
            item_id="TEST-001",
            validation_hint="Start debugging from line 5",
        )

        assert written is True
        content = files.read_tier(KnowledgeTier.STYLE)
        assert "Iteration 1 (TEST-001)" in content
        assert "Prefer one logical change per commit" in content
        assert "Start debugging from line 5" in content


class TestTierWriterRecent:
    """Test TierWriter recent entry writing."""

    def test_write_recent_entry_creates_content(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        writer = TierWriter(files)

        written = writer.write_recent_entry(
            iteration=1,
            item_id="TEST-001",
            item_title="Test Item",
            work_mode="feature",
            agent_exit=0,
            validate_status="passed",
            outcome="progressed",
        )

        assert written is True
        content = files.read_tier(KnowledgeTier.RECENT)
        assert "Iteration 1" in content
        assert "TEST-001 - Test Item" in content
        assert "Mode: feature" in content
        assert "Outcome: progressed" in content


class TestBoundedAppend:
    """Test bounded append behavior."""

    def test_bounded_append_respects_max_entries(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        policy = WritePolicy(max_entries=3)
        writer = TierWriter(files, policy)

        # Write 5 entries
        for i in range(5):
            writer.write_recent_entry(
                iteration=i + 1,
                item_id=f"TEST-{i + 1:03d}",
                item_title=f"Test Item {i + 1}",
                work_mode="feature",
                agent_exit=0,
                validate_status="passed",
                outcome="progressed",
            )

        content = files.read_tier(KnowledgeTier.RECENT)
        # Should only have 3 entries (entries 3, 4, 5)
        assert content.count("## ") == 3
        assert "TEST-001" not in content  # Oldest should be removed
        assert "TEST-003" in content
        assert "TEST-005" in content


class TestValidation:
    """Test tier file validation."""

    def test_validate_empty_content(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        writer = TierWriter(files)

        errors = writer.validate_tier_content(KnowledgeTier.GUARDRAILS, "")
        assert len(errors) == 1
        assert "Missing required section" in errors[0]

    def test_validate_valid_content(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        writer = TierWriter(files)

        valid_content = "# Guardrails\n\n## 2024-01-01T00:00:00Z Iteration 1\n- Test entry\n"
        errors = writer.validate_tier_content(KnowledgeTier.GUARDRAILS, valid_content)
        assert len(errors) == 0

    def test_validate_detects_malformed_header(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        writer = TierWriter(files)

        content = "# Guardrails\n\n## \n- Empty header\n"
        errors = writer.validate_tier_content(KnowledgeTier.GUARDRAILS, content)
        assert any("Malformed section header" in e for e in errors)


class TestLintTierFile:
    """Test tier file linting."""

    def test_lint_valid_content(self):
        content = (
            "# Guardrails\n\n"
            "## 2024-01-15T10:30:00Z Iteration 1\n"
            "- Test entry with sufficient content\n"
        )
        errors, warnings = lint_tier_file(KnowledgeTier.GUARDRAILS, content)
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_lint_empty_content(self):
        errors, warnings = lint_tier_file(KnowledgeTier.GUARDRAILS, "")
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_lint_missing_header(self):
        content = "## Some entry\n- Content\n"
        errors, warnings = lint_tier_file(KnowledgeTier.GUARDRAILS, content)
        assert any("header" in e.lower() for e in errors)

    def test_lint_low_signal_entry(self):
        content = "# Guardrails\n\n## 2024-01-15T10:30:00Z Iteration 1\n- x\n"
        errors, warnings = lint_tier_file(KnowledgeTier.GUARDRAILS, content)
        assert any("low-signal" in w.lower() for w in warnings)

    def test_lint_malformed_timestamp(self):
        content = "# Guardrails\n\n## 2024-99-99T99:99:99Z Iteration 1\n- Content\n"
        errors, warnings = lint_tier_file(KnowledgeTier.GUARDRAILS, content)
        assert any("timestamp" in w.lower() for w in warnings)

    def test_lint_strict_mode(self):
        content = "# Guardrails\n\n## 2024-01-15T10:30:00Z Iteration 1\n- x\n"
        errors, warnings = lint_tier_file(KnowledgeTier.GUARDRAILS, content, strict=True)
        # In strict mode, warnings become errors
        assert len(errors) > 0 or len(warnings) == 0


class TestTierStatistics:
    """Test tier file statistics."""

    def test_empty_content_stats(self):
        stats = get_tier_statistics("")
        assert stats["entry_count"] == 0
        assert stats["content_size"] == 0
        assert stats["line_count"] == 1  # Empty string still has one line
        assert stats["date_range"]["earliest"] is None
        assert stats["date_range"]["latest"] is None

    def test_content_with_entries(self):
        content = """# Guardrails

## 2024-01-15T10:00:00Z Iteration 1
- Entry one

## 2024-01-16T11:00:00Z Iteration 2
- Entry two
- More content here
"""
        stats = get_tier_statistics(content)
        assert stats["entry_count"] == 2
        assert stats["line_count"] == 9
        assert stats["content_size"] > 0
        assert stats["date_range"]["earliest"] is not None
        assert stats["date_range"]["latest"] is not None

    def test_date_range_extraction(self):
        content = """# Guardrails

## 2024-01-15T10:00:00Z Iteration 1
- Entry one

## 2024-01-20T15:30:00Z Iteration 2
- Entry two
"""
        stats = get_tier_statistics(content)
        earliest = stats["date_range"]["earliest"]
        latest = stats["date_range"]["latest"]
        assert earliest is not None and "2024-01-15" in earliest
        assert latest is not None and "2024-01-20" in latest


class TestReplaceSectionMode:
    """Test replace-section write mode."""

    def test_replace_section_updates_existing(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)

        # Initial content
        initial = """# Guardrails

## Section A
- Old content A

## Section B
- Content B
"""
        files.write_tier(KnowledgeTier.GUARDRAILS, initial)

        policy = WritePolicy(mode=WriteMode.REPLACE_SECTION, section_target="Section A")
        writer = TierWriter(files, policy)

        # This should replace Section A
        new_content = "## Section A\n- New content A"
        result = writer._replace_section(initial, new_content, "Section A")

        assert "New content A" in result
        assert "Old content A" not in result
        assert "Content B" in result  # Section B should remain


class TestTierWriterNoDeduplicate:
    """Test TierWriter without deduplication."""

    def test_no_deduplicate_allows_duplicates(self, tmp_path: Path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        files = FileStorage(agent_dir)
        policy = WritePolicy(deduplicate=False)
        writer = TierWriter(files, policy)

        # Write same entry twice
        writer.write_guardrails_entry(
            iteration=1,
            item_id="TEST-001",
            item_title="Test Item",
            reason="validation_failed",
        )

        writer.write_guardrails_entry(
            iteration=1,
            item_id="TEST-001",
            item_title="Test Item",
            reason="validation_failed",
        )

        content = files.read_tier(KnowledgeTier.GUARDRAILS)
        # Should have 2 entries since deduplication is disabled
        assert content.count("Iteration 1 (TEST-001)") == 2
