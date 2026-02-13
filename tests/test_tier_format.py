"""Tests for tier_format module: merge_tier_content, split_tier_by_format, round-trip."""

from __future__ import annotations

from agent_recall.core.tier_format import (
    EntryFormat,
    ParsedEntry,
    TierContent,
    merge_tier_content,
    parse_tier_content,
    split_tier_by_format,
)


class TestMergeTierContent:
    """Test merge_tier_content() output format and behavior."""

    def test_empty_tier_content_returns_single_newline(self):
        """merge_tier_content() with empty TierContent returns '\\n'."""
        content = TierContent()
        result = merge_tier_content(content)
        assert result == "\n"

    def test_only_preamble_returns_preamble_plus_newline(self):
        """merge_tier_content() with only preamble returns preamble text + '\\n'."""
        content = TierContent(preamble=["# Guardrails", "", "Rules and warnings."])
        result = merge_tier_content(content)
        assert result == "# Guardrails\n\nRules and warnings.\n"

    def test_only_bullet_entries_returns_bullet_lines_plus_newline(self):
        """merge_tier_content() with only bullet entries returns bullet lines + '\\n'."""
        content = TierContent(
            bullet_entries=[
                ParsedEntry(
                    EntryFormat.BULLET,
                    "- [GOTCHA] Avoid weak passwords",
                    kind="GOTCHA",
                    text="Avoid weak passwords",
                ),
                ParsedEntry(
                    EntryFormat.BULLET,
                    "- [PATTERN] Use DTOs",
                    kind="PATTERN",
                    text="Use DTOs",
                ),
            ]
        )
        result = merge_tier_content(content)
        assert result == "- [GOTCHA] Avoid weak passwords\n- [PATTERN] Use DTOs\n"

    def test_only_ralph_entries_returns_ralph_blocks_plus_newline(self):
        """merge_tier_content() with only Ralph entries returns Ralph blocks + '\\n'."""
        content = TierContent(
            ralph_entries=[
                ParsedEntry(
                    EntryFormat.RALPH,
                    "## 2026-02-12T10:00:00Z Iteration 1 (AR-001)\n- Error detail",
                    timestamp="2026-02-12T10:00:00Z",
                    iteration=1,
                    item_id="AR-001",
                ),
            ]
        )
        result = merge_tier_content(content)
        assert result == "## 2026-02-12T10:00:00Z Iteration 1 (AR-001)\n- Error detail\n"

    def test_preamble_first_then_bullets_then_ralph(self):
        """Output order: preamble first, then bullet entries, then Ralph entries."""
        content = TierContent(
            preamble=["# Guardrails"],
            bullet_entries=[
                ParsedEntry(EntryFormat.BULLET, "- [GOTCHA] Test", kind="GOTCHA", text="Test"),
            ],
            ralph_entries=[
                ParsedEntry(
                    EntryFormat.RALPH,
                    "## 2026-02-12T10:00:00Z Iteration 1 (AR-001)\n- Detail",
                    timestamp="2026-02-12T10:00:00Z",
                    iteration=1,
                    item_id="AR-001",
                ),
            ],
        )
        result = merge_tier_content(content)
        assert result.startswith("# Guardrails")
        assert "- [GOTCHA] Test" in result
        assert "## 2026-02-12T10:00:00Z Iteration 1 (AR-001)" in result
        # Order: preamble, bullets, ralph
        preamble_end = result.find("# Guardrails") + len("# Guardrails")
        bullet_pos = result.find("- [GOTCHA] Test")
        ralph_pos = result.find("## 2026-02-12T10:00:00Z")
        assert preamble_end < bullet_pos < ralph_pos

    def test_sections_separated_by_double_newline(self):
        """merge_tier_content() separates sections with double-newline ('\\n\\n')."""
        content = TierContent(
            preamble=["# Header"],
            bullet_entries=[
                ParsedEntry(EntryFormat.BULLET, "- [X] Y", kind="X", text="Y"),
            ],
        )
        result = merge_tier_content(content)
        assert "\n\n" in result

    def test_appends_trailing_newline(self):
        """merge_tier_content() appends a trailing newline to the output."""
        content = TierContent(preamble=["Hello"])
        result = merge_tier_content(content)
        assert result.endswith("\n")

    def test_accepts_preserve_order_parameter(self):
        """merge_tier_content() accepts preserve_order parameter without error."""
        content = TierContent()
        result = merge_tier_content(content, preserve_order=True)
        assert result == "\n"
        result = merge_tier_content(content, preserve_order=False)
        assert result == "\n"


class TestSplitTierByFormat:
    """Test split_tier_by_format() backward-compatible API."""

    def test_returns_three_tuple(self):
        """split_tier_by_format() returns a 3-tuple of (list[str], list[str], list[str])."""
        content = "# Guardrails\n\n- [GOTCHA] Test"
        result = split_tier_by_format(content)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
        assert isinstance(result[2], list)
        assert all(isinstance(x, str) for x in result[0])
        assert all(isinstance(x, str) for x in result[1])
        assert all(isinstance(x, str) for x in result[2])

    def test_preamble_matches_parse_tier_content(self):
        """split_tier_by_format() preamble list matches parse_tier_content().preamble."""
        content = "# Guardrails\n\nRules here.\n\n- [GOTCHA] Test"
        parsed = parse_tier_content(content)
        preamble_lines, _, _ = split_tier_by_format(content)
        assert preamble_lines == parsed.preamble

    def test_bullet_list_contains_raw_content(self):
        """split_tier_by_format() bullet list has raw_content from bullet ParsedEntries."""
        content = "- [GOTCHA] One\n- [PATTERN] Two"
        parsed = parse_tier_content(content)
        _, bullet_strings, _ = split_tier_by_format(content)
        expected = [e.raw_content for e in parsed.bullet_entries]
        assert bullet_strings == expected

    def test_ralph_list_contains_raw_content(self):
        """split_tier_by_format() ralph list has raw_content from ralph ParsedEntries."""
        content = """## 2026-02-12T10:00:00Z Iteration 1 (AR-001)
- Error detail

## 2026-02-12T11:00:00Z Iteration 2 (AR-002)
- Another"""
        parsed = parse_tier_content(content)
        _, _, ralph_strings = split_tier_by_format(content)
        expected = [e.raw_content for e in parsed.ralph_entries]
        assert ralph_strings == expected


class TestRoundTrip:
    """Test parse → merge round-trip preserves all entries."""

    def _round_trip(self, content: str) -> str:
        """Parse then merge; return merged string."""
        tier_content = parse_tier_content(content)
        return merge_tier_content(tier_content)

    def test_preserves_bullet_count(self):
        """Round-trip: parsing then merging a mixed-format file preserves all bullet entry count."""
        content = """# Guardrails

- [GOTCHA] One
- [PATTERN] Two
- [GOTCHA] Three

## 2026-02-12T10:00:00Z Iteration 1 (AR-001)
- Ralph detail"""
        parsed_before = parse_tier_content(content)
        merged = self._round_trip(content)
        parsed_after = parse_tier_content(merged)
        assert len(parsed_after.bullet_entries) == len(parsed_before.bullet_entries)

    def test_preserves_ralph_count(self):
        """Round-trip: parsing then merging a mixed-format file preserves all Ralph entry count."""
        content = """# Guardrails

## 2026-02-12T10:00:00Z Iteration 1 (AR-001)
- First

## 2026-02-12T11:00:00Z Iteration 2 (AR-002)
- Second"""
        parsed_before = parse_tier_content(content)
        merged = self._round_trip(content)
        parsed_after = parse_tier_content(merged)
        assert len(parsed_after.ralph_entries) == len(parsed_before.ralph_entries)

    def test_preserves_preamble_content(self):
        """Round-trip: parsing then merging preserves preamble content."""
        content = """# Guardrails

Rules and warnings learned during development.

- [GOTCHA] Test"""
        merged = self._round_trip(content)
        parsed = parse_tier_content(merged)
        assert "# Guardrails" in "\n".join(parsed.preamble)
        assert "Rules and warnings" in "\n".join(parsed.preamble)

    def test_no_entries_duplicated_or_lost(self):
        """Round-trip: no entries are duplicated or lost during round-trip."""
        content = """# Header

- [GOTCHA] A
- [PATTERN] B

## 2026-02-12T10:00:00Z Iteration 1 (AR-001)
- Ralph A

## 2026-02-12T11:00:00Z Iteration 2 (AR-002)
- Ralph B"""
        parsed_before = parse_tier_content(content)
        merged = self._round_trip(content)
        parsed_after = parse_tier_content(merged)

        assert len(parsed_after.bullet_entries) == len(parsed_before.bullet_entries)
        assert len(parsed_after.ralph_entries) == len(parsed_before.ralph_entries)

        bullet_texts_before = {e.raw_content.rstrip() for e in parsed_before.bullet_entries}
        bullet_texts_after = {e.raw_content.rstrip() for e in parsed_after.bullet_entries}
        assert bullet_texts_after == bullet_texts_before

        ralph_texts_before = {e.raw_content.rstrip() for e in parsed_before.ralph_entries}
        ralph_texts_after = {e.raw_content.rstrip() for e in parsed_after.ralph_entries}
        assert ralph_texts_after == ralph_texts_before
