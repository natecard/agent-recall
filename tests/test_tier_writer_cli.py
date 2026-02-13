"""Tests for tier file CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from agent_recall.cli.main import app

runner = CliRunner()


class TestWriteGuardrailsCommand:
    """Test write-guardrails CLI command."""

    def test_write_guardrails_success(self):
        with runner.isolated_filesystem():
            # Initialize first
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0

            result = runner.invoke(
                app,
                [
                    "write-guardrails",
                    "--iteration",
                    "1",
                    "--item-id",
                    "TEST-001",
                    "--item-title",
                    "Test Item",
                    "--reason",
                    "validation_failed",
                    "--validation-hint",
                    "pytest error",
                ],
            )

            assert result.exit_code == 0
            assert "Wrote guardrails entry" in result.output

            guardrails_path = Path(".agent") / "GUARDRAILS.md"
            content = guardrails_path.read_text()
            assert "Iteration 1 (TEST-001)" in content
            assert "pytest error" in content

    def test_write_guardrails_skips_duplicate(self):
        with runner.isolated_filesystem():
            # Initialize first
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0

            # Write first entry
            runner.invoke(
                app,
                [
                    "write-guardrails",
                    "--iteration",
                    "1",
                    "--item-id",
                    "TEST-001",
                    "--item-title",
                    "Test Item",
                ],
            )

            # Try to write duplicate
            result = runner.invoke(
                app,
                [
                    "write-guardrails",
                    "--iteration",
                    "1",
                    "--item-id",
                    "TEST-001",
                    "--item-title",
                    "Test Item",
                ],
            )

            assert result.exit_code == 0
            assert "Skipped duplicate" in result.output


class TestWriteGuardrailsFailureCommand:
    """Test write-guardrails-failure CLI command."""

    def test_write_hard_failure(self):
        with runner.isolated_filesystem():
            # Initialize first
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0

            result = runner.invoke(
                app,
                [
                    "write-guardrails-failure",
                    "--iteration",
                    "1",
                    "--item-id",
                    "TEST-001",
                    "--item-title",
                    "Test Item",
                    "--error",
                    "Error 1",
                    "--error",
                    "Error 2",
                    "--validation-hint",
                    "pytest failed",
                ],
            )

            assert result.exit_code == 0
            assert "Wrote hard failure" in result.output

            guardrails_path = Path(".agent") / "GUARDRAILS.md"
            content = guardrails_path.read_text()
            assert "HARD FAILURE" in content
            assert "Error 1" in content
            assert "Error 2" in content


class TestWriteStyleCommand:
    """Test write-style CLI command."""

    def test_write_style_success(self):
        with runner.isolated_filesystem():
            # Initialize first
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0

            result = runner.invoke(
                app,
                [
                    "write-style",
                    "--iteration",
                    "1",
                    "--item-id",
                    "TEST-001",
                    "--validation-hint",
                    "Start at line 5",
                ],
            )

            assert result.exit_code == 0
            assert "Wrote style entry" in result.output

            style_path = Path(".agent") / "STYLE.md"
            content = style_path.read_text()
            assert "Iteration 1 (TEST-001)" in content
            assert "Start at line 5" in content


class TestWriteRecentCommand:
    """Test write-recent CLI command."""

    def test_write_recent_success(self):
        with runner.isolated_filesystem():
            # Initialize first
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0

            result = runner.invoke(
                app,
                [
                    "write-recent",
                    "--iteration",
                    "1",
                    "--item-id",
                    "TEST-001",
                    "--item-title",
                    "Test Item",
                    "--mode",
                    "feature",
                    "--agent-exit",
                    "0",
                    "--validate-status",
                    "passed",
                    "--outcome",
                    "progressed",
                ],
            )

            assert result.exit_code == 0
            assert "Wrote recent entry" in result.output

            recent_path = Path(".agent") / "RECENT.md"
            content = recent_path.read_text()
            assert "Iteration 1" in content
            assert "TEST-001 - Test Item" in content
            assert "Mode: feature" in content


class TestLintTiersCommand:
    """Test lint-tiers CLI command."""

    def test_lint_all_valid(self):
        with runner.isolated_filesystem():
            # Initialize first
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0

            # Create valid tier files
            guardrails_path = Path(".agent") / "GUARDRAILS.md"
            guardrails_path.write_text(
                "# Guardrails\n\n## 2024-01-15T10:00:00Z Iteration 1\n- Valid entry\n"
            )
            style_path = Path(".agent") / "STYLE.md"
            style_path.write_text("# Style\n\n## 2024-01-15T10:00:00Z Iteration 1\n- Valid entry\n")
            recent_path = Path(".agent") / "RECENT.md"
            recent_path.write_text(
                "# Recent\n\n## 2024-01-15T10:00:00Z Iteration 1\n- Valid entry\n"
            )

            result = runner.invoke(app, ["lint-tiers"])

            assert result.exit_code == 0
            assert "All tier files passed" in result.output

    def test_lint_with_errors(self):
        with runner.isolated_filesystem():
            # Initialize first
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0

            # Create invalid tier file (missing header)
            guardrails_path = Path(".agent") / "GUARDRAILS.md"
            guardrails_path.write_text("## Entry\n- Content\n")

            result = runner.invoke(
                app,
                ["lint-tiers", "--tier", "guardrails"],
            )

            assert result.exit_code == 1
            assert "Some tier files have issues" in result.output

    def test_lint_single_tier(self):
        with runner.isolated_filesystem():
            # Initialize first
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0

            guardrails_path = Path(".agent") / "GUARDRAILS.md"
            guardrails_path.write_text(
                "# Guardrails\n\n## 2024-01-15T10:00:00Z Iteration 1\n- Valid entry\n"
            )

            result = runner.invoke(
                app,
                ["lint-tiers", "--tier", "guardrails"],
            )

            assert result.exit_code == 0
            assert "GUARDRAILS" in result.output


class TestTierStatsCommand:
    """Test tier-stats CLI command."""

    def test_tier_stats_output(self):
        with runner.isolated_filesystem():
            # Initialize first
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0

            # Create tier files with content
            guardrails_path = Path(".agent") / "GUARDRAILS.md"
            guardrails_path.write_text(
                "# Guardrails\n\n## 2024-01-15T10:00:00Z Iteration 1\n- Entry\n"
            )
            style_path = Path(".agent") / "STYLE.md"
            style_path.write_text("# Style\n\n## 2024-01-15T10:00:00Z Iteration 1\n- Entry\n")
            recent_path = Path(".agent") / "RECENT.md"
            recent_path.write_text("# Recent\n\n## 2024-01-15T10:00:00Z Iteration 1\n- Entry\n")

            result = runner.invoke(app, ["tier-stats"])

            assert result.exit_code == 0
            assert "GUARDRAILS" in result.output
            assert "STYLE" in result.output
            assert "RECENT" in result.output
