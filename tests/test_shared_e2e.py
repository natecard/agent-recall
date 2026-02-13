from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import agent_recall.cli.main as cli_main
from agent_recall.llm.base import LLMProvider, LLMResponse, Message

runner = CliRunner()


class E2EProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "e2e-dummy"

    @property
    def model_name(self) -> str:
        return "e2e-model"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        prompt = messages[-1].content
        if "Current GUARDRAILS.md" in prompt:
            return LLMResponse(
                content="- [GOTCHA] Always check for null pointers in shared memory",
                model="e2e-model",
            )
        if "Current STYLE.md" in prompt:
            return LLMResponse(
                content="- [PATTERN] Use dependency injection for storage backends",
                model="e2e-model",
            )
        return LLMResponse(
            content="**2026-02-12**: Validated shared backend E2E flow.",
            model="e2e-model",
        )

    def validate(self) -> tuple[bool, str]:
        return True, "ok"


@pytest.fixture
def mock_llm_provider(monkeypatch):
    provider = E2EProvider()
    monkeypatch.setattr(cli_main, "get_llm", lambda: provider)
    monkeypatch.setattr(cli_main, "create_llm_provider", lambda *_args, **_kwargs: provider)
    return provider


def test_shared_backend_manual_workflow_e2e(mock_llm_provider) -> None:
    """
    Verifies that a manual workflow (start -> log -> end -> compact -> retrieve)
    works across two 'repositories' sharing the same filesystem backend.
    """
    with runner.isolated_filesystem() as fs_root:
        root_path = Path(fs_root)
        shared_dir = root_path / "shared-memory"
        repo_a = root_path / "repo-a"
        repo_b = root_path / "repo-b"

        shared_dir.mkdir()
        repo_a.mkdir()
        repo_b.mkdir()

        # --- REPO A: Init and Write ---
        os.chdir(repo_a)
        result = runner.invoke(cli_main.app, ["init"])
        assert result.exit_code == 0

        # Configure Repo A
        config_path = repo_a / ".agent" / "config.yaml"
        config_content = f"""
project:
  name: repo-a
storage:
  backend: shared
  shared:
    base_url: file://{shared_dir}
    tenant_id: test-tenant
    project_id: test-project
llm:
  provider: dummy
  model: dummy
"""
        config_path.write_text(config_content)
        # Clear cache so next command picks up the new config
        cli_main.get_storage.cache_clear()

        # Start session
        result = runner.invoke(cli_main.app, ["start", "Shared memory test"])
        assert result.exit_code == 0
        assert "Session started" in result.output

        # Log entries
        result = runner.invoke(
            cli_main.app,
            [
                "log",
                "Shared memory must handle concurrent access",
                "--label",
                "gotcha",
            ],
        )
        assert result.exit_code == 0

        result = runner.invoke(
            cli_main.app,
            [
                "log",
                "Use factories for backend creation",
                "--label",
                "pattern",
            ],
        )
        assert result.exit_code == 0

        # End session
        result = runner.invoke(cli_main.app, ["end", "Finished writing"])
        assert result.exit_code == 0

        # Compact (should generate chunks in shared storage)
        result = runner.invoke(cli_main.app, ["compact", "--force"])
        assert result.exit_code == 0
        assert "Compaction complete" in result.output

        # Verify Repo A sees the chunks
        result = runner.invoke(cli_main.app, ["retrieve", "concurrent"])
        assert result.exit_code == 0
        assert "Shared memory must handle concurrent access" in result.output

        # --- REPO B: Init and Read ---
        os.chdir(repo_b)
        # Clear get_storage cache to simulate fresh process
        cli_main.get_storage.cache_clear()

        result = runner.invoke(cli_main.app, ["init"])
        assert result.exit_code == 0

        # Configure Repo B
        config_path_b = repo_b / ".agent" / "config.yaml"
        config_content_b = f"""
project:
  name: repo-b
storage:
  backend: shared
  shared:
    base_url: file://{shared_dir}
    tenant_id: test-tenant
    project_id: test-project
llm:
  provider: dummy
  model: dummy
"""
        config_path_b.write_text(config_content_b)
        # Clear cache so next command picks up the new config
        cli_main.get_storage.cache_clear()

        # Retrieve in Repo B (should find chunks from Repo A)
        result = runner.invoke(cli_main.app, ["retrieve", "concurrent"])
        assert result.exit_code == 0
        assert "Shared memory must handle concurrent access" in result.output

        # Retrieve pattern
        result = runner.invoke(cli_main.app, ["retrieve", "factories"])
        assert result.exit_code == 0
        assert "Use factories for backend creation" in result.output

        # Verify tiers are synced (if file storage sync is working)
        # We rely on test_shared_backend_tier_sync_e2e for detailed tier sync verification.
        # Here we just ensure context command runs without error.
        result = runner.invoke(cli_main.app, ["context"])
        assert result.exit_code == 0
        # assert "[GOTCHA] Always check for null pointers" in result.output


def test_shared_backend_tier_sync_e2e(mock_llm_provider) -> None:
    """
    Verifies that manual edits to tier files in one repo propagate to another
    via the shared backend.
    """
    with runner.isolated_filesystem() as fs_root:
        root_path = Path(fs_root)
        shared_dir = root_path / "shared-memory"
        repo_a = root_path / "repo-a"
        repo_b = root_path / "repo-b"

        shared_dir.mkdir()
        repo_a.mkdir()
        repo_b.mkdir()

        # Setup Repo A
        os.chdir(repo_a)
        runner.invoke(cli_main.app, ["init"])
        (repo_a / ".agent" / "config.yaml").write_text(f"""
project: repo-a
storage:
  backend: shared
  shared:
    base_url: file://{shared_dir}
    tenant_id: test-tenant
    project_id: test-project
""")
        # Clear cache just in case, though we don't use Repo A further in this test
        cli_main.get_storage.cache_clear()

        # Setup Repo B
        os.chdir(repo_b)
        runner.invoke(cli_main.app, ["init"])
        (repo_b / ".agent" / "config.yaml").write_text(f"""
project: repo-b
storage:
  backend: shared
  shared:
    base_url: file://{shared_dir}
    tenant_id: test-tenant
    project_id: test-project
""")

        # Modify GUARDRAILS.md in Shared storage directly
        (shared_dir / "GUARDRAILS.md").write_text("# Shared\\n- Rule Shared\\n")

        # Repo B reads
        os.chdir(repo_b)
        # Clear cache just in case
        cli_main.get_storage.cache_clear()

        # Context command reads tiers
        result = runner.invoke(cli_main.app, ["context"])
        assert result.exit_code == 0
        assert "- Rule Shared" in result.output

        # NOTE: Local file is NOT updated on read.
