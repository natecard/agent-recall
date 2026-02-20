from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

import agent_recall.cli.main as cli_main
from agent_recall.cli.ralph import build_agent_cmd_from_ralph_config
from agent_recall.ralph.costs import summarize_costs
from agent_recall.ralph.notifications import build_notification_content
from agent_recall.storage.models import RalphNotificationEvent

runner = CliRunner()


def test_cli_ralph_enable_creates_state_file() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "enable"])
        assert result.exit_code == 0
        state_path = Path(".agent") / "ralph" / "ralph_state.json"
        assert state_path.exists()
        payload = json.loads(state_path.read_text())
        assert payload["status"] == "ENABLED"


def test_cli_ralph_enable_with_prd_counts_items() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        prd_path = Path("prd.json")
        prd_path.write_text(
            json.dumps(
                {
                    "items": [
                        {"id": "AR-1", "passes": False},
                        {"id": "AR-2", "passes": False},
                    ]
                }
            )
        )
        result = runner.invoke(cli_main.app, ["ralph", "enable", "--prd", "prd.json"])
        assert result.exit_code == 0
        assert "2 PRD items" in result.output


def test_cli_ralph_enable_missing_prd_errors() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "enable", "--prd", "missing.json"])
        assert result.exit_code == 1
        assert "PRD file not found" in result.output


def test_cli_ralph_disable_updates_state() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["ralph", "enable"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "disable"])
        assert result.exit_code == 0
        state_path = Path(".agent") / "ralph" / "ralph_state.json"
        payload = json.loads(state_path.read_text())
        assert payload["status"] == "DISABLED"


def test_cli_ralph_set_agent_stores_in_config() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            ["ralph", "set-agent", "--cli", "claude-code", "--model", "claude-opus-4-6"],
        )
        assert result.exit_code == 0
        import yaml

        config_path = Path(".agent") / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        assert config["ralph"]["coding_cli"] == "claude-code"
        assert config["ralph"]["cli_model"] == "claude-opus-4-6"


def test_cli_ralph_set_agent_rejects_invalid() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "set-agent", "--cli", "invalid-tool"])
        assert result.exit_code == 1
        assert "Invalid coding CLI" in result.output


def test_build_agent_cmd_from_ralph_config_with_model() -> None:
    cmd = build_agent_cmd_from_ralph_config({"coding_cli": "codex", "cli_model": "gpt-5.3-codex"})
    assert cmd == "codex exec --model gpt-5.3-codex -"


def test_build_agent_cmd_from_ralph_config_codex_without_model() -> None:
    cmd = build_agent_cmd_from_ralph_config({"coding_cli": "codex"})
    assert cmd == "codex exec -"


def test_build_agent_cmd_from_ralph_config_opencode_with_model() -> None:
    cmd = build_agent_cmd_from_ralph_config(
        {"coding_cli": "opencode", "cli_model": "github-copilot/gpt-5.3-codex"}
    )
    assert cmd == 'opencode run -m github-copilot/gpt-5.3-codex "$(cat {prompt_file})"'


def test_build_agent_cmd_from_ralph_config_opencode_without_model() -> None:
    cmd = build_agent_cmd_from_ralph_config({"coding_cli": "opencode"})
    assert cmd == 'opencode run "$(cat {prompt_file})"'


def test_opencode_plugin_cli_install_and_uninstall(tmp_path: Path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    result = runner.invoke(
        cli_main.app,
        [
            "ralph",
            "plugin",
            "opencode-install",
            "--project-dir",
            str(project_dir),
        ],
    )
    assert result.exit_code == 0
    plugin_path = project_dir / ".opencode" / "plugins" / "agent-recall-ralph.js"
    assert plugin_path.exists()

    result = runner.invoke(
        cli_main.app,
        [
            "ralph",
            "plugin",
            "opencode-uninstall",
            "--project-dir",
            str(project_dir),
        ],
    )
    assert result.exit_code == 0
    assert not plugin_path.exists()


def test_build_agent_cmd_from_ralph_config_missing_cli_returns_none() -> None:
    cmd = build_agent_cmd_from_ralph_config({"cli_model": "gpt-5.3-codex"})
    assert cmd is None


def test_build_notification_content_iteration_complete() -> None:
    info = build_notification_content(
        RalphNotificationEvent.ITERATION_COMPLETE,
        iteration=2,
    )
    assert "Iteration 2" in info.message


def test_cost_report_aggregates_iteration_tokens() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        agent_dir = Path(".agent")
        ralph_dir = agent_dir / "ralph"
        ralph_dir.mkdir(parents=True, exist_ok=True)
        report_path = ralph_dir / "iterations" / "001.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "iteration": 1,
                    "item_id": "AR-705",
                    "item_title": "Token & Cost Tracking Dashboard",
                    "token_usage": {"prompt_tokens": 1000, "completion_tokens": 500},
                    "token_model": "gpt-4o",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        store = cli_main.IterationReportStore(agent_dir / "ralph")
        summary = summarize_costs(store.load_all())
        assert summary.total_tokens == 1500


def test_ralph_cost_report_json_output() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        agent_dir = Path(".agent")
        ralph_dir = agent_dir / "ralph" / "iterations"
        ralph_dir.mkdir(parents=True, exist_ok=True)
        (ralph_dir / "001.json").write_text(
            json.dumps(
                {
                    "iteration": 1,
                    "item_id": "AR-705",
                    "item_title": "Token & Cost Tracking Dashboard",
                    "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "token_model": "gpt-4o",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        result = runner.invoke(cli_main.app, ["ralph", "cost-report", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["total_tokens"] == 15


def test_ralph_set_budget_updates_config() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "set-budget", "--cost-usd", "12.5"])
        assert result.exit_code == 0
        config_path = Path(".agent") / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["ralph"]["cost_budget_usd"] == 12.5


def test_ralph_set_budget_rejects_negative() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "set-budget", "--cost-usd", "-1"])
        assert result.exit_code == 1
        assert "Cost budget must be" in result.output


def test_ralph_run_loop_emits_output_line_when_no_cli() -> None:
    """When no coding_cli is set, run_loop emits an output_line event."""
    import asyncio

    from agent_recall.ralph.loop import RalphLoop
    from agent_recall.storage.files import FileStorage
    from agent_recall.storage.sqlite import SQLiteStorage

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        prd_path = Path(".agent") / "ralph" / "prd.json"
        prd_path.parent.mkdir(parents=True, exist_ok=True)
        prd_path.write_text(
            json.dumps({"items": [{"id": "T-1", "title": "Test", "passes": False}]})
        )
        agent_dir = Path(".agent")
        storage = SQLiteStorage(agent_dir / "state.db")
        files = FileStorage(agent_dir)
        loop = RalphLoop(agent_dir, storage, files)

        # Enable the loop first.
        loop.enable()

        events: list[dict] = []

        def cb(event: dict) -> None:
            events.append(event)

        summary = asyncio.run(
            loop.run_loop(
                progress_callback=cb,
                coding_cli=None,
                cli_model=None,
            )
        )
        assert summary["total_iterations"] == 1
        output_events = [e for e in events if e.get("event") == "output_line"]
        assert len(output_events) >= 1
        assert "No coding CLI configured" in output_events[0]["line"]


def test_ralph_run_loop_passes_coding_cli_params() -> None:
    """run_loop accepts coding_cli and cli_model without error."""
    import asyncio

    from agent_recall.ralph.loop import RalphLoop
    from agent_recall.storage.files import FileStorage
    from agent_recall.storage.sqlite import SQLiteStorage

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        prd_path = Path(".agent") / "ralph" / "prd.json"
        prd_path.parent.mkdir(parents=True, exist_ok=True)
        prd_path.write_text(
            json.dumps({"items": [{"id": "T-2", "title": "Test 2", "passes": False}]})
        )
        agent_dir = Path(".agent")
        storage = SQLiteStorage(agent_dir / "state.db")
        files = FileStorage(agent_dir)
        loop = RalphLoop(agent_dir, storage, files)
        loop.enable()

        events: list[dict] = []

        def cb(event: dict) -> None:
            events.append(event)

        # Use a non-existent binary — should emit output_line with error.
        _summary = asyncio.run(
            loop.run_loop(
                progress_callback=cb,
                coding_cli="claude-code",
                cli_model="claude-sonnet-4-20250514",
            )
        )
        # The agent should have "failed" since the binary isn't on the test PATH.
        output_events = [e for e in events if e.get("event") == "output_line"]
        assert len(output_events) >= 1
        agent_events = [e for e in events if e.get("event") == "agent_complete"]
        assert len(agent_events) == 1


def test_ralph_run_loop_uses_codex_exec(monkeypatch) -> None:
    """_run_agent_subprocess uses codex exec (not --print) for Codex CLI."""
    import asyncio

    from agent_recall.ralph.loop import RalphLoop
    from agent_recall.storage.files import FileStorage
    from agent_recall.storage.sqlite import SQLiteStorage

    class _FakeStdout:
        def __init__(self) -> None:
            self._lines = [b"ok\n", b""]

        async def readline(self) -> bytes:
            return self._lines.pop(0)

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = _FakeStdout()

        async def wait(self) -> int:
            return 0

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        agent_dir = Path(".agent")
        storage = SQLiteStorage(agent_dir / "state.db")
        files = FileStorage(agent_dir)
        loop = RalphLoop(agent_dir, storage, files)

        recorded: list[str] = []
        events: list[dict] = []

        async def _fake_create_subprocess_exec(*cmd, **_kwargs):  # noqa: ANN002, ANN003
            recorded.extend(str(part) for part in cmd)
            return _FakeProc()

        monkeypatch.setattr("agent_recall.ralph.loop.shutil.which", lambda _bin: "/usr/bin/codex")
        monkeypatch.setattr(
            "agent_recall.ralph.loop.asyncio.create_subprocess_exec",
            _fake_create_subprocess_exec,
        )

        exit_code = asyncio.run(
            loop._run_agent_subprocess(  # noqa: SLF001
                coding_cli="codex",
                cli_model="gpt-5.3-codex",
                item_title="Test 3",
                iteration=1,
                item_id="T-3",
                progress_callback=events.append,
            )
        )

        assert exit_code == 0
        assert recorded == [
            "/usr/bin/codex",
            "exec",
            "--model",
            "gpt-5.3-codex",
            "Work on PRD item T-3: Test 3",
        ]
        output_lines = [
            str(event.get("line") or "") for event in events if event.get("event") == "output_line"
        ]
        assert any(
            "codex exec --model gpt-5.3-codex Work on PRD item T-3: Test 3" in line
            for line in output_lines
        )


def test_ralph_run_shell_mode_streams_via_shared_pipeline(monkeypatch) -> None:
    with runner.isolated_filesystem():
        script_path = Path("ralph-agent-recall-loop.sh")
        script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        monkeypatch.setattr("agent_recall.cli.ralph.get_default_script_path", lambda: script_path)
        monkeypatch.setenv("AGENT_RECALL_RALPH_STREAM_DEBUG", "0")

        prd_path = Path("agent_recall") / "ralph" / "prd.json"
        prd_path.parent.mkdir(parents=True, exist_ok=True)
        prd_path.write_text(
            '{"items":[{"id":"AR-1","title":"Test","passes":false}]}',
            encoding="utf-8",
        )

        captured_cmd: list[str] = []

        def _fake_stream_runner(cmd, **kwargs):  # noqa: ANN001
            captured_cmd[:] = list(cmd)
            kwargs["on_emit"]("stream fragment 1\n")
            kwargs["on_emit"]("stream fragment 2\n")
            return 0

        monkeypatch.setattr("agent_recall.cli.ralph.run_streaming_command", _fake_stream_runner)

        result = runner.invoke(
            cli_main.app,
            [
                "ralph",
                "run",
                "--agent-cmd",
                "echo test",
                "--max-iterations",
                "1",
                "--sleep-seconds",
                "0",
            ],
        )

        assert result.exit_code == 0
        assert captured_cmd
        assert captured_cmd[0] == str(script_path)
        assert "--agent-cmd" in captured_cmd
        assert "--agent-transport" in captured_cmd
        assert captured_cmd[captured_cmd.index("--agent-transport") + 1] == "pipe"
        assert "stream fragment 1" in result.output
        assert "stream fragment 2" in result.output
        assert "Ralph loop completed successfully." in result.output


def test_ralph_run_shell_mode_passes_agent_transport_override(monkeypatch) -> None:
    with runner.isolated_filesystem():
        script_path = Path("ralph-agent-recall-loop.sh")
        script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        monkeypatch.setattr("agent_recall.cli.ralph.get_default_script_path", lambda: script_path)
        monkeypatch.setenv("AGENT_RECALL_RALPH_STREAM_DEBUG", "0")

        prd_path = Path("agent_recall") / "ralph" / "prd.json"
        prd_path.parent.mkdir(parents=True, exist_ok=True)
        prd_path.write_text(
            '{"items":[{"id":"AR-1","title":"Test","passes":false}]}',
            encoding="utf-8",
        )

        captured_cmd: list[str] = []

        def _fake_stream_runner(cmd, **kwargs):  # noqa: ANN001
            captured_cmd[:] = list(cmd)
            return 0

        monkeypatch.setattr("agent_recall.cli.ralph.run_streaming_command", _fake_stream_runner)

        result = runner.invoke(
            cli_main.app,
            [
                "ralph",
                "run",
                "--agent-cmd",
                "echo test",
                "--agent-transport",
                "auto",
                "--max-iterations",
                "1",
                "--sleep-seconds",
                "0",
            ],
        )

        assert result.exit_code == 0
        assert captured_cmd
        assert "--agent-transport" in captured_cmd
        assert captured_cmd[captured_cmd.index("--agent-transport") + 1] == "auto"
