from __future__ import annotations

import io

import yaml

from agent_recall.cli.textual_tui import (
    AgentRecallTextualApp,
    _build_command_suggestions,
    _clean_optional_text,
    _is_knowledge_run_command,
    _is_palette_cli_command_redundant,
    get_palette_actions,
)


def test_build_command_suggestions_excludes_slash_variants() -> None:
    suggestions = _build_command_suggestions(["status", "config setup", "sync"])

    assert "status" in suggestions
    assert "config setup" in suggestions
    assert "sync" in suggestions
    assert "settings" in suggestions
    assert "config settings" in suggestions
    assert "config model" in suggestions
    assert "ralph enable" in suggestions
    assert "ralph disable" in suggestions
    assert "ralph status" in suggestions


def test_palette_contains_ralph_loop_controls() -> None:
    action_ids = {action.action_id for action in get_palette_actions()}
    assert "ralph-enable" in action_ids
    assert "ralph-disable" in action_ids
    assert "ralph-status" in action_ids


def test_palette_contains_ralph_config_action() -> None:
    action_ids = {action.action_id for action in get_palette_actions()}
    assert "ralph-config" in action_ids


def test_palette_contains_ralph_run_action() -> None:
    action_ids = {action.action_id for action in get_palette_actions()}
    assert "ralph-run" in action_ids


def test_palette_contains_terminal_toggle() -> None:
    action_ids = {action.action_id for action in get_palette_actions()}
    assert "ralph-terminal" in action_ids


def test_clean_optional_text_handles_none_variants() -> None:
    assert _clean_optional_text(None) == ""
    assert _clean_optional_text("None") == ""
    assert _clean_optional_text("null") == ""
    assert _clean_optional_text(" value ") == "value"


def test_palette_cli_command_redundancy_filter() -> None:
    assert _is_palette_cli_command_redundant("open") is True
    assert _is_palette_cli_command_redundant("status") is True
    assert _is_palette_cli_command_redundant("run") is True
    assert _is_palette_cli_command_redundant("sources") is True
    assert _is_palette_cli_command_redundant("sessions") is True
    assert _is_palette_cli_command_redundant("theme list") is True
    assert _is_palette_cli_command_redundant("theme show") is False
    assert _is_palette_cli_command_redundant("config model") is True
    assert _is_palette_cli_command_redundant("providers") is False


def test_is_knowledge_run_command() -> None:
    assert _is_knowledge_run_command("run")
    assert _is_knowledge_run_command("compact")
    assert _is_knowledge_run_command("sync")
    assert _is_knowledge_run_command("sync --source cursor")
    assert not _is_knowledge_run_command("sync --no-compact")
    assert not _is_knowledge_run_command("status")


def test_tui_ralph_run_streams_shell_loop_with_configured_agent_cmd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent_dir = tmp_path / ".agent"
    (agent_dir / "ralph").mkdir(parents=True)

    config = {
        "ralph": {
            "enabled": True,
            "coding_cli": "codex",
            "cli_model": "gpt-5.3-codex",
            "max_iterations": 3,
            "sleep_seconds": 0,
            "compact_mode": "off",
            "selected_prd_ids": ["AR-1", "AR-2"],
        }
    }
    (agent_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (agent_dir / "ralph" / "prd.json").write_text(
        '{"items":[{"id":"AR-1","title":"One","passes":false}]}', encoding="utf-8"
    )
    fake_script = tmp_path / "ralph-agent-recall-loop.sh"
    fake_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    def theme_defaults() -> tuple[list[str], str]:
        themes = ["dark+"]
        return themes, "dark+"

    def discover_models(*_args: object, **_kwargs: object) -> tuple[list[str], str | None]:
        models = ["gpt-test"]
        return models, None

    app = AgentRecallTextualApp(
        render_dashboard=lambda *_args, **_kwargs: "",
        execute_command=lambda *_args, **_kwargs: (False, []),
        list_sessions_for_picker=lambda *_args, **_kwargs: [],
        run_setup_payload=lambda *_args, **_kwargs: (False, []),
        run_model_config=lambda *_args, **_kwargs: [],
        theme_defaults_provider=theme_defaults,
        theme_runtime_provider=None,
        theme_resolve_provider=None,
        model_defaults_provider=lambda: {},
        setup_defaults_provider=lambda: {},
        discover_models=discover_models,
        providers=[],
        list_prd_items_for_picker=None,
        cli_commands=[],
    )

    captured_activity: list[str] = []
    captured_cmd: list[str] = []
    worker_result: dict[str, object] = {}

    class _FakePopen:
        def __init__(self, cmd, **_kwargs) -> None:  # noqa: ANN001
            captured_cmd[:] = list(cmd)
            self.stdout = io.StringIO("cli output line 1\ncli output line 2\n")
            self.returncode = 0

        def wait(self) -> int:
            return int(self.returncode)

    monkeypatch.setattr(
        "agent_recall.cli.ralph.get_default_script_path",
        lambda: fake_script,
    )
    monkeypatch.setattr("agent_recall.cli.textual_tui.subprocess.Popen", _FakePopen)
    monkeypatch.setattr(app, "_append_activity", lambda line: captured_activity.append(line))
    monkeypatch.setattr(app, "call_from_thread", lambda fn, *args: fn(*args))

    def _run_worker_inline(fn, **_kwargs):  # noqa: ANN001
        worker_result.update(fn())

        class _Worker:
            pass

        return _Worker()

    monkeypatch.setattr(app, "run_worker", _run_worker_inline)

    app.action_run_ralph_loop()

    assert captured_cmd
    assert captured_cmd[0] == str(fake_script)
    assert "--agent-cmd" in captured_cmd
    agent_cmd = captured_cmd[captured_cmd.index("--agent-cmd") + 1]
    assert agent_cmd == "codex --print --model gpt-5.3-codex"
    assert "--max-iterations" in captured_cmd
    assert captured_cmd[captured_cmd.index("--max-iterations") + 1] == "3"
    assert "--compact-mode" in captured_cmd
    assert captured_cmd[captured_cmd.index("--compact-mode") + 1] == "off"
    assert "--prd-ids" in captured_cmd
    assert captured_cmd[captured_cmd.index("--prd-ids") + 1] == "AR-1,AR-2"
    assert "cli output line 1" in captured_activity
    assert "cli output line 2" in captured_activity
    assert worker_result["exit_code"] == 0
