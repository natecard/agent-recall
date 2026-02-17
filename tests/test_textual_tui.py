from __future__ import annotations

from typing import Any, cast

import yaml

from agent_recall.cli.tui import (
    AgentRecallTextualApp,
    _build_command_suggestions,
    _clean_optional_text,
    _is_knowledge_run_command,
    _is_palette_cli_command_redundant,
    _sanitize_activity_fragment,
    get_palette_actions,
)


def _build_test_app() -> AgentRecallTextualApp:
    def theme_defaults() -> tuple[list[str], str]:
        themes = ["dark+"]
        return themes, "dark+"

    def discover_models(*_args: object, **_kwargs: object) -> tuple[list[str], str | None]:
        models = ["gpt-test"]
        return models, None

    return AgentRecallTextualApp(
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


def test_palette_contains_timeline_view() -> None:
    action_ids = {action.action_id for action in get_palette_actions()}
    assert "view:timeline" in action_ids


def test_clean_optional_text_handles_none_variants() -> None:
    assert _clean_optional_text(None) == ""
    assert _clean_optional_text("None") == ""
    assert _clean_optional_text("null") == ""
    assert _clean_optional_text(" value ") == "value"


def test_sanitize_activity_fragment_strips_terminal_control_sequences() -> None:
    raw = "\x1b[7mhello\x1b[0m\rabc\b!\x1b]8;;https://example.com\x07link\x1b]8;;\x07"
    cleaned = _sanitize_activity_fragment(raw)
    assert cleaned == "hello\nab!link"
    assert "\x1b" not in cleaned
    assert "\b" not in cleaned


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

    app = _build_test_app()

    captured_activity: list[str] = []
    captured_cmd: list[str] = []
    worker_result: dict[str, object] = {}

    monkeypatch.setattr(
        "agent_recall.cli.ralph.get_default_script_path",
        lambda: fake_script,
    )
    monkeypatch.setenv("AGENT_RECALL_RALPH_STREAM_DEBUG", "0")
    monkeypatch.setattr(
        "agent_recall.cli.tui.logic.ralph_mixin.run_streaming_command",
        lambda cmd, **kwargs: (
            captured_cmd.__setitem__(slice(None), list(cmd)),
            kwargs["on_emit"]("cli output line 1\n"),
            kwargs["on_emit"]("cli output line 2\n"),
            0,
        )[-1],
    )
    monkeypatch.setattr(app, "_append_activity", lambda line: captured_activity.append(line))
    monkeypatch.setattr(app, "call_from_thread", lambda fn, *args: fn(*args))
    dummy_widget = type("DummyWidget", (), {"display": False})()
    monkeypatch.setattr(app, "query_one", lambda *_args, **_kwargs: dummy_widget)

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
    assert any("cli output line 1" in line for line in captured_activity)
    assert any("cli output line 2" in line for line in captured_activity)
    assert worker_result["exit_code"] == 0


def test_activity_scroll_keys_work_without_active_worker(monkeypatch) -> None:
    app = _build_test_app()

    class _DummyActivityWidget:
        def __init__(self) -> None:
            self.scroll_x = 0
            self.scroll_y = 10
            self.max_scroll_y = 100

        @property
        def is_vertical_scroll_end(self) -> bool:
            return int(self.scroll_y) >= int(self.max_scroll_y)

        def scroll_to(self, *, x: int, y: int, animate: bool, force: bool) -> None:
            _ = animate, force
            self.scroll_x = x
            self.scroll_y = y

        def scroll_end(self, animate: bool = False) -> None:
            _ = animate
            self.scroll_y = self.max_scroll_y

    class _FakeKeyEvent:
        def __init__(self, key: str) -> None:
            self.key = key
            self.prevented = False
            self.stopped = False

        def prevent_default(self) -> None:
            self.prevented = True

        def stop(self) -> None:
            self.stopped = True

    widget = _DummyActivityWidget()
    monkeypatch.setattr(app, "query_one", lambda *_args, **_kwargs: widget)

    app._worker_context.clear()
    app.on_key(cast(Any, _FakeKeyEvent("up")))
    assert widget.scroll_y == 7
    assert app._activity_follow_tail is False

    event = _FakeKeyEvent("end")
    app.on_key(cast(Any, event))
    assert widget.scroll_y == widget.max_scroll_y
    assert app._activity_follow_tail is True
    assert event.prevented is True
    assert event.stopped is True
