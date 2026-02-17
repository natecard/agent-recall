from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from rich.console import Console
from rich.panel import Panel

from agent_recall.cli.tui import (
    AgentRecallTextualApp,
    _build_command_suggestions,
    _clean_optional_text,
    _is_knowledge_run_command,
    _is_palette_cli_command_redundant,
    _sanitize_activity_fragment,
    get_palette_actions,
)
from agent_recall.cli.tui.views import DashboardPanels
from agent_recall.cli.tui.views.dashboard_context import DashboardRenderContext


def _build_test_app() -> AgentRecallTextualApp:
    def theme_defaults() -> tuple[list[str], str]:
        themes = ["dark+"]
        return themes, "dark+"

    def discover_models(*_args: object, **_kwargs: object) -> tuple[list[str], str | None]:
        models = ["gpt-test"]
        return models, None

    class _ThemeManager:
        def get_theme_name(self) -> str:
            return "dark+"

    class _Storage:
        def get_stats(self) -> dict[str, int]:
            return {}

        def get_last_processed_at(self) -> None:
            return None

    class _Files:
        def read_tier(self, _tier: object) -> str:
            return ""

        def read_config(self) -> dict[str, object]:
            return {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-test",
                    "temperature": 0.2,
                    "max_tokens": 1024,
                    "base_url": None,
                }
            }

    class _Ingester:
        source_name = "cursor"

        def discover_sessions(self) -> list[object]:
            return []

    class _CostSummary:
        total_tokens: int = 0
        total_cost_usd: float = 0.0

    dashboard_context = DashboardRenderContext(
        console=Console(width=120, record=True),
        theme_manager=_ThemeManager(),
        agent_dir=Path("/tmp"),
        ralph_max_iterations=10,
        get_storage=lambda: _Storage(),
        get_files=lambda: _Files(),
        get_repo_selected_sources=lambda _files: None,
        resolve_repo_root_for_display=lambda: Path("/tmp"),
        filter_ingesters_by_sources=lambda ingesters, _selected: ingesters,
        get_default_ingesters=lambda **_kwargs: [_Ingester()],
        render_iteration_timeline=lambda _store, max_entries: ["line"] * max_entries,
        summarize_costs=lambda _reports: _CostSummary(),
        format_usd=lambda amount: f"${amount:.2f}",
        is_interactive_terminal=lambda: False,
        help_lines_provider=lambda: ["/help"],
    )

    return AgentRecallTextualApp(
        render_dashboard=lambda *_args, **_kwargs: "",
        dashboard_context=dashboard_context,
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
    assert _is_palette_cli_command_redundant("theme show") is True
    assert _is_palette_cli_command_redundant("ralph status") is True
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


def test_tui_successful_ralph_run_clears_selected_prd_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir(parents=True)
    config_path = agent_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"ralph": {"selected_prd_ids": ["AR-1", "AR-2"]}}),
        encoding="utf-8",
    )

    app = _build_test_app()
    captured_activity: list[str] = []
    monkeypatch.setattr(app, "_append_activity", lambda line: captured_activity.append(line))
    monkeypatch.setattr(app, "_refresh_dashboard_panel", lambda: None)

    app._handle_worker_success("ralph_run", {"exit_code": 0})

    payload = cast(dict[str, Any], yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    ralph_cfg = cast(dict[str, Any], payload.get("ralph") or {})
    assert ralph_cfg.get("selected_prd_ids") is None
    assert any(
        "Cleared PRD selection after successful Ralph run." in line for line in captured_activity
    )


def test_tui_unsuccessful_ralph_run_keeps_selected_prd_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir(parents=True)
    config_path = agent_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"ralph": {"selected_prd_ids": ["AR-1", "AR-2"]}}),
        encoding="utf-8",
    )

    app = _build_test_app()
    captured_activity: list[str] = []
    monkeypatch.setattr(app, "_append_activity", lambda line: captured_activity.append(line))
    monkeypatch.setattr(app, "_refresh_dashboard_panel", lambda: None)

    app._handle_worker_success("ralph_run", {"exit_code": 2})

    payload = cast(dict[str, Any], yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    ralph_cfg = cast(dict[str, Any], payload.get("ralph") or {})
    assert ralph_cfg.get("selected_prd_ids") == ["AR-1", "AR-2"]
    assert not any(
        "Cleared PRD selection after successful Ralph run." in line for line in captured_activity
    )


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


def test_selecting_source_sync_option_runs_sync(monkeypatch) -> None:
    app = _build_test_app()
    captured: list[str] = []

    def _capture_sync(source_name: str) -> None:
        captured.append(source_name)

    monkeypatch.setattr(app, "_run_source_sync", _capture_sync)

    class _OptionList:
        id = "activity_result_list"

    class _Option:
        id = "sync-source:cursor"

    class _Event:
        option_list = _OptionList()
        option = _Option()

    app.on_option_list_option_selected(cast(Any, _Event()))

    assert captured == ["cursor"]


def test_refresh_dashboard_reuses_layout_without_remove_children(monkeypatch) -> None:
    app = _build_test_app()
    app.current_view = "overview"
    app._dashboard_layout_view = "overview"

    panels = DashboardPanels(
        header=Panel("header"),
        knowledge=Panel("knowledge"),
        llm=Panel("llm"),
        sources=Panel("sources"),
        sources_compact=Panel("sources_compact"),
        settings=Panel("settings"),
        timeline=Panel("timeline"),
        ralph=Panel("ralph"),
        slash_console=None,
        source_names=["cursor"],
    )

    class _FakeDashboard:
        def __init__(self) -> None:
            self.remove_children_calls = 0

        def remove_children(self):  # noqa: ANN204
            self.remove_children_calls += 1

            class _Awaitable:
                def __await__(self):  # noqa: ANN204
                    if False:
                        yield None
                    return None

            return _Awaitable()

    class _FakeStatic:
        def __init__(self) -> None:
            self.updates: list[object] = []

        def update(self, renderable: object) -> None:
            self.updates.append(renderable)

    dashboard = _FakeDashboard()
    header = _FakeStatic()
    knowledge = _FakeStatic()
    sources = _FakeStatic()
    by_selector: dict[str, object] = {
        "#dashboard": dashboard,
        "#dashboard_header": header,
        "#dashboard_knowledge": knowledge,
        "#dashboard_sources": sources,
    }

    def _query_one(selector: str, *_args: object, **_kwargs: object) -> object:
        return by_selector[selector]

    monkeypatch.setattr(
        "agent_recall.cli.tui.app.build_dashboard_panels",
        lambda *args, **kwargs: panels,
    )
    monkeypatch.setattr(app, "query_one", _query_one)
    monkeypatch.setattr(app, "_sync_runtime_theme", lambda: None)
    monkeypatch.setattr(app, "_refresh_activity_panel", lambda: None)

    app._refresh_dashboard_panel()
    app._refresh_dashboard_panel()

    assert dashboard.remove_children_calls == 0

    # Verify we got Panel objects with the expected content
    assert len(header.updates) == 2
    assert isinstance(header.updates[0], Panel)
    assert header.updates[0].renderable == "header"

    assert len(knowledge.updates) == 2
    assert isinstance(knowledge.updates[0], Panel)
    assert knowledge.updates[0].renderable == "knowledge"

    assert len(sources.updates) == 2
    assert isinstance(sources.updates[0], Panel)
    assert sources.updates[0].renderable == "sources_compact"
