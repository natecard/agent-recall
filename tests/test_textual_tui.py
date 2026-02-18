from __future__ import annotations

from collections.abc import Callable
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
        ralph_enabled=False,
        ralph_running=False,
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
        discover_coding_models=lambda _cli: ([], None),
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


def test_palette_contains_layout_action() -> None:
    action_ids = {action.action_id for action in get_palette_actions()}
    assert "layout" in action_ids


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


def test_layout_modal_updates_visibility_and_banner(monkeypatch) -> None:
    app = _build_test_app()
    app.tui_widget_visibility = {
        "knowledge": True,
        "sources": True,
        "timeline": True,
        "ralph": True,
        "llm": True,
        "settings": True,
    }
    captured: dict[str, bool] = {}
    monkeypatch.setattr(app, "_apply_tui_layout_settings", lambda: captured.setdefault("ok", True))

    class _LayoutModule:
        @staticmethod
        def default_widget_visibility() -> dict[str, bool]:
            return {
                "knowledge": True,
                "sources": True,
                "timeline": True,
                "ralph": True,
                "llm": True,
                "settings": True,
            }

        @staticmethod
        def normalize_banner_size(value: object) -> str:
            return "compact" if str(value).strip().lower() == "compact" else "normal"

    app._layout_module = None
    monkeypatch.setattr(app, "_load_layout_module", lambda: _LayoutModule)

    app._apply_layout_modal_result(
        {
            "widgets": {"knowledge": False, "sources": True},
            "banner_size": "compact",
        }
    )

    assert app.tui_widget_visibility["knowledge"] is False
    assert app.tui_widget_visibility["sources"] is True
    assert app.tui_banner_size == "compact"
    assert captured.get("ok") is True


def test_dashboard_mount_skips_hidden_widgets() -> None:
    app = _build_test_app()
    app.current_view = "overview"
    app.tui_banner_size = "normal"
    app.tui_widget_visibility = {
        "knowledge": False,
        "sources": False,
        "timeline": True,
        "ralph": True,
        "llm": True,
        "settings": True,
    }

    cast(Any, app)._build_overview_row = lambda panels: None

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
        source_names=[],
    )

    class _FakeDashboard:
        def __init__(self) -> None:
            self.mounted: list[object] = []

        def mount(self, widget: object) -> None:
            self.mounted.append(widget)

    dashboard = _FakeDashboard()

    app._mount_dashboard_widgets(cast(Any, dashboard), panels)

    mounted_ids = {getattr(widget, "id", None) for widget in dashboard.mounted}
    assert "dashboard_knowledge" not in mounted_ids
    assert "dashboard_sources" not in mounted_ids
    assert "dashboard_header" in mounted_ids


def test_refresh_dashboard_reuses_layout_without_remove_children(monkeypatch) -> None:
    app = _build_test_app()
    app.current_view = "overview"
    app._dashboard_layout_view = "overview"
    app.tui_widget_visibility = {
        "knowledge": True,
        "sources": True,
        "timeline": True,
        "ralph": True,
        "llm": True,
        "settings": True,
    }
    app.tui_banner_size = "normal"
    app._last_layout_signature = (
        app.tui_banner_size,
        tuple(sorted(app.tui_widget_visibility.items())),
    )
    app._layout_module = None

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
            self.classes = ""

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

    assert app._last_layout_signature is not None

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


def test_tui_ralph_run_falls_back_to_python_loop_when_script_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent_dir = tmp_path / ".agent"
    (agent_dir / "ralph").mkdir(parents=True)

    config = {
        "ralph": {
            "enabled": True,
            "coding_cli": "codex",
            "cli_model": "gpt-5.3-codex",
            "max_iterations": 2,
            "sleep_seconds": 0,
            "compact_mode": "off",
            "selected_prd_ids": ["AR-1"],
        }
    }
    (agent_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (agent_dir / "ralph" / "prd.json").write_text(
        '{"items":[{"id":"AR-1","title":"One","passes":false}]}', encoding="utf-8"
    )

    app = _build_test_app()
    captured_activity: list[str] = []
    worker_result: dict[str, object] = {}

    class _DummyWidget:
        display = False
        value = ""
        cursor_position = 0

        def focus(self) -> None:
            pass

    class _FakeLoop:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run_loop(self, **kwargs: object) -> dict[str, int]:
            progress_callback = kwargs.get("progress_callback")
            if callable(progress_callback):
                cast(Callable[[dict[str, str]], None], progress_callback)(
                    {"event": "output_line", "line": "fallback line"}
                )
            return {"total_iterations": 1, "passed": 1, "failed": 0}

    monkeypatch.setattr(app, "_append_activity", lambda line: captured_activity.append(line))
    monkeypatch.setattr(app, "call_from_thread", lambda fn, *args: fn(*args))
    monkeypatch.setattr(app, "query_one", lambda *_args, **_kwargs: _DummyWidget())
    monkeypatch.setattr(
        "agent_recall.cli.ralph.get_default_script_path",
        lambda: tmp_path / "missing-script.sh",
    )
    monkeypatch.setattr(
        "agent_recall.cli.ralph.get_ralph_components",
        lambda: (agent_dir, object(), object()),
    )
    monkeypatch.setattr("agent_recall.ralph.loop.RalphLoop", _FakeLoop)

    def _run_worker_inline(fn, **_kwargs):  # noqa: ANN001
        worker_result.update(fn())

        class _Worker:
            pass

        return _Worker()

    monkeypatch.setattr(app, "run_worker", _run_worker_inline)

    app.action_run_ralph_loop()

    assert any("Falling back to built-in loop mode." in line for line in captured_activity)
    assert any("fallback line" in line for line in captured_activity)
    assert worker_result["exit_code"] == 0


def test_cli_input_handles_slash_command_refresh(monkeypatch) -> None:
    app = _build_test_app()
    captured_activity: list[str] = []
    captured_command: list[str] = []
    monkeypatch.setattr(app, "_append_activity", lambda line: captured_activity.append(line))
    monkeypatch.setattr(app, "_refresh_dashboard_panel", lambda: None)

    class _FakeInput:
        id = "cli_input"
        value = ""
        cursor_position = 0

        def focus(self) -> None:
            pass

    class _FakeSuggestions:
        id = "cli_suggestions"
        display = False

        def clear_options(self) -> None:
            pass

    def _mock_query_one(selector: str, *args: Any, **kwargs: Any) -> Any:
        if "cli_input" in selector:
            return _FakeInput()
        return _FakeSuggestions()

    monkeypatch.setattr(app, "query_one", _mock_query_one)

    # Mock the backend command runner to capture what command would be run
    monkeypatch.setattr(app, "_run_backend_command", lambda cmd: captured_command.append(cmd))

    class _Submitted:
        value = "/refresh"

    app.on_input_submitted(cast(Any, _Submitted()))

    assert any("> /refresh" in line for line in captured_activity)
    assert captured_command == ["status"]


def test_cli_input_handles_help_command(monkeypatch) -> None:
    app = _build_test_app()
    captured_activity: list[str] = []
    monkeypatch.setattr(app, "_append_activity", lambda line: captured_activity.append(line))

    class _FakeInput:
        id = "cli_input"
        value = ""
        cursor_position = 0

        def focus(self) -> None:
            pass

    class _FakeSuggestions:
        id = "cli_suggestions"
        display = False

        def clear_options(self) -> None:
            pass

    def _mock_query_one(selector: str, *args: Any, **kwargs: Any) -> Any:
        if "cli_input" in selector:
            return _FakeInput()
        return _FakeSuggestions()

    monkeypatch.setattr(app, "query_one", _mock_query_one)

    class _Submitted:
        value = "/help"

    app.on_input_submitted(cast(Any, _Submitted()))

    assert any("Available slash commands:" in line for line in captured_activity)
    assert any("/quit" in line for line in captured_activity)
    assert any("/refresh" in line for line in captured_activity)


def test_cli_input_handles_unknown_command(monkeypatch) -> None:
    app = _build_test_app()
    captured_activity: list[str] = []
    monkeypatch.setattr(app, "_append_activity", lambda line: captured_activity.append(line))

    class _FakeInput:
        id = "cli_input"
        value = ""
        cursor_position = 0

        def focus(self) -> None:
            pass

    class _FakeSuggestions:
        id = "cli_suggestions"
        display = False

        def clear_options(self) -> None:
            pass

    def _mock_query_one(selector: str, *args: Any, **kwargs: Any) -> Any:
        if "cli_input" in selector:
            return _FakeInput()
        return _FakeSuggestions()

    monkeypatch.setattr(app, "query_one", _mock_query_one)

    class _Submitted:
        value = "/unknowncmd"

    app.on_input_submitted(cast(Any, _Submitted()))

    assert any("Unknown command: /unknowncmd" in line for line in captured_activity)


def test_cli_input_requires_slash_prefix(monkeypatch) -> None:
    app = _build_test_app()
    captured_activity: list[str] = []
    monkeypatch.setattr(app, "_append_activity", lambda line: captured_activity.append(line))

    class _FakeInput:
        id = "cli_input"
        value = ""
        cursor_position = 0

        def focus(self) -> None:
            pass

    class _FakeSuggestions:
        id = "cli_suggestions"
        display = False

        def clear_options(self) -> None:
            pass

    def _mock_query_one(selector: str, *args: Any, **kwargs: Any) -> Any:
        if "cli_input" in selector:
            return _FakeInput()
        return _FakeSuggestions()

    monkeypatch.setattr(app, "query_one", _mock_query_one)

    class _Submitted:
        value = "status"

    app.on_input_submitted(cast(Any, _Submitted()))

    assert any("> status" in line for line in captured_activity)
    assert any("Commands must start with /" in line for line in captured_activity)


def test_cli_input_focus_action(monkeypatch) -> None:
    app = _build_test_app()

    class _FakeInput:
        def __init__(self) -> None:
            self.value = ""
            self.cursor_position = 0
            self.focused = False

        def focus(self) -> None:
            self.focused = True

    fake_input = _FakeInput()
    monkeypatch.setattr(app, "query_one", lambda *_args, **_kwargs: fake_input)

    app.action_focus_cli_input()

    assert fake_input.focused is True
    assert fake_input.value == "/"
    assert fake_input.cursor_position == 1


def test_context_aware_dashboard_ralph_disabled() -> None:
    """When Ralph is disabled, overview shows Knowledge + Sources side-by-side."""
    from agent_recall.cli.tui.views import build_dashboard_panels
    from agent_recall.cli.tui.views.dashboard_context import DashboardRenderContext

    class _ThemeManager:
        def get_theme_name(self) -> str:
            return "dark+"

    class _Storage:
        def get_stats(self) -> dict[str, int]:
            return {"processed_sessions": 5}

        def get_last_processed_at(self) -> None:
            return None

    class _Files:
        def read_tier(self, _tier: object) -> str:
            return "test content"

        def read_config(self) -> dict[str, object]:
            return {"llm": {"provider": "openai", "model": "gpt-test", "temperature": 0.2}}

    class _Ingester:
        source_name = "cursor"

        def discover_sessions(self) -> list[object]:
            return []

    class _CostSummary:
        total_tokens: int = 0
        total_cost_usd: float = 0.0

    context = DashboardRenderContext(
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
        ralph_enabled=False,
        ralph_running=False,
    )

    panels = build_dashboard_panels(context, view="overview")
    assert panels.knowledge is not None
    assert panels.sources_compact is not None
    # When Ralph disabled, we should have knowledge and sources panels
    assert "Knowledge" in str(panels.knowledge.title)
    assert "Sources" in str(panels.sources_compact.title)


def test_context_aware_dashboard_ralph_enabled_running() -> None:
    """When Ralph is enabled and running, overview shows Ralph + Timeline."""
    from agent_recall.cli.tui.views import build_dashboard_panels
    from agent_recall.cli.tui.views.dashboard_context import DashboardRenderContext

    class _ThemeManager:
        def get_theme_name(self) -> str:
            return "dark+"

    class _Storage:
        def get_stats(self) -> dict[str, int]:
            return {"processed_sessions": 5}

        def get_last_processed_at(self) -> None:
            return None

    class _Files:
        def read_tier(self, _tier: object) -> str:
            return "test content"

        def read_config(self) -> dict[str, object]:
            return {"llm": {"provider": "openai", "model": "gpt-test", "temperature": 0.2}}

    class _Ingester:
        source_name = "cursor"

        def discover_sessions(self) -> list[object]:
            return []

    class _CostSummary:
        total_tokens: int = 0
        total_cost_usd: float = 0.0

    context = DashboardRenderContext(
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
        ralph_enabled=True,
        ralph_running=True,
    )

    panels = build_dashboard_panels(context, view="overview")
    assert panels.ralph is not None
    assert panels.timeline is not None
    assert "Ralph" in str(panels.ralph.title)
    assert "Timeline" in str(panels.timeline.title)


def test_context_aware_dashboard_ralph_enabled_idle() -> None:
    """When Ralph is enabled but idle, overview shows Ralph + Knowledge."""
    from agent_recall.cli.tui.views import build_dashboard_panels
    from agent_recall.cli.tui.views.dashboard_context import DashboardRenderContext

    class _ThemeManager:
        def get_theme_name(self) -> str:
            return "dark+"

    class _Storage:
        def get_stats(self) -> dict[str, int]:
            return {"processed_sessions": 5}

        def get_last_processed_at(self) -> None:
            return None

    class _Files:
        def read_tier(self, _tier: object) -> str:
            return "test content"

        def read_config(self) -> dict[str, object]:
            return {"llm": {"provider": "openai", "model": "gpt-test", "temperature": 0.2}}

    class _Ingester:
        source_name = "cursor"

        def discover_sessions(self) -> list[object]:
            return []

    class _CostSummary:
        total_tokens: int = 0
        total_cost_usd: float = 0.0

    context = DashboardRenderContext(
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
        ralph_enabled=True,
        ralph_running=False,
    )

    panels = build_dashboard_panels(context, view="overview")
    assert panels.ralph is not None
    assert panels.knowledge is not None
    assert "Ralph" in str(panels.ralph.title)
    assert "Knowledge" in str(panels.knowledge.title)


def test_context_aware_dashboard_header_badge_ralph_enabled() -> None:
    """Header shows Ralph badge when Ralph is enabled."""
    from agent_recall.cli.tui.views import build_dashboard_panels
    from agent_recall.cli.tui.views.dashboard_context import DashboardRenderContext

    class _ThemeManager:
        def get_theme_name(self) -> str:
            return "dark+"

    class _Storage:
        def get_stats(self) -> dict[str, int]:
            return {"processed_sessions": 5}

        def get_last_processed_at(self) -> None:
            return None

    class _Files:
        def read_tier(self, _tier: object) -> str:
            return "test content"

        def read_config(self) -> dict[str, object]:
            return {"llm": {"provider": "openai", "model": "gpt-test", "temperature": 0.2}}

    class _Ingester:
        source_name = "cursor"

        def discover_sessions(self) -> list[object]:
            return []

    class _CostSummary:
        total_tokens: int = 0
        total_cost_usd: float = 0.0

    # Test with Ralph enabled and running
    context_running = DashboardRenderContext(
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
        ralph_enabled=True,
        ralph_running=True,
    )

    panels_running = build_dashboard_panels(context_running, view="overview")
    assert panels_running.header is not None
    # Header title should contain "Ralph Active" badge
    header_title = str(panels_running.header.title)
    assert "Ralph" in header_title

    # Test with Ralph enabled but idle
    context_idle = DashboardRenderContext(
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
        ralph_enabled=True,
        ralph_running=False,
    )

    panels_idle = build_dashboard_panels(context_idle, view="overview")
    assert panels_idle.header is not None
    header_title_idle = str(panels_idle.header.title)
    assert "Ralph" in header_title_idle


def test_filter_command_suggestions_prefix_match() -> None:
    """Test that prefix matches are prioritized over substring matches."""
    from agent_recall.cli.tui.commands.help_text import filter_command_suggestions

    commands = ["/refresh", "/ralph-run", "/ralph-enable", "/run", "/help"]
    suggestions = filter_command_suggestions("/r", commands, max_results=8)

    # Should include /refresh, /ralph-run, /ralph-enable, /run
    assert "/refresh" in suggestions
    assert "/run" in suggestions
    # Prefix matches should come before substring matches
    assert (
        suggestions.index("/refresh") < suggestions.index("/help")
        if "/help" in suggestions
        else True
    )


def test_filter_command_suggestions_includes_layout() -> None:
    """Test that /layout appears in suggestions."""
    from agent_recall.cli.tui.commands.help_text import filter_command_suggestions

    commands = ["/layout", "/settings", "/help"]
    suggestions = filter_command_suggestions("/l", commands, max_results=8)

    assert "/layout" in suggestions


def test_filter_command_suggestions_max_results() -> None:
    """Test that suggestions are limited to max_results."""
    from agent_recall.cli.tui.commands.help_text import filter_command_suggestions

    commands = [
        "/cmd1",
        "/cmd2",
        "/cmd3",
        "/cmd4",
        "/cmd5",
        "/cmd6",
        "/cmd7",
        "/cmd8",
        "/cmd9",
        "/cmd10",
    ]
    suggestions = filter_command_suggestions("/cmd", commands, max_results=8)

    assert len(suggestions) <= 8


def test_filter_command_suggestions_no_slash_prefix() -> None:
    """Test that non-slash input returns empty suggestions."""
    from agent_recall.cli.tui.commands.help_text import filter_command_suggestions

    commands = ["/refresh", "/help"]
    suggestions = filter_command_suggestions("refresh", commands)

    assert suggestions == []


def test_filter_command_suggestions_empty_input() -> None:
    """Test that empty input returns empty suggestions."""
    from agent_recall.cli.tui.commands.help_text import filter_command_suggestions

    commands = ["/refresh", "/help"]
    suggestions = filter_command_suggestions("", commands)

    assert suggestions == []


def test_cli_autocomplete_shows_suggestions_on_input(monkeypatch) -> None:
    """Test that typing /r shows matching suggestions."""
    app = _build_test_app()

    class _FakeInput:
        id = "cli_input"
        value = "/r"
        cursor_position = 2

        def focus(self) -> None:
            pass

    class _FakeSuggestions:
        id = "cli_suggestions"
        display = False
        options: list[Any] = []

        def clear_options(self) -> None:
            self.options = []

        def add_option(self, option: Any) -> None:
            self.options.append(option)

    fake_input = _FakeInput()
    fake_suggestions = _FakeSuggestions()

    def _mock_query_one(selector: str, *args: Any, **kwargs: Any) -> Any:
        if "cli_input" in selector:
            return fake_input
        return fake_suggestions

    monkeypatch.setattr(app, "query_one", _mock_query_one)

    class _Changed:
        value = "/r"

    app.on_input_changed(cast(Any, _Changed()))

    # Suggestions should be visible and have matching commands
    assert fake_suggestions.display is True
    assert len(fake_suggestions.options) > 0


def test_cli_autocomplete_esc_dismisses_suggestions(monkeypatch) -> None:
    """Test that pressing Esc dismisses suggestions without submitting."""
    app = _build_test_app()

    class _FakeSuggestions:
        id = "cli_suggestions"
        display = True

        def clear_options(self) -> None:
            pass

    fake_suggestions = _FakeSuggestions()

    def _mock_query_one(selector: str, *args: Any, **kwargs: Any) -> Any:
        return fake_suggestions

    monkeypatch.setattr(app, "query_one", _mock_query_one)

    app._suggestions_visible = True

    # Test _hide_suggestions directly
    app._hide_suggestions()

    assert fake_suggestions.display is False
    assert app._suggestions_visible is False


def test_cli_autocomplete_tab_completes_suggestion(monkeypatch) -> None:
    """Test that _apply_suggestion completes the highlighted suggestion."""
    app = _build_test_app()

    class _FakeInput:
        id = "cli_input"
        value = "/ref"
        cursor_position = 4

        def focus(self) -> None:
            pass

    class _FakeOption:
        prompt = "/refresh"

    class _FakeSuggestions:
        id = "cli_suggestions"
        display = True
        option_count = 1
        _highlighted = 0

        @property
        def highlighted(self) -> int:
            return self._highlighted

        @highlighted.setter
        def highlighted(self, value: int) -> None:
            self._highlighted = value

        def clear_options(self) -> None:
            pass

        def get_option_at_index(self, index: int) -> Any:
            return _FakeOption()

    fake_input = _FakeInput()
    fake_suggestions = _FakeSuggestions()

    def _mock_query_one(selector: str, *args: Any, **kwargs: Any) -> Any:
        if "cli_input" in selector:
            return fake_input
        return fake_suggestions

    monkeypatch.setattr(app, "query_one", _mock_query_one)

    app._suggestions_visible = True
    app._highlighted_suggestion_index = 0

    # Test _apply_suggestion directly
    app._apply_suggestion()

    assert fake_input.value == "/refresh"
    assert fake_input.cursor_position == len("/refresh")
    assert app._suggestions_visible is False


def test_interactive_sources_widget_displays_sources() -> None:
    """Test that InteractiveSourcesWidget displays sources with correct data."""
    from agent_recall.cli.tui.widgets import InteractiveSourcesWidget

    sources = [
        {"name": "cursor", "status": "available", "sessions": 5, "available": True},
        {"name": "codex", "status": "empty", "sessions": 0, "available": False},
    ]

    sync_calls: list[str] = []

    def on_sync(name: str) -> None:
        sync_calls.append(name)

    widget = InteractiveSourcesWidget(
        sources=sources,
        on_sync=on_sync,
        last_synced="2024-01-15 10:30 UTC",
    )

    assert widget.sources == sources
    assert widget.last_synced == "2024-01-15 10:30 UTC"
    assert widget.on_sync == on_sync


def test_interactive_sources_widget_mark_sync_complete() -> None:
    """Test that mark_sync_complete updates widget state correctly."""
    from agent_recall.cli.tui.widgets import InteractiveSourcesWidget

    sources = [
        {"name": "cursor", "status": "available", "sessions": 5, "available": True},
    ]

    widget = InteractiveSourcesWidget(
        sources=sources,
        on_sync=lambda _name: None,
        last_synced="2024-01-15 10:30 UTC",
    )

    # Simulate a sync in progress
    widget._syncing_sources.add("cursor")

    # Mark as complete with success
    widget.mark_sync_complete("cursor", success=True)

    assert "cursor" not in widget._syncing_sources


def test_interactive_sources_widget_mark_sync_complete_failure() -> None:
    """Test that mark_sync_complete handles failure case."""
    from agent_recall.cli.tui.widgets import InteractiveSourcesWidget

    sources = [
        {"name": "cursor", "status": "available", "sessions": 5, "available": True},
    ]

    widget = InteractiveSourcesWidget(
        sources=sources,
        on_sync=lambda _name: None,
        last_synced="2024-01-15 10:30 UTC",
    )

    # Simulate a sync in progress
    widget._syncing_sources.add("cursor")

    # Mark as complete with failure
    widget.mark_sync_complete("cursor", success=False)

    assert "cursor" not in widget._syncing_sources


def test_interactive_sources_widget_update_sources() -> None:
    """Test that update_sources refreshes the sources data."""
    from agent_recall.cli.tui.widgets import InteractiveSourcesWidget

    initial_sources = [
        {"name": "cursor", "status": "available", "sessions": 5, "available": True},
    ]

    widget = InteractiveSourcesWidget(
        sources=initial_sources,
        on_sync=lambda _name: None,
        last_synced="2024-01-15 10:30 UTC",
    )

    new_sources = [
        {"name": "cursor", "status": "available", "sessions": 10, "available": True},
        {"name": "codex", "status": "available", "sessions": 3, "available": True},
    ]

    widget.update_sources(new_sources, "2024-01-15 11:00 UTC")

    assert widget.sources == new_sources
    assert widget.last_synced == "2024-01-15 11:00 UTC"


def test_build_sources_data_returns_correct_format() -> None:
    """Test that build_sources_data returns sources in expected format."""
    from rich.console import Console

    from agent_recall.cli.tui.views import build_sources_data
    from agent_recall.cli.tui.views.dashboard_context import DashboardRenderContext

    class _MockIngester:
        source_name = "cursor"

        def discover_sessions(self) -> list[object]:
            return [object(), object(), object()]  # 3 sessions

    class _MockStorage:
        def get_last_processed_at(self) -> None:
            return None

    class _MockFiles:
        def read_config(self) -> dict[str, object]:
            return {}

    class _ThemeManager:
        def get_theme_name(self) -> str:
            return "dark+"

    context = DashboardRenderContext(
        console=Console(width=120, record=True),
        theme_manager=_ThemeManager(),
        agent_dir=Path("/tmp"),
        ralph_max_iterations=10,
        get_storage=lambda: _MockStorage(),
        get_files=lambda: _MockFiles(),
        get_repo_selected_sources=lambda _files: None,
        resolve_repo_root_for_display=lambda: Path("/tmp"),
        filter_ingesters_by_sources=lambda ingesters, _selected: ingesters,
        get_default_ingesters=lambda **_kwargs: [_MockIngester()],
        render_iteration_timeline=lambda _store, max_entries: [],
        summarize_costs=lambda _reports: type(
            "CostSummary", (), {"total_tokens": 0, "total_cost_usd": 0.0}
        )(),
        format_usd=lambda amount: f"${amount:.2f}",
        is_interactive_terminal=lambda: False,
        help_lines_provider=lambda: [],
    )

    sources_data, last_synced = build_sources_data(context, all_cursor_workspaces=False)

    assert len(sources_data) == 1
    assert sources_data[0]["name"] == "cursor"
    assert sources_data[0]["sessions"] == 3
    assert sources_data[0]["available"] is True
    assert last_synced == "Never"


def test_open_iteration_detail_builds_modal(monkeypatch) -> None:
    app = _build_test_app()

    class _FakeModal:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    captured: dict[str, Any] = {}

    def _capture_push(screen: Any) -> None:
        captured["screen"] = screen

    class _FakeReport:
        iteration = 5
        item_id = "AR-1004"
        item_title = "Enhanced Knowledge"
        summary = "Did the thing."
        outcome = type("Outcome", (), {"value": "COMPLETED"})()

    class _FakeStore:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def load_diff_for_iteration(self, _iteration: int) -> str:
            return "diff --git a/file b/file"

    monkeypatch.setattr(
        "agent_recall.cli.tui.app.IterationDetailModal",
        _FakeModal,
    )
    monkeypatch.setattr(
        "agent_recall.cli.tui.app.IterationReportStore",
        _FakeStore,
    )
    monkeypatch.setattr(app, "push_screen", _capture_push)

    app._open_iteration_detail(cast(Any, _FakeReport()))

    screen = captured.get("screen")
    assert isinstance(screen, _FakeModal)
    assert screen.kwargs["title"] == "Iteration 5 Detail"
    assert screen.kwargs["summary_text"] == "Did the thing."
    assert screen.kwargs["outcome_text"] == "COMPLETED"
    assert "AR-1004" in screen.kwargs["item_text"]
