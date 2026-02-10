from __future__ import annotations

import os
import re
import shlex
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.theme import Theme
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Footer, Header, Input, OptionList, Select, Static
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from agent_recall.core.onboarding import API_KEY_ENV_BY_PROVIDER

DiscoverModelsFn = Callable[[str, str | None, str | None], tuple[list[str], str | None]]

_PROVIDER_BASE_URL_DEFAULTS = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai-compatible": "http://localhost:8080/v1",
}


def _strip_rich_markup(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text)


def _clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def _build_command_suggestions(cli_commands: list[str]) -> list[str]:
    base = [
        "help",
        "status",
        "sync --no-compact",
        "compact",
        "sources",
        "settings",
        "preferences",
        "config settings",
        "config setup",
        "config setup --force",
        "config setup --quick",
        "config model --provider ollama --model llama3.1",
        "config model --temperature 0.2 --max-tokens 8192",
        "view overview",
        "view sources",
        "view llm",
        "view knowledge",
        "view settings",
        "view console",
        "view all",
        "menu overview",
        "quit",
    ]

    suggestions: list[str] = []
    seen: set[str] = set()

    for value in [*base, *cli_commands]:
        cleaned = value.strip()
        if not cleaned:
            continue
        if cleaned.startswith("/"):
            cleaned = cleaned[1:].strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        suggestions.append(cleaned)

    return suggestions


@dataclass(frozen=True)
class PaletteAction:
    action_id: str
    title: str
    group: str
    shortcut: str = ""
    keywords: str = ""


class CommandPaletteModal(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, actions: list[PaletteAction]):
        super().__init__()
        self.actions = actions
        self.query_text = ""

    def compose(self) -> ComposeResult:
        with Container(id="palette_overlay"):
            with Vertical(id="palette_card"):
                yield Static("Commands", id="palette_title")
                yield Input(placeholder="Search commands...", id="palette_search")
                yield OptionList(id="palette_options")
                yield Static("Enter run  Up/Down move  Esc close", id="palette_hint")

    def on_mount(self) -> None:
        self.query_one("#palette_search", Input).focus()
        self._rebuild_options()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "palette_search":
            self.query_text = event.value
            self._rebuild_options()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "palette_search":
            return
        option_list = self.query_one("#palette_options", OptionList)
        highlighted = option_list.highlighted
        if highlighted is None:
            query = self.query_text.strip()
            if query:
                self.dismiss(f"run:{query}")
            return
        option = option_list.get_option_at_index(highlighted)
        option_id = option.id or ""
        if option_id == "action:run-query":
            query = self.query_text.strip()
            if query:
                self.dismiss(f"run:{query}")
            return
        if option_id.startswith("action:"):
            self.dismiss(option_id.split(":", 1)[1])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id or ""
        if option_id == "action:run-query":
            query = self.query_text.strip()
            if query:
                self.dismiss(f"run:{query}")
            return
        if option_id.startswith("action:"):
            self.dismiss(option_id.split(":", 1)[1])

    def _rebuild_options(self) -> None:
        query = self.query_text.strip().lower()
        grouped: dict[str, list[PaletteAction]] = defaultdict(list)
        for action in self.actions:
            haystack = f"{action.title} {action.group} {action.keywords} {action.shortcut}".lower()
            if query and query not in haystack:
                continue
            grouped[action.group].append(action)

        options: list[Option] = []
        if query:
            options.append(Option("[b]Run[/b]", id="heading:run", disabled=True))
            options.append(Option(f"Run: {self.query_text.strip()}", id="action:run-query"))
        grouped_order = ["Suggested", "Session", "CLI", "Agent", "Provider", "Views", "Maintenance"]
        for group in grouped_order:
            items = grouped.get(group, [])
            if not items:
                continue

            if not query:
                options.append(Option(f"[b]{group}[/b]", id=f"heading:{group}", disabled=True))
            for action in items:
                shortcut_text = f" [dim]{action.shortcut}[/dim]" if action.shortcut else ""
                options.append(
                    Option(
                        f"{action.title}{shortcut_text}",
                        id=f"action:{action.action_id}",
                    )
                )

        option_list = self.query_one("#palette_options", OptionList)
        option_list.set_options(options)

        for index, option in enumerate(options):
            if not option.disabled:
                option_list.highlighted = index
                break


class SetupModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(
        self,
        *,
        providers: list[str],
        defaults: dict[str, Any],
        discover_models: DiscoverModelsFn,
    ):
        super().__init__()
        self.providers = providers
        self.defaults = defaults
        self.discover_models = discover_models

    def compose(self) -> ComposeResult:
        default_provider = str(self.defaults.get("provider", self.providers[0]))
        if default_provider not in self.providers:
            default_provider = self.providers[0]

        selected_agents = {
            source for source in self.defaults.get("selected_agents", []) if isinstance(source, str)
        }
        default_model = _clean_optional_text(self.defaults.get("model", ""))
        base_url = _clean_optional_text(self.defaults.get("base_url", ""))
        if not base_url:
            base_url = _PROVIDER_BASE_URL_DEFAULTS.get(default_provider, "")
        repo_path = str(self.defaults.get("repository_path", Path.cwd()))

        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("Repository Setup", classes="modal_title")
                yield Static(repo_path, id="setup_repo_path")
                with Horizontal(classes="field_row"):
                    yield Checkbox(
                        "Use this repository",
                        value=bool(self.defaults.get("repository_verified", True)),
                        id="setup_repository_verified",
                    )
                    yield Checkbox(
                        "Force reconfigure",
                        value=bool(self.defaults.get("force", False)),
                        id="setup_force",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Provider", classes="field_label")
                    yield Select(
                        [(provider, provider) for provider in self.providers],
                        value=default_provider,
                        allow_blank=False,
                        id="setup_provider",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Base URL", classes="field_label")
                    yield Input(
                        value=base_url,
                        placeholder="Base URL (optional)",
                        id="setup_base_url",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("API key", classes="field_label")
                    yield Input(
                        value="",
                        placeholder="Provider API key (optional)",
                        password=True,
                        id="setup_api_key",
                        classes="field_input",
                    )
                yield Static("", id="setup_api_hint")
                with Horizontal(classes="field_row"):
                    yield Static("Model list", classes="field_label")
                    yield Select(
                        [("Manual entry", "__manual__")],
                        value="__manual__",
                        allow_blank=False,
                        id="setup_model_picker",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Model", classes="field_label")
                    yield Input(
                        value=default_model,
                        placeholder="Model",
                        id="setup_model",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Temperature", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("temperature", 0.3)),
                        placeholder="0.0-2.0",
                        id="setup_temperature",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Max tokens", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("max_tokens", 4096)),
                        placeholder=">0",
                        id="setup_max_tokens",
                        classes="field_input",
                    )
                with Horizontal(classes="setup_agents field_row"):
                    yield Checkbox(
                        "Cursor", value="cursor" in selected_agents, id="setup_agent_cursor"
                    )
                    yield Checkbox(
                        "Claude Code",
                        value="claude-code" in selected_agents,
                        id="setup_agent_claude",
                    )
                    yield Checkbox(
                        "Validate provider connection",
                        value=bool(self.defaults.get("validate", False)),
                        id="setup_validate",
                    )
                yield Static("", id="setup_status")
                with Horizontal(classes="modal_actions"):
                    yield Button("Refresh Models", id="setup_refresh_models")
                    yield Button("Apply Setup", variant="primary", id="setup_apply")
                    yield Button("Cancel", id="setup_cancel")

    def on_mount(self) -> None:
        self._update_provider_hints(set_default_base_url=True)
        self._refresh_models()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "setup_provider":
            self._update_provider_hints(set_default_base_url=True)
            self._refresh_models()
            return
        if event.select.id != "setup_model_picker":
            return

        selected_value = event.value
        if selected_value == Select.BLANK:
            return
        if str(selected_value) == "__manual__":
            return
        self.query_one("#setup_model", Input).value = str(selected_value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup_cancel":
            self.dismiss(None)
            return

        if event.button.id == "setup_refresh_models":
            self._refresh_models()
            return

        if event.button.id != "setup_apply":
            return

        status = self.query_one("#setup_status", Static)

        provider_value = self.query_one("#setup_provider", Select).value
        if provider_value == Select.BLANK:
            status.update("[red]Provider is required[/red]")
            return
        provider = str(provider_value)

        repository_verified = bool(self.query_one("#setup_repository_verified", Checkbox).value)
        if not repository_verified:
            status.update("[red]Repository must be confirmed[/red]")
            return

        selected_agents: list[str] = []
        if self.query_one("#setup_agent_cursor", Checkbox).value:
            selected_agents.append("cursor")
        if self.query_one("#setup_agent_claude", Checkbox).value:
            selected_agents.append("claude-code")
        if not selected_agents:
            status.update("[red]Choose at least one agent source[/red]")
            return

        model = _clean_optional_text(self.query_one("#setup_model", Input).value)
        if not model:
            status.update("[red]Model is required[/red]")
            return

        base_url = _clean_optional_text(self.query_one("#setup_base_url", Input).value) or None
        if provider == "openai-compatible" and not base_url:
            status.update("[red]Base URL is required for openai-compatible[/red]")
            return

        try:
            temperature = float(self.query_one("#setup_temperature", Input).value)
        except ValueError:
            status.update("[red]Temperature must be a number[/red]")
            return
        if temperature < 0.0 or temperature > 2.0:
            status.update("[red]Temperature must be between 0.0 and 2.0[/red]")
            return

        try:
            max_tokens = int(self.query_one("#setup_max_tokens", Input).value)
        except ValueError:
            status.update("[red]Max tokens must be an integer[/red]")
            return
        if max_tokens <= 0:
            status.update("[red]Max tokens must be > 0[/red]")
            return

        self.dismiss(
            {
                "force": bool(self.query_one("#setup_force", Checkbox).value),
                "repository_verified": repository_verified,
                "selected_agents": selected_agents,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "validate": bool(self.query_one("#setup_validate", Checkbox).value),
                "api_key": _clean_optional_text(self.query_one("#setup_api_key", Input).value)
                or None,
            }
        )

    def _refresh_models(self) -> None:
        provider = self._selected_provider()
        status = self.query_one("#setup_status", Static)

        input_model = _clean_optional_text(self.query_one("#setup_model", Input).value)
        base_url_value = _clean_optional_text(
            self.query_one("#setup_base_url", Input).value
        ) or None
        base_url = base_url_value or _PROVIDER_BASE_URL_DEFAULTS.get(provider)

        env_var = API_KEY_ENV_BY_PROVIDER.get(provider)
        entered_key = _clean_optional_text(self.query_one("#setup_api_key", Input).value)

        previous_env_value = os.environ.get(env_var) if env_var else None
        should_override_env = bool(env_var and entered_key)
        if env_var and should_override_env:
            os.environ[env_var] = entered_key

        try:
            models, error_message = self.discover_models(provider, base_url, env_var)
        finally:
            if env_var and should_override_env:
                if previous_env_value is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = previous_env_value

        options: list[tuple[str, str]] = [("Manual entry", "__manual__")]
        options.extend((model_name, model_name) for model_name in models)

        picker = self.query_one("#setup_model_picker", Select)
        picker.set_options(options)

        selected_model = input_model or _clean_optional_text(self.defaults.get("model", ""))
        if selected_model in models:
            picker.value = selected_model
        else:
            picker.value = "__manual__"

        if models:
            status.update(f"[green]Loaded {len(models)} live model(s)[/green]")
        elif error_message:
            status.update(f"[yellow]Live model discovery unavailable: {error_message}[/yellow]")
        else:
            status.update("[yellow]No live models returned; use manual model entry.[/yellow]")

    def _selected_provider(self) -> str:
        value = self.query_one("#setup_provider", Select).value
        if value == Select.BLANK:
            return self.providers[0]
        return str(value)

    def _update_provider_hints(self, *, set_default_base_url: bool) -> None:
        provider = self._selected_provider()
        env_var = API_KEY_ENV_BY_PROVIDER.get(provider)

        api_hint = self.query_one("#setup_api_hint", Static)
        api_input = self.query_one("#setup_api_key", Input)
        if env_var:
            api_hint.update(f"API key env: {env_var}")
            api_input.placeholder = f"API key ({env_var}) optional"
        else:
            api_hint.update("This provider typically does not require an API key.")
            api_input.placeholder = "Provider API key (optional)"

        base_url_input = self.query_one("#setup_base_url", Input)
        default_base_url = _PROVIDER_BASE_URL_DEFAULTS.get(provider, "")
        if set_default_base_url:
            base_url_input.value = default_base_url
        if provider == "openai-compatible":
            base_url_input.placeholder = "Base URL (required)"
        elif default_base_url:
            base_url_input.placeholder = "Base URL (auto-filled)"
        else:
            base_url_input.placeholder = "Base URL (optional)"


class ModelConfigModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(
        self,
        providers: list[str],
        defaults: dict[str, Any],
        discover_models: DiscoverModelsFn,
    ):
        super().__init__()
        self.providers = providers
        self.defaults = defaults
        self.discover_models = discover_models

    def compose(self) -> ComposeResult:
        default_provider = _clean_optional_text(self.defaults.get("provider", self.providers[0]))
        if default_provider not in self.providers:
            default_provider = self.providers[0]
        base_url = _clean_optional_text(self.defaults.get("base_url", ""))
        if not base_url:
            base_url = _PROVIDER_BASE_URL_DEFAULTS.get(default_provider, "")
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("Model Configuration", classes="modal_title")
                with Horizontal(classes="field_row"):
                    yield Static("Provider", classes="field_label")
                    yield Select(
                        [(provider, provider) for provider in self.providers],
                        value=default_provider,
                        allow_blank=False,
                        id="model_provider",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Model", classes="field_label")
                    yield Input(
                        value=_clean_optional_text(self.defaults.get("model", "")),
                        placeholder="Model",
                        id="model_name",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Base URL", classes="field_label")
                    yield Input(
                        value=base_url,
                        placeholder="Base URL (optional)",
                        id="model_base_url",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("API key", classes="field_label")
                    yield Input(
                        value="",
                        placeholder="Provider API key (optional)",
                        password=True,
                        id="model_api_key",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Model list", classes="field_label")
                    yield Select(
                        [("Manual entry", "__manual__")],
                        value="__manual__",
                        allow_blank=False,
                        id="model_picker",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Temperature", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("temperature", 0.3)),
                        placeholder="0.0-2.0",
                        id="model_temperature",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Max tokens", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("max_tokens", 4096)),
                        placeholder=">0",
                        id="model_max_tokens",
                        classes="field_input",
                    )
                yield Checkbox("Validate after apply", value=True, id="model_validate")
                yield Static("", id="model_discovery_status")
                yield Static("", id="model_error")
                with Horizontal(classes="modal_actions"):
                    yield Button("Refresh models", id="model_refresh")
                    yield Button("Apply", variant="primary", id="model_apply")
                    yield Button("Cancel", id="model_cancel")

    def on_mount(self) -> None:
        self._apply_base_url_default(set_default_base_url=False)
        self._refresh_models()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model_provider":
            self._apply_base_url_default(set_default_base_url=True)
            self._refresh_models()
            return
        if event.select.id != "model_picker":
            return
        selected_value = event.value
        if selected_value == Select.BLANK:
            return
        if str(selected_value) == "__manual__":
            return
        self.query_one("#model_name", Input).value = str(selected_value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "model_api_key":
            self._refresh_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "model_cancel":
            self.dismiss(None)
            return
        if event.button.id == "model_refresh":
            self._refresh_models()
            return
        if event.button.id != "model_apply":
            return

        error_widget = self.query_one("#model_error", Static)

        provider_widget = self.query_one("#model_provider", Select)
        provider = provider_widget.value
        if provider == Select.BLANK:
            error_widget.update("[red]Provider is required[/red]")
            return

        try:
            temperature = float(self.query_one("#model_temperature", Input).value)
        except ValueError:
            error_widget.update("[red]Temperature must be a number[/red]")
            return
        if temperature < 0.0 or temperature > 2.0:
            error_widget.update("[red]Temperature must be between 0.0 and 2.0[/red]")
            return

        try:
            max_tokens = int(self.query_one("#model_max_tokens", Input).value)
        except ValueError:
            error_widget.update("[red]Max tokens must be an integer[/red]")
            return
        if max_tokens <= 0:
            error_widget.update("[red]Max tokens must be > 0[/red]")
            return

        self.dismiss(
            {
                "provider": provider,
                "model": _clean_optional_text(self.query_one("#model_name", Input).value) or None,
                "base_url": _clean_optional_text(self.query_one("#model_base_url", Input).value)
                or None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "validate": bool(self.query_one("#model_validate", Checkbox).value),
            }
        )

    def _apply_base_url_default(self, *, set_default_base_url: bool) -> None:
        provider_value = self.query_one("#model_provider", Select).value
        if provider_value == Select.BLANK:
            provider = self.providers[0]
        else:
            provider = str(provider_value)

        base_url_input = self.query_one("#model_base_url", Input)
        default_base_url = _PROVIDER_BASE_URL_DEFAULTS.get(provider, "")
        if set_default_base_url:
            base_url_input.value = default_base_url

        if provider == "openai-compatible":
            base_url_input.placeholder = "Base URL (required)"
        elif default_base_url:
            base_url_input.placeholder = "Base URL (auto-filled)"
        else:
            base_url_input.placeholder = "Base URL (optional)"

    def _refresh_models(self) -> None:
        provider_value = self.query_one("#model_provider", Select).value
        if provider_value == Select.BLANK:
            provider = self.providers[0]
        else:
            provider = str(provider_value)

        input_model = _clean_optional_text(self.query_one("#model_name", Input).value)
        base_url_value = _clean_optional_text(
            self.query_one("#model_base_url", Input).value
        ) or None
        base_url = base_url_value or _PROVIDER_BASE_URL_DEFAULTS.get(provider)

        env_var = API_KEY_ENV_BY_PROVIDER.get(provider)
        entered_key = _clean_optional_text(self.query_one("#model_api_key", Input).value)

        previous_env_value = os.environ.get(env_var) if env_var else None
        should_override_env = bool(env_var and entered_key)
        if env_var and should_override_env:
            os.environ[env_var] = entered_key

        try:
            models, error_message = self.discover_models(provider, base_url, env_var)
        finally:
            if env_var and should_override_env:
                if previous_env_value is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = previous_env_value

        picker = self.query_one("#model_picker", Select)
        options: list[tuple[str, str]] = [("Manual entry", "__manual__")]
        options.extend((model_name, model_name) for model_name in models)
        picker.set_options(options)

        selected_model = input_model or _clean_optional_text(self.defaults.get("model", ""))
        if selected_model in models:
            picker.value = selected_model
        else:
            picker.value = "__manual__"

        status = self.query_one("#model_discovery_status", Static)
        if models:
            status.update(f"[green]Loaded {len(models)} live model(s)[/green]")
        elif error_message:
            status.update(f"[yellow]Live model discovery unavailable: {error_message}[/yellow]")
        else:
            status.update("[yellow]No live models returned; use manual model entry.[/yellow]")


class SettingsModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, current_view: str, refresh_seconds: float, all_cursor_workspaces: bool):
        super().__init__()
        self.current_view = current_view
        self.refresh_seconds = refresh_seconds
        self.all_cursor_workspaces = all_cursor_workspaces

    def compose(self) -> ComposeResult:
        views = ["overview", "sources", "llm", "knowledge", "settings", "console", "all"]
        default_view = self.current_view if self.current_view in views else "overview"
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("TUI Settings", classes="modal_title")
                with Horizontal(classes="field_row"):
                    yield Static("Default view", classes="field_label")
                    yield Select(
                        [(view_name, view_name) for view_name in views],
                        value=default_view,
                        allow_blank=False,
                        id="settings_view",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Refresh", classes="field_label")
                    yield Input(
                        value=str(self.refresh_seconds),
                        placeholder="seconds (>=0.2)",
                        id="settings_refresh",
                        classes="field_input",
                    )
                yield Checkbox(
                    "Include all Cursor workspaces",
                    value=self.all_cursor_workspaces,
                    id="settings_all_cursor",
                )
                yield Static("", id="settings_error")
                with Horizontal(classes="modal_actions"):
                    yield Button("Save", variant="primary", id="settings_save")
                    yield Button("Cancel", id="settings_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings_cancel":
            self.dismiss(None)
            return
        if event.button.id != "settings_save":
            return

        error_widget = self.query_one("#settings_error", Static)

        view_widget = self.query_one("#settings_view", Select)
        selected_view = view_widget.value
        if selected_view == Select.BLANK:
            error_widget.update("[red]View selection is required[/red]")
            return

        try:
            refresh_seconds = float(self.query_one("#settings_refresh", Input).value)
        except ValueError:
            error_widget.update("[red]Refresh must be a number[/red]")
            return
        if refresh_seconds < 0.2:
            error_widget.update("[red]Refresh must be >= 0.2[/red]")
            return

        self.dismiss(
            {
                "view": selected_view,
                "refresh_seconds": refresh_seconds,
                "all_cursor_workspaces": bool(
                    self.query_one("#settings_all_cursor", Checkbox).value
                ),
            }
        )


class AgentRecallTextualApp(App[None]):
    CSS = """
    #root {
        height: 100%;
        width: 100%;
    }
    #dashboard {
        height: 1fr;
        overflow: auto;
    }
    #activity {
        height: 7;
        overflow: auto;
    }
    #palette_overlay, #modal_overlay {
        align: center middle;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.55);
    }
    #palette_card {
        width: 74%;
        max-width: 96;
        height: auto;
        max-height: 78%;
        padding: 0 1;
        background: $panel;
        border: heavy $accent;
        overflow: auto;
    }
    #modal_card {
        width: 72%;
        max-width: 92;
        height: auto;
        max-height: 82%;
        padding: 1;
        background: $panel;
        border: heavy $accent;
        overflow: auto;
    }
    #palette_title, .modal_title {
        text-style: bold;
        margin-bottom: 1;
    }
    #palette_search {
        margin-bottom: 0;
    }
    #palette_options {
        height: 1fr;
        margin-bottom: 0;
    }
    #palette_hint, #setup_api_hint, #setup_repo_path {
        color: $text-muted;
    }
    .field_row {
        height: auto;
        margin: 0;
    }
    .field_label {
        width: 14;
        color: $text-muted;
        padding-top: 1;
    }
    .field_input {
        width: 1fr;
    }
    .setup_agents {
        height: auto;
        margin-bottom: 0;
    }
    .modal_actions {
        margin-top: 0;
        height: auto;
    }
    #setup_status, #model_error, #settings_error {
        margin-top: 0;
    }
    #model_discovery_status {
        margin-top: 0;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("ctrl+comma", "open_settings_modal", "Settings"),
        Binding("ctrl+r", "refresh_now", "Refresh"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        render_dashboard: Callable[..., Any],
        execute_command: Callable[[str], tuple[bool, list[str]]],
        run_setup_payload: Callable[[dict[str, object]], tuple[bool, list[str]]],
        run_model_config: Callable[
            [str | None, str | None, str | None, float | None, int | None, bool],
            list[str],
        ],
        model_defaults_provider: Callable[[], dict[str, Any]],
        setup_defaults_provider: Callable[[], dict[str, Any]],
        discover_models: DiscoverModelsFn,
        providers: list[str],
        cli_commands: list[str] | None = None,
        rich_theme: Theme | None = None,
        initial_view: str = "overview",
        refresh_seconds: float = 2.0,
        all_cursor_workspaces: bool = False,
    ):
        super().__init__()
        self._render_dashboard = render_dashboard
        self._execute_command = execute_command
        self._run_setup_payload = run_setup_payload
        self._run_model_config = run_model_config
        self._model_defaults_provider = model_defaults_provider
        self._setup_defaults_provider = setup_defaults_provider
        self._discover_models = discover_models
        self._providers = providers
        self._cli_commands = [
            command.strip() for command in (cli_commands or []) if command.strip()
        ]
        self._command_suggestions = _build_command_suggestions(self._cli_commands)
        self._rich_theme = rich_theme
        self.current_view = initial_view
        self.refresh_seconds = refresh_seconds
        self.all_cursor_workspaces = all_cursor_workspaces
        self.status = "Ready. Press Ctrl+P for commands."
        self.activity: deque[str] = deque(maxlen=24)
        self._refresh_timer = None
        self._worker_context: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="root"):
            yield Static(id="dashboard")
            yield Static(id="activity")
        yield Footer()

    def on_mount(self) -> None:
        if self._rich_theme is not None:
            self.console.push_theme(self._rich_theme)
        self._configure_refresh_timer(self.refresh_seconds)
        self._append_activity("Opened TUI. Press Ctrl+P for command palette.")
        self._refresh_dashboard_panel()

    def action_command_palette(self) -> None:
        self.action_open_command_palette()

    def action_open_command_palette(self) -> None:
        self.push_screen(
            CommandPaletteModal(self._palette_actions()),
            self._handle_palette_action,
        )

    def action_open_settings_modal(self) -> None:
        self.push_screen(
            SettingsModal(
                current_view=self.current_view,
                refresh_seconds=self.refresh_seconds,
                all_cursor_workspaces=self.all_cursor_workspaces,
            ),
            self._apply_settings_modal_result,
        )

    def action_open_setup_modal(self) -> None:
        self.push_screen(
            SetupModal(
                providers=self._providers,
                defaults=self._setup_defaults_provider(),
                discover_models=self._discover_models,
            ),
            self._apply_setup_modal_result,
        )

    def action_open_model_modal(self) -> None:
        self.push_screen(
            ModelConfigModal(
                self._providers,
                self._model_defaults_provider(),
                self._discover_models,
            ),
            self._apply_model_modal_result,
        )

    def action_refresh_now(self) -> None:
        self._refresh_dashboard_panel()
        self._append_activity("Manual refresh complete.")

    def _configure_refresh_timer(self, refresh_seconds: float) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(refresh_seconds, self._refresh_dashboard_panel)

    def _refresh_dashboard_panel(self) -> None:
        renderable = self._render_dashboard(
            all_cursor_workspaces=self.all_cursor_workspaces,
            include_banner_header=True,
            view=self.current_view,
            refresh_seconds=self.refresh_seconds,
            show_slash_console=False,
        )
        self.query_one("#dashboard", Static).update(renderable)
        self._refresh_activity_panel()

    def _refresh_activity_panel(self) -> None:
        lines = "\n".join(self.activity) if self.activity else "No activity yet."
        subtitle = (
            f"view={self.current_view} refresh={self.refresh_seconds:.1f}s "
            f"all_cursor_workspaces={'yes' if self.all_cursor_workspaces else 'no'}"
        )
        self.query_one("#activity", Static).update(
            Panel(lines, title=self.status, subtitle=subtitle)
        )

    def _append_activity(self, line: str) -> None:
        self.activity.append(line)
        self._refresh_activity_panel()

    def _palette_actions(self) -> list[PaletteAction]:
        actions = [
            PaletteAction(
                "setup",
                "Run setup wizard",
                "Suggested",
                "",
                "onboarding config setup",
            ),
            PaletteAction(
                "model",
                "Configure model",
                "Suggested",
                "",
                "provider model temperature max tokens",
            ),
            PaletteAction("settings", "Open settings", "Suggested", "", "settings preferences"),
            PaletteAction("status", "Refresh status", "Session", "", "status dashboard"),
            PaletteAction("sync", "Sync sessions", "Session", "", "sync ingest"),
            PaletteAction("compact", "Run compaction", "Session", "", "compact"),
            PaletteAction("sources", "Inspect sources", "Session", "", "sources cursor claude"),
            PaletteAction("view:overview", "Overview view", "Views", "", "view overview"),
            PaletteAction("view:sources", "Sources view", "Views", "", "view sources"),
            PaletteAction("view:llm", "LLM view", "Views", "", "view llm"),
            PaletteAction("view:knowledge", "Knowledge view", "Views", "", "view knowledge"),
            PaletteAction("view:settings", "Settings view", "Views", "", "view settings"),
            PaletteAction("view:console", "Console view", "Views", "", "view console"),
            PaletteAction("view:all", "All panels view", "Views", "", "view all"),
            PaletteAction("quit", "Quit", "Maintenance", "Ctrl+Q", "quit exit"),
        ]
        for command in self._command_suggestions:
            if command in {"tui"}:
                continue
            actions.append(
                PaletteAction(
                    f"cmd:{command}",
                    command,
                    "CLI",
                    "",
                    f"cli command {command}",
                )
            )
        return actions

    def _handle_palette_action(self, action_id: str | None) -> None:
        if not action_id:
            return
        if action_id.startswith("view:"):
            self.current_view = action_id.split(":", 1)[1]
            self.status = f"View: {self.current_view}"
            self._append_activity(f"Switched to {self.current_view} view.")
            self._refresh_dashboard_panel()
            return

        if action_id == "setup":
            self.action_open_setup_modal()
            return
        if action_id == "model":
            self.action_open_model_modal()
            return
        if action_id == "settings":
            self.action_open_settings_modal()
            return
        if action_id == "quit":
            self.exit()
            return
        if action_id.startswith("run:"):
            self._run_backend_command(action_id.split(":", 1)[1])
            return
        if action_id.startswith("cmd:"):
            self._run_backend_command(action_id.split(":", 1)[1])
            return

        command_by_action = {
            "status": "status",
            "sync": "sync --no-compact",
            "compact": "compact",
            "sources": "sources",
        }
        command = command_by_action.get(action_id)
        if command:
            self._run_backend_command(command)

    def _run_backend_command(self, command: str) -> None:
        if self._handle_local_command(command):
            return
        self._append_activity(f"> {command}")
        self.status = f"Running: {command}"
        worker = self.run_worker(
            lambda: self._execute_command(command),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "command"

    def _handle_local_command(self, raw: str) -> bool:
        command = raw.strip()
        if not command:
            return True
        normalized = command[1:] if command.startswith("/") else command
        try:
            parts = shlex.split(normalized)
        except ValueError as exc:
            self.status = "Invalid command"
            self._append_activity(f"Invalid command: {exc}")
            return True
        if not parts:
            return True

        action = parts[0].lower()
        second = parts[1].lower() if len(parts) > 1 else ""

        if action in {"setup", "onboarding"}:
            self.action_open_setup_modal()
            self.status = "Setup"
            return True

        if action == "model":
            self.action_open_model_modal()
            self.status = "Model configuration"
            return True

        if action == "config" and second == "setup":
            self.action_open_setup_modal()
            self.status = "Setup"
            return True

        if action == "config" and second == "model":
            self.action_open_model_modal()
            self.status = "Model configuration"
            return True

        if action in {"settings", "preferences"}:
            self.action_open_settings_modal()
            self.status = "Settings"
            self._append_activity("Opened settings dialog.")
            return True

        if action == "config" and second in {"settings", "preferences", "prefs"}:
            self.action_open_settings_modal()
            self.status = "Settings"
            self._append_activity("Opened settings dialog.")
            return True

        if action in {"view", "menu"}:
            valid = {"overview", "sources", "llm", "knowledge", "settings", "console", "all"}
            if len(parts) == 1:
                self._append_activity(
                    f"Current view: {self.current_view}. Available: {', '.join(sorted(valid))}"
                )
                return True
            requested = parts[1].strip().lower()
            if requested not in valid:
                self._append_activity(f"Unknown view '{requested}'.")
                return True
            self.current_view = requested
            self.status = f"View: {requested}"
            self._append_activity(f"Switched to {requested} view.")
            self._refresh_dashboard_panel()
            return True

        return False

    def _apply_setup_modal_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        self.status = "Applying setup"
        self._append_activity("Applying setup from modal...")
        worker = self.run_worker(
            lambda: self._run_setup_payload(result),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "setup"

    def _apply_model_modal_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        self.status = "Applying model configuration"
        self._append_activity("Applying model configuration...")
        worker = self.run_worker(
            lambda: self._run_model_config(
                result.get("provider"),
                result.get("model"),
                result.get("base_url"),
                result.get("temperature"),
                result.get("max_tokens"),
                bool(result.get("validate", True)),
            ),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "model"

    def _apply_settings_modal_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        self.current_view = str(result.get("view", self.current_view))
        self.refresh_seconds = float(result.get("refresh_seconds", self.refresh_seconds))
        self.all_cursor_workspaces = bool(
            result.get("all_cursor_workspaces", self.all_cursor_workspaces)
        )
        self._configure_refresh_timer(self.refresh_seconds)
        self.status = "Settings updated"
        self._append_activity("Settings updated.")
        self._refresh_dashboard_panel()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        worker_key = id(event.worker)
        context = self._worker_context.get(worker_key)
        if context is None:
            return

        if event.state == WorkerState.RUNNING:
            return

        if event.state == WorkerState.SUCCESS:
            self._handle_worker_success(context, event.worker.result)
            self._worker_context.pop(worker_key, None)
            return

        if event.state == WorkerState.CANCELLED:
            self.status = "Operation cancelled"
            self._append_activity("Previous operation cancelled.")
            self._worker_context.pop(worker_key, None)
            self._refresh_dashboard_panel()
            return

        if event.state == WorkerState.ERROR:
            self.status = "Operation failed"
            error = event.worker.error
            if error is None:
                self._append_activity("Operation failed with an unknown error.")
            else:
                self._append_activity(f"Error: {error}")
            self._worker_context.pop(worker_key, None)
            self._refresh_dashboard_panel()

    def _handle_worker_success(self, context: str, result: object) -> None:
        if context == "command":
            should_exit = False
            lines: list[str] = []
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], bool)
                and isinstance(result[1], list)
            ):
                should_exit = result[0]
                lines = [line for line in result[1] if isinstance(line, str)]

            for line in lines:
                cleaned = _strip_rich_markup(line).strip()
                if cleaned:
                    self._append_activity(cleaned)

            self.status = "Last command executed"
            self._refresh_dashboard_panel()
            if should_exit:
                self.exit()
            return

        if context == "setup":
            changed = False
            lines: list[str] = []
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], bool)
                and isinstance(result[1], list)
            ):
                changed = result[0]
                lines = [line for line in result[1] if isinstance(line, str)]

            for line in lines:
                cleaned = _strip_rich_markup(line).strip()
                if cleaned:
                    self._append_activity(cleaned)

            self.status = "Setup completed" if changed else "Setup unchanged"
            self._append_activity(
                "Setup completed." if changed else "Setup already complete for this repository."
            )
            self._refresh_dashboard_panel()
            return

        if context == "model":
            lines = (
                [line for line in result if isinstance(line, str)]
                if isinstance(result, list)
                else []
            )
            for line in lines:
                cleaned = _strip_rich_markup(line).strip()
                if cleaned:
                    self._append_activity(cleaned)

            self.status = "Model configuration updated"
            self._append_activity("Model configuration updated.")
            self._refresh_dashboard_panel()
