from __future__ import annotations

import os
import re
import shlex
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.theme import Theme
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Footer, Header, Input, OptionList, Select, Static
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from agent_recall.core.onboarding import API_KEY_ENV_BY_PROVIDER
from agent_recall.ingest.sources import SOURCE_DEFINITIONS

DiscoverModelsFn = Callable[[str, str | None, str | None], tuple[list[str], str | None]]
ThemeDefaultsFn = Callable[[], tuple[list[str], str]]
ThemeRuntimeFn = Callable[[], tuple[str, Theme]]
ExecuteCommandFn = Callable[[str, int, int], tuple[bool, list[str]]]
ListSessionsForPickerFn = Callable[[int, bool], list[dict[str, Any]]]
ThemeResolveFn = Callable[[str], Theme | None]
_LOADING_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

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


def _source_checkbox_id(source_name: str) -> str:
    return f"setup_agent_{source_name.replace('-', '_')}"


def _build_command_suggestions(cli_commands: list[str]) -> list[str]:
    base = [
        "help",
        "status",
        "open",
        "run",
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


def _normalize_palette_command(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned.startswith("/"):
        cleaned = cleaned[1:].strip()
    return cleaned


def _is_palette_cli_command_redundant(command: str) -> bool:
    normalized = _normalize_palette_command(command)
    if not normalized:
        return True

    # Distinct subcommands that provide unique value in palette UX.
    distinct_subcommands = {
        "theme show",
    }
    if normalized in distinct_subcommands:
        return False

    exact = {
        "open",
        "status",
        "run",
        "sync",
        "compact",
        "sources",
        "sessions",
        "settings",
        "preferences",
        "setup",
        "model",
        "theme",
    }
    if normalized in exact:
        return True

    prefixes = (
        "view ",
        "menu ",
        "theme ",
        "config setup",
        "config model",
        "config settings",
        "config preferences",
    )
    return normalized.startswith(prefixes)


def _is_knowledge_run_command(command: str) -> bool:
    value = command.strip()
    if not value:
        return False
    try:
        parts = shlex.split(value)
    except ValueError:
        return False
    if not parts:
        return False
    action = parts[0].lower()
    if action in {"run", "compact"}:
        return True
    if action == "sync" and "--no-compact" not in parts:
        return True
    return False


@dataclass(frozen=True)
class PaletteAction:
    action_id: str
    title: str
    description: str
    group: str
    shortcut: str = ""
    binding: str = ""
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
                with Horizontal(id="palette_header"):
                    yield Static("Commands", id="palette_title")
                    yield Static("esc", id="palette_close_hint")
                yield Input(
                    placeholder="Search actions, or type a CLI command and press Enter",
                    id="palette_search",
                )
                yield OptionList(id="palette_options")
                yield Static(
                    "Enter to run · Type any CLI command to execute directly",
                    id="palette_hint",
                )

    def on_mount(self) -> None:
        self.query_one("#palette_search", Input).focus()
        self._rebuild_options()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "palette_search":
            self.query_text = event.value
            self._rebuild_options()

    def on_resize(self, event: events.Resize) -> None:
        _ = event
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
            haystack = (
                f"{action.title} {action.description} {action.group} "
                f"{action.keywords} {action.shortcut} {action.binding}"
            ).lower()
            if query and query not in haystack:
                continue
            grouped[action.group].append(action)

        option_list = self.query_one("#palette_options", OptionList)
        list_width = int(option_list.size.width) if int(option_list.size.width or 0) > 0 else 72
        options: list[Option] = []
        if query:
            options.append(Option("[dim]Run typed command[/dim]", id="heading:run", disabled=True))
            options.append(
                Option(
                    f"Run exactly: [bold]{self.query_text.strip()}[/bold]",
                    id="action:run-query",
                )
            )
        grouped_order = [
            ("Core", "Suggested"),
            ("Sessions", "Session"),
            ("Views", "Views"),
            ("Settings", "Settings"),
            ("System", "System"),
        ]
        for index, (group, label) in enumerate(grouped_order):
            items = grouped.get(group, [])
            if not items:
                continue

            if not query:
                if index > 0:
                    options.append(Option("", id=f"heading:spacer:{group}", disabled=True))
                options.append(
                    Option(
                        f"[bold accent]{label}[/bold accent]",
                        id=f"heading:{group}",
                        disabled=True,
                    )
                )
            for action in items:
                line = action.title
                if action.shortcut:
                    line = f"{line} [dim]{action.shortcut}[/dim]"
                if query:
                    line = f"{line} [dim]· {action.description}[/dim]"
                if action.binding:
                    left_plain = _strip_rich_markup(line)
                    binding_plain = _strip_rich_markup(action.binding)
                    spacer_width = max(2, list_width - len(left_plain) - len(binding_plain) - 2)
                    line = f"{line}{' ' * spacer_width}[dim]{action.binding}[/dim]"
                options.append(
                    Option(
                        line,
                        id=f"action:{action.action_id}",
                    )
                )

        option_list.set_options(options)

        for index, option in enumerate(options):
            if not option.disabled:
                option_list.highlighted = index
                break


class SetupModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, *, defaults: dict[str, Any]):
        super().__init__()
        self.defaults = defaults

    def compose(self) -> ComposeResult:
        selected_agents = {
            source for source in self.defaults.get("selected_agents", []) if isinstance(source, str)
        }
        repo_path = str(self.defaults.get("repository_path", Path.cwd()))

        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("Repository Setup (1/2)", classes="modal_title")
                yield Static(
                    "Step 1 of 2 · Repository and session sources",
                    classes="modal_subtitle",
                )
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
                with Horizontal(classes="setup_agents field_row"):
                    for source in SOURCE_DEFINITIONS:
                        yield Checkbox(
                            source.display_name,
                            value=source.name in selected_agents,
                            id=_source_checkbox_id(source.name),
                        )
                yield Static("", id="setup_status")
                with Horizontal(classes="modal_actions"):
                    yield Button("Next", variant="primary", id="setup_next")
                    yield Button("Cancel", id="setup_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup_cancel":
            self.dismiss(None)
            return
        if event.button.id != "setup_next":
            return

        status = self.query_one("#setup_status", Static)

        repository_verified = bool(self.query_one("#setup_repository_verified", Checkbox).value)
        if not repository_verified:
            status.update("[red]Repository must be confirmed[/red]")
            return

        selected_agents: list[str] = []
        for source in SOURCE_DEFINITIONS:
            checkbox = self.query_one(f"#{_source_checkbox_id(source.name)}", Checkbox)
            if checkbox.value:
                selected_agents.append(source.name)
        if not selected_agents:
            status.update("[red]Choose at least one agent source[/red]")
            return

        self.dismiss(
            {
                "force": bool(self.query_one("#setup_force", Checkbox).value),
                "repository_verified": repository_verified,
                "selected_agents": selected_agents,
            }
        )


class ModelConfigModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(
        self,
        providers: list[str],
        defaults: dict[str, Any],
        discover_models: DiscoverModelsFn,
        *,
        onboarding_step: bool = False,
    ):
        super().__init__()
        self.providers = providers
        self.defaults = defaults
        self.discover_models = discover_models
        self.onboarding_step = onboarding_step

    def compose(self) -> ComposeResult:
        default_provider = _clean_optional_text(self.defaults.get("provider", self.providers[0]))
        if default_provider not in self.providers:
            default_provider = self.providers[0]
        base_url = _clean_optional_text(self.defaults.get("base_url", ""))
        if not base_url:
            base_url = _PROVIDER_BASE_URL_DEFAULTS.get(default_provider, "")
        validate_default = bool(self.defaults.get("validate", not self.onboarding_step))
        title = "Model Setup (2/2)" if self.onboarding_step else "Model Configuration"
        primary_label = "Finish setup" if self.onboarding_step else "Apply"
        subtitle = (
            "Step 2 of 2 · Provider, model, and generation defaults"
            if self.onboarding_step
            else "Provider and generation defaults"
        )
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static(title, classes="modal_title")
                yield Static(subtitle, classes="modal_subtitle")
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
                yield Static("", id="model_api_hint")
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
                yield Checkbox("Validate after apply", value=validate_default, id="model_validate")
                yield Static("", id="model_discovery_status")
                yield Static("", id="model_error")
                with Horizontal(classes="modal_actions"):
                    yield Button("Refresh models", id="model_refresh")
                    if self.onboarding_step:
                        yield Button("Back", id="model_back")
                    yield Button(primary_label, variant="primary", id="model_apply")
                    yield Button("Cancel", id="model_cancel")

    def on_mount(self) -> None:
        self._apply_base_url_default(set_default_base_url=False)
        self._update_api_key_hint()
        self._refresh_models()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model_provider":
            self._apply_base_url_default(set_default_base_url=True)
            self._update_api_key_hint()
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
        if event.button.id == "model_back":
            self.dismiss({"_action": "back"})
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
                "api_key": _clean_optional_text(self.query_one("#model_api_key", Input).value)
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

    def _update_api_key_hint(self) -> None:
        provider_value = self.query_one("#model_provider", Select).value
        provider = self.providers[0] if provider_value == Select.BLANK else str(provider_value)

        env_var = API_KEY_ENV_BY_PROVIDER.get(provider)
        hint = self.query_one("#model_api_hint", Static)
        api_input = self.query_one("#model_api_key", Input)

        if not env_var:
            hint.update("[dim]This provider typically does not require an API key.[/dim]")
            api_input.placeholder = "Provider API key (optional)"
            return

        if os.environ.get(env_var):
            hint.update(
                f"[dim]Using shared {env_var} from local secrets. Leave blank to keep it.[/dim]"
            )
            api_input.placeholder = f"Leave blank to keep stored {env_var}"
            return

        hint.update(f"[dim]No {env_var} found. Enter one to store it for all repositories.[/dim]")
        api_input.placeholder = f"API key ({env_var}) optional"

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


class ThemePickerModal(ModalScreen[dict[str, str] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close"), Binding("enter", "apply", "Apply")]

    def __init__(
        self,
        themes: list[str],
        current_theme: str,
        on_preview: Callable[[str], None],
    ):
        super().__init__()
        self.themes = themes
        self.current_theme = current_theme
        self.on_preview = on_preview

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("Theme", classes="modal_title")
                yield Static(
                    "Preview updates as you move. Enter applies, Esc cancels.",
                    classes="modal_subtitle",
                )
                yield OptionList(id="theme_modal_options")
                yield Static("Use ↑↓ to preview themes", id="theme_modal_hint")

    def on_mount(self) -> None:
        option_list = self.query_one("#theme_modal_options", OptionList)
        options = [
            Option(
                (
                    f"{theme_name} [green]✓ Current[/green]"
                    if theme_name == self.current_theme
                    else theme_name
                ),
                id=f"theme:{theme_name}",
            )
            for theme_name in self.themes
        ]
        option_list.set_options(options)

        selected_index = 0
        for index, option in enumerate(option_list.options):
            option_id = option.id or ""
            if option_id == f"theme:{self.current_theme}":
                selected_index = index
                break
        option_list.highlighted = selected_index
        option_list.focus()

        if self.themes:
            self.on_preview(self.themes[selected_index])

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "theme_modal_options":
            return
        option_id = event.option.id or ""
        if not option_id.startswith("theme:"):
            return
        theme_name = option_id.split(":", 1)[1]
        if theme_name:
            self.on_preview(theme_name)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "theme_modal_options":
            return
        option_id = event.option.id or ""
        if not option_id.startswith("theme:"):
            return
        self.dismiss({"theme": option_id.split(":", 1)[1]})

    def action_apply(self) -> None:
        option_list = self.query_one("#theme_modal_options", OptionList)
        highlighted = option_list.highlighted
        if highlighted is None:
            self.dismiss(None)
            return
        option = option_list.get_option_at_index(highlighted)
        option_id = option.id or ""
        if option_id.startswith("theme:"):
            self.dismiss({"theme": option_id.split(":", 1)[1]})
            return
        self.dismiss(None)


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


class SessionRunModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
        Binding("space", "toggle_selection", "Toggle"),
        Binding("enter", "apply", "Run"),
    ]

    def __init__(self, sessions: list[dict[str, Any]]):
        super().__init__()
        self.sessions = sessions
        self.filter_query = ""
        self.selected_session_ids: set[str] = set()

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("Run Knowledge Update", classes="modal_title")
                yield Input(
                    placeholder="Filter conversations...",
                    id="run_sessions_filter",
                    classes="field_input",
                )
                yield OptionList(id="run_sessions_list")
                yield Static("", id="run_sessions_status")
                with Horizontal(classes="modal_actions"):
                    yield Button("Run Selected", variant="primary", id="run_sessions_apply")
                    yield Button("Cancel", id="run_sessions_cancel")

    def on_mount(self) -> None:
        self.query_one("#run_sessions_filter", Input).focus()
        self._rebuild_options()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "run_sessions_filter":
            return
        self.filter_query = event.value
        self._rebuild_options()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "run_sessions_list":
            return
        self._toggle_highlighted()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run_sessions_cancel":
            self.dismiss(None)
            return
        if event.button.id == "run_sessions_apply":
            self._submit()

    def action_toggle_selection(self) -> None:
        self._toggle_highlighted()

    def action_apply(self) -> None:
        self._submit()

    def _toggle_highlighted(self) -> None:
        option_list = self.query_one("#run_sessions_list", OptionList)
        highlighted = option_list.highlighted
        if highlighted is None:
            return
        option = option_list.get_option_at_index(highlighted)
        option_id = option.id or ""
        if not option_id.startswith("session:"):
            return
        session_id = option_id.split(":", 1)[1]
        if not session_id:
            return
        if session_id in self.selected_session_ids:
            self.selected_session_ids.remove(session_id)
        else:
            self.selected_session_ids.add(session_id)
        self._rebuild_options()

    def _submit(self) -> None:
        ordered_selection = [
            str(session.get("session_id") or "")
            for session in self.sessions
            if str(session.get("session_id") or "") in self.selected_session_ids
        ]
        ordered_selection = [item for item in ordered_selection if item]
        if not ordered_selection:
            self.query_one("#run_sessions_status", Static).update(
                "[red]Select at least one conversation.[/red]"
            )
            return
        self.dismiss({"session_ids": ordered_selection})

    def _rebuild_options(self) -> None:
        query = self.filter_query.strip().lower()
        visible_sessions: list[dict[str, Any]] = []
        for session in self.sessions:
            title = str(session.get("title") or "").lower()
            started = str(session.get("started") or "").lower()
            source = str(session.get("source") or "").lower()
            if query and query not in f"{title} {started} {source}":
                continue
            visible_sessions.append(session)

        option_list = self.query_one("#run_sessions_list", OptionList)
        options: list[Option] = []
        for session in visible_sessions:
            session_id = str(session.get("session_id") or "")
            if not session_id:
                continue
            selected = session_id in self.selected_session_ids
            options.append(
                Option(
                    self._line_for_session(session, selected),
                    id=f"session:{session_id}",
                )
            )

        if not options:
            options = [Option("No matching conversations.", id="session:", disabled=True)]

        option_list.set_options(options)
        for index, option in enumerate(options):
            if option.disabled:
                continue
            option_list.highlighted = index
            break

        self.query_one("#run_sessions_status", Static).update(
            f"[dim]{len(visible_sessions)} shown · {len(self.selected_session_ids)} selected"
            " · Space toggles, Enter runs[/dim]"
        )

    def _line_for_session(self, session: dict[str, Any], selected: bool) -> str:
        marker = "[green]✓[/green]" if selected else "[dim]○[/dim]"
        title = str(session.get("title") or "Untitled conversation")
        started = str(session.get("started") or "-")
        message_count = int(session.get("message_count", 0))
        processed = "processed" if bool(session.get("processed")) else "new"
        return (
            f"{marker} {title} "
            f"[dim]({started} · {message_count} msg · {processed})[/dim]"
        )


class AgentRecallTextualApp(App[None]):
    CSS = """
    #root {
        height: 1fr;
        width: 100%;
        align: center top;
    }
    #app_shell {
        width: 96%;
        max-width: 210;
        height: 100%;
    }
    #dashboard {
        height: auto;
        overflow: auto;
        min-height: 0;
    }
    #activity {
        height: 1fr;
        min-height: 4;
        overflow: auto;
    }
    #activity_log {
        height: 1fr;
        overflow: auto;
    }
    #activity_result_list {
        height: 1fr;
        overflow: auto;
        display: none;
    }
    #palette_overlay, #modal_overlay {
        align: center middle;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.30);
    }
    #palette_card {
        width: 74%;
        max-width: 94;
        height: auto;
        max-height: 78%;
        padding: 1 3;
        background: $panel;
        border: none;
        overflow: auto;
    }
    #modal_card {
        width: 64%;
        max-width: 84;
        height: auto;
        max-height: 82%;
        padding: 1 2;
        background: $panel;
        border: round $accent;
        overflow: auto;
    }
    #palette_header {
        layout: horizontal;
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }
    #palette_title, .modal_title {
        text-style: bold;
        margin-bottom: 0;
    }
    #palette_title {
        width: auto;
        text-wrap: nowrap;
    }
    #palette_close_hint {
        width: 1fr;
        color: $text-muted;
        text-align: right;
        text-wrap: nowrap;
    }
    .modal_subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }
    .modal_title {
        padding-top: 1;
    }
    #palette_search {
        margin-bottom: 1;
    }
    #palette_options {
        height: 1fr;
        margin-bottom: 1;
    }
    #palette_hint, #setup_api_hint, #setup_repo_path, #model_api_hint {
        color: $text-muted;
    }
    #palette_hint {
        margin-top: 1;
    }
    .field_row {
        height: auto;
        margin: 0 0 1 0;
    }
    .field_label {
        width: 15;
        color: $text-muted;
        padding-top: 1;
    }
    .field_input {
        width: 1fr;
    }
    .setup_agents {
        height: auto;
        margin-bottom: 1;
    }
    .modal_actions {
        margin-top: 1;
        padding-top: 1;
        height: auto;
    }
    #setup_status, #model_api_hint, #model_error, #settings_error {
        margin-top: 0;
    }
    #model_discovery_status {
        margin-top: 0;
        color: $text-muted;
    }
    #theme_picker_hint, #theme_modal_hint {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("f1", "open_command_palette", "Commands", show=False),
        Binding("ctrl+g", "open_settings_modal", "Settings"),
        Binding("ctrl+r", "refresh_now", "Refresh"),
        Binding("ctrl+k", "run_knowledge_update", "Run"),
        Binding("ctrl+y", "sync_conversations", "Sync"),
        Binding("ctrl+t", "open_theme_modal", "Theme"),
        Binding("ctrl+c", "request_quit", "Quit", show=False, priority=True),
        Binding("escape", "close_inline_picker", "Close picker", show=False),
        Binding("ctrl+q", "request_quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        *,
        render_dashboard: Callable[..., Any],
        execute_command: ExecuteCommandFn,
        list_sessions_for_picker: ListSessionsForPickerFn,
        run_setup_payload: Callable[[dict[str, object]], tuple[bool, list[str]]],
        run_model_config: Callable[
            [str | None, str | None, str | None, float | None, int | None, bool],
            list[str],
        ],
        theme_defaults_provider: ThemeDefaultsFn,
        theme_runtime_provider: ThemeRuntimeFn | None = None,
        theme_resolve_provider: ThemeResolveFn | None = None,
        model_defaults_provider: Callable[[], dict[str, Any]],
        setup_defaults_provider: Callable[[], dict[str, Any]],
        discover_models: DiscoverModelsFn,
        providers: list[str],
        cli_commands: list[str] | None = None,
        rich_theme: Theme | None = None,
        initial_view: str = "overview",
        refresh_seconds: float = 2.0,
        all_cursor_workspaces: bool = False,
        onboarding_required: bool = False,
    ):
        super().__init__()
        self._render_dashboard = render_dashboard
        self._execute_command = execute_command
        self._list_sessions_for_picker = list_sessions_for_picker
        self._run_setup_payload = run_setup_payload
        self._run_model_config = run_model_config
        self._theme_defaults_provider = theme_defaults_provider
        self._theme_runtime_provider = theme_runtime_provider
        self._theme_resolve_provider = theme_resolve_provider
        self._model_defaults_provider = model_defaults_provider
        self._setup_defaults_provider = setup_defaults_provider
        self._discover_models = discover_models
        self._providers = providers
        self._cli_commands = [
            command.strip() for command in (cli_commands or []) if command.strip()
        ]
        self._command_suggestions = _build_command_suggestions(self._cli_commands)
        self._rich_theme = rich_theme
        self._active_theme_name: str | None = None
        self.current_view = initial_view
        self.refresh_seconds = refresh_seconds
        self.all_cursor_workspaces = all_cursor_workspaces
        self.onboarding_required = onboarding_required
        self.status = "Ready. Press Ctrl+P for commands."
        self.activity: deque[str] = deque(maxlen=2000)
        self._theme_preview_active = False
        self._theme_commit_inflight = False
        self._theme_preview_origin: str | None = None
        self._result_list_open = False
        self._refresh_timer = None
        self._resize_refresh_timer = None
        self._worker_context: dict[int, str] = {}
        self._knowledge_run_workers: set[int] = set()
        self._pending_setup_payload: dict[str, Any] | None = None
        self._last_activity_render: tuple[str, str, str] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="root"):
            with Vertical(id="app_shell"):
                yield Static(id="dashboard")
                with Vertical(id="activity"):
                    yield Static(id="activity_log")
                    yield OptionList(id="activity_result_list")
        yield Footer()

    def on_mount(self) -> None:
        if self._theme_runtime_provider is not None:
            self._sync_runtime_theme()
        elif self._rich_theme is not None:
            self.console.push_theme(self._rich_theme)
            self._active_theme_name = "__initial__"
        self._configure_refresh_timer(self.refresh_seconds)
        self._append_activity("TUI ready. Press Ctrl+P for commands.")
        self._refresh_dashboard_panel()
        if self.onboarding_required:
            self.status = "Onboarding required"
            self._append_activity("Onboarding required. Opening setup wizard...")
            self.call_after_refresh(self.action_open_setup_modal)

    def on_unmount(self) -> None:
        self._teardown_runtime()

    def action_command_palette(self) -> None:
        self.action_open_command_palette()

    def action_request_quit(self) -> None:
        self.status = "Closing..."
        self._append_activity("Stopping background operations...")
        self._teardown_runtime()
        self.exit()

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
        self._pending_setup_payload = None
        self._open_setup_step_one_modal(self._setup_defaults_provider())

    def action_open_model_modal(self) -> None:
        self.push_screen(
            ModelConfigModal(
                self._providers,
                self._model_defaults_provider(),
                self._discover_models,
            ),
            self._apply_model_modal_result,
        )

    def _open_setup_step_one_modal(self, defaults: dict[str, Any]) -> None:
        self.push_screen(
            SetupModal(defaults=defaults),
            self._apply_setup_modal_result,
        )

    def _open_setup_step_two_modal(self, defaults: dict[str, Any]) -> None:
        self.push_screen(
            ModelConfigModal(
                self._providers,
                defaults,
                self._discover_models,
                onboarding_step=True,
            ),
            self._apply_setup_model_modal_result,
        )

    def action_close_inline_picker(self) -> None:
        if self._result_list_open:
            self._close_inline_result_list(announce=False)

    def on_resize(self, event: events.Resize) -> None:
        _ = event
        if self._resize_refresh_timer is not None:
            self._resize_refresh_timer.stop()
        # Debounce resize-driven refreshes to avoid event-loop saturation.
        self._resize_refresh_timer = self.set_timer(0.12, self._flush_resize_refresh)

    def _flush_resize_refresh(self) -> None:
        self._resize_refresh_timer = None
        self._refresh_dashboard_panel()

    def action_refresh_now(self) -> None:
        self._refresh_dashboard_panel()
        self._append_activity("Manual refresh complete.")

    def action_sync_conversations(self) -> None:
        self._run_backend_command("sync --no-compact")

    def action_run_knowledge_update(self) -> None:
        self._run_backend_command("run")

    def _configure_refresh_timer(self, refresh_seconds: float) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(refresh_seconds, self._refresh_dashboard_panel)

    def _teardown_runtime(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None
        if self._resize_refresh_timer is not None:
            self._resize_refresh_timer.stop()
            self._resize_refresh_timer = None
        self.workers.cancel_all()
        self._worker_context.clear()
        self._knowledge_run_workers.clear()

    def _refresh_dashboard_panel(self) -> None:
        self._sync_runtime_theme()
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
        history_lines = list(self.activity) if self.activity else ["No activity yet."]
        lines = "\n".join(history_lines)
        subtitle = f"{self.current_view} · {self.refresh_seconds:.1f}s · Ctrl+P commands"
        title = self.status
        if self._knowledge_run_workers:
            frame_index = int(time.time() * 6) % len(_LOADING_FRAMES)
            frame = _LOADING_FRAMES[frame_index]
            title = f"Knowledge run in progress {frame} Loading"
            subtitle = f"{subtitle} · running synthesis"
        if self._worker_context:
            title = f"{title} …"
        render_key = (title, subtitle, lines)
        if render_key == self._last_activity_render:
            return

        activity_widget = self.query_one("#activity_log", Static)
        was_at_bottom = activity_widget.is_vertical_scroll_end
        previous_scroll_x = activity_widget.scroll_x
        previous_scroll_y = activity_widget.scroll_y

        activity_widget.update(Panel(lines, title=title, subtitle=subtitle))
        self._last_activity_render = render_key

        if was_at_bottom:
            activity_widget.scroll_end(animate=False)
            return

        activity_widget.scroll_to(
            x=previous_scroll_x,
            y=previous_scroll_y,
            animate=False,
            force=True,
        )

    def _append_activity(self, line: str) -> None:
        self.activity.append(line)
        self._refresh_activity_panel()

    def _show_inline_result_list(self, lines: list[str]) -> None:
        if not lines:
            return

        picker = self.query_one("#activity_result_list", OptionList)
        picker.set_options(
            [Option(line, id=f"output:{index}") for index, line in enumerate(lines, start=1)]
        )
        picker.highlighted = 0
        picker.display = True
        self.query_one("#activity_log", Static).display = False
        self._result_list_open = True
        self.status = "Command output list"
        picker.focus()

    def _close_inline_result_list(self, announce: bool = True) -> None:
        if not self._result_list_open:
            return
        picker = self.query_one("#activity_result_list", OptionList)
        picker.display = False
        self.query_one("#activity_log", Static).display = True
        self._result_list_open = False
        if announce:
            self._append_activity("Closed command output list.")

    def _sync_runtime_theme(self) -> None:
        if self._theme_preview_active:
            return
        if self._theme_runtime_provider is None:
            return
        try:
            theme_name, theme = self._theme_runtime_provider()
        except Exception:  # noqa: BLE001
            return
        self._apply_theme(theme_name, theme)

    def _apply_theme(self, theme_name: str, theme: Theme) -> None:
        if self._active_theme_name == theme_name:
            return
        if self._active_theme_name is not None:
            self.console.pop_theme()
        self.console.push_theme(theme)
        self._active_theme_name = theme_name

    def action_open_theme_modal(self) -> None:
        themes, current_theme = self._theme_defaults_provider()
        if not themes:
            self.status = "No themes available"
            self._append_activity("No themes available.")
            return

        if self._result_list_open:
            self._close_inline_result_list(announce=False)

        runtime_theme_name = current_theme
        if self._theme_runtime_provider is not None:
            try:
                runtime_theme_name, _ = self._theme_runtime_provider()
            except Exception:  # noqa: BLE001
                runtime_theme_name = current_theme
        selected_theme = runtime_theme_name if runtime_theme_name in themes else current_theme

        self._theme_preview_origin = runtime_theme_name
        self._theme_preview_active = True
        self._theme_commit_inflight = False
        self.status = "Theme picker"
        self._append_activity("Theme picker opened. Preview follows selection.")
        self.push_screen(
            ThemePickerModal(
                themes,
                selected_theme,
                self._preview_theme_from_modal,
            ),
            self._apply_theme_modal_result,
        )

    def _resolve_theme_by_name(self, theme_name: str) -> Theme | None:
        if not theme_name.strip():
            return None
        if self._theme_resolve_provider is None:
            return None
        try:
            return self._theme_resolve_provider(theme_name)
        except Exception:  # noqa: BLE001
            return None

    def _preview_theme_from_modal(self, theme_name: str) -> None:
        theme = self._resolve_theme_by_name(theme_name)
        if theme is None:
            return
        self._apply_theme(theme_name, theme)
        self._refresh_dashboard_panel()

    def _restore_preview_origin(self) -> None:
        origin = self._theme_preview_origin
        if not origin:
            return
        theme = self._resolve_theme_by_name(origin)
        if theme is None:
            return
        self._apply_theme(origin, theme)
        self._refresh_dashboard_panel()

    def _apply_theme_modal_result(self, result: dict[str, str] | None) -> None:
        if result is None:
            self._theme_preview_active = False
            self._theme_commit_inflight = False
            self._restore_preview_origin()
            self._theme_preview_origin = None
            self.status = "Theme unchanged"
            self._append_activity("Theme picker closed without changes.")
            return

        selected_theme = str(result.get("theme", "")).strip()
        if not selected_theme:
            self._theme_preview_active = False
            self._theme_commit_inflight = False
            self._restore_preview_origin()
            self._theme_preview_origin = None
            self.status = "Theme unchanged"
            return

        self.status = f"Applying theme: {selected_theme}"
        self._append_activity(f"Applying theme '{selected_theme}'...")
        self._theme_commit_inflight = True
        self._run_backend_command(
            f"theme set {shlex.quote(selected_theme)}",
            bypass_local=True,
        )

    def _finalize_theme_commit(self, *, success: bool) -> None:
        if not self._theme_commit_inflight and not self._theme_preview_active:
            return
        if success:
            self._theme_commit_inflight = False
            self._theme_preview_active = False
            self._theme_preview_origin = None
            return
        self._theme_commit_inflight = False
        self._theme_preview_active = False
        self._restore_preview_origin()
        self._theme_preview_origin = None

    def _palette_actions(self) -> list[PaletteAction]:
        actions = [
            PaletteAction(
                "setup",
                "Setup",
                "Configure sources and model defaults for this repo",
                "Core",
                shortcut="configure repository setup",
                keywords="onboarding setup",
            ),
            PaletteAction(
                "knowledge-run",
                "Run Knowledge Update",
                "Ingest conversations and synthesize GUARDRAILS, STYLE, and RECENT",
                "Core",
                shortcut="ingest + synthesize",
                binding="Ctrl+K",
                keywords="run compact synthesis llm",
            ),
            PaletteAction(
                "run:select",
                "Run Selected Conversations",
                "Choose specific conversations for a targeted knowledge update",
                "Core",
                shortcut="targeted run",
                keywords="sessions select run llm",
            ),
            PaletteAction(
                "sync",
                "Sync Conversations",
                "Ingest from enabled sources without running synthesis",
                "Core",
                shortcut="ingest only",
                binding="Ctrl+Y",
                keywords="sync ingest",
            ),
            PaletteAction(
                "status",
                "Refresh Dashboard",
                "Reload all dashboard panels now",
                "Core",
                shortcut="refresh now",
                binding="Ctrl+R",
                keywords="status dashboard refresh",
            ),
            PaletteAction(
                "sources",
                "Source Health",
                "Check source availability and discovered conversation counts",
                "Sessions",
                shortcut="availability + counts",
                keywords="sources cursor claude",
            ),
            PaletteAction(
                "theme",
                "Theme",
                "Switch themes instantly with arrows and Enter",
                "Sessions",
                shortcut="preview + apply",
                binding="Ctrl+T",
                keywords="theme list set",
            ),
            PaletteAction(
                "sessions",
                "Conversations",
                "Browse discovered conversations for this repository",
                "Sessions",
                shortcut="browse conversation list",
                keywords="sessions conversations history",
            ),
            PaletteAction(
                "view:overview",
                "Overview",
                "High-level repository status and health",
                "Views",
                keywords="view overview",
            ),
            PaletteAction(
                "view:sources",
                "Sources View",
                "Source connectivity and ingestion status",
                "Views",
                keywords="view sources",
            ),
            PaletteAction(
                "view:llm",
                "LLM View",
                "Provider, model, and synthesis configuration",
                "Views",
                keywords="view llm",
            ),
            PaletteAction(
                "view:knowledge",
                "Knowledge View",
                "Knowledge base artifacts and indexed chunks",
                "Views",
                keywords="view knowledge",
            ),
            PaletteAction(
                "view:settings",
                "Settings View",
                "Runtime and interface settings",
                "Views",
                keywords="view settings",
            ),
            PaletteAction(
                "view:console",
                "Console View",
                "Recent command output and activity history",
                "Views",
                keywords="view console",
            ),
            PaletteAction(
                "view:all",
                "All Views",
                "Show all dashboard panels together",
                "Views",
                keywords="view all",
            ),
            PaletteAction(
                "model",
                "Model Preferences",
                "Adjust provider, model, base URL, and generation defaults",
                "Settings",
                shortcut="provider + model config",
                keywords="provider model temperature max tokens",
            ),
            PaletteAction(
                "settings",
                "Workspace Preferences",
                "Change default view, refresh speed, and workspace scope",
                "Settings",
                shortcut="refresh/view/workspaces",
                binding="Ctrl+G",
                keywords="settings preferences",
            ),
            PaletteAction(
                "quit",
                "Quit",
                "Exit the TUI",
                "System",
                binding="Ctrl+Q",
                keywords="quit exit",
            ),
        ]
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
        if action_id == "theme":
            self.action_open_theme_modal()
            return
        if action_id == "run:select":
            self.action_open_session_run_modal()
            return
        if action_id == "quit":
            self.action_request_quit()
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
            "knowledge-run": "run",
            "sources": "sources",
            "sessions": "sessions",
        }
        command = command_by_action.get(action_id)
        if command:
            self._run_backend_command(command)

    def action_open_session_run_modal(self) -> None:
        self.status = "Loading conversations"
        self._append_activity("Loading conversations for selection...")
        worker = self.run_worker(
            lambda: self._list_sessions_for_picker(200, self.all_cursor_workspaces),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "session_picker"

    def _run_backend_command(self, command: str, *, bypass_local: bool = False) -> None:
        if not bypass_local and self._handle_local_command(command):
            return
        if self._result_list_open:
            self._close_inline_result_list(announce=False)
        self._append_activity(f"> {command}")
        self.status = f"Running: {command}"
        if _is_knowledge_run_command(command):
            self._append_activity("Knowledge run started. Loading...")
        viewport_width = max(int(self.size.width or 0), 80)
        viewport_height = max(int(self.size.height or 0), 24)
        worker = self.run_worker(
            lambda: self._execute_command(command, viewport_width, viewport_height),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        worker_key = id(worker)
        self._worker_context[worker_key] = "command"
        if _is_knowledge_run_command(command):
            self._knowledge_run_workers.add(worker_key)

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

        if action == "theme":
            if len(parts) == 1:
                self.action_open_theme_modal()
                return True
            if second == "list":
                self.action_open_theme_modal()
                return True
            if second == "set" and len(parts) == 2:
                self.action_open_theme_modal()
                return True
        if action == "run" and len(parts) > 1 and parts[1].lower() == "select":
            self.action_open_session_run_modal()
            self.status = "Select conversations"
            return True

        return False

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "activity_result_list":
            return

    def _apply_setup_modal_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            self._pending_setup_payload = None
            return
        self._pending_setup_payload = dict(result)
        model_defaults = dict(self._setup_defaults_provider())
        self.status = "Setup (step 2/2)"
        self._append_activity("Step 1 complete. Configure provider and model settings.")
        self._open_setup_step_two_modal(model_defaults)

    def _apply_setup_model_modal_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            self._pending_setup_payload = None
            self.status = "Setup cancelled"
            self._append_activity("Setup cancelled.")
            return

        action = str(result.get("_action") or "").strip().lower()
        if action == "back":
            setup_defaults = dict(self._setup_defaults_provider())
            if isinstance(self._pending_setup_payload, dict):
                setup_defaults.update(self._pending_setup_payload)
            self.status = "Setup (step 1/2)"
            self._append_activity("Returned to setup step 1.")
            self._open_setup_step_one_modal(setup_defaults)
            return

        payload: dict[str, Any] = {}
        if isinstance(self._pending_setup_payload, dict):
            payload.update(self._pending_setup_payload)
        payload.update(result)
        self._pending_setup_payload = None

        self.status = "Applying setup"
        self._append_activity("Applying setup from wizard...")
        worker = self.run_worker(
            lambda: self._run_setup_payload(payload),
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

    def _apply_session_run_modal_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        session_ids = [
            str(item).strip() for item in result.get("session_ids", []) if str(item).strip()
        ]
        if not session_ids:
            self._append_activity("No conversations selected.")
            return
        command_parts = ["sync", "--verbose"]
        for session_id in session_ids:
            command_parts.append("--session-id")
            command_parts.append(shlex.quote(session_id))
        command = " ".join(command_parts)
        self._append_activity(f"Selected {len(session_ids)} conversation(s) for knowledge run.")
        self._run_backend_command(command)

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
            self._complete_knowledge_worker(worker_key, success=True)
            return

        if event.state == WorkerState.CANCELLED:
            self.status = "Operation cancelled"
            self._append_activity("Previous operation cancelled.")
            self._finalize_theme_commit(success=False)
            self._worker_context.pop(worker_key, None)
            self._complete_knowledge_worker(worker_key, success=False)
            self._refresh_dashboard_panel()
            return

        if event.state == WorkerState.ERROR:
            self.status = "Operation failed"
            error = event.worker.error
            if error is None:
                self._append_activity("Operation failed with an unknown error.")
            else:
                self._append_activity(f"Error: {error}")
            self._finalize_theme_commit(success=False)
            self._worker_context.pop(worker_key, None)
            self._complete_knowledge_worker(worker_key, success=False)
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

            cleaned_lines: list[str] = []
            for line in lines:
                cleaned = _strip_rich_markup(line).strip()
                if cleaned:
                    cleaned_lines.append(cleaned)
                    self._append_activity(cleaned)

            command_header = cleaned_lines[0] if cleaned_lines else ""
            if cleaned_lines and (
                len(cleaned_lines) > 12
                or "/sources" in command_header
                or "/sessions" in command_header
            ):
                self._show_inline_result_list(cleaned_lines)

            self.status = "Last command executed"
            self._finalize_theme_commit(success=True)
            self._refresh_dashboard_panel()
            if should_exit:
                self.action_request_quit()
            return

        if context == "session_picker":
            sessions: list[dict[str, Any]] = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        sessions.append({str(key): value for key, value in item.items()})
            if not sessions:
                self.status = "No conversations found"
                self._append_activity("No conversations available to select.")
                self._refresh_dashboard_panel()
                return
            self.status = "Select conversations"
            self._append_activity(f"Loaded {len(sessions)} conversation(s).")
            self.push_screen(
                SessionRunModal(sessions),
                self._apply_session_run_modal_result,
            )
            self._refresh_dashboard_panel()
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

            self.onboarding_required = False
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

    def _complete_knowledge_worker(self, worker_key: int, *, success: bool) -> None:
        if worker_key not in self._knowledge_run_workers:
            return
        self._knowledge_run_workers.discard(worker_key)
        if self._knowledge_run_workers:
            return
        if success:
            self._append_activity("Knowledge run completed.")
        else:
            self._append_activity("Knowledge run ended with errors.")
