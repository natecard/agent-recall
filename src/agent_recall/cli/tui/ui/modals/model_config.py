from __future__ import annotations

import os
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static

from agent_recall.cli.tui.constants import _PROVIDER_BASE_URL_DEFAULTS
from agent_recall.cli.tui.logic.text_sanitizers import _clean_optional_text
from agent_recall.cli.tui.types import DiscoverModelsFn
from agent_recall.core.onboarding import API_KEY_ENV_BY_PROVIDER


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
        base_url_value = (
            _clean_optional_text(self.query_one("#model_base_url", Input).value) or None
        )
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
