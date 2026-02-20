"""Compact LLM configuration modal for onboarding step 2a."""

from __future__ import annotations

import os
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static

from agent_recall.cli.tui.constants import (
    _PROVIDER_BASE_URL_DEFAULTS,
)
from agent_recall.cli.tui.logic.text_sanitizers import _clean_optional_text
from agent_recall.cli.tui.types import DiscoverModelsFn
from agent_recall.core.onboarding import API_KEY_ENV_BY_PROVIDER


class LLMConfigStepModal(ModalScreen[dict[str, Any] | None]):
    """Focused LLM config: provider, model, temperature, max tokens. No scroll."""

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
        validate_default = bool(self.defaults.get("validate", False))

        with Container(id="modal_overlay"):
            with Vertical(id="modal_card", classes="modal_compact"):
                yield Static("LLM Configuration", classes="modal_title")
                yield Static("Step 2 of 3 · Compaction model", classes="modal_subtitle")
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Provider", classes="field_label")
                    yield Select(
                        [(p, p) for p in self.providers],
                        value=default_provider,
                        allow_blank=False,
                        id="model_provider",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Model", classes="field_label")
                    yield Select(
                        [("Manual entry", "__manual__")],
                        value="__manual__",
                        allow_blank=False,
                        id="model_picker",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("", classes="field_label")
                    yield Input(
                        value=_clean_optional_text(self.defaults.get("model", "")),
                        placeholder="Select above or type",
                        id="model_name",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Base URL", classes="field_label")
                    yield Input(
                        value=base_url,
                        placeholder="Optional",
                        id="model_base_url",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("API key", classes="field_label")
                    yield Input(
                        value="",
                        placeholder="Optional",
                        password=True,
                        id="model_api_key",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Temp", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("temperature", 0.3)),
                        placeholder="0.0-2.0",
                        id="model_temperature",
                        classes="field_input",
                    )
                    yield Static("Tokens", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("max_tokens", 4096)),
                        placeholder=">0",
                        id="model_max_tokens",
                        classes="field_input",
                    )
                yield Static("", id="model_api_hint")
                yield Checkbox("Validate connection", value=validate_default, id="model_validate")
                yield Static("", id="model_discovery_status")
                yield Static("", id="model_error")
                with Horizontal(classes="modal_actions"):
                    yield Button("Refresh", id="model_refresh")
                    yield Button("Back", id="model_back")
                    yield Button("Next", variant="primary", id="model_next")
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
        if event.select.id == "model_picker":
            val = event.value
            if val == Select.BLANK or str(val) == "__manual__":
                return
            self.query_one("#model_name", Input).value = str(val)

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
        if event.button.id != "model_next":
            return

        err = self.query_one("#model_error", Static)
        provider = self.query_one("#model_provider", Select).value
        if provider == Select.BLANK:
            err.update("[error]Provider required[/error]")
            return
        try:
            temp = float(self.query_one("#model_temperature", Input).value)
        except ValueError:
            err.update("[error]Invalid temperature[/error]")
            return
        if temp < 0.0 or temp > 2.0:
            err.update("[error]Temperature 0.0-2.0[/error]")
            return
        try:
            tokens = int(self.query_one("#model_max_tokens", Input).value)
        except ValueError:
            err.update("[error]Invalid max tokens[/error]")
            return
        if tokens <= 0:
            err.update("[error]Max tokens > 0[/error]")
            return

        self.dismiss(
            {
                "_action": "next",
                "provider": str(provider),
                "model": _clean_optional_text(self.query_one("#model_name", Input).value) or None,
                "base_url": _clean_optional_text(self.query_one("#model_base_url", Input).value)
                or None,
                "api_key": _clean_optional_text(self.query_one("#model_api_key", Input).value)
                or None,
                "temperature": temp,
                "max_tokens": tokens,
                "validate": bool(self.query_one("#model_validate", Checkbox).value),
            }
        )

    def _apply_base_url_default(self, *, set_default_base_url: bool) -> None:
        p = str(self.query_one("#model_provider", Select).value or self.providers[0])
        inp = self.query_one("#model_base_url", Input)
        default = _PROVIDER_BASE_URL_DEFAULTS.get(p, "")
        if set_default_base_url:
            inp.value = default
        inp.placeholder = "Required" if p == "openai-compatible" else "Optional"

    def _update_api_key_hint(self) -> None:
        p = str(self.query_one("#model_provider", Select).value or self.providers[0])
        env_var = API_KEY_ENV_BY_PROVIDER.get(p)
        hint = self.query_one("#model_api_hint", Static)
        if not env_var:
            hint.update("")
            return
        if os.environ.get(env_var):
            hint.update(f"[dim]Using {env_var}[/dim]")
        else:
            hint.update(f"[dim]Set {env_var} if needed[/dim]")

    def _refresh_models(self) -> None:
        p = str(self.query_one("#model_provider", Select).value or self.providers[0])
        base = _clean_optional_text(self.query_one("#model_base_url", Input).value) or None
        base = base or _PROVIDER_BASE_URL_DEFAULTS.get(p)
        env_var = API_KEY_ENV_BY_PROVIDER.get(p)
        key = _clean_optional_text(self.query_one("#model_api_key", Input).value)
        prev = os.environ.get(env_var) if env_var else None
        if env_var and key:
            os.environ[env_var] = key
        try:
            models, err_msg = self.discover_models(p, base, env_var)
        finally:
            if env_var and key:
                if prev is not None:
                    os.environ[env_var] = prev
                else:
                    os.environ.pop(env_var, None)

        picker = self.query_one("#model_picker", Select)
        opts: list[tuple[str, str]] = [("Manual entry", "__manual__")]
        opts.extend((m, m) for m in models)
        picker.set_options(opts)

        current = _clean_optional_text(self.query_one("#model_name", Input).value)
        current = current or _clean_optional_text(self.defaults.get("model", ""))
        picker.value = current if current in models else "__manual__"

        status = self.query_one("#model_discovery_status", Static)
        status.update(
            f"[success]{len(models)} models[/success]"
            if models
            else f"[warning]{err_msg or 'No models'}[/warning]"
        )
