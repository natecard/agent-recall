"""Vector memory configuration modal for onboarding step 3."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static


class VectorMemoryStepModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, defaults: dict[str, Any]):
        super().__init__()
        self.defaults = defaults

    def compose(self) -> ComposeResult:
        backend = str(self.defaults.get("vector_backend") or "local").strip().lower() or "local"
        if backend not in {"local", "turbopuffer"}:
            backend = "local"
        backend_options: list[tuple[str, str]] = [
            ("Local semantic memory (Recommended)", "local"),
            ("TurboPuffer", "turbopuffer"),
        ]

        with Container(id="modal_overlay"):
            with Vertical(id="modal_card", classes="modal_compact"):
                yield Static("Vector Memory", classes="modal_title")
                yield Static("Step 3 of 4 · Semantic memory backend", classes="modal_subtitle")
                yield Checkbox(
                    "Enable semantic vector memory",
                    value=bool(self.defaults.get("vector_enabled", True)),
                    id="vector_enabled",
                )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Backend", classes="field_label")
                    yield Select[str](
                        backend_options,
                        value=backend,
                        allow_blank=False,
                        id="vector_backend",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Local model", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("local_model_name") or "all-MiniLM-L6-v2"),
                        placeholder="all-MiniLM-L6-v2",
                        id="vector_local_model_name",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Model path", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("local_model_path") or ""),
                        placeholder="Optional local path",
                        id="vector_local_model_path",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Cache dir", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("local_model_cache_dir") or ""),
                        placeholder="Optional cache dir",
                        id="vector_local_model_cache_dir",
                        classes="field_input",
                    )
                yield Checkbox(
                    "Download local model during setup if missing",
                    value=bool(self.defaults.get("local_model_auto_download", True)),
                    id="vector_local_model_auto_download",
                )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Turbo URL", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("turbopuffer_base_url") or ""),
                        placeholder="https://...",
                        id="vector_turbopuffer_base_url",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Turbo env", classes="field_label")
                    yield Input(
                        value=str(
                            self.defaults.get("turbopuffer_api_key_env") or "TURBOPUFFER_API_KEY"
                        ),
                        placeholder="TURBOPUFFER_API_KEY",
                        id="vector_turbopuffer_api_key_env",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Embed URL", classes="field_label")
                    yield Input(
                        value=str(self.defaults.get("external_embedding_base_url") or ""),
                        placeholder="https://...",
                        id="vector_external_embedding_base_url",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Embed env", classes="field_label")
                    yield Input(
                        value=str(
                            self.defaults.get("external_embedding_api_key_env") or "OPENAI_API_KEY"
                        ),
                        placeholder="OPENAI_API_KEY",
                        id="vector_external_embedding_api_key_env",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Embed model", classes="field_label")
                    yield Input(
                        value=str(
                            self.defaults.get("external_embedding_model")
                            or "text-embedding-3-small"
                        ),
                        placeholder="text-embedding-3-small",
                        id="vector_external_embedding_model",
                        classes="field_input",
                    )
                yield Static("", id="vector_error")
                with Horizontal(classes="modal_actions"):
                    yield Button("Back", id="vector_back")
                    yield Button("Skip", id="vector_skip")
                    yield Button("Next", variant="primary", id="vector_next")
                    yield Button("Cancel", id="vector_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "vector_cancel":
            self.dismiss(None)
            return
        if event.button.id == "vector_back":
            self.dismiss({"_action": "back"})
            return
        if event.button.id == "vector_skip":
            self.dismiss(
                {
                    "_action": "next",
                    "vector_enabled": False,
                    "vector_backend": "local",
                }
            )
            return
        if event.button.id != "vector_next":
            return

        enabled = bool(self.query_one("#vector_enabled", Checkbox).value)
        backend = str(self.query_one("#vector_backend", Select).value or "local").strip().lower()
        if backend not in {"local", "turbopuffer"}:
            backend = "local"
        error = self.query_one("#vector_error", Static)

        payload = {
            "_action": "next",
            "vector_enabled": enabled,
            "vector_backend": backend,
            "local_model_name": self.query_one("#vector_local_model_name", Input).value.strip()
            or "all-MiniLM-L6-v2",
            "local_model_path": self.query_one("#vector_local_model_path", Input).value.strip()
            or None,
            "local_model_cache_dir": self.query_one(
                "#vector_local_model_cache_dir", Input
            ).value.strip()
            or None,
            "local_model_auto_download": bool(
                self.query_one("#vector_local_model_auto_download", Checkbox).value
            ),
            "turbopuffer_base_url": self.query_one(
                "#vector_turbopuffer_base_url", Input
            ).value.strip()
            or None,
            "turbopuffer_api_key_env": self.query_one(
                "#vector_turbopuffer_api_key_env", Input
            ).value.strip()
            or "TURBOPUFFER_API_KEY",
            "external_embedding_base_url": self.query_one(
                "#vector_external_embedding_base_url", Input
            ).value.strip()
            or None,
            "external_embedding_api_key_env": self.query_one(
                "#vector_external_embedding_api_key_env", Input
            ).value.strip()
            or "OPENAI_API_KEY",
            "external_embedding_model": self.query_one(
                "#vector_external_embedding_model", Input
            ).value.strip()
            or "text-embedding-3-small",
        }

        if enabled and backend == "turbopuffer":
            missing: list[str] = []
            if payload["turbopuffer_base_url"] is None:
                missing.append("TurboPuffer base URL")
            if payload["external_embedding_base_url"] is None:
                missing.append("External embedding base URL")
            if not payload["external_embedding_model"]:
                missing.append("External embedding model")
            if missing:
                error.update("[error]" + ", ".join(missing) + " required[/error]")
                return

        self.dismiss(payload)
