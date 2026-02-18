"""First-launch screen for delta diff renderer setup."""

from __future__ import annotations

from typing import Any, cast

from textual import on
from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Label, ProgressBar, Static

from agent_recall.cli.tui.delta import (
    DELTA_MANUAL_INSTALL_URL,
    DeltaDownloadError,
    check_network,
    download_delta,
    write_delta_setup_declined,
)


class DeltaDownloadComplete(Message):
    """Posted when delta download worker completes."""

    def __init__(self, success: bool, error: str | None = None) -> None:
        super().__init__()
        self.success = success
        self.error = error


class FirstLaunchScreen(Screen[None]):
    """Blocking splash screen for one-time delta download setup."""

    DEFAULT_CSS = """
    FirstLaunchScreen {
        align: center middle;
    }
    FirstLaunchScreen #first_launch_card {
        width: 60;
        height: auto;
        padding: 2 3;
        border: solid $accent;
        background: $surface;
    }
    FirstLaunchScreen #first_launch_title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    FirstLaunchScreen #first_launch_message {
        text-align: center;
        margin-bottom: 2;
        color: $text;
    }
    FirstLaunchScreen #first_launch_actions {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    FirstLaunchScreen #first_launch_progress {
        margin: 1 0;
    }
    FirstLaunchScreen #first_launch_error {
        color: $error;
        margin: 1 0;
        text-align: center;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._downloading = False
        self._error: str | None = None

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="first_launch_card"):
                yield Label("Delta Diff Renderer", id="first_launch_title")
                yield Static(
                    "Delta is not installed. Download it for better diff display? (one-time setup)",
                    id="first_launch_message",
                )
                yield Static("", id="first_launch_error")
                yield ProgressBar(id="first_launch_progress", show_eta=False)
                with Horizontal(id="first_launch_actions"):
                    yield Button("Download", variant="primary", id="first_launch_download")
                    yield Button("Skip", id="first_launch_skip")

    def on_mount(self) -> None:
        self.query_one("#first_launch_progress", ProgressBar).display = False
        self._check_offline()

    def _check_offline(self) -> None:
        if not check_network():
            self._error = "No network. Install manually."
            self._show_error()

    def _show_error(self) -> None:
        error_widget = self.query_one("#first_launch_error", Static)
        manual_msg = f"\n{DELTA_MANUAL_INSTALL_URL}"
        error_widget.update(f"[red]{self._error}{manual_msg}[/red]")
        error_widget.display = True

    def _start_download(self) -> None:
        self._downloading = True
        self._error = None
        self.query_one("#first_launch_error", Static).update("")
        self.query_one("#first_launch_error", Static).display = False
        self.query_one("#first_launch_download", Button).display = False
        self.query_one("#first_launch_skip", Button).display = False
        progress_bar = self.query_one("#first_launch_progress", ProgressBar)
        progress_bar.display = True
        progress_bar.update(progress=0.0)

        def _progress_callback(progress: float, _msg: str) -> None:
            self.app.call_from_thread(lambda: progress_bar.update(progress=progress))

        def _download_worker() -> object:
            try:
                download_delta(progress_callback=_progress_callback)
                return True
            except DeltaDownloadError as e:
                return str(e)

        worker = self.app.run_worker(
            _download_worker,
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        cast(Any, self.app)._worker_context[id(worker)] = "delta_download"

    @on(DeltaDownloadComplete)
    def _on_download_complete(self, event: DeltaDownloadComplete) -> None:
        self._on_download_complete_impl(event.success, event.error)

    def _on_download_complete_impl(self, success: bool, error: str | None) -> None:
        self._downloading = False
        progress_bar = self.query_one("#first_launch_progress", ProgressBar)
        progress_bar.display = False

        if success:
            self.app.pop_screen()
        else:
            self._error = error or "Download failed"
            self._show_error()
            self.query_one("#first_launch_download", Button).display = True
            self.query_one("#first_launch_skip", Button).display = True
            self.query_one("#first_launch_download", Button).label = "Retry"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "first_launch_skip":
            write_delta_setup_declined()
            self.app.pop_screen()
            return
        if event.button.id == "first_launch_download" and not self._downloading:
            if not check_network():
                self._error = "No network. Install manually."
                self._show_error()
                return
            self._start_download()
