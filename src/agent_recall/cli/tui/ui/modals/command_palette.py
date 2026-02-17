from __future__ import annotations

from collections import defaultdict

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from agent_recall.cli.tui.commands.palette_actions import (
    PaletteAction,
    _is_palette_cli_command_redundant,
    _normalize_palette_command,
)
from agent_recall.cli.tui.logic.text_sanitizers import _strip_rich_markup


class CommandPaletteModal(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, actions: list[PaletteAction], cli_commands: list[str]):
        super().__init__()
        self.actions = actions
        self.cli_commands = cli_commands
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
                    "Type to filter, use ↑↓ to navigate, Enter to run",
                    id="palette_hint",
                )

    def on_mount(self) -> None:
        self.query_one("#palette_search", Input).focus()
        self._rebuild_options()

    def on_key(self, event: events.Key) -> None:
        if event.key not in {"up", "down"}:
            return
        event.prevent_default()
        event.stop()
        direction = -1 if event.key == "up" else 1
        self._move_highlight(direction)

    def _move_highlight(self, direction: int) -> None:
        option_list = self.query_one("#palette_options", OptionList)
        options = option_list.options
        if not options:
            return

        highlighted = option_list.highlighted
        if highlighted is None:
            index = 0 if direction > 0 else len(options) - 1
        else:
            index = highlighted + direction

        while 0 <= index < len(options):
            if not options[index].disabled:
                option_list.highlighted = index
                break
            index += direction

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

        cli_commands: list[str] = []
        for command in self.cli_commands:
            if _is_palette_cli_command_redundant(command):
                continue
            normalized = _normalize_palette_command(command)
            if query and query not in normalized:
                continue
            cli_commands.append(command)

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

        if cli_commands:
            options.append(Option("", id="heading:spacer:cli", disabled=True))
            options.append(
                Option(
                    "[bold accent]CLI Commands[/bold accent]",
                    id="heading:cli",
                    disabled=True,
                )
            )
            for command in cli_commands:
                command_text = _normalize_palette_command(command)
                line = command_text
                if query:
                    line = f"{line} [dim]- raw CLI command[/dim]"
                options.append(Option(line, id=f"action:cmd:{command_text}"))

        option_list.set_options(options)

        for index, option in enumerate(options):
            if not option.disabled:
                option_list.highlighted = index
                break
