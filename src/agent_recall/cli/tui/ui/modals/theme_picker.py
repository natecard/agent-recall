from __future__ import annotations

from collections.abc import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


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
