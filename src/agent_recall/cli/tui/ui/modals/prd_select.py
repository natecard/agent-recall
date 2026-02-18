from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option


class PRDSelectModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
        Binding("space", "toggle_selection", "Toggle"),
        Binding("enter", "apply", "Apply"),
    ]

    def __init__(
        self,
        prd_items: list[dict[str, Any]],
        selected_ids: list[str],
        max_iterations: int,
    ):
        super().__init__()
        self.prd_items = prd_items
        self.selected_prd_ids: set[str] = set(selected_ids)
        self.max_iterations = max_iterations
        self.filter_query = ""

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static(
                    "Select PRD Items (optional — leave empty for all items; model decides)",
                    classes="modal_title",
                )
                yield Input(
                    placeholder="Filter PRD items...",
                    id="prd_select_filter",
                    classes="field_input",
                )
                yield OptionList(id="prd_select_list")
                yield Static("", id="prd_select_status")
                with Horizontal(classes="modal_actions"):
                    yield Button("Apply", variant="primary", id="prd_select_apply")
                    yield Button("Cancel", id="prd_select_cancel")

    def on_mount(self) -> None:
        self.query_one("#prd_select_filter", Input).focus()
        self._rebuild_options()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prd_select_filter":
            return
        self.filter_query = event.value
        self._rebuild_options()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "prd_select_list":
            return
        self._toggle_highlighted()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prd_select_cancel":
            self.dismiss(None)
            return
        if event.button.id == "prd_select_apply":
            self._submit()

    def action_toggle_selection(self) -> None:
        self._toggle_highlighted()

    def action_apply(self) -> None:
        self._submit()

    def _toggle_highlighted(self) -> None:
        option_list = self.query_one("#prd_select_list", OptionList)
        highlighted = option_list.highlighted
        if highlighted is None:
            return
        option = option_list.get_option_at_index(highlighted)
        option_id = option.id or ""
        if not option_id.startswith("prd:"):
            return
        prd_id = option_id.split(":", 1)[1]
        if not prd_id:
            return
        if prd_id in self.selected_prd_ids:
            self.selected_prd_ids.remove(prd_id)
        else:
            self.selected_prd_ids.add(prd_id)
        self._rebuild_options()

    def _submit(self) -> None:
        ordered_selection = [
            str(item.get("id") or "")
            for item in self.prd_items
            if str(item.get("id") or "") in self.selected_prd_ids
        ]
        ordered_selection = [x for x in ordered_selection if x]
        if ordered_selection and len(ordered_selection) > self.max_iterations:
            self.query_one("#prd_select_status", Static).update(
                f"[red]Selected {len(ordered_selection)} PRDs exceeds max_iterations "
                f"({self.max_iterations}). Select at most {self.max_iterations}.[/red]"
            )
            return
        self.dismiss({"selected_prd_ids": ordered_selection or None})

    def _rebuild_options(self) -> None:
        query = self.filter_query.strip().lower()
        visible_items: list[dict[str, Any]] = []
        for item in self.prd_items:
            title = str(item.get("title") or "").lower()
            item_id = str(item.get("id") or "").lower()
            if query and query not in f"{title} {item_id}":
                continue
            visible_items.append(item)

        option_list = self.query_one("#prd_select_list", OptionList)
        options: list[Option] = []
        seen_ids: set[str] = set()
        for item in visible_items:
            item_id = str(item.get("id") or "")
            if not item_id:
                continue
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            selected = item_id in self.selected_prd_ids
            options.append(
                Option(
                    self._line_for_prd_item(item, selected),
                    id=f"prd:{item_id}",
                )
            )

        if not options:
            options = [Option("No matching PRD items.", id="prd:", disabled=True)]

        option_list.set_options(options)
        for index, option in enumerate(options):
            if option.disabled:
                continue
            option_list.highlighted = index
            break

        count_ok = (
            len(self.selected_prd_ids) <= self.max_iterations if self.selected_prd_ids else True
        )
        limit_hint = f" (max {self.max_iterations})" if self.selected_prd_ids else ""
        empty_hint = " · Empty = all items (model decides)" if not self.selected_prd_ids else ""
        self.query_one("#prd_select_status", Static).update(
            f"[dim]{len(visible_items)} shown · {len(self.selected_prd_ids)} selected"
            f"{limit_hint}{empty_hint} · Space toggles, Enter applies[/dim]"
            + ("" if count_ok else " [red]· Exceeds limit![/red]")
        )

    def _line_for_prd_item(self, item: dict[str, Any], selected: bool) -> str:
        marker = "[green]✓[/green]" if selected else "[dim]○[/dim]"
        title = str(item.get("title") or "Untitled")
        item_id = str(item.get("id") or "-")
        priority = int(item.get("priority") or 0)
        status = "passed" if bool(item.get("passes")) else "pending"
        return f"{marker} {title} [dim]({item_id} · priority {priority} · {status})[/dim]"
