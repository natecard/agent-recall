from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Static

from agent_recall.storage.curation_queue import (
    CurationQueueItem,
    CurationQueueStatus,
    CurationQueueStore,
)


@dataclass(frozen=True)
class CurationQueueWidget:
    store: CurationQueueStore

    def render(self) -> Panel:
        items = self.store.get_pending()
        if not items:
            return Panel(
                "[dim]No pending items awaiting curation.[/dim]\n\n"
                "[dim]When curation_mode is enabled in config, newly ingested "
                "sessions will appear here for approval before compaction.[/dim]",
                title="Curation Queue",
                border_style="accent",
            )

        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Source", style="table_header", width=12, no_wrap=True)
        table.add_column("Label", style="table_header", width=14, no_wrap=True)
        table.add_column("Preview", overflow="fold")
        table.add_column("Time", width=16, no_wrap=True)

        for item in items[:20]:
            preview = item.content_preview
            if len(preview) > 80:
                preview = preview[:77] + "..."
            timestamp = item.timestamp.strftime("%Y-%m-%d %H:%M")
            table.add_row(
                item.source,
                f"[accent]{item.proposed_label}[/accent]",
                preview,
                f"[dim]{timestamp}[/dim]",
            )

        if len(items) > 20:
            remaining = len(items) - 20
            table.add_row("", "", f"[dim]... and {remaining} more items[/dim]", "")

        count_text = f"{len(items)} item{'s' if len(items) != 1 else ''} pending"
        return Panel(
            Group(table, Text(), Text(f"[dim]{count_text}[/dim]", justify="right")),
            title="Curation Queue",
            border_style="accent",
        )


class ApproveAllConfirmModal(ModalScreen[bool]):
    DEFAULT_CSS = """
    ApproveAllConfirmModal {
        align: center middle;
    }
    ApproveAllConfirmModal #modal_card {
        width: 60;
        padding: 2;
        background: $surface;
        border: thick $accent;
    }
    ApproveAllConfirmModal #modal_title {
        text-align: center;
        margin-bottom: 1;
    }
    ApproveAllConfirmModal #modal_body {
        margin-bottom: 1;
    }
    ApproveAllConfirmModal #button_row {
        align: center middle;
        height: auto;
    }
    ApproveAllConfirmModal Button {
        margin: 0 2;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss(False)", "Cancel"),
    ]

    def __init__(self, count: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._count = count

    def compose(self):
        with Vertical(id="modal_card"):
            yield Static(
                f"Approve {self._count} Item{'s' if self._count != 1 else ''}?",
                id="modal_title",
                classes="modal_title",
            )
            yield Static(
                f"All {self._count} pending items will be approved and sent to the "
                "compaction pipeline.\n\nThis action cannot be undone.",
                id="modal_body",
            )
            with Vertical(id="button_row"):
                yield Button("Cancel", id="cancel_btn", variant="default")
                yield Button("Approve All", id="confirm_btn", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm_btn":
            self.dismiss(True)
        elif event.button.id == "cancel_btn":
            self.dismiss(False)

    def on_key(self, event) -> None:
        if event.key == "enter":
            self.dismiss(True)
        elif event.key == "escape":
            self.dismiss(False)


class InteractiveCurationQueueWidget(Vertical):
    DEFAULT_CSS = """
    InteractiveCurationQueueWidget {
        height: 100%;
        padding: 0;
    }
    InteractiveCurationQueueWidget #queue_table {
        height: 1fr;
        border: solid $accent;
    }
    InteractiveCurationQueueWidget #label_edit_container {
        height: auto;
        display: none;
        padding: 2;
        background: $surface-darken-2;
    }
    InteractiveCurationQueueWidget #label_edit_container.visible {
        display: block;
    }
    InteractiveCurationQueueWidget #label_edit_input {
        width: 100%;
    }
    InteractiveCurationQueueWidget #help_footer {
        height: auto;
        padding: 2;
        background: $surface;
        text-align: center;
    }
    """

    class ItemApproved(Message):
        def __init__(self, chunk_id: str, entry_id: str | None) -> None:
            super().__init__()
            self.chunk_id = chunk_id
            self.entry_id = entry_id

    class ItemRejected(Message):
        def __init__(self, chunk_id: str) -> None:
            super().__init__()
            self.chunk_id = chunk_id

    class AllApproved(Message):
        def __init__(self, count: int) -> None:
            super().__init__()
            self.count = count

    BINDINGS = [
        Binding("y", "approve_item", "Approve"),
        Binding("n", "reject_item", "Reject"),
        Binding("e", "edit_label", "Edit Label"),
        Binding("a", "approve_all", "Approve All"),
        Binding("escape", "cancel_action", "Cancel"),
    ]

    def __init__(
        self,
        store: CurationQueueStore,
        on_approve: Callable[[CurationQueueItem], None] | None = None,
        on_reject: Callable[[str], None] | None = None,
        on_approve_all: Callable[[int], None] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._store = store
        self._on_approve = on_approve
        self._on_reject = on_reject
        self._on_approve_all = on_approve_all
        self._items: list[CurationQueueItem] = []
        self._label_edit_active = False
        self._editing_chunk_id: str | None = None

    def compose(self):
        yield DataTable(id="queue_table", zebra_stripes=True)
        with Vertical(id="label_edit_container"):
            yield Input(placeholder="Enter new label...", id="label_edit_input")
        yield Static(
            "[dim]y[/dim] Approve  [dim]n[/dim] Reject  "
            "[dim]e[/dim] Edit Label  [dim]a[/dim] Approve All  "
            "[dim]Escape[/dim] Cancel",
            id="help_footer",
        )

    def on_mount(self) -> None:
        self._load_items()
        self._render_table()
        table = self.query_one("#queue_table", DataTable)
        table.focus()

    def _load_items(self) -> None:
        self._items = self._store.get_pending()

    def _render_table(self) -> None:
        table = self.query_one("#queue_table", DataTable)
        table.clear(columns=True)
        table.add_column("Source", width=12)
        table.add_column("Label", width=14)
        table.add_column("Preview", width=50)
        table.add_column("Time", width=16)

        if not self._items:
            table.add_row("", "", "[dim]No pending items[/dim]", "")
            return

        for item in self._items:
            preview = item.content_preview
            if len(preview) > 50:
                preview = preview[:47] + "..."
            timestamp = item.timestamp.strftime("%Y-%m-%d %H:%M")
            table.add_row(
                item.source,
                f"[accent]{item.proposed_label}[/accent]",
                preview,
                f"[dim]{timestamp}[/dim]",
            )

    def refresh_queue(self) -> None:
        self._load_items()
        self._render_table()

    def action_approve_item(self) -> None:
        if self._label_edit_active:
            return
        table = self.query_one("#queue_table", DataTable)
        row = table.cursor_row
        if row is None or row >= len(self._items):
            return
        item = self._items[row]
        self._store.update_status(item.chunk_id, CurationQueueStatus.APPROVED)
        self.post_message(self.ItemApproved(item.chunk_id, item.entry_id))
        if self._on_approve:
            self._on_approve(item)
        self.refresh_queue()

    def action_reject_item(self) -> None:
        if self._label_edit_active:
            return
        table = self.query_one("#queue_table", DataTable)
        row = table.cursor_row
        if row is None or row >= len(self._items):
            return
        item = self._items[row]
        self._store.update_status(item.chunk_id, CurationQueueStatus.REJECTED)
        self.post_message(self.ItemRejected(item.chunk_id))
        if self._on_reject:
            self._on_reject(item.chunk_id)
        self.refresh_queue()

    def action_edit_label(self) -> None:
        if self._label_edit_active:
            return
        table = self.query_one("#queue_table", DataTable)
        row = table.cursor_row
        if row is None or row >= len(self._items):
            return
        item = self._items[row]
        self._editing_chunk_id = item.chunk_id
        container = self.query_one("#label_edit_container", Vertical)
        input_widget = self.query_one("#label_edit_input", Input)
        input_widget.value = item.proposed_label
        container.add_class("visible")
        self._label_edit_active = True
        input_widget.focus()

    def action_approve_all(self) -> None:
        if self._label_edit_active:
            return
        if not self._items:
            return

        def _handle_confirm(confirmed: bool | None) -> None:
            if confirmed:
                count = self._store.approve_all()
                self.post_message(self.AllApproved(count))
                if self._on_approve_all:
                    self._on_approve_all(count)
                self.refresh_queue()

        self.app.push_screen(
            ApproveAllConfirmModal(len(self._items)),
            _handle_confirm,
        )

    def action_cancel_action(self) -> None:
        if self._label_edit_active:
            container = self.query_one("#label_edit_container", Vertical)
            container.remove_class("visible")
            self._label_edit_active = False
            self._editing_chunk_id = None
            table = self.query_one("#queue_table", DataTable)
            table.focus()

    @on(Input.Submitted, "#label_edit_input")
    def _on_label_submitted(self, event: Input.Submitted) -> None:
        if not self._label_edit_active or not self._editing_chunk_id:
            return
        new_label = event.value.strip()
        if new_label:
            self._store.update_label(self._editing_chunk_id, new_label)
        self.action_cancel_action()
        self.refresh_queue()
