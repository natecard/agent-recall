from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from textual import on
from textual.containers import Vertical
from textual.coordinate import Coordinate
from textual.message import Message
from textual.widgets import DataTable, Input, Static

from agent_recall.core.tier_format import ParsedEntry, parse_tier_content
from agent_recall.storage.files import FileStorage, KnowledgeTier


@dataclass(frozen=True)
class KnowledgeEntry:
    tier: KnowledgeTier
    entry: ParsedEntry
    line_index: int


class InteractiveKnowledgeWidget(Vertical):
    DEFAULT_CSS = """
    InteractiveKnowledgeWidget {
        height: 100%;
        padding: 0;
    }
    InteractiveKnowledgeWidget #knowledge_table {
        height: 1fr;
        border: solid $accent;
    }
    InteractiveKnowledgeWidget #search_container {
        height: auto;
        display: none;
        padding: 2;
        background: $surface-darken-1;
    }
    InteractiveKnowledgeWidget #search_container.visible {
        display: block;
    }
    InteractiveKnowledgeWidget #search_input {
        width: 100%;
    }
    InteractiveKnowledgeWidget #edit_container {
        height: auto;
        display: none;
        padding: 2;
        background: $surface-darken-2;
    }
    InteractiveKnowledgeWidget #edit_container.visible {
        display: block;
    }
    InteractiveKnowledgeWidget #edit_input {
        width: 100%;
    }
    InteractiveKnowledgeWidget #help_footer {
        height: auto;
        padding: 2;
        background: $surface;
        text-align: center;
    }
    """

    class EntryEdited(Message):
        def __init__(self, tier: KnowledgeTier, entry: KnowledgeEntry, new_text: str) -> None:
            super().__init__()
            self.tier = tier
            self.entry = entry
            self.new_text = new_text

    class EntryDeleted(Message):
        def __init__(self, tier: KnowledgeTier, count: int) -> None:
            super().__init__()
            self.tier = tier
            self.count = count

    BINDINGS = [
        ("/", "toggle_search", "Search"),
        ("e", "edit_entry", "Edit"),
        ("d", "toggle_delete", "Mark Delete"),
        ("D", "confirm_deletions", "Confirm Delete"),
        ("escape", "cancel_action", "Cancel"),
    ]

    def __init__(
        self,
        files: FileStorage,
        on_edit: Callable[[KnowledgeTier, KnowledgeEntry, str], None] | None = None,
        on_delete: Callable[[KnowledgeTier, int], None] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._files = files
        self._on_edit = on_edit
        self._on_delete = on_delete
        self._entries: list[KnowledgeEntry] = []
        self._filtered_entries: list[KnowledgeEntry] = []
        self._search_active = False
        self._edit_active = False
        self._selected_index = 0
        self._search_query = ""
        self._editing_entry: KnowledgeEntry | None = None
        self._pending_deletions: set[tuple[KnowledgeTier, int]] = set()

    def compose(self):
        yield DataTable(id="knowledge_table", zebra_stripes=True)
        with Vertical(id="search_container"):
            yield Input(placeholder="Search entries...", id="search_input")
        with Vertical(id="edit_container"):
            yield Input(placeholder="Edit entry text...", id="edit_input")
        yield Static(
            "[dim]/[/dim] Search  [dim]e[/dim] Edit  "
            "[dim]d[/dim] Delete  [dim]D[/dim] Confirm  "
            "[dim]Escape[/dim] Cancel",
            id="help_footer",
        )

    def on_mount(self) -> None:
        self._load_entries()
        self._render_table()
        table = self.query_one("#knowledge_table", DataTable)
        table.focus()

    def _load_entries(self) -> None:
        self._entries = []
        for tier in [KnowledgeTier.GUARDRAILS, KnowledgeTier.STYLE, KnowledgeTier.RECENT]:
            content = self._files.read_tier(tier)
            parsed = parse_tier_content(content)
            for i, entry in enumerate(parsed.bullet_entries):
                self._entries.append(KnowledgeEntry(tier=tier, entry=entry, line_index=i))
        self._filtered_entries = list(self._entries)

    def _render_table(self) -> None:
        table = self.query_one("#knowledge_table", DataTable)
        table.clear(columns=True)
        table.add_column("Tier", width=12)
        table.add_column("Kind", width=12)
        table.add_column("Entry", width=60)

        if not self._filtered_entries:
            table.add_row("", "", "[dim]No knowledge entries[/dim]")
            return

        for entry in self._filtered_entries:
            tier_name = entry.tier.value
            kind = entry.entry.kind or "—"
            text = entry.entry.text or entry.entry.raw_content[:80]
            if len(text) > 80:
                text = text[:77] + "..."
            entry_key = (entry.tier, entry.line_index)
            if entry_key in self._pending_deletions:
                text = f"[s]{text}[/s] [dim red](delete)[/dim red]"
            table.add_row(tier_name, kind, text)

        if self._filtered_entries:
            table.cursor_type = "row"
            if self._selected_index < len(self._filtered_entries):
                table.cursor_coordinate = Coordinate(self._selected_index, 0)

    def action_toggle_search(self) -> None:
        if self._edit_active:
            return
        self._search_active = not self._search_active
        container = self.query_one("#search_container", Vertical)
        if self._search_active:
            container.add_class("visible")
            search_input = self.query_one("#search_input", Input)
            search_input.value = ""
            search_input.focus()
        else:
            container.remove_class("visible")
            self._filtered_entries = list(self._entries)
            self._selected_index = 0
            self._render_table()
            self.query_one("#knowledge_table", DataTable).focus()

    def action_edit_entry(self) -> None:
        if self._search_active:
            return
        if not self._filtered_entries:
            return
        table = self.query_one("#knowledge_table", DataTable)
        if table.cursor_coordinate is None:
            return
        row_index = table.cursor_coordinate.row
        if row_index >= len(self._filtered_entries):
            return
        self._editing_entry = self._filtered_entries[row_index]
        self._edit_active = True
        container = self.query_one("#edit_container", Vertical)
        container.add_class("visible")
        edit_input = self.query_one("#edit_input", Input)
        edit_input.value = self._editing_entry.entry.text or ""
        edit_input.focus()

    def action_toggle_delete(self) -> None:
        if self._search_active or self._edit_active:
            return
        if not self._filtered_entries:
            return
        table = self.query_one("#knowledge_table", DataTable)
        if table.cursor_coordinate is None:
            return
        row_index = table.cursor_coordinate.row
        if row_index >= len(self._filtered_entries):
            return
        entry = self._filtered_entries[row_index]
        entry_key = (entry.tier, entry.line_index)
        if entry_key in self._pending_deletions:
            self._pending_deletions.remove(entry_key)
        else:
            self._pending_deletions.add(entry_key)
        self._render_table()

    def action_confirm_deletions(self) -> None:
        if self._search_active or self._edit_active:
            return
        if not self._pending_deletions:
            return
        self._execute_deletions()
        self._pending_deletions.clear()
        self._load_entries()
        self._render_table()
        self.query_one("#knowledge_table", DataTable).focus()

    def _execute_deletions(self) -> None:
        deletions_by_tier: dict[KnowledgeTier, list[int]] = {}
        for tier, line_index in self._pending_deletions:
            if tier not in deletions_by_tier:
                deletions_by_tier[tier] = []
            deletions_by_tier[tier].append(line_index)

        for tier, line_indices in deletions_by_tier.items():
            content = self._files.read_tier(tier)
            parsed = parse_tier_content(content)
            sorted_indices = sorted(line_indices, reverse=True)
            for idx in sorted_indices:
                if idx < len(parsed.bullet_entries):
                    parsed.bullet_entries.pop(idx)
            from agent_recall.core.tier_format import merge_tier_content

            new_content = merge_tier_content(parsed)
            self._files.write_tier(tier, new_content)
            if self._on_delete:
                self._on_delete(tier, len(line_indices))

    def action_cancel_action(self) -> None:
        if self._edit_active:
            self._edit_active = False
            self._editing_entry = None
            container = self.query_one("#edit_container", Vertical)
            container.remove_class("visible")
            self.query_one("#knowledge_table", DataTable).focus()
            return
        if self._search_active:
            self._search_active = False
            container = self.query_one("#search_container", Vertical)
            container.remove_class("visible")
            self._filtered_entries = list(self._entries)
            self._selected_index = 0
            self._render_table()
            self.query_one("#knowledge_table", DataTable).focus()
            return
        if self._pending_deletions:
            self._pending_deletions.clear()
            self._render_table()
            self.query_one("#knowledge_table", DataTable).focus()

    @on(Input.Changed, "#search_input")
    def _on_search_changed(self, event: Input.Changed) -> None:
        query = event.value.lower().strip()
        self._search_query = query
        if not query:
            self._filtered_entries = list(self._entries)
        else:
            self._filtered_entries = [
                e
                for e in self._entries
                if query in (e.entry.text or "").lower()
                or query in (e.entry.kind or "").lower()
                or query in e.tier.value.lower()
            ]
        self._selected_index = 0
        self._render_table()

    @on(Input.Submitted, "#edit_input")
    def _on_edit_submitted(self, event: Input.Submitted) -> None:
        if self._editing_entry is None:
            return
        new_text = event.value.strip()
        if new_text:
            self._write_edit(self._editing_entry, new_text)
            if self._on_edit:
                self._on_edit(self._editing_entry.tier, self._editing_entry, new_text)
        self._edit_active = False
        self._editing_entry = None
        container = self.query_one("#edit_container", Vertical)
        container.remove_class("visible")
        self._load_entries()
        self._render_table()
        self.query_one("#knowledge_table", DataTable).focus()

    def _write_edit(self, entry: KnowledgeEntry, new_text: str) -> None:
        content = self._files.read_tier(entry.tier)
        parsed = parse_tier_content(content)
        if entry.line_index < len(parsed.bullet_entries):
            old_entry = parsed.bullet_entries[entry.line_index]
            kind = old_entry.kind or "NOTE"
            new_line = f"- [{kind}] {new_text}"
            parsed.bullet_entries[entry.line_index] = ParsedEntry(
                format=old_entry.format,
                raw_content=new_line,
                kind=kind,
                text=new_text,
            )
        from agent_recall.core.tier_format import merge_tier_content

        new_content = merge_tier_content(parsed)
        self._files.write_tier(entry.tier, new_content)

    def refresh_entries(self) -> None:
        self._load_entries()
        if self._search_query:
            self._filtered_entries = [
                e
                for e in self._entries
                if self._search_query in (e.entry.text or "").lower()
                or self._search_query in (e.entry.kind or "").lower()
                or self._search_query in e.tier.value.lower()
            ]
        else:
            self._filtered_entries = list(self._entries)
        self._render_table()
