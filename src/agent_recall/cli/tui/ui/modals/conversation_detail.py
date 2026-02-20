from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown

from agent_recall.ingest.base import RawSession


class ConversationDetailModal(ModalScreen[None]):
    """Modal to display a detailed conversational view of a single session."""

    DEFAULT_CSS = """
    ConversationDetailModal {
        align: center middle;
        background: $background 50%;
    }

    #conversation_detail_overlay {
        width: 90vw;
        height: 90vh;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        border-title-color: $accent;
    }

    #conversation_detail_actions {
        dock: bottom;
        height: auto;
        padding-top: 1;
        align: right middle;
    }

    #conversation_detail_content {
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, session: RawSession) -> None:
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        with Container(id="conversation_detail_overlay"):
            with ScrollableContainer(id="conversation_detail_content"):
                yield Markdown(self._build_markdown())
            with Container(id="conversation_detail_actions"):
                yield Button("Close", id="conversation_detail_close", variant="primary")

    def _build_markdown(self) -> str:
        lines: list[str] = [f"# {self.session.title or 'Untitled Conversation'}", ""]
        if self.session.started_at:
            lines.append(f"**Started**: {self.session.started_at}")
        lines.append(f"**Source**: {self.session.source}")
        lines.append("")

        for msg in self.session.messages:
            lines.append(f"### {msg.role.capitalize()}")
            lines.append(str(msg.content))
            lines.append("")

            for tc in msg.tool_calls:
                lines.append("```json")
                lines.append(f"// Tool: {tc.tool}")
                try:
                    args_str = json.dumps(tc.args, indent=2)
                except Exception:
                    args_str = str(tc.args)
                lines.append(args_str)
                if tc.result:
                    lines.append(f"// Result:\n{tc.result}")
                lines.append("```")
                lines.append("")

        return "\n".join(lines)

    def on_mount(self) -> None:
        overlay = self.query_one("#conversation_detail_overlay", Container)
        overlay.border_title = "Conversation Details"
        self.query_one("#conversation_detail_close", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "conversation_detail_close":
            self.dismiss(None)
