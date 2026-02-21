from __future__ import annotations

from typing import TYPE_CHECKING

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

if TYPE_CHECKING:
    pass


class ResizeHandle(Static):
    DEFAULT_CSS = """
    ResizeHandle {
        width: 1;
        height: 1fr;
        background: $panel;
        content-align: center middle;
    }
    ResizeHandle:hover {
        background: $accent;
    }
    ResizeHandle.dragging {
        background: $accent;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__("│", name=name, id=id, classes=classes)
        self._dragging = False

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._dragging = True
        self.add_class("dragging")
        self.capture_mouse()
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.remove_class("dragging")
            self.release_mouse()
            event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging:
            self.post_message(self.Dragged(event.screen_x))
            event.stop()

    def on_leave(self, event: events.Leave) -> None:
        if not self._dragging:
            return
        event.stop()

    class Dragged(Message):
        def __init__(self, screen_x: int) -> None:
            super().__init__()
            self.screen_x = screen_x


class ResizableSplit(Horizontal):
    DEFAULT_CSS = """
    ResizableSplit {
        height: 1fr;
        width: 100%;
    }
    ResizableSplit > Container#split_left {
        height: 1fr;
        overflow: hidden;
    }
    ResizableSplit > ResizeHandle {
        height: 1fr;
    }
    ResizableSplit > Container#split_right {
        height: 1fr;
        width: 1fr;
        overflow: hidden;
    }
    """

    left_width: reactive[int] = reactive(32, init=False)

    class WidthChanged(Message):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.width = width

    def __init__(
        self,
        *,
        initial_width: int = 32,
        min_width: int = 20,
        max_width: int = 80,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.left_width = initial_width
        self._min_width = min_width
        self._max_width = max_width
        self._split_start_x: int = 0

    def compose(self) -> ComposeResult:
        with Container(id="split_left"):
            pass
        yield ResizeHandle(id="split_handle")
        with Container(id="split_right"):
            pass

    def on_mount(self) -> None:
        self._apply_width()

    def watch_left_width(self, old_width: int, new_width: int) -> None:
        if old_width != new_width:
            self._apply_width()
            self.post_message(self.WidthChanged(new_width))

    def _apply_width(self) -> None:
        try:
            left = self.query_one("#split_left", Container)
            left.styles.width = self.left_width
        except Exception:
            pass

    @on(ResizeHandle.Dragged)
    def _on_handle_dragged(self, event: ResizeHandle.Dragged) -> None:
        try:
            left = self.query_one("#split_left", Container)
            region = left.region
            if region:
                new_width = event.screen_x - region.x
                new_width = max(self._min_width, min(self._max_width, new_width))
                self.left_width = new_width
        except Exception:
            pass
        event.stop()
