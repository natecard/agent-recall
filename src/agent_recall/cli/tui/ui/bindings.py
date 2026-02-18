from textual.binding import Binding

TUI_BINDINGS = [
    Binding("ctrl+p", "command_palette", "Commands"),
    Binding("f1", "open_command_palette", "Commands", show=False),
    Binding("ctrl+g", "open_settings_modal", "Settings"),
    Binding("ctrl+r", "refresh_now", "Refresh"),
    Binding("ctrl+k", "run_knowledge_update", "Run"),
    Binding("ctrl+y", "sync_conversations", "Sync"),
    Binding("ctrl+t", "open_theme_modal", "Theme"),
    Binding("ctrl+l", "open_layout_modal", "Layout"),
    Binding("ctrl+c", "request_quit", "Quit", show=False, priority=True),
    Binding("escape", "close_inline_picker", "Close picker", show=False),
    Binding("ctrl+q", "request_quit", "Quit", priority=True),
    Binding("ctrl+`", "toggle_terminal_panel", "Terminal", show=False),
    Binding("/", "focus_cli_input", "CLI", show=False),
    Binding("enter", "focus_cli_input", "CLI", show=False),
]
