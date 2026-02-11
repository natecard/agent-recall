from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from agent_recall.cli.theme import ThemeManager

# Full dramatic banner (for terminals >= 90 columns)
BANNER_FULL = r"""
    ·
    ·    ★    ·
    ·    ·    ·    ·    ·
     ╔════════════════════════════════════════════════════════════════════════╗
     ║  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ║
     ║  ░░                                                                ░░  ║
     ║  ░░          █████╗  ██████╗ ███████╗███╗   ██╗████████╗           ░░  ║
     ║  ░░         ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝           ░░  ║
     ║  ░░         ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║              ░░  ║
     ║  ░░         ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║              ░░  ║
     ║  ░░         ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║              ░░  ║
     ║  ░░         ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝              ░░  ║
     ║  ░░                                                                ░░  ║
     ║  ░░                             ◆ ◆ ◆                              ░░  ║
     ║  ░░                                                                ░░  ║
     ║  ░░         ██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗            ░░  ║
     ║  ░░         ██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║            ░░  ║
     ║  ░░         ██████╔╝█████╗  ██║     ███████║██║     ██║            ░░  ║
     ║  ░░         ██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║            ░░  ║
     ║  ░░         ██║  ██║███████╗╚██████╗██║  ██║███████╗███████╗       ░░  ║
     ║  ░░         ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝       ░░  ║
     ║  ░░                                                                ░░  ║
     ║  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ║
     ╠════════════════════════════════════════════════════════════════════════╣
     ║   ◈  A G E N T I C   M E M O R Y   R E T R I E V A L   S Y S T E M  ◈  ║
     ╚════════════════════════════════════════════════════════════════════════╝
    ·    ·    ·    ·    ·
    ·    ★    ·
    ·
"""

# Compact banner (for terminals 50-89 columns)
BANNER_COMPACT = r"""
      ═════════════════════════════════════
      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ┃
      ┃  ░░    ▄▀█ █▀▀ █▀▀ █▄ █ ▀█▀      ░░  ┃
      ┃  ░░    █▀█ █▄█ ██▄ █ ▀█  █       ░░  ┃
      ┃  ░░              ◆ ◆ ◆           ░░  ┃
      ┃  ░░    █▀█ █▀▀ █▀▀ ▄▀█ █   █     ░░  ┃
      ┃  ░░    █▀▄ ██▄ █▄▄ █▀█ █▄▄ █▄▄   ░░  ┃
      ┃  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ┃
      ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
      ┃      ◈  MEMORY RETRIEVAL SYSTEM  ◈   ┃
      ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
     ═════════════════════════════════════
"""

# Minimal banner (for terminals < 50 columns)
BANNER_MINIMAL = r"""
  ·  ★  ·
  ══════════════════
  ┃  AGENT RECALL  ┃
  ┃────────────────┃
  ┃     ◆ ◆ ◆      ┃
  ┃────────────────┃
  ┃ Memory System  ┃
  ══════════════════
    ·  ★  ·
"""

# Simpler TUI header for narrower terminals or cleaner look
BANNER_TUI_HEADER_SIMPLE = r"""
                                 ═══════════════════════════════════════════════════════════
                               ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                               ┃  ░░   █████╗    ██████╗   ███████╗  ███╗   ██╗  ████████╗ ░░  ┃
                               ┃  ░░  ██╔══██╗  ██╔════╝   ██╔════╝  ████╗  ██║  ╚══██╔══╝ ░░  ┃
                               ┃  ░░  ███████║  ██║  ███╗  █████╗    ██╔██╗ ██║     ██║    ░░  ┃
                               ┃  ░░  ██╔══██║  ██║   ██║  ██╔══╝    ██║╚██╗██║     ██║    ░░  ┃
                               ┃  ░░  ██║  ██║  ╚██████╔╝  ███████╗  ██║ ╚████║     ██║    ░░  ┃
                               ┃  ░░                       ◆ ◆ ◆                           ░░  ┃
                               ┃  ░░  ██████╗  ███████╗  ██████╗  █████╗  ██╗     ██╗      ░░  ┃
                               ┃  ░░  ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗ ██║     ██║      ░░  ┃
                               ┃  ░░  ██████╔╝ █████╗   ██║      ███████║ ██║     ██║      ░░  ┃
                               ┃  ░░  ██╔══██╗ ██╔══╝   ██║      ██╔══██║ ██║     ██║      ░░  ┃
                               ┃  ░░  ██║  ██║ ███████╗ ╚██████╗ ██║  ██║ ███████╗███████╗ ░░  ┃
                               ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
                               ┃     ◈  M E M O R Y   R E T R I E V A L   S Y S T E M  ◈       ┃
                               ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                 ════════════════════════════════════════════════════════════
"""

# Clean minimal TUI header
BANNER_TUI_HEADER_MINIMAL = r"""
        ════════════════════════════
         ★·┃ ░░ AGENT RECALL  ░░ |·★
         ★·┃ ░░     ◆ ◆ ◆     ░░ |·★
         ★·┃ ░░ Memory System ░░ |·★
        ════════════════════════════
"""

# Character to style mapping
CHAR_STYLE_MAP = {
    "★": "banner.star",
    "◆": "banner.diamond",
    "◈": "banner.diamond",
    "·": "banner.dot",
    "░": "banner.shade",
    "▄": "banner.letter",
    "█": "banner.letter",
    "▀": "banner.letter",
    "▐": "banner.letter",
    "▌": "banner.letter",
}

BORDER_CHARS = set("╔╗╚╝║╠╣┏┓┗┛┃┣┫━─")
ACCENT_CHARS = set("═")


class BannerRenderer:
    """Renders the ASCII banner with theme-aware styling."""

    def __init__(
        self,
        console: Console,
        theme_manager: ThemeManager | None = None,
    ):
        self.console = console
        self.theme_manager = theme_manager

    def get_appropriate_banner(self, width: int | None = None) -> str:
        """Select banner based on terminal width."""
        if width is None:
            width = self.console.width

        if width >= 90:
            return BANNER_FULL
        elif width >= 50:
            return BANNER_COMPACT
        else:
            return BANNER_MINIMAL

    def get_tui_header_banner(self, width: int | None = None) -> str:
        """Select TUI header banner based on terminal width."""
        if width is None:
            width = self.console.width

        if width >= 100:
            return BANNER_TUI_HEADER_SIMPLE
        elif width >= 70:
            return BANNER_TUI_HEADER_MINIMAL
        else:
            return BANNER_TUI_HEADER_MINIMAL

    def stylize_banner(self, banner: str) -> Text:
        """Apply theme styles to banner text."""
        text = Text(banner)

        # Taglines to style
        taglines = [
            "M E M O R Y   R E T R I E V A L   S Y S T E M",
            "M E M O R Y   R E T R I E V A L",
            "MEMORY RETRIEVAL SYSTEM",
            "Memory System",
            "AGENT RECALL",
            "RECALL",
        ]

        for i, char in enumerate(banner):
            style = None

            if char in CHAR_STYLE_MAP:
                style = CHAR_STYLE_MAP[char]
            elif char in BORDER_CHARS:
                style = "banner.border"
            elif char in ACCENT_CHARS:
                style = "banner.accent"

            if style:
                text.stylize(style, i, i + 1)

        # Style taglines
        for tagline in taglines:
            start = 0
            while True:
                idx = banner.find(tagline, start)
                if idx == -1:
                    break
                text.stylize("banner.tagline", idx, idx + len(tagline))
                start = idx + len(tagline)

        return text

    def render(self, centered: bool = True) -> None:
        """Render the banner to console."""
        width = self.console.width
        banner = self.get_appropriate_banner(width)
        styled_banner = self.stylize_banner(banner)

        if centered:
            self.console.print(styled_banner, justify="center")
        else:
            self.console.print(styled_banner)

    def render_animated(self, delay: float = 0.015) -> None:
        """Render banner with a line-by-line animation effect."""
        width = self.console.width
        banner = self.get_appropriate_banner(width)
        lines = banner.strip("\n").split("\n")

        for line in lines:
            styled_line = self.stylize_banner(line)
            self.console.print(styled_line, justify="center")
            time.sleep(delay)

    def get_styled_text(self) -> Text:
        """Get the styled banner as a Rich Text object."""
        banner = self.get_appropriate_banner()
        return self.stylize_banner(banner)

    def get_tui_header_text(self) -> Text:
        """Get the styled TUI header banner as a Rich Text object."""
        banner = self.get_tui_header_banner()
        return self.stylize_banner(banner)


def print_banner(
    console: Console,
    theme_manager: ThemeManager | None = None,
    animated: bool = False,
    delay: float = 0.015,
) -> None:
    """Convenience function to print the banner."""
    renderer = BannerRenderer(console, theme_manager)
    if animated:
        renderer.render_animated(delay=delay)
    else:
        renderer.render()


def get_tui_header(
    console: Console,
    theme_manager: ThemeManager | None = None,
) -> Text:
    """Get a styled Text object for the TUI header."""
    renderer = BannerRenderer(console, theme_manager)
    return renderer.get_tui_header_text()
