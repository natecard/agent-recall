from __future__ import annotations

from rich.theme import Theme

# Theme definitions based on VSCode and popular themes
THEMES: dict[str, dict[str, str]] = {
    "dark+": {
        # VSCode Dark+ theme
        "error": "bright_red",
        "success": "bright_green",
        "warning": "bright_yellow",
        "info": "bright_cyan",
        "dim": "dim white",
        "bold": "bold white",
        "accent": "cyan",
        "table_header": "cyan",
        # Banner styles
        "banner.border": "bold bright_magenta",
        "banner.letter": "bold bright_cyan",
        "banner.shade": "dim blue",
        "banner.star": "bold bright_yellow",
        "banner.diamond": "bold yellow",
        "banner.tagline": "bold bright_white",
        "banner.accent": "bright_green",
        "banner.dot": "dim cyan",
    },
    "light+": {
        # VSCode Light+ theme
        "error": "red",
        "success": "green",
        "warning": "yellow",
        "info": "blue",
        "dim": "dim black",
        "bold": "bold black",
        "accent": "blue",
        "table_header": "blue",
        # Banner styles
        "banner.border": "bold blue",
        "banner.letter": "bold bright_blue",
        "banner.shade": "dim bright_black",
        "banner.star": "bold yellow",
        "banner.diamond": "bold magenta",
        "banner.tagline": "bold black",
        "banner.accent": "green",
        "banner.dot": "dim bright_black",
    },
    "monokai": {
        # Monokai theme
        "error": "#f92672",  # Pink-red
        "success": "#a6e22e",  # Green
        "warning": "#fd971f",  # Orange
        "info": "#66d9ef",  # Cyan-blue
        "dim": "dim #75715e",  # Dimmed gray
        "bold": "bold #f8f8f2",  # Bold off-white
        "accent": "#ae81ff",  # Purple
        "table_header": "#66d9ef",  # Cyan-blue
        # Banner styles
        "banner.border": "bold #f92672",  # Pink-red border
        "banner.letter": "bold #a6e22e",  # Green letters
        "banner.shade": "dim #75715e",  # Dimmed gray
        "banner.star": "bold #fd971f",  # Orange stars
        "banner.diamond": "bold #ae81ff",  # Purple diamonds
        "banner.tagline": "bold #f8f8f2",  # Off-white tagline
        "banner.accent": "#66d9ef",  # Cyan accent
        "banner.dot": "dim #75715e",  # Dimmed dots
    },
    "dracula": {
        # Dracula theme
        "error": "#ff5555",  # Red
        "success": "#50fa7b",  # Green
        "warning": "#f1fa8c",  # Yellow
        "info": "#8be9fd",  # Cyan
        "dim": "dim #6272a4",  # Dimmed comment color
        "bold": "bold #f8f8f2",  # Bold foreground
        "accent": "#bd93f9",  # Purple
        "table_header": "#8be9fd",  # Cyan
        # Banner styles
        "banner.border": "bold #ff79c6",  # Pink border
        "banner.letter": "bold #8be9fd",  # Cyan letters
        "banner.shade": "dim #6272a4",  # Comment color shade
        "banner.star": "bold #f1fa8c",  # Yellow stars
        "banner.diamond": "bold #bd93f9",  # Purple diamonds
        "banner.tagline": "bold #f8f8f2",  # Foreground tagline
        "banner.accent": "#50fa7b",  # Green accent
        "banner.dot": "dim #6272a4",  # Dimmed dots
    },
    "high-contrast-dark": {
        # High Contrast Dark (accessible)
        "error": "#f48771",  # Bright red-orange (high contrast)
        "success": "#89d185",  # Bright green (high contrast)
        "warning": "#dcdcaa",  # Bright yellow (high contrast)
        "info": "#4ec9b0",  # Bright cyan (high contrast)
        "dim": "dim #cccccc",  # Dimmed light gray
        "bold": "bold #ffffff",  # Bold white
        "accent": "#4fc1ff",  # Bright blue (high contrast)
        "table_header": "#4fc1ff",  # Bright blue
        # Banner styles
        "banner.border": "bold #ffffff",  # White border (max contrast)
        "banner.letter": "bold #4fc1ff",  # Bright blue letters
        "banner.shade": "#cccccc",  # Light gray shade
        "banner.star": "bold #dcdcaa",  # Yellow stars
        "banner.diamond": "bold #f48771",  # Red-orange diamonds
        "banner.tagline": "bold #ffffff",  # White tagline
        "banner.accent": "#89d185",  # Green accent
        "banner.dot": "#cccccc",  # Light gray dots
    },
    "high-contrast-light": {
        # High Contrast Light (accessible)
        "error": "#cd3131",  # Dark red (high contrast)
        "success": "#00bc00",  # Dark green (high contrast)
        "warning": "#949800",  # Dark yellow (high contrast)
        "info": "#0451a5",  # Dark blue (high contrast)
        "dim": "dim #333333",  # Dimmed dark gray
        "bold": "bold #000000",  # Bold black
        "accent": "#007acc",  # Dark blue (high contrast)
        "table_header": "#007acc",  # Dark blue
        # Banner styles
        "banner.border": "bold #000000",  # Black border (max contrast)
        "banner.letter": "bold #0451a5",  # Dark blue letters
        "banner.shade": "#333333",  # Dark gray shade
        "banner.star": "bold #949800",  # Yellow stars
        "banner.diamond": "bold #cd3131",  # Red diamonds
        "banner.tagline": "bold #000000",  # Black tagline
        "banner.accent": "#00bc00",  # Green accent
        "banner.dot": "#333333",  # Dark gray dots
    },
    "cyberpunk": {
        # Cyberpunk neon theme
        "error": "#ff2a6d",
        "success": "#05d9e8",
        "warning": "#f9c80e",
        "info": "#01c5c4",
        "dim": "dim #444444",
        "bold": "bold #ffffff",
        "accent": "#d300c5",
        "table_header": "#05d9e8",
        # Banner styles
        "banner.border": "bold #d300c5",  # Hot pink border
        "banner.letter": "bold #05d9e8",  # Electric cyan letters
        "banner.shade": "dim #7b2cbf",  # Purple shade
        "banner.star": "bold #f9c80e",  # Yellow neon stars
        "banner.diamond": "bold #ff2a6d",  # Neon red diamonds
        "banner.tagline": "bold #ffffff",  # White tagline
        "banner.accent": "#01c5c4",  # Teal accent
        "banner.dot": "dim #7b2cbf",  # Purple dots
    },
    "nord": {
        # Nord theme
        "error": "#bf616a",
        "success": "#a3be8c",
        "warning": "#ebcb8b",
        "info": "#88c0d0",
        "dim": "dim #4c566a",
        "bold": "bold #eceff4",
        "accent": "#81a1c1",
        "table_header": "#88c0d0",
        # Banner styles
        "banner.border": "bold #81a1c1",  # Frost blue border
        "banner.letter": "bold #88c0d0",  # Frost cyan letters
        "banner.shade": "dim #4c566a",  # Polar night shade
        "banner.star": "bold #ebcb8b",  # Aurora yellow stars
        "banner.diamond": "bold #b48ead",  # Aurora purple diamonds
        "banner.tagline": "bold #eceff4",  # Snow storm tagline
        "banner.accent": "#a3be8c",  # Aurora green accent
        "banner.dot": "dim #4c566a",  # Polar night dots
    },
}

DEFAULT_THEME = "dark+"


class ThemeManager:
    """Manages CLI themes and provides Rich Theme objects."""

    _instance: ThemeManager | None = None

    def __new__(cls, theme_name: str = DEFAULT_THEME) -> ThemeManager:
        """Singleton pattern for global theme access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, theme_name: str = DEFAULT_THEME):
        """Initialize theme manager with a theme name."""
        if self._initialized:
            return
        self.theme_name = theme_name
        self._theme = self._create_theme(theme_name)
        self._initialized = True

    @staticmethod
    def get_available_themes() -> list[str]:
        """Return list of available theme names."""
        return list(THEMES.keys())

    @staticmethod
    def is_valid_theme(theme_name: str) -> bool:
        """Check if a theme name is valid."""
        return theme_name in THEMES

    @staticmethod
    def get_theme_colors(theme_name: str) -> dict[str, str]:
        """Get raw color definitions for a theme."""
        if theme_name not in THEMES:
            theme_name = DEFAULT_THEME
        return THEMES[theme_name].copy()

    def _create_theme(self, theme_name: str) -> Theme:
        """Create a Rich Theme from theme definition."""
        if theme_name not in THEMES:
            theme_name = DEFAULT_THEME

        theme_def = THEMES[theme_name]
        return Theme(theme_def, inherit=True)

    def get_theme(self) -> Theme:
        """Get the current Rich Theme object."""
        return self._theme

    def set_theme(self, theme_name: str) -> None:
        """Change the current theme."""
        if not self.is_valid_theme(theme_name):
            available = ", ".join(self.get_available_themes())
            raise ValueError(f"Invalid theme: {theme_name}. Available: {available}")
        self.theme_name = theme_name
        self._theme = self._create_theme(theme_name)

    def get_theme_name(self) -> str:
        """Get the current theme name."""
        return self.theme_name

    def get_color(self, color_name: str) -> str:
        """Fetch a color dynamically from the current theme."""
        return THEMES[self.theme_name].get(color_name, "")

    def get_banner_styles(self) -> dict[str, str]:
        """Get banner-specific styles for current theme."""
        theme_colors = THEMES.get(self.theme_name, THEMES[DEFAULT_THEME])
        return {k: v for k, v in theme_colors.items() if k.startswith("banner.")}
