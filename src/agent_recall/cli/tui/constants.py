from __future__ import annotations

from pathlib import Path

_LOADING_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_DEBUG_LOG_PATH = Path("/Users/natecard/OnHere/Repos/self-docs/.cursor/debug.log")
_RALPH_STREAM_FLUSH_SECONDS = 0.08
_RALPH_STREAM_FLUSH_MAX_CHARS = 8192

_PROVIDER_BASE_URL_DEFAULTS = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai-compatible": "http://localhost:8080/v1",
}

_RALPH_CLI_OPTIONS = [
    ("claude-code", "claude-code"),
    ("codex", "codex"),
    ("opencode", "opencode"),
]

_RALPH_CLI_DEFAULT_MODELS: dict[str, list[str]] = {
    "claude-code": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "codex": [
        "o4-mini",
        "gpt-4o",
        "gpt-5.3-codex",
        "gpt-5-codex",
    ],
    "opencode": [],
}

_COMPACT_MODE_OPTIONS: list[tuple[str, str]] = [
    ("always", "always"),
    ("on-failure", "on-failure"),
    ("off", "off"),
]
