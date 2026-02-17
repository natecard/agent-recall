from __future__ import annotations

import re
from typing import Any

_ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_OSC_ESCAPE_PATTERN = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")


def _strip_rich_markup(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text)


def _sanitize_activity_fragment(fragment: str) -> str:
    """Normalize raw terminal fragments for stable TUI rendering."""
    if not fragment:
        return ""
    text = _OSC_ESCAPE_PATTERN.sub("", fragment)
    text = _ANSI_ESCAPE_PATTERN.sub("", text)
    text = text.replace("\r", "\n")
    out: list[str] = []
    for char in text:
        if char == "\b":
            if out:
                out.pop()
            continue
        if ord(char) < 32 and char not in {"\n", "\t"}:
            continue
        out.append(char)
    return "".join(out)


def _clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def _source_checkbox_id(source_name: str) -> str:
    return f"setup_agent_{source_name.replace('-', '_')}"
