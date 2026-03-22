from __future__ import annotations

from typing import Any

from textual.widgets import Select

_MISSING = object()
_SELECT_NULL = getattr(Select, "NULL", _MISSING)
_SELECT_BLANK = getattr(Select, "BLANK", _MISSING)


def select_empty_value() -> Any:
    """Return the Textual no-selection sentinel across Select API variants."""
    if _SELECT_NULL is not _MISSING:
        return _SELECT_NULL
    if _SELECT_BLANK is not _MISSING:
        return _SELECT_BLANK
    return None


def is_select_empty(value: object) -> bool:
    """Return True when a Select value represents 'no selection'."""
    if value is None:
        return True
    if _SELECT_NULL is not _MISSING and value == _SELECT_NULL:
        return True
    if _SELECT_BLANK is not _MISSING and value == _SELECT_BLANK:
        return True
    return False
