from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.theme import Theme

DiscoverModelsFn = Callable[[str, str | None, str | None], tuple[list[str], str | None]]
DiscoverCodingModelsFn = Callable[[str], tuple[list[str], str | None]]
ThemeDefaultsFn = Callable[[], tuple[list[str], str]]
ThemeRuntimeFn = Callable[[], tuple[str, Theme]]
ExecuteCommandFn = Callable[[str, int, int], tuple[bool, list[str]]]
ListSessionsForPickerFn = Callable[[int, bool], list[dict[str, Any]]]
ListPrdItemsForPickerFn = Callable[[], dict[str, Any]]
ThemeResolveFn = Callable[[str], Theme | None]
