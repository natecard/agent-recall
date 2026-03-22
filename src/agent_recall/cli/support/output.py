from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import typer
from rich.console import Console

from agent_recall.cli.support.errors import CliError

SUPPORTED_FORMATS = ("table", "json", "md")


def normalize_output_format(
    value: str,
    *,
    allowed: Iterable[str],
    default: str,
) -> str:
    normalized = (value or default).strip().lower()
    allowed_set = {item.strip().lower() for item in allowed}
    if normalized in allowed_set:
        return normalized
    sorted_allowed = ", ".join(sorted(allowed_set))
    raise CliError(
        f"Invalid format. Use one of: {sorted_allowed}.",
        kind="usage",
        exit_code=2,
        details={"format": normalized, "allowed": sorted(allowed_set)},
    )


def print_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2))


def print_success_json(data: dict[str, Any], *, exit_code: int = 0) -> None:
    payload = {
        "status": "ok",
        "data": data,
        "exit_code": int(exit_code),
    }
    print_json(payload)


def emit_cli_error(
    console: Console,
    error: CliError,
    *,
    output_format: str = "table",
) -> None:
    normalized = output_format.strip().lower()
    if normalized == "json":
        print_json(error.to_payload())
        return
    console.print(f"[error]{error.message}[/error]")
    if error.details:
        for key, value in sorted(error.details.items()):
            console.print(f"[dim]- {key}: {value}[/dim]")
