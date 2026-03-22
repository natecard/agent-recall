from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import typer

ErrorKind = Literal["usage", "config", "validation", "runtime"]


@dataclass
class CliError(Exception):
    message: str
    kind: ErrorKind = "runtime"
    exit_code: int = 1
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "error": {
                "kind": self.kind,
                "message": self.message,
            },
            "exit_code": int(self.exit_code),
        }
        if self.details:
            payload["error"]["details"] = self.details
        return payload


def exit_with_cli_error(error: CliError) -> None:
    raise typer.Exit(int(error.exit_code))


def as_cli_error(exc: Exception, *, default_code: int = 1) -> CliError:
    if isinstance(exc, CliError):
        return exc
    return CliError(str(exc), kind="runtime", exit_code=default_code)
