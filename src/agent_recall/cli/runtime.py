from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent_recall.cli.theme import DEFAULT_THEME, ThemeManager
from agent_recall.core.config import load_config
from agent_recall.core.onboarding import inject_stored_api_keys
from agent_recall.external_compaction import ExternalCompactionService
from agent_recall.external_compaction.service import WriteTarget
from agent_recall.external_compaction.write_guard import ExternalWriteScopeGuard
from agent_recall.llm import LLMConfig, create_llm_provider, ensure_provider_dependency
from agent_recall.storage import create_storage_backend
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage
from agent_recall.storage.remote import resolve_shared_db_path

AGENT_DIR = Path(".agent")
DB_PATH = AGENT_DIR / "state.db"

T = TypeVar("T")

_theme_manager = ThemeManager(DEFAULT_THEME)
console = Console(theme=_theme_manager.get_theme())


def get_console() -> Console:
    return console


def get_theme_manager() -> ThemeManager:
    """Get or initialize theme manager from config."""
    global _theme_manager
    global console
    if AGENT_DIR.exists():
        files = FileStorage(AGENT_DIR)
        config_dict = files.read_config()
        theme_name = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        if theme_name != _theme_manager.get_theme_name():
            _theme_manager.set_theme(theme_name)
            console = Console(theme=_theme_manager.get_theme())
    return _theme_manager


def ensure_initialized() -> None:
    if AGENT_DIR.exists():
        return
    get_theme_manager()
    console.print("[error]Not initialized. Run 'agent-recall init' first.[/error]")
    raise typer.Exit(1)


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    ensure_initialized()
    config = load_config(AGENT_DIR)
    return create_storage_backend(config, DB_PATH)


def get_files() -> FileStorage:
    ensure_initialized()
    config = load_config(AGENT_DIR)
    shared_tiers_dir: Path | None = None
    if config.storage.backend == "shared":
        try:
            resolved = resolve_shared_db_path(config.storage.shared.base_url)
            if isinstance(resolved, Path):
                shared_tiers_dir = resolved.parent
        except (NotImplementedError, ValueError):
            shared_tiers_dir = None
    return FileStorage(AGENT_DIR, shared_tiers_dir=shared_tiers_dir)


def get_llm():
    inject_stored_api_keys()
    files = get_files()
    config_dict = files.read_config()
    llm_config = LLMConfig(**config_dict.get("llm", {}))
    dependency_ok, dependency_message = ensure_provider_dependency(
        llm_config.provider,
        auto_install=True,
    )
    if not dependency_ok:
        raise RuntimeError(dependency_message or "Provider dependency setup failed.")
    if dependency_message:
        console.print(f"[dim]{dependency_message}[/dim]")
    return create_llm_provider(llm_config)


def run_with_spinner(description: str, action: Callable[[], T]) -> T:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description, total=None)
        return action()


def build_external_compaction_service(
    storage: Storage,
    files: FileStorage,
) -> ExternalCompactionService:
    return ExternalCompactionService(
        storage,
        files,
        agent_dir=AGENT_DIR,
        repo_root=Path.cwd(),
    )


def resolve_external_write_target(files: FileStorage) -> WriteTarget:
    guard = ExternalWriteScopeGuard.from_config(
        repo_root=Path.cwd(),
        config=files.read_config(),
    )
    return guard.resolve_target()


def resolve_external_write_target_override(
    files: FileStorage,
    override: str | None,
) -> WriteTarget:
    guard = ExternalWriteScopeGuard.from_config(
        repo_root=Path.cwd(),
        config=files.read_config(),
    )
    return guard.resolve_target(override)


def format_session_time(value: datetime | None) -> str:
    if value is None:
        return "-"
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return normalized.strftime("%Y-%m-%d %H:%M UTC")
