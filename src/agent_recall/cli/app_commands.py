from __future__ import annotations

import asyncio
import io
import json
import shutil
import textwrap
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import UUID

import httpx
import typer
from packaging.version import Version
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.theme import Theme
from typer.testing import CliRunner

from agent_recall import __version__
from agent_recall.cli import ralph as ralph_cli
from agent_recall.cli.banner import print_banner
from agent_recall.cli.command_contract import (
    command_parity_report,
    get_command_contract,
    get_registered_cli_command_paths,
)
from agent_recall.cli.commands.context_flow import (
    ContextBundleWriteRequest,
    ContextRequest,
    assemble_standard_context,
    context_result_json_payload,
    execute_context_request,
    render_context_result_markdown,
    write_context_bundle,
)
from agent_recall.cli.commands.external_compaction_queue_flow import (
    ExternalCompactionQueueAdapter,
    QueueListRequest,
    QueueTransitionRequest,
    load_queue_rows,
    render_queue_table,
    transition_queue_rows,
)
from agent_recall.cli.commands.status_flow import StatusInputs, base_status_lines
from agent_recall.cli.commands.sync_flow import build_sync_request, run_sync
from agent_recall.cli.ralph import (
    load_prd_items,
    ralph_app,
    read_ralph_config,
    render_ralph_status,
)
from agent_recall.cli.support.errors import CliError
from agent_recall.cli.support.output import emit_cli_error, normalize_output_format, print_json
from agent_recall.cli.theme import DEFAULT_THEME, ThemeManager
from agent_recall.cli.tui import get_palette_actions
from agent_recall.cli.tui import router as tui_router
from agent_recall.cli.tui.commands.help_text import build_tui_help_lines
from agent_recall.cli.tui.views import DashboardRenderContext, build_tui_dashboard
from agent_recall.core.adapters import get_default_adapters
from agent_recall.core.background_sync import BackgroundSyncManager
from agent_recall.core.compact import CompactionEngine
from agent_recall.core.config import load_config
from agent_recall.core.context import ContextAssembler
from agent_recall.core.embedding_diagnostics import EmbeddingDiagnostics
from agent_recall.core.embedding_indexer import EmbeddingIndexer
from agent_recall.core.ingest import TranscriptIngestor
from agent_recall.core.log import LogWriter
from agent_recall.core.memory_pack import (
    PACK_FORMAT,
    PACK_VERSION,
    MergeStrategy,
    build_memory_pack,
    import_memory_pack,
    read_memory_pack,
    validate_memory_pack,
    write_memory_pack,
)
from agent_recall.core.onboarding import (
    apply_repo_setup,
    discover_coding_cli_models,
    discover_provider_models,
    ensure_repo_onboarding,
    get_onboarding_defaults,
    get_repo_preferred_sources,
    inject_stored_api_keys,
    is_interactive_terminal,
    is_repo_onboarding_complete,
)
from agent_recall.core.retrieval_feedback import evaluate_feedback_impact
from agent_recall.core.retrieve import Retriever
from agent_recall.core.rule_confidence import snapshot_rules
from agent_recall.core.session import SessionManager
from agent_recall.core.sync import AutoSync
from agent_recall.core.telemetry import PipelineTelemetry
from agent_recall.core.tier_compaction import (
    TierCompactionConfig,
    TierCompactionHook,
)
from agent_recall.core.tier_writer import (
    TierValidationError,
    TierWriter,
    WriteMode,
    WritePolicy,
    get_tier_statistics,
    lint_tier_file,
)
from agent_recall.core.topic_threads import build_topic_threads
from agent_recall.external_compaction import (
    ExternalCompactionService,
    run_external_compaction_mcp,
)
from agent_recall.external_compaction.models import ExternalNotesValidationError
from agent_recall.external_compaction.service import WriteTarget
from agent_recall.external_compaction.write_guard import ExternalWriteScopeGuard
from agent_recall.ingest import get_default_ingesters, get_ingester
from agent_recall.ingest.sources import (
    VALID_SOURCE_NAMES,
    normalize_source_name,
    resolve_source_location_hint,
)
from agent_recall.llm import (
    LLMProvider,
    Message,
    create_llm_provider,
    ensure_provider_dependency,
    get_available_providers,
    validate_provider_config,
)
from agent_recall.llm.coding_cli import CodingCLIProvider
from agent_recall.memory import (
    LocalVectorStore,
    MemoryPolicy,
)
from agent_recall.memory.migration import VectorMigrationRequest, VectorMigrationService
from agent_recall.ralph.costs import format_usd, summarize_costs
from agent_recall.ralph.iteration_store import IterationOutcome, IterationReportStore
from agent_recall.storage import create_storage_backend
from agent_recall.storage.base import Storage, UnsupportedStorageCapabilityError
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.metadata import attribution_fields as resolve_attribution_fields
from agent_recall.storage.migrations.migrate_to_embeddings import (
    get_migration_preview,
    migrate_database,
)
from agent_recall.storage.models import (
    CurationStatus,
    LLMConfig,
    LogEntry,
    PipelineEventAction,
    PipelineStage,
    RetrievalConfig,
    SemanticLabel,
)
from agent_recall.storage.remote import resolve_shared_db_path

app = typer.Typer(help="Agent Memory System - Persistent knowledge for AI coding agents")
_slash_runner = CliRunner()


def _print_version_and_maybe_update() -> None:
    """Print installed version and update hint if available."""
    current = __version__
    console.print(f"agent-recall {current}")

    latest = _fetch_latest_version()
    if latest is None:
        return

    try:
        if Version(latest) > Version(current):
            console.print(
                f"[dim]A newer version ({latest}) is available. "
                "Upgrade with: uv tool upgrade agent-recall[/dim]"
            )
    except Exception:  # noqa: BLE001
        pass


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    if version_flag:
        _print_version_and_maybe_update()
        raise typer.Exit()


config_app = typer.Typer(help="Manage onboarding and model configuration")
embedding_app = typer.Typer(help="Manage embeddings and semantic search")
context_app = typer.Typer(help="Build and refresh context bundles")
sync_app = typer.Typer(help="Synchronize native session sources")
tiers_app = typer.Typer(help="Manage tier files")
tiers_write_app = typer.Typer(help="Write structured entries to tier files")


# Initialize theme manager and console
_theme_manager = ThemeManager(DEFAULT_THEME)
console = Console(theme=_theme_manager.get_theme())

AGENT_DIR = Path(".agent")
DB_PATH = AGENT_DIR / "state.db"
T = TypeVar("T")
SOURCE_CHOICES_TEXT = ", ".join(VALID_SOURCE_NAMES)


INITIAL_GUARDRAILS = """# Guardrails

Rules and warnings for this codebase. Entries added automatically from agent sessions.
    """

INITIAL_RULES = """# Rules

User-authored operating rules for coding agents in this repository.
"""

INITIAL_STYLE = """# Style Guide

Coding patterns and preferences for this codebase. Entries added automatically from agent sessions.
"""

INITIAL_RECENT = """# Recent Sessions

Summaries of recent agent sessions.
"""

INITIAL_CONFIG = """# Agent Memory Configuration

llm:
  provider: anthropic
  model: claude-sonnet-4-20250514

compaction:
  backend: llm
  max_hours_before_compact: 24
  external:
    write_target: runtime
    allow_template_writes: false
    conflict_policy: prefer_newest
    pending_limit: 20
  max_recent_tokens: 1500
  max_sessions_before_compact: 5
  promote_pattern_after_occurrences: 3
  index_decision_entries: true
  index_decision_min_confidence: 0.7
  index_exploration_entries: true
  index_exploration_min_confidence: 0.7
  index_narrative_entries: false
  index_narrative_min_confidence: 0.8
  archive_sessions_older_than_days: 30

tier_compaction:
  auto_run: true
  max_entries_per_tier: 50
  strict_deduplication: false
  summary_threshold_entries: 40
  summary_max_entries: 20

retrieval:
  backend: fts5
  top_k: 5
  fusion_k: 60
  rerank_enabled: false
  rerank_candidate_k: 20
  semantic_index_enabled: false
  embedding_dimensions: 64

memory:
  mode: markdown
  vector_backend: local
  embedding_provider: local
  fusion_fts_weight: 0.4
  fusion_semantic_weight: 0.6
  feedback_weight: 0.2
  migration_batch_size: 100
  local_model_path: null
  external_embedding_base_url: null
  external_embedding_api_key_env: OPENAI_API_KEY
  external_embedding_model: text-embedding-3-small
  external_embedding_timeout_seconds: 10.0
  cost:
    max_external_embedding_usd: 1.0
    max_vector_records: 20000
  privacy:
    redaction_patterns: []
    retention_days: 90
  turbopuffer:
    base_url: null
    api_key_env: TURBOPUFFER_API_KEY
    timeout_seconds: 10.0
    retry_attempts: 2

storage:
  backend: local
  shared:
    base_url: null
    api_key_env: AGENT_RECALL_SHARED_API_KEY
    require_api_key: false
    timeout_seconds: 10.0
    retry_attempts: 2

telemetry:
  enabled: true

guardrails:
  enforcement:
    enabled: false

theme:
  name: dark+

ralph:
  enabled: false
  max_iterations: 10
  sleep_seconds: 2
  compact_mode: always
  forecast:
    window: 5
    use_llm: false
    llm_on_consecutive_failures: 2
    llm_model: null
  synthesis:
    auto_after_loop: true
    max_guardrails: 30
    max_style: 30

adapters:
  enabled: false
  output_dir: .agent/context
  default_token_budget: null
  per_adapter_token_budget: {}
"""


def _get_theme_manager() -> ThemeManager:
    """Get or initialize theme manager from config."""
    global _theme_manager
    if AGENT_DIR.exists():
        try:
            files = FileStorage(AGENT_DIR)
            config_dict = files.read_config()
            theme_name = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
            if theme_name != _theme_manager.get_theme_name():
                _theme_manager.set_theme(theme_name)
                # Update console with new theme
                global console
                console = Console(theme=_theme_manager.get_theme())
        except Exception:  # noqa: BLE001
            pass
    return _theme_manager


def ensure_initialized() -> None:
    if AGENT_DIR.exists():
        return
    _get_theme_manager()  # Ensure theme is loaded
    console.print("[error]Not initialized. Run 'agent-recall init' first.[/error]")
    raise typer.Exit(1)


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    """Get a database connection."""
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


def _build_external_compaction_service(
    storage: Storage,
    files: FileStorage,
) -> ExternalCompactionService:
    return ExternalCompactionService(
        storage,
        files,
        agent_dir=AGENT_DIR,
        repo_root=Path.cwd(),
    )


def _resolve_external_write_target(files: FileStorage) -> WriteTarget:
    guard = ExternalWriteScopeGuard.from_config(
        repo_root=Path.cwd(),
        config=files.read_config(),
    )
    return guard.resolve_target()


def _resolve_external_write_target_override(
    files: FileStorage,
    override: str | None,
) -> WriteTarget:
    guard = ExternalWriteScopeGuard.from_config(
        repo_root=Path.cwd(),
        config=files.read_config(),
    )
    return guard.resolve_target(override)


def _require_storage_capability(
    storage: Storage,
    capability: str,
    *,
    message: str,
) -> None:
    if bool(getattr(storage.capabilities, capability, False)):
        return
    console.print(f"[error]{message}[/error]")
    raise typer.Exit(1)


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


def _load_retrieval_config(files: FileStorage) -> RetrievalConfig:
    config_dict = files.read_config()
    retrieval_data = config_dict.get("retrieval", {})
    if retrieval_data is None:
        retrieval_data = {}
    if not isinstance(retrieval_data, dict):
        raise ValueError("Invalid retrieval configuration: 'retrieval' must be a mapping.")
    try:
        return RetrievalConfig.model_validate(retrieval_data)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid retrieval configuration: {exc}") from exc


def _normalize_retrieval_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"fts5", "hybrid", "vector_primary"}:
        return normalized
    raise ValueError("Invalid retrieval backend. Use 'fts5', 'hybrid', or 'vector_primary'.")


def _build_retriever(
    storage: Storage,
    files: FileStorage,
    *,
    backend: str | None = None,
    fusion_k: int | None = None,
    rerank: bool | None = None,
    rerank_candidate_k: int | None = None,
) -> tuple[Retriever, RetrievalConfig]:
    retrieval_cfg = _load_retrieval_config(files)
    config_dict = files.read_config()
    memory_cfg = config_dict.get("memory", {}) if isinstance(config_dict, dict) else {}
    if not isinstance(memory_cfg, dict):
        memory_cfg = {}
    mode_backend = {
        "markdown": "fts5",
        "hybrid": "hybrid",
        "vector_primary": "vector_primary",
    }
    configured_mode = str(memory_cfg.get("mode", "")).strip().lower()
    default_backend = retrieval_cfg.backend
    if retrieval_cfg.backend == "fts5":
        default_backend = mode_backend.get(configured_mode, retrieval_cfg.backend)
    selected_backend = (
        _normalize_retrieval_backend(backend) if backend is not None else default_backend
    )
    selected_fusion_k = fusion_k if fusion_k is not None else retrieval_cfg.fusion_k
    selected_rerank = retrieval_cfg.rerank_enabled if rerank is None else rerank
    selected_rerank_candidate_k = (
        rerank_candidate_k if rerank_candidate_k is not None else retrieval_cfg.rerank_candidate_k
    )
    try:
        fts_weight = float(memory_cfg.get("fusion_fts_weight", 0.4))
    except (TypeError, ValueError):
        fts_weight = 0.4
    try:
        semantic_weight = float(memory_cfg.get("fusion_semantic_weight", 0.6))
    except (TypeError, ValueError):
        semantic_weight = 0.6
    try:
        feedback_weight = float(memory_cfg.get("feedback_weight", 0.2))
    except (TypeError, ValueError):
        feedback_weight = 0.2
    retriever = Retriever(
        storage,
        backend=selected_backend,
        fusion_k=selected_fusion_k,
        rerank_enabled=selected_rerank,
        rerank_candidate_k=selected_rerank_candidate_k,
        fts_weight=fts_weight,
        semantic_weight=semantic_weight,
        feedback_weight=feedback_weight,
    )
    return retriever, retrieval_cfg


def run_with_spinner(description: str, action: Callable[[], T]) -> T:
    """Run a blocking action with a transient spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description, total=None)
        return action()


def _resolve_output_format(
    value: str,
    *,
    allowed: tuple[str, ...] = ("table", "json"),
    default: str = "table",
) -> str:
    try:
        return normalize_output_format(value, allowed=allowed, default=default)
    except CliError as exc:
        emit_cli_error(console, exc, output_format=value)
        raise typer.Exit(exc.exit_code) from None


def _as_clickable_uri(location: str) -> str:
    try:
        path = Path(location)
        if path.is_absolute():
            return path.as_uri()
    except (OSError, ValueError):
        return location
    return location


def _format_session_time(value: datetime | None) -> str:
    if value is None:
        return "-"
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return normalized.strftime("%Y-%m-%d %H:%M UTC")


def _compact_session_ref(session_id: str, max_chars: int = 18) -> str:
    cleaned = session_id.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    head = max_chars // 2 - 1
    tail = max_chars - head - 1
    return f"{cleaned[:head]}…{cleaned[-tail:]}"


def _conversation_table_mode(width: int) -> str:
    if width >= 150:
        return "wide"
    if width >= 120:
        return "medium"
    return "compact"


def _resolve_repo_root_for_display(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def _get_repo_selected_sources(files: FileStorage) -> list[str] | None:
    try:
        return get_repo_preferred_sources(files)
    except Exception:  # noqa: BLE001
        return None


def _filter_ingesters_by_sources(ingesters, selected_sources: list[str] | None):
    if not selected_sources:
        return ingesters
    allowed = set(selected_sources)
    filtered = [ingester for ingester in ingesters if ingester.source_name in allowed]
    return filtered if filtered else ingesters


def _tui_help_lines() -> list[str]:
    return build_tui_help_lines()


def _collect_registered_cli_command_paths() -> list[str]:
    return get_registered_cli_command_paths()


def get_default_prd_path() -> Path:
    return ralph_cli.get_default_prd_path()


def get_default_script_path() -> Path:
    return ralph_cli.get_default_script_path()


@app.command("command-inventory")
def command_inventory() -> None:
    """Print command inventory for CLI, TUI, and palette."""
    cli_commands = set(_collect_registered_cli_command_paths())
    tui_slash_commands: set[str] = set()
    for contract in get_command_contract():
        if "tui" not in contract.surfaces:
            continue
        tui_slash_commands.add(contract.command)
        tui_slash_commands.update(contract.aliases)
    palette_actions = {action.action_id for action in get_palette_actions()}
    parity = command_parity_report(
        cli_commands=cli_commands,
        tui_commands=tui_slash_commands,
    )

    table = Table(title="Command Inventory", box=box.SIMPLE)
    table.add_column("Surface")
    table.add_column("Count", justify="right")
    table.add_row("CLI", str(len(cli_commands)))
    table.add_row("TUI Slash", str(len(tui_slash_commands)))
    table.add_row("Palette", str(len(palette_actions)))
    console.print(table)

    contract_table = Table(title="Contract Commands", box=box.SIMPLE)
    contract_table.add_column("Command")
    contract_table.add_column("Aliases")
    contract_table.add_column("Surfaces")
    for contract in get_command_contract():
        aliases = ", ".join(contract.aliases) if contract.aliases else "-"
        surfaces = ", ".join(contract.surfaces)
        contract_table.add_row(contract.command, aliases, surfaces)
    console.print(contract_table)

    if (
        parity["missing_in_tui"]
        or parity["missing_in_cli"]
        or parity["extra_in_tui"]
        or parity["extra_in_cli"]
    ):
        parity_table = Table(title="Parity Gaps", box=box.SIMPLE)
        parity_table.add_column("Category")
        parity_table.add_column("Commands")
        parity_table.add_row(
            "Missing in TUI",
            ", ".join(sorted(parity["missing_in_tui"])) or "-",
        )
        parity_table.add_row(
            "Missing in CLI",
            ", ".join(sorted(parity["missing_in_cli"])) or "-",
        )
        parity_table.add_row(
            "Extra in TUI",
            ", ".join(sorted(parity["extra_in_tui"])) or "-",
        )
        parity_table.add_row(
            "Extra in CLI",
            ", ".join(sorted(parity["extra_in_cli"])) or "-",
        )
        console.print(parity_table)

    palette_table = Table(title="Palette Actions", box=box.SIMPLE)
    palette_table.add_column("Action")
    for action_id in sorted(palette_actions):
        palette_table.add_row(action_id)
    console.print(palette_table)


def _read_adapter_config(files: FileStorage) -> dict[str, Any]:
    config_dict = files.read_config()
    adapter_config = config_dict.get("adapters", {}) if isinstance(config_dict, dict) else {}
    return adapter_config if isinstance(adapter_config, dict) else {}


def _write_adapter_config(files: FileStorage, updates: dict[str, object]) -> dict[str, object]:
    config_dict = files.read_config()
    if "adapters" not in config_dict or not isinstance(config_dict.get("adapters"), dict):
        config_dict["adapters"] = {}
    adapter_config = config_dict["adapters"]
    if isinstance(adapter_config, dict):
        adapter_config.update(updates)
    config_dict["adapters"] = adapter_config
    files.write_config(config_dict)
    return config_dict


def _format_per_adapter_token_budgets(adapter_config: dict[str, object]) -> str:
    budgets = adapter_config.get("per_adapter_token_budget")
    if not isinstance(budgets, dict):
        return "none"
    normalized: list[str] = []
    for name, budget in budgets.items():
        if not isinstance(name, str):
            continue
        if not isinstance(budget, int):
            continue
        normalized.append(f"{name}={budget}")
    return ", ".join(normalized) if normalized else "none"


def _format_default_adapter_token_budget(adapter_config: dict[str, object]) -> str:
    value = adapter_config.get("default_token_budget")
    return str(value) if value is not None else "none"


def _format_named_token_budget_map(values: object) -> str:
    if not isinstance(values, dict):
        return "none"
    normalized: list[str] = []
    for name, budget in values.items():
        if not isinstance(name, str):
            continue
        if not isinstance(budget, int):
            continue
        normalized.append(f"{name}={budget}")
    return ", ".join(normalized) if normalized else "none"


def _parse_named_token_budget_map(raw_value: str, *, label: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    if not raw_value.strip():
        return parsed
    for raw in raw_value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            console.print(f"[error]Invalid {label} budget '{item}'. Use name=tokens.[/error]")
            raise typer.Exit(1)
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            console.print(f"[error]{label.title()} name cannot be empty.[/error]")
            raise typer.Exit(1)
        try:
            tokens = int(value)
        except ValueError:
            console.print(f"[error]Invalid token budget '{value}'. Use an integer.[/error]")
            raise typer.Exit(1)
        if tokens <= 0:
            console.print(f"[error]Token budget must be > 0 (got {tokens}).[/error]")
            raise typer.Exit(1)
        parsed[name] = tokens
    return parsed


def _entry_attribution_fields(entry: LogEntry) -> tuple[str, str, str]:
    return resolve_attribution_fields(
        entry.metadata,
        fallback_agent_source=entry.source.value,
        fallback_provider="unknown",
        fallback_model="unknown",
    )


def _collect_entries_for_attribution(
    storage: Storage,
    *,
    limit_per_status: int = 300,
) -> list[LogEntry]:
    entries: list[LogEntry] = []
    seen: set[UUID] = set()
    for status in (CurationStatus.APPROVED, CurationStatus.PENDING, CurationStatus.REJECTED):
        for entry in storage.list_entries_by_curation_status(status=status, limit=limit_per_status):
            if entry.id in seen:
                continue
            seen.add(entry.id)
            entries.append(entry)
    return entries


def _format_source_session_attribution(
    storage: Storage,
    source_session_ids: list[str],
) -> str:
    counts: dict[str, int] = {}
    for source_session_id in source_session_ids:
        entries = storage.get_entries_by_source_session(source_session_id, limit=200)
        for entry in entries:
            agent_source, provider, _model = _entry_attribution_fields(entry)
            key = f"{agent_source}/{provider}"
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "-"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top = [f"{name}:{count}" for name, count in ranked[:2]]
    return ", ".join(top)


def _read_memory_config(files: FileStorage) -> dict[str, Any]:
    config = files.read_config()
    memory_cfg = config.get("memory", {}) if isinstance(config, dict) else {}
    return memory_cfg if isinstance(memory_cfg, dict) else {}


def _memory_policy(files: FileStorage) -> MemoryPolicy:
    return MemoryPolicy.from_memory_config(_read_memory_config(files))


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
    no_splash: bool = typer.Option(False, "--no-splash", help="Skip splash screen"),
):
    """Initialize agent memory in the current repository."""
    _get_theme_manager()

    # Show splash banner unless disabled
    if not no_splash:
        console.clear()
        # Use 'medium' as default for init to ensure it shows properly
        print_banner(console, _theme_manager, animated=True, delay=0.012)
        console.print()  # Add spacing after banner

    if AGENT_DIR.exists() and not force:
        console.print("[warning].agent/ already exists. Use --force to reinitialize.[/warning]")
        raise typer.Exit(1)

    if AGENT_DIR.exists() and force:
        shutil.rmtree(AGENT_DIR)

    AGENT_DIR.mkdir(exist_ok=True)
    (AGENT_DIR / "logs").mkdir(exist_ok=True)
    (AGENT_DIR / "archive").mkdir(exist_ok=True)

    (AGENT_DIR / "RULES.md").write_text(INITIAL_RULES)
    (AGENT_DIR / "GUARDRAILS.md").write_text(INITIAL_GUARDRAILS)
    (AGENT_DIR / "STYLE.md").write_text(INITIAL_STYLE)
    (AGENT_DIR / "RECENT.md").write_text(INITIAL_RECENT)
    (AGENT_DIR / "config.yaml").write_text(INITIAL_CONFIG)

    # Ensure storage points at the newly initialized repository path.
    get_storage.cache_clear()
    get_storage()

    console.print(
        Panel.fit(
            "[success]✓ Initialized .agent/ directory[/success]\n\n"
            "Files created:\n"
            "  • config.yaml - Configuration\n"
            "  • RULES.md - User-authored agent rules\n"
            "  • GUARDRAILS.md - Hard rules and warnings\n"
            "  • STYLE.md - Patterns and preferences\n"
            "  • RECENT.md - Session summaries\n"
            "  • state.db - Session and log storage\n\n"
            "Next: [bold]agent-recall open[/bold] (TUI onboarding + dashboard)",
            title="agent-recall",
        )
    )


@app.command()
def splash(
    animated: bool = typer.Option(
        True,
        "--animated/--no-animated",
        "-a/-A",
        help="Enable/disable line-by-line animation",
    ),
    delay: float = typer.Option(
        0.015,
        "--delay",
        "-d",
        min=0.0,
        max=0.5,
        help="Animation delay between lines (seconds)",
    ),
    size: str = typer.Option(
        None,
        "--size",
        "-s",
        help="Force banner size: full, medium, compact, minimal (default: auto-detect)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show debug information about terminal size",
    ),
):
    """Display the Agent Recall splash banner."""
    _get_theme_manager()

    if debug:
        console.print(f"[dim]Terminal width: {console.width}[/dim]")
        console.print(f"[dim]Terminal height: {console.height}[/dim]")
        console.print(f"[dim]Force size: {size or 'auto'}[/dim]")
        console.print()

    console.clear()
    print_banner(console, _theme_manager, animated=animated, delay=delay)


@app.command()
def start(task: str = typer.Argument(..., help="Description of what this session is working on")):
    """Start a new session and output context for the agent."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()

    session_mgr = SessionManager(storage)
    try:
        retriever, retrieval_cfg = _build_retriever(storage, files)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None
    context_asm = ContextAssembler(
        storage,
        files,
        retriever=retriever,
        retrieval_top_k=retrieval_cfg.top_k,
    )

    session = session_mgr.start(task)
    context = context_asm.assemble(task=task)

    console.print(f"[dim]Session started: {session.id}[/dim]\n")
    console.print(context)


@app.command()
def log(
    content: str = typer.Argument(..., help="The observation or learning to log"),
    label: str = typer.Option(..., "--label", "-l", help="Semantic label"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """Log an observation or learning from the current session."""
    storage = get_storage()
    session_mgr = SessionManager(storage)
    log_writer = LogWriter(storage)

    _get_theme_manager()  # Ensure theme is loaded
    try:
        semantic_label = SemanticLabel(label)
    except ValueError:
        valid = ", ".join(item.value for item in SemanticLabel)
        console.print(f"[error]Invalid label '{label}'. Valid: {valid}[/error]")
        raise typer.Exit(1) from None

    active = session_mgr.get_active()
    session_id = active.id if active else None

    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []

    log_writer.log(
        content=content,
        label=semantic_label,
        session_id=session_id,
        tags=tag_list,
    )

    session_note = f" (session: {session_id})" if session_id else " (no active session)"
    console.print(f"[success]✓ Logged [{label}]{session_note}[/success]")


@app.command()
def end(summary: str = typer.Argument(..., help="Summary of what was accomplished")):
    """End the current session."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    session_mgr = SessionManager(storage)

    active = session_mgr.get_active()
    if not active:
        console.print("[warning]No active session to end.[/warning]")
        raise typer.Exit(1)

    session = session_mgr.end(active.id, summary)
    console.print(f"[success]✓ Session ended: {session.id}[/success]")
    console.print(f"[dim]Summary: {summary}[/dim]")
    console.print(f"[dim]Entries logged: {session.entry_count}[/dim]")


@context_app.callback(invoke_without_command=True)
def context(
    ctx: typer.Context,
    task: str | None = typer.Option(None, "--task", "-t", help="Task for relevant retrieval"),
    output_format: str = typer.Option("md", "--format", "-f", help="Output format: md or json"),
    for_pr: bool = typer.Option(
        False,
        "--for-pr",
        help="Render PR-scoped context using git diff and code-review template",
    ),
    base_ref: str = typer.Option(
        "HEAD~1",
        "--base-ref",
        help="Base git ref for --for-pr scope extraction",
    ),
    head_ref: str = typer.Option(
        "HEAD",
        "--head-ref",
        help="Head git ref for --for-pr scope extraction",
    ),
    max_diff_files: int = typer.Option(
        200,
        "--max-diff-files",
        min=1,
        help="Maximum changed files to include in --for-pr scope",
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        "-k",
        min=1,
        help="Override maximum number of retrieved chunks (default from config)",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Override retrieval backend (fts5 or hybrid)",
    ),
    fusion_k: int | None = typer.Option(
        None,
        "--fusion-k",
        min=1,
        help="Override hybrid fusion constant (default from config)",
    ),
    rerank: bool | None = typer.Option(
        None,
        "--rerank/--no-rerank",
        help="Override reranking behavior (default from config)",
    ),
    rerank_candidate_k: int | None = typer.Option(
        None,
        "--rerank-candidate-k",
        min=1,
        help="Override rerank candidate pool size (default from config)",
    ),
):
    """Output current context (guardrails, style, recent, relevant)."""
    if ctx.invoked_subcommand is not None:
        return
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()
    try:
        retriever, retrieval_cfg = _build_retriever(
            storage,
            files,
            backend=backend,
            fusion_k=fusion_k,
            rerank=rerank,
            rerank_candidate_k=rerank_candidate_k,
        )
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None
    request = ContextRequest(
        task=task,
        for_pr=for_pr,
        base_ref=base_ref,
        head_ref=head_ref,
        max_diff_files=max_diff_files,
        top_k=top_k,
    )
    result = execute_context_request(
        request=request,
        storage=storage,
        files=files,
        retriever=retriever,
        retrieval_top_k=retrieval_cfg.top_k,
    )

    try:
        normalized_format = normalize_output_format(
            output_format, allowed=("md", "json"), default="md"
        )
    except CliError as exc:
        emit_cli_error(console, exc, output_format=output_format)
        raise typer.Exit(exc.exit_code) from None

    if normalized_format == "md":
        render_context_result_markdown(result, console=console)
        return

    if normalized_format == "json":
        print_json({"status": "ok", "data": context_result_json_payload(result), "exit_code": 0})
        return


@context_app.command("refresh")
def refresh_context(
    task: str | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Task for relevant retrieval (defaults to active session task)",
    ),
    output_dir: Path = typer.Option(
        AGENT_DIR / "context",
        "--output-dir",
        help="Directory where refreshed context bundle files are written",
    ),
    adapter_payloads: bool | None = typer.Option(
        None,
        "--adapter-payloads/--no-adapter-payloads",
        help="Write adapter-ready context payloads for supported agents",
    ),
):
    """Refresh context bundle files for active task and repository state."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()

    session_mgr = SessionManager(storage)
    active = session_mgr.get_active()
    resolved_task = task or (active.task if active else None)

    try:
        retriever, retrieval_cfg = _build_retriever(storage, files)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None
    config = load_config(AGENT_DIR)
    adapter_cfg = config.adapters
    llm_cfg = config.llm
    if adapter_payloads is None:
        adapter_payloads = bool(adapter_cfg.enabled)
    markdown_path = output_dir / "context.md"
    json_path = output_dir / "context.json"

    retry_attempts = 3
    retry_backoff_seconds = 1.0
    diagnostics: list[str] = []
    output = ""
    adapters_written: dict[str, Path] = {}

    for attempt in range(1, retry_attempts + 1):
        try:
            output = assemble_standard_context(
                storage=storage,
                files=files,
                retriever=retriever,
                retrieval_top_k=retrieval_cfg.top_k,
                task=resolved_task,
            )
            bundle = write_context_bundle(
                ContextBundleWriteRequest(
                    context=output,
                    task=resolved_task,
                    active_session_id=str(active.id) if active else None,
                    repo_path=Path.cwd().resolve(),
                    output_dir=output_dir,
                    refreshed_at=datetime.now(UTC),
                    adapter_payloads=bool(adapter_payloads),
                    adapter_output_dir=Path(adapter_cfg.output_dir),
                    adapters=get_default_adapters(),
                    token_budget=adapter_cfg.default_token_budget,
                    per_adapter_budgets=adapter_cfg.per_adapter_token_budget,
                    per_provider_budgets=adapter_cfg.per_provider_token_budget,
                    per_model_budgets=adapter_cfg.per_model_token_budget,
                    provider=llm_cfg.provider,
                    model=llm_cfg.model,
                )
            )
            markdown_path = bundle.markdown_path
            json_path = bundle.json_path
            adapters_written = bundle.adapters_written
            break
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                f"Attempt {attempt}/{retry_attempts} failed: {type(exc).__name__}: {exc}"
            )
            if attempt < retry_attempts:
                time.sleep(retry_backoff_seconds * attempt)
                continue

            console.print(
                f"[error]Context refresh failed after {retry_attempts} attempt(s).[/error]"
            )
            for detail in diagnostics:
                console.print(f"[dim]- {detail}[/dim]")
            raise typer.Exit(1) from None

    lines = [
        "[success]✓ Context bundle refreshed[/success]",
        f"  Markdown: {markdown_path}",
        f"  JSON:     {json_path}",
    ]
    if adapter_payloads:
        adapter_names = ", ".join(adapters_written) if adapters_written else "-"
        adapter_dir = Path(adapter_cfg.output_dir)
        lines.append(f"  Adapters: {adapter_names}")
        lines.append(f"  Adapter dir: {adapter_dir}")
    if diagnostics:
        lines.append(f"  Retries:  {len(diagnostics)}")
    if resolved_task:
        lines.append(f"  Task:     {resolved_task}")
    else:
        lines.append("  Task:     none (set --task or start a session for task retrieval)")

    console.print("\n".join(lines))


@app.command()
def compact(force: bool = typer.Option(False, "--force", "-f", help="Force compaction")):
    """Run compaction to update knowledge tiers from logs."""
    storage = get_storage()
    files = get_files()

    _get_theme_manager()

    config_dict = files.read_config()
    compaction_cfg = config_dict.get("compaction", {}) if isinstance(config_dict, dict) else {}
    backend = compaction_cfg.get("backend", "llm") if isinstance(compaction_cfg, dict) else "llm"
    backend = str(backend).strip().lower()

    if backend == "mcp_external":
        service = _build_external_compaction_service(storage, files)
        raw_pending_limit = compaction_cfg.get("max_sessions_before_compact", 5)
        try:
            pending_limit = int(raw_pending_limit)
        except (TypeError, ValueError):
            pending_limit = 5
        pending = service.list_imported_conversations(
            limit=max(1, pending_limit),
            pending_only=True,
        )
        console.print(
            Panel.fit(
                "[success]✓ External compaction mode active[/success]\n\n"
                "No in-process synthesis was executed.\n"
                "Use an external agent via MCP or JSON payload workflow:\n"
                "  1) agent-recall external-compaction export --pending-only\n"
                "  2) Agent generates notes JSON\n"
                "  3) agent-recall external-compaction apply --input <notes.json>\n"
                "  4) Optional: agent-recall external-compaction mcp-server\n\n"
                f"Pending imported conversations: {len(pending)}",
                title="Compaction Results",
            )
        )
        return

    llm: LLMProvider
    if backend == "coding_cli":
        ralph_cfg = config_dict.get("ralph", {}) if isinstance(config_dict, dict) else {}
        coding_cli = ralph_cfg.get("coding_cli") if isinstance(ralph_cfg, dict) else None
        cli_model = ralph_cfg.get("cli_model") if isinstance(ralph_cfg, dict) else None

        if not coding_cli:
            console.print(
                "[error]compaction.backend is 'coding_cli' "
                "but ralph.coding_cli is not configured[/error]"
            )
            raise typer.Exit(1)

        try:
            llm = CodingCLIProvider(coding_cli=str(coding_cli), model=cli_model)
            console.print(f"[dim]Using coding CLI backend: {coding_cli}[/dim]")
        except Exception as exc:
            console.print(f"[error]Coding CLI provider error: {exc}[/error]")
            raise typer.Exit(1) from None
    else:
        try:
            llm = get_llm()
        except Exception as exc:
            console.print(f"[error]LLM configuration error: {exc}[/error]")
            raise typer.Exit(1) from None

    engine = CompactionEngine(storage, files, llm)
    results = run_with_spinner(
        "Running compaction...",
        lambda: asyncio.run(engine.compact(force=force)),
    )

    console.print(
        Panel.fit(
            f"[success]✓ Compaction complete[/success]\n\n"
            f"Backend: {backend}\n"
            f"Guardrails updated: {results['guardrails_updated']}\n"
            f"Style updated: {results['style_updated']}\n"
            f"Recent updated: {results['recent_updated']}\n"
            f"LLM requests: {results.get('llm_requests', 0)} "
            f"(responses: {results.get('llm_responses', 0)})\n"
            f"Chunks indexed: {results['chunks_indexed']}",
            title="Results",
        )
    )


@sync_app.callback(invoke_without_command=True)
def sync(
    ctx: typer.Context,
    compact: bool = typer.Option(
        True,
        "--compact/--no-compact",
        help="Run compaction after sync",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help=f"Only sync from specific source: {SOURCE_CHOICES_TEXT}",
    ),
    since_days: int | None = typer.Option(
        None,
        "--since-days",
        "-d",
        help="Only sync sessions from the last N days",
    ),
    session_id: list[str] = typer.Option(
        None,
        "--session-id",
        help="Only sync specific source session ID(s); repeat option to include multiple",
    ),
    max_sessions: int | None = typer.Option(
        None,
        "--max-sessions",
        "-n",
        min=1,
        help="Limit sync to the most recent N discovered sessions",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force compaction even if no new learnings",
    ),
    skip_embeddings: bool = typer.Option(
        False,
        "--skip-embeddings",
        help="Skip computing semantic embeddings for chunks",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    cursor_db_path: Path | None = typer.Option(
        None,
        "--cursor-db-path",
        help="Path to a specific Cursor state.vscdb for testing/override",
    ),
    cursor_storage_dir: Path | None = typer.Option(
        None,
        "--cursor-storage-dir",
        help="Override Cursor workspaceStorage root path",
    ),
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Include Cursor sessions from all workspaces (not only current repo match)",
    ),
):
    """Automatically discover and process native agent sessions."""
    if ctx.invoked_subcommand is not None:
        return
    storage = get_storage()
    files = get_files()

    _get_theme_manager()  # Ensure theme is loaded
    try:
        llm = get_llm()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]LLM configuration error: {exc}[/error]")
        console.print("[dim]Check .agent/config.yaml or required API key env vars.[/dim]")
        raise typer.Exit(1) from None

    normalized_source = normalize_source_name(source) if source else None
    selected_sources = None if source else _get_repo_selected_sources(files)
    request = build_sync_request(
        compact=compact,
        skip_embeddings=skip_embeddings,
        force_compact=force,
        source=source,
        since_days=since_days,
        session_id=session_id,
        max_sessions=max_sessions,
        normalized_source=normalized_source,
        selected_sources=selected_sources,
    )

    if source:
        ingester_source = normalized_source or source
        try:
            ingesters = [
                get_ingester(
                    ingester_source,
                    cursor_db_path=cursor_db_path,
                    workspace_storage_dir=cursor_storage_dir,
                    cursor_all_workspaces=all_cursor_workspaces,
                )
            ]
        except ValueError as exc:
            console.print(f"[error]{exc}[/error]")
            raise typer.Exit(1) from None
    else:
        ingesters = get_default_ingesters(
            cursor_db_path=cursor_db_path,
            workspace_storage_dir=cursor_storage_dir,
            cursor_all_workspaces=all_cursor_workspaces,
        )
        ingesters = _filter_ingesters_by_sources(ingesters, selected_sources)

    def _sync_progress_logger(event: dict[str, Any]) -> None:
        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        event_name = str(event.get("event", ""))
        if event_name == "extraction_session_started":
            source_name = str(event.get("source", "?"))
            source_session_id = str(event.get("session_id", "?"))
            total_messages = event.get("messages_total")
            message_text = str(_as_int(total_messages, -1)) if total_messages is not None else "?"
            if message_text == "-1":
                message_text = "?"
            batch_size = event.get("messages_per_batch")
            batch_text = (
                f", batch size {str(_as_int(batch_size, -1))}"
                if batch_size is not None and _as_int(batch_size, -1) > 0
                else ""
            )
            console.print(
                f"[dim]{source_name}:{source_session_id} extraction started "
                f"({message_text} messages{batch_text}).[/dim]"
            )
            return

        if event_name == "extraction_batch_size_adjusted":
            source_name = str(event.get("source", "?"))
            source_session_id = str(event.get("session_id", "?"))
            old_size = _as_int(event.get("old_messages_per_batch"), 0)
            new_size = _as_int(event.get("new_messages_per_batch"), 0)
            attempt = _as_int(event.get("attempt"), 0)
            max_attempts = _as_int(event.get("max_attempts"), 0)
            console.print(
                f"[warning]{source_name}:{source_session_id} rate limited; reducing batch size "
                f"from {old_size} to {new_size} (attempt {attempt}/{max_attempts}).[/warning]"
            )
            return

        if event_name == "extraction_retry_scheduled":
            source_name = str(event.get("source", "?"))
            source_session_id = str(event.get("session_id", "?"))
            reason = str(event.get("reason", "retry")).replace("_", " ")
            next_attempt = _as_int(event.get("next_attempt"), 0)
            max_attempts = _as_int(event.get("max_attempts"), 0)
            delay_seconds = _as_float(event.get("delay_seconds"), 0.0)
            retry_after = event.get("retry_after_seconds")
            retry_after_text = ""
            if retry_after is not None:
                retry_after_value = _as_float(retry_after, 0.0)
                if retry_after_value > 0:
                    retry_after_text = f", retry-after={retry_after_value:.1f}s"
            console.print(
                f"[warning]{source_name}:{source_session_id} {reason}; retrying in "
                f"{delay_seconds:.1f}s (attempt {next_attempt}/{max_attempts}{retry_after_text})."
                "[/warning]"
            )
            return

        if event_name != "extraction_batch_complete":
            return

        source_name = str(event.get("source", "?"))
        source_session_id = str(event.get("session_id", "?"))
        processed = _as_int(event.get("messages_processed"), 0)
        total = _as_int(event.get("messages_total"), 0)
        batch_index = _as_int(event.get("batch_index"), 0)
        batch_count = _as_int(event.get("batch_count"), 0)
        learnings = _as_int(event.get("batch_learnings"), 0)
        attempt = _as_int(event.get("attempt"), 0)
        max_attempts = _as_int(event.get("max_attempts"), 0)
        attempt_suffix = (
            f", attempt {attempt}/{max_attempts}" if attempt > 1 and max_attempts > 1 else ""
        )
        console.print(
            f"[dim]{source_name}:{source_session_id} sent {processed}/{total} messages "
            f"to LLM (batch {batch_index}/{batch_count}, learnings={learnings}"
            f"{attempt_suffix}).[/dim]"
        )

    auto_sync = AutoSync(storage, files, llm, ingesters)
    if hasattr(auto_sync, "progress_callback"):
        auto_sync.progress_callback = _sync_progress_logger

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Syncing sessions...", total=None)

        try:
            results = run_sync(auto_sync, request)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[error]Sync failed: {exc}[/error]")
            raise typer.Exit(1) from None

    lines = [""]
    if request.selected_sources and not request.source:
        lines.append(f"Using configured sources: {', '.join(request.selected_sources)}")
    if request.session_ids:
        lines.append(f"Requested session IDs: {len(request.session_ids)}")
    if request.max_sessions is not None:
        lines.append(f"Max sessions: {request.max_sessions}")
    lines.append(f"Sessions discovered: {results['sessions_discovered']}")
    lines.append(f"Sessions processed:  {results['sessions_processed']}")
    if int(results["sessions_skipped"]) > 0:
        lines.append(
            f"Sessions skipped:    {results['sessions_skipped']} "
            f"(already processed: {int(results.get('sessions_already_processed', 0))}, "
            f"empty: {int(results.get('empty_sessions', 0))})"
        )
    lines.append(f"Learnings extracted: {results['learnings_extracted']}")
    lines.append(f"LLM extraction requests: {results.get('llm_requests', 0)}")
    if request.session_ids and int(results["sessions_processed"]) == 0:
        already_processed = int(results.get("sessions_already_processed", 0))
        if already_processed == int(results["sessions_discovered"]) and already_processed > 0:
            lines.append(
                "[warning]Selected sessions were already processed; ingestion did not "
                "rerun.[/warning]"
            )
            lines.append(
                "[dim]Use `agent-recall sync reset --session-id <id>` to reprocess a session.[/dim]"
            )

    if verbose and results.get("by_source"):
        lines.append("")
        lines.append("[bold]By source:[/bold]")
        by_source = results["by_source"]
        for source_name, source_stats in by_source.items():
            lines.append(
                f"  {source_name}: {source_stats['processed']}/{source_stats['discovered']} "
                f"processed, {source_stats['learnings']} learnings"
            )
            lines.append(f"    [dim]LLM extraction requests: {source_stats['llm_batches']}[/dim]")
            if int(source_stats.get("already_processed", 0)) > 0:
                lines.append(
                    f"    [dim]{source_stats['already_processed']} session(s) skipped as already "
                    "processed[/dim]"
                )
            if int(source_stats.get("empty", 0)) > 0:
                lines.append(
                    f"    [dim]{source_stats['empty']} discovered session(s) had no parseable "
                    "messages and were skipped[/dim]"
                )
    session_diagnostics = results.get("session_diagnostics") or []
    if verbose and session_diagnostics:
        lines.append("")
        lines.append("[bold]Session diagnostics:[/bold]")
        for item in session_diagnostics[:30]:
            source_name = str(item.get("source", "?"))
            source_session_id = str(item.get("session_id", "?"))
            status = str(item.get("status", "unknown"))
            message_count = item.get("message_count")
            learnings = int(item.get("learnings_extracted", 0))
            message_text = (
                f"messages={message_count}" if isinstance(message_count, int) else "messages=?"
            )
            lines.append(
                f"  {source_name}:{source_session_id}  status={status}  "
                f"{message_text}  learnings={learnings}"
            )
            warning = item.get("warning")
            if warning:
                lines.append(f"    [warning]{warning}[/warning]")
        if len(session_diagnostics) > 30:
            lines.append(f"  [dim]... and {len(session_diagnostics) - 30} more session(s)[/dim]")

    if "compaction" in results:
        comp = results["compaction"]
        lines.append("")
        lines.append("[bold]Knowledge synthesis:[/bold]")
        lines.append(f"  Backend:            {comp.get('backend', 'llm')}")
        lines.append(
            f"  LLM requests:       {comp.get('llm_requests', 0)} "
            f"(responses: {comp.get('llm_responses', 0)})"
        )
        lines.append(f"  Guardrails updated: {'✓' if comp.get('guardrails_updated') else '-'}")
        lines.append(f"  Style updated:      {'✓' if comp.get('style_updated') else '-'}")
        lines.append(f"  Recent updated:     {'✓' if comp.get('recent_updated') else '-'}")
        lines.append(f"  Chunks indexed:     {comp.get('chunks_indexed', 0)}")
        if comp.get("deferred"):
            reason = str(comp.get("deferred_reason", "threshold"))
            session_threshold = int(comp.get("session_threshold", 0))
            sessions_processed = int(comp.get("sessions_processed", 0))
            token_threshold = int(comp.get("token_threshold", 0))
            recent_tokens = int(comp.get("recent_tokens", 0))
            age_threshold = comp.get("age_threshold_hours")
            hours_since_recent = comp.get("hours_since_recent")
            lines.append(
                "  [warning]Status: deferred until auto-compaction threshold is met.[/warning]"
            )
            lines.append(f"  Deferred reason:    {reason}")
            lines.append(
                f"  Session threshold:  {sessions_processed}/{session_threshold} sessions processed"
            )
            lines.append(f"  Token threshold:    {recent_tokens}/{token_threshold} recent tokens")
            if isinstance(age_threshold, int | float):
                if isinstance(hours_since_recent, int | float):
                    lines.append(
                        f"  Age threshold:      {float(hours_since_recent):.1f}/"
                        f"{float(age_threshold):.1f} hours since RECENT update"
                    )
                else:
                    lines.append(
                        f"  Age threshold:      unknown/{float(age_threshold):.1f} hours "
                        "since RECENT update"
                    )
            lines.append("  Next step: agent-recall sync --compact --force")
        if comp.get("external_required"):
            lines.append("  External compaction required: yes")
            lines.append(
                f"  Pending conversations: {int(comp.get('pending_external_conversations', 0))}"
            )
            lines.append(
                "  [warning]Synthesis deferred until external notes are applied.[/warning]"
            )
            lines.append("  Next step: agent-recall external-compaction export --pending-only")
            lines.append(
                "  Remediation: agent-recall external-compaction apply "
                "--input <notes.json> --commit"
            )
        changed_files = []
        if comp.get("guardrails_updated"):
            changed_files.append(".agent/GUARDRAILS.md")
        if comp.get("style_updated"):
            changed_files.append(".agent/STYLE.md")
        if comp.get("recent_updated"):
            changed_files.append(".agent/RECENT.md")
        lines.append(
            "  Updated files:      " + (", ".join(changed_files) if changed_files else "none")
        )
    elif compact:
        lines.append("")
        lines.append("[bold]Knowledge synthesis:[/bold]")
        lines.append("  Status: skipped (no new learnings extracted during this sync run)")
        lines.append("  Remediation: agent-recall sync --compact --force")
        lines.append(
            "  Review pending sessions: agent-recall external-compaction list --pending-only"
        )

    if results.get("errors"):
        errors = results["errors"]
        lines.append("")
        lines.append(f"[warning]Warnings: {len(errors)}[/warning]")
        if verbose:
            for error in errors[:10]:
                lines.append(f"  [dim]- {error}[/dim]")
            if len(errors) > 10:
                lines.append(f"  [dim]... and {len(errors) - 10} more[/dim]")

    lines.append("")

    if int(results["learnings_extracted"]) > 0:
        title = "[success]✓ Sync Complete[/success]"
    elif int(results["sessions_processed"]) > 0:
        title = "[warning]Sync Complete (no learnings extracted)[/warning]"
    else:
        title = "[dim]Sync Complete (no new sessions)[/dim]"

    console.print(Panel.fit("\n".join(lines), title=title))


@embedding_app.command("stats")
def embedding_stats(
    stale_days: int = typer.Option(
        90,
        "--stale-days",
        min=0,
        help="Threshold in days to consider embeddings stale",
    ),
):
    """Show embedding coverage, similarity diagnostics, and size estimates."""
    storage = get_storage()
    diagnostics = EmbeddingDiagnostics(storage)

    coverage = diagnostics.get_coverage_stats()
    similarity = diagnostics.get_similarity_distribution()
    stale = diagnostics.check_stale_embeddings(threshold_days=stale_days)
    size = diagnostics.estimate_embedding_size()

    lines = [
        f"Total chunks:      {coverage['total_chunks']}",
        f"Embedded chunks:   {coverage['embedded_chunks']}",
        f"Coverage:          {float(coverage['coverage_percent']):.1f}%",
        f"Pending:           {coverage['pending']}",
        "",
        "Similarity distribution:",
        (
            f"  mean={similarity['mean']:.4f} median={similarity['median']:.4f} "
            f"std={similarity['std']:.4f} min={similarity['min']:.4f} "
            f"max={similarity['max']:.4f}"
        ),
        "",
        f"Stale chunks (>{stale['threshold_days']}d): {stale['stale_chunks']}",
        (
            f"Estimated embedding bytes: {size['total_bytes']} "
            f"(~{float(size['per_chunk_kb']):.2f} KB/chunk)"
        ),
    ]
    console.print(Panel.fit("\n".join(lines), title="Embedding Stats"))


@embedding_app.command("reindex")
def embedding_reindex(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-indexing all chunks (mark all as stale first)",
    ),
    max_chunks: int = typer.Option(
        0,
        "--max-chunks",
        "-n",
        min=0,
        help="Limit number of chunks to re-index (0 = all)",
    ),
):
    """Re-index all embeddings (or a subset) for semantic search."""
    storage = get_storage()

    if not force:
        console.print("[error]--force flag is required to re-index embeddings.[/error]")
        console.print("[dim]This will re-embed all chunks. Use --force to confirm.[/dim]")
        raise typer.Exit(1)

    _get_theme_manager()

    indexer = EmbeddingIndexer(storage)

    stats_before = indexer.get_indexing_stats()
    console.print(
        f"[dim]Before: {stats_before['embedded_chunks']}/{stats_before['total_chunks']} "
        "chunks embedded[/dim]"
    )

    if max_chunks > 0:
        console.print(f"[dim]Limiting to {max_chunks} chunks...[/dim]")

    console.print("[dim]Re-indexing chunks without embeddings...[/dim]")

    result = indexer.index_missing_embeddings(max_chunks=max_chunks)

    stats_after = indexer.get_indexing_stats()
    console.print(
        Panel.fit(
            f"Indexed: {result['indexed']}\n"
            f"Skipped: {result['skipped']}\n"
            f"Total embedded: {stats_after['embedded_chunks']}/{stats_after['total_chunks']}",
            title="Reindex Complete",
        )
    )


@embedding_app.command("search")
def embedding_search(
    query: str | None = typer.Option(
        None,
        "--query",
        "-q",
        help="Search query (omit for interactive mode)",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        min=1,
        max=20,
        help="Number of results to return",
    ),
):
    """Test semantic search interactively."""
    _get_theme_manager()
    storage = get_storage()

    if query is None:
        query = typer.prompt("Enter search query")

    retriever = Retriever(storage)

    console.print(f"[dim]Searching for: {query}[/dim]")

    results = retriever.search_hybrid(query=str(query), top_k=top_k)

    if not results:
        console.print("[warning]No results found.[/warning]")
        return

    table = Table(title=f"Top {len(results)} Results")
    table.add_column("#", style="dim", width=4)
    table.add_column("Content", style="default")

    for i, chunk in enumerate(results, 1):
        content_preview = chunk.content[:100].replace("\n", " ")
        if len(chunk.content) > 100:
            content_preview += "..."
        table.add_row(str(i), content_preview)

    console.print(table)


EMBEDDING_TEST_QUERIES = [
    "JWT authentication",
    "database connection",
    "API error handling",
    "user login",
    "cache invalidation",
    "file upload",
    "password reset",
    "session management",
    "rate limiting",
    "error logging",
]


@embedding_app.command("test-quality")
def embedding_test_quality():
    """Run test queries to evaluate embedding search quality."""
    _get_theme_manager()
    storage = get_storage()

    diagnostics = EmbeddingDiagnostics(storage)
    coverage = diagnostics.get_coverage_stats()

    if coverage["embedded_chunks"] == 0:
        console.print(
            "[error]No embeddings found. Run 'agent-recall embedding reindex --force' "
            "first.[/error]"
        )
        raise typer.Exit(1)

    retriever = Retriever(storage)

    console.print(
        Panel.fit(
            f"Testing embedding quality with {len(EMBEDDING_TEST_QUERIES)} queries",
            title="Embedding Quality Test",
        )
    )

    results: list[dict[str, str | float]] = []

    for query in EMBEDDING_TEST_QUERIES:
        search_results = retriever.search_hybrid(query=query, top_k=1)
        if search_results:
            result_text = search_results[0].content[:60].replace("\n", " ")
            results.append(
                {
                    "query": query,
                    "top_result": result_text,
                    "has_result": True,
                }
            )
            console.print(f"[dim]{query}:[/dim] {result_text}")
        else:
            results.append(
                {
                    "query": query,
                    "top_result": "(no results)",
                    "has_result": False,
                }
            )
            console.print(f"[dim]{query}:[/dim] [warning](no results)[/warning]")

    total_with_results = sum(1 for r in results if r["has_result"])
    success_rate = (total_with_results / len(results)) * 100

    console.print(
        Panel.fit(
            f"Queries with results: {total_with_results}/{len(results)} ({success_rate:.0f}%)",
            title="Quality Test Summary",
        )
    )


@embedding_app.command("benchmark")
def embedding_benchmark_cmd(
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Run quick benchmark (1K chunks only)",
    ),
    size: str = typer.Option(
        "1k",
        "--size",
        help="Chunk count: 1k, 10k, or 100k",
    ),
):
    """Run performance benchmarks for embedding pipeline."""
    import shutil
    import tempfile
    import time

    from agent_recall.core.embedding_indexer import EmbeddingIndexer
    from agent_recall.core.retrieve import Retriever
    from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel
    from agent_recall.storage.sqlite import SQLiteStorage

    _get_theme_manager()

    chunk_counts = {"1k": 1000, "10k": 10000, "100k": 100000}

    if quick:
        sizes = [1000]
    else:
        size_lower = size.lower().strip()
        if size_lower not in chunk_counts:
            console.print(f"[error]Invalid size: {size}. Use 1k, 10k, or 100k.[/error]")
            raise typer.Exit(1)
        sizes = [chunk_counts[size_lower]]

    console.print(
        Panel.fit(
            "Embedding Pipeline Benchmark",
            title="Benchmark",
        )
    )

    for n_chunks in sizes:
        console.print(f"\n[bold]Running benchmark for {n_chunks} chunks...[/bold]")

        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "benchmark.db"

        try:
            storage = SQLiteStorage(db_path)

            topics = [
                "authentication",
                "database",
                "api",
                "cache",
                "security",
                "performance",
            ]

            console.print(f"  Creating {n_chunks} test chunks...")
            for i in range(n_chunks):
                topic = topics[i % len(topics)]
                content = f"Benchmark chunk {i} about {topic} with sample content for testing."
                chunk = Chunk(
                    source=ChunkSource.MANUAL,
                    source_ids=[],
                    content=content,
                    label=SemanticLabel.PATTERN,
                )
                storage.store_chunk(chunk)

            console.print("  Indexing embeddings...")
            indexer = EmbeddingIndexer(storage, batch_size=32)

            start_time = time.time()
            indexer.index_missing_embeddings()
            indexing_time = time.time() - start_time

            chunks_per_sec = n_chunks / indexing_time if indexing_time > 0 else 0

            console.print("  Running retrieval queries...")
            retriever = Retriever(storage, backend="hybrid")

            queries = [
                "authentication token",
                "database connection",
                "api endpoint",
            ]

            start_time = time.time()
            for _ in range(10):
                for query in queries:
                    retriever.search_hybrid(query, top_k=5)
            retrieval_time = time.time() - start_time
            queries_run = 10 * len(queries)
            ms_per_query = (retrieval_time / queries_run) * 1000

            db_size_mb = db_path.stat().st_size / (1024 * 1024)
            kb_per_chunk = (db_path.stat().st_size / n_chunks) / 1024

            table = Table(title=f"Results: {n_chunks} chunks")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Indexing Time", f"{indexing_time:.2f}s")
            table.add_row("Indexing Speed", f"{chunks_per_sec:.1f} chunks/sec")
            table.add_row("Retrieval Time", f"{retrieval_time:.2f}s")
            table.add_row("Latency per Query", f"{ms_per_query:.1f} ms")
            table.add_row("DB Size", f"{db_size_mb:.2f} MB")
            table.add_row("Size per Chunk", f"{kb_per_chunk:.2f} KB")

            console.print(table)

        finally:
            shutil.rmtree(temp_dir)

    console.print("\n[dim]For full benchmark results, see docs/BENCHMARKS.md[/dim]")


@embedding_app.command("migrate-embeddings")
def migrate_embeddings_cmd(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview migration without making changes",
    ),
    no_backup: bool = typer.Option(
        False,
        "--no-backup",
        help="Skip database backup before migration",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
):
    """Migrate existing database to support semantic embeddings."""
    from agent_recall.storage.sqlite import SQLiteStorage

    storage = get_storage()

    if not isinstance(storage, SQLiteStorage):
        console.print("[error]Migration is only supported for SQLite storage backends.[/error]")
        raise typer.Exit(1)

    preview = get_migration_preview(storage)

    if preview["needs_migration"]:
        console.print(
            Panel.fit(
                "Schema migration required\n"
                f"Total chunks: {preview['total_chunks']}\n"
                f"Pending embeddings: {preview['pending_chunks']}",
                title="Migration Preview",
            )
        )
    else:
        console.print(
            Panel.fit(
                "Database already has embedding support\n"
                f"Total chunks: {preview['total_chunks']}\n"
                f"Embedded: {preview['embedded_chunks']}\n"
                f"Pending: {preview['pending_chunks']}",
                title="Migration Preview",
            )
        )

    if dry_run:
        console.print("[dim]Dry run complete. No changes made.[/dim]")
        return

    if not force:
        confirm = typer.prompt(
            "Proceed with migration? (backup will be created unless --no-backup is set)",
            default="n",
        )
        if confirm.lower() not in ("y", "yes"):
            console.print("[dim]Migration cancelled.[/dim]")
            return

    console.print("[dim]Starting migration...[/dim]")

    result = migrate_database(storage, backup=not no_backup)

    if result.get("error"):
        console.print(f"[error]Migration failed: {result['error']}[/error]")
        raise typer.Exit(1)

    if result["already_migrated"]:
        console.print(
            Panel.fit(
                f"Indexing complete\n"
                f"Embedded: {result['indexed']} new chunks\n"
                f"Total embedded: {result['embedded_after']}/{result['chunks_after']}",
                title="Migration Complete",
            )
        )
    else:
        backup_info = (
            f"\nBackup: {result['backup_path']}"
            if result["backup_path"]
            else "\n(No backup created)"
        )
        console.print(
            Panel.fit(
                f"Migration complete{backup_info}\n"
                f"Indexed: {result['indexed']} chunks\n"
                f"Total embedded: {result['embedded_after']}/{result['chunks_after']}\n\n"
                "To restore from backup:\n"
                f"  cp {result['backup_path']} <db-path>",
                title="Migration Complete",
            )
        )


@sync_app.command("background")
def sync_background(
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help=f"Only sync from specific source: {SOURCE_CHOICES_TEXT}",
    ),
    max_sessions: int | None = typer.Option(
        None,
        "--max-sessions",
        "-n",
        min=1,
        help="Limit sync to the most recent N discovered sessions",
    ),
    no_compact: bool = typer.Option(
        False,
        "--no-compact",
        help="Skip compaction after sync",
    ),
):
    """Run sync in background with safe locking (prevents duplicate syncs)."""
    storage = get_storage()
    files = get_files()

    _get_theme_manager()

    # Check if already running
    bg_manager = BackgroundSyncManager(storage, files, auto_sync=None)
    can_start, reason = bg_manager.can_start_sync()

    if not can_start:
        console.print(f"[warning]Cannot start background sync: {reason}[/warning]")
        raise typer.Exit(1)

    try:
        llm = get_llm()
    except Exception as exc:
        console.print(f"[error]LLM configuration error: {exc}[/error]")
        raise typer.Exit(1) from None

    selected_sources = _get_repo_selected_sources(files)
    sources = None
    if source:
        sources = [normalize_source_name(source)]
    elif selected_sources:
        sources = selected_sources

    ingesters = get_default_ingesters()
    ingesters = _filter_ingesters_by_sources(ingesters, selected_sources)
    auto_sync = AutoSync(storage, files, llm, ingesters)

    # Create manager with actual auto_sync
    bg_manager = BackgroundSyncManager(storage, files, auto_sync)

    # Run sync
    result = asyncio.run(
        bg_manager.run_sync(
            sources=sources,
            max_sessions=max_sessions,
            compact=not no_compact,
        )
    )

    if result.was_already_running:
        console.print("[warning]Sync already running in another process[/warning]")
        raise typer.Exit(1)

    if result.success:
        console.print(
            f"[success]✓ Background sync complete[/success]\n"
            f"  Sessions processed: {result.sessions_processed}\n"
            f"  Learnings extracted: {result.learnings_extracted}"
        )
    else:
        console.print("[error]Sync failed[/error]")
        if result.error_message:
            console.print(f"[error]{result.error_message}[/error]")
        if result.diagnostics:
            console.print("[dim]Failure diagnostics:[/dim]")
            for detail in result.diagnostics:
                console.print(f"[dim]- {detail}[/dim]")
        raise typer.Exit(1)


@app.command("sessions")
def sessions(
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help=f"Only list sessions from specific source: {SOURCE_CHOICES_TEXT}",
    ),
    since_days: int | None = typer.Option(
        None,
        "--since-days",
        "-d",
        help="Only list sessions from the last N days",
    ),
    session_id: list[str] = typer.Option(
        None,
        "--session-id",
        help="Only include specific source session ID(s); repeat option to include multiple",
    ),
    max_sessions: int | None = typer.Option(
        None,
        "--max-sessions",
        "-n",
        min=1,
        help="Limit list to the most recent N discovered sessions",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
    cursor_db_path: Path | None = typer.Option(
        None,
        "--cursor-db-path",
        help="Path to a specific Cursor state.vscdb for testing/override",
    ),
    cursor_storage_dir: Path | None = typer.Option(
        None,
        "--cursor-storage-dir",
        help="Override Cursor workspaceStorage root path",
    ),
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Include Cursor sessions from all workspaces (not only current repo match)",
    ),
):
    """List discoverable agent sessions with source IDs and inferred titles."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()

    since = datetime.now(UTC) - timedelta(days=since_days) if since_days else None
    normalized_source = normalize_source_name(source) if source else None
    selected_sources = None if source else _get_repo_selected_sources(files)
    sources = [normalized_source] if normalized_source else selected_sources
    selected_session_ids = [item.strip() for item in (session_id or []) if item.strip()]
    session_ids = selected_session_ids or None

    if source:
        ingester_source = normalized_source or source
        try:
            ingesters = [
                get_ingester(
                    ingester_source,
                    cursor_db_path=cursor_db_path,
                    workspace_storage_dir=cursor_storage_dir,
                    cursor_all_workspaces=all_cursor_workspaces,
                )
            ]
        except ValueError as exc:
            console.print(f"[error]{exc}[/error]")
            raise typer.Exit(1) from None
    else:
        ingesters = get_default_ingesters(
            cursor_db_path=cursor_db_path,
            workspace_storage_dir=cursor_storage_dir,
            cursor_all_workspaces=all_cursor_workspaces,
        )
        ingesters = _filter_ingesters_by_sources(ingesters, selected_sources)

    auto_sync = AutoSync(storage, files, llm=None, ingesters=ingesters)
    results = run_with_spinner(
        "Discovering sessions...",
        lambda: auto_sync.list_sessions(
            since=since,
            sources=sources,
            session_ids=session_ids,
            max_sessions=max_sessions,
        ),
    )

    output_format = format.strip().lower()
    if output_format == "json":
        payload = {
            "sessions_discovered": results["sessions_discovered"],
            "by_source": results["by_source"],
            "errors": results["errors"],
            "sessions": [
                {
                    **{
                        key: value
                        for key, value in session_row.items()
                        if key not in {"started_at", "ended_at", "session_path", "project_path"}
                    },
                    "started_at": (
                        session_row["started_at"].isoformat()
                        if isinstance(session_row.get("started_at"), datetime)
                        else None
                    ),
                    "ended_at": (
                        session_row["ended_at"].isoformat()
                        if isinstance(session_row.get("ended_at"), datetime)
                        else None
                    ),
                    "session_path": str(session_row["session_path"]),
                    "project_path": (
                        str(session_row["project_path"])
                        if isinstance(session_row.get("project_path"), Path)
                        else None
                    ),
                }
                for session_row in results["sessions"]
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)

    if selected_sources and not source:
        console.print(f"[dim]Configured sources: {', '.join(selected_sources)}[/dim]")
    if session_ids:
        console.print(f"[dim]Requested session IDs: {len(session_ids)}[/dim]")
    if max_sessions is not None:
        console.print(f"[dim]Max sessions: {max_sessions}[/dim]")

    table = Table(
        title="Discovered Sessions",
        box=box.SQUARE,
        show_lines=True,
        expand=True,
    )
    table.add_column("Source", style="table_header")
    table.add_column("Session ID", overflow="fold")
    table.add_column("Title", overflow="fold")
    table.add_column("Started", overflow="fold")
    table.add_column("Messages", justify="right")
    table.add_column("Processed", justify="center")

    for session_row in results["sessions"]:
        title_text = str(session_row.get("title") or "-")
        table.add_row(
            str(session_row["source"]),
            str(session_row["session_id"]),
            title_text,
            _format_session_time(session_row.get("started_at")),
            str(session_row.get("message_count", 0)),
            "[success]✓[/success]" if session_row.get("processed") else "[warning]X[/warning]",
        )

    console.print(table)

    errors = results.get("errors") or []
    if errors:
        warning_lines = ["", f"[warning]Warnings: {len(errors)}[/warning]"]
        for error in errors[:10]:
            warning_lines.append(f"[dim]- {error}[/dim]")
        if len(errors) > 10:
            warning_lines.append(f"[dim]... and {len(errors) - 10} more[/dim]")
        console.print("\n".join(warning_lines))


@app.command()
def sources(
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Show Cursor sessions from all workspaces, not only current repo match",
    ),
    max_sessions: int | None = typer.Option(
        None,
        "--max-sessions",
        "-n",
        min=1,
        help="Limit detailed session list to the most recent N sessions",
    ),
):
    """Show source status and discovered sessions/conversations."""
    _get_theme_manager()  # Ensure theme is loaded
    ensure_initialized()
    storage = get_storage()
    files = get_files()

    selected_sources = _get_repo_selected_sources(files)
    ingesters = get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces)
    ingesters = _filter_ingesters_by_sources(ingesters, selected_sources)

    table = Table(title="Session Sources")
    table.add_column("Source", style="table_header")
    table.add_column("Status")
    table.add_column("Sessions", justify="right")
    table.add_column("Location", overflow="fold")

    rows: list[tuple[str, str, str, str]] = []

    def _collect_rows() -> None:
        for ingester in ingesters:
            try:
                sessions = ingester.discover_sessions()
                available = len(sessions) > 0
                status_text = "✓ Available" if available else "- No sessions"
                status_style = "success" if available else "dim"

                if available:
                    location = str(sessions[0].resolve())
                else:
                    hint = resolve_source_location_hint(ingester)
                    location = str(hint) if hint else "Unknown"

                rows.append(
                    (
                        ingester.source_name,
                        f"[{status_style}]{status_text}[/{status_style}]",
                        str(len(sessions)),
                        _as_clickable_uri(location),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    (
                        ingester.source_name,
                        "[error]✗ Error[/error]",
                        "-",
                        f"[error]{str(exc)[:40]}[/error]",
                    )
                )

    run_with_spinner("Discovering session sources...", _collect_rows)

    if selected_sources:
        console.print(f"[dim]Configured sources: {', '.join(selected_sources)}[/dim]")

    for row in rows:
        table.add_row(*row)

    console.print(table)

    auto_sync = AutoSync(storage, files, llm=None, ingesters=ingesters)
    session_results = run_with_spinner(
        "Loading session details...",
        lambda: auto_sync.list_sessions(max_sessions=max_sessions),
    )

    sessions = session_results.get("sessions", [])
    if not sessions:
        console.print("[dim]No session conversations discovered.[/dim]")
        return

    mode = _conversation_table_mode(console.width)
    sessions_table = Table(
        title="Discovered Conversations",
        box=box.SQUARE,
        show_lines=True,
        expand=True,
    )
    sessions_table.add_column("#", justify="right", style="table_header", no_wrap=True, width=3)
    sessions_table.add_column("Conversation", overflow="fold")

    if mode == "wide":
        sessions_table.add_column("Started", no_wrap=True, width=20)
        sessions_table.add_column("Messages", justify="right", no_wrap=True, width=8)
        sessions_table.add_column("Processed", justify="center", no_wrap=True, width=9)
        sessions_table.add_column("Ref", overflow="fold", no_wrap=True, width=13)
    elif mode == "medium":
        sessions_table.add_column("Started", no_wrap=True, width=20)
        sessions_table.add_column("Messages", justify="right", no_wrap=True, width=8)
        sessions_table.add_column("Processed", justify="center", no_wrap=True, width=9)
    else:
        sessions_table.add_column("Messages", justify="right", no_wrap=True, width=8)
        sessions_table.add_column("Processed", justify="center", no_wrap=True, width=9)

    for index, session_row in enumerate(sessions, start=1):
        title = str(session_row.get("title") or "").strip()
        conversation = title if title else "Untitled conversation"
        processed = (
            "[success]✓[/success]" if session_row.get("processed") else "[warning]X[/warning]"
        )

        if mode == "wide":
            sessions_table.add_row(
                str(index),
                conversation,
                _format_session_time(session_row.get("started_at")),
                str(session_row.get("message_count", 0)),
                processed,
                _compact_session_ref(str(session_row["session_id"])),
            )
        elif mode == "medium":
            sessions_table.add_row(
                str(index),
                conversation,
                _format_session_time(session_row.get("started_at")),
                str(session_row.get("message_count", 0)),
                processed,
            )
        else:
            sessions_table.add_row(
                str(index),
                conversation,
                str(session_row.get("message_count", 0)),
                processed,
            )

    console.print(sessions_table)

    errors = session_results.get("errors") or []
    if errors:
        console.print(f"[warning]Warnings: {len(errors)}[/warning]")
        for error in errors[:10]:
            console.print(f"[dim]- {error}[/dim]")
        if len(errors) > 10:
            console.print(f"[dim]... and {len(errors) - 10} more[/dim]")


@app.command()
def status():
    """Show agent-recall status and statistics."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()
    config_dict = files.read_config()

    stats = storage.get_stats()
    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    style = files.read_tier(KnowledgeTier.STYLE)
    recent = files.read_tier(KnowledgeTier.RECENT)
    selected_sources = _get_repo_selected_sources(files)
    adapter_config = _read_adapter_config(files)
    storage_cfg = config_dict.get("storage", {}) if isinstance(config_dict, dict) else {}
    compaction_cfg = config_dict.get("compaction", {}) if isinstance(config_dict, dict) else {}
    shared_cfg = storage_cfg.get("shared", {}) if isinstance(storage_cfg, dict) else {}
    storage_backend = (
        str(storage_cfg.get("backend", "local")).strip().lower()
        if isinstance(storage_cfg, dict)
        else "local"
    )
    shared_base_url = (
        str(shared_cfg.get("base_url", "")).strip() if isinstance(shared_cfg, dict) else ""
    )
    compaction_backend = (
        str(compaction_cfg.get("backend", "llm")).strip().lower()
        if isinstance(compaction_cfg, dict)
        else "llm"
    )
    lines = base_status_lines(
        StatusInputs(
            stats=stats,
            guardrails_chars=len(guardrails),
            style_chars=len(style),
            recent_chars=len(recent),
            onboarding_complete=is_repo_onboarding_complete(files),
            selected_sources=selected_sources,
            storage_backend=storage_backend,
            shared_base_url=shared_base_url,
            compaction_backend=compaction_backend,
            adapter_enabled=bool(adapter_config.get("enabled")),
            adapter_output_dir=str(adapter_config.get("output_dir") or ".agent/context"),
            adapter_token_budget=_format_default_adapter_token_budget(adapter_config),
            adapter_per_adapter_budgets=_format_per_adapter_token_budgets(adapter_config),
        )
    )

    if storage.capabilities.rule_confidence:
        rule_summary = storage.get_rule_confidence_summary()
    else:
        rule_summary = None

    if isinstance(rule_summary, dict):
        lines.extend(
            [
                f"  Rules tracked: {int(rule_summary.get('total_rules', 0))}",
                f"  Stale rules: {int(rule_summary.get('stale_rules', 0))}",
                (f"  Low confidence: {int(rule_summary.get('low_confidence_rules', 0))}"),
                (f"  Average confidence: {float(rule_summary.get('average_confidence', 0.0)):.2f}"),
            ]
        )
        oldest_signal = rule_summary.get("oldest_signal_at")
        if oldest_signal:
            lines.append(f"  Oldest signal: {oldest_signal}")
    else:
        lines.append("  Rules tracked: n/a")

    lines.extend(
        [
            "",
            "[bold]Session Sources:[/bold]",
        ]
    )

    source_lines: list[str] = []

    def _collect_source_lines() -> None:
        ingesters = _filter_ingesters_by_sources(
            get_default_ingesters(),
            selected_sources,
        )
        for ingester in ingesters:
            try:
                available = len(ingester.discover_sessions())
                icon = "✓" if available else "-"
                source_lines.append(
                    f"  {icon} {ingester.source_name}: {available} sessions available"
                )
            except Exception as exc:  # noqa: BLE001
                source_lines.append(
                    f"  ✗ {ingester.source_name}: [error]Error - {str(exc)[:30]}[/error]"
                )

    run_with_spinner("Collecting source status...", _collect_source_lines)
    lines.extend(source_lines)

    # Add background sync status
    lines.append("")
    lines.append("[bold]Background Sync:[/bold]")
    bg_manager = BackgroundSyncManager(storage, files, None)
    bg_status = bg_manager.get_status()
    if bg_status.is_running:
        lines.append(f"  Status: [warning]Running (PID: {bg_status.pid})[/warning]")
    else:
        lines.append("  Status: [dim]Idle[/dim]")
    if bg_status.started_at:
        lines.append(f"  Last started:  {bg_status.started_at.strftime('%Y-%m-%d %H:%M UTC')}")
    if bg_status.completed_at:
        lines.append(f"  Last completed: {bg_status.completed_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"  Last sessions:  {bg_status.sessions_processed}")
        lines.append(f"  Last learnings: {bg_status.learnings_extracted}")
    if bg_status.error_message:
        lines.append(f"  [error]Last error: {bg_status.error_message}[/error]")

    if compaction_backend == "mcp_external":
        try:
            service = _build_external_compaction_service(storage, files)
            pending = service.list_imported_conversations(limit=50, pending_only=True)
            lines.append("")
            lines.append("[bold]External Compaction:[/bold]")
            lines.append(f"  Pending conversations: {len(pending)}")
            if pending:
                lines.append("  Next step: agent-recall external-compaction export --pending-only")
                lines.append(
                    "  Apply notes: agent-recall external-compaction apply "
                    "--input <notes.json> --commit"
                )
            else:
                lines.append("  Status: no pending external compaction conversations.")
        except Exception as exc:  # noqa: BLE001
            lines.append("")
            lines.append("[bold]External Compaction:[/bold]")
            lines.append(f"  [warning]Unable to load pending status: {exc}[/warning]")

    lines.extend(render_ralph_status(config_dict))

    console.print(Panel.fit("\n".join(lines), title="Agent Recall Status"))


def _fetch_latest_version() -> str | None:
    """Fetch the latest version from PyPI. Returns None on any error."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get("https://pypi.org/pypi/agent-recall/json")
            resp.raise_for_status()
            data = resp.json()
            return data.get("info", {}).get("version")
    except Exception:  # noqa: BLE001
        return None


def _format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "-"
    if duration_seconds < 60:
        return f"{duration_seconds:.0f}s"
    minutes = duration_seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


def _render_iteration_timeline(store: IterationReportStore, *, max_entries: int) -> list[str]:
    if max_entries <= 0:
        return ["[dim]No entries to display.[/dim]"]
    reports = store.load_recent(count=max_entries)
    if not reports:
        return ["[dim]No iteration history yet.[/dim]"]

    lines: list[str] = []
    for report in reports:
        outcome = report.outcome
        symbol = "✓" if outcome == IterationOutcome.COMPLETED else "✗"
        outcome_text = outcome.value if outcome is not None else "UNKNOWN"
        duration = _format_duration(report.duration_seconds)
        summary = report.summary or report.failure_reason or report.validation_hint or ""
        summary = _truncate(summary, 64)
        item_label = report.item_id or "unknown"
        headline = f"{report.iteration:03d} {symbol} {item_label} ({duration}) {outcome_text}"
        lines.append(headline)
        if summary:
            lines.append(f"  {summary}")
    return lines


def _load_ralph_state_payload() -> dict:
    """Load ralph_state.json and return parsed payload or empty dict."""
    try:
        state_path = AGENT_DIR / "ralph" / "ralph_state.json"
        if not state_path.exists():
            return {}
        import json

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _dashboard_render_context() -> DashboardRenderContext:
    ralph_max_iterations: int | None = None
    try:
        ralph_config = read_ralph_config(get_files())
        max_iter_value = ralph_config.get("max_iterations")
        if isinstance(max_iter_value, int | float) and max_iter_value >= 1:
            ralph_max_iterations = int(max_iter_value)
    except Exception:  # noqa: BLE001
        ralph_max_iterations = None

    # Load Ralph state for context-aware dashboard
    ralph_state = _load_ralph_state_payload()
    ralph_enabled = ralph_state.get("enabled", False) is True
    ralph_running = ralph_state.get("status", "").lower() == "running"

    return DashboardRenderContext(
        console=console,
        theme_manager=_theme_manager,
        agent_dir=AGENT_DIR,
        ralph_max_iterations=ralph_max_iterations,
        get_storage=get_storage,
        get_files=get_files,
        get_repo_selected_sources=_get_repo_selected_sources,
        resolve_repo_root_for_display=_resolve_repo_root_for_display,
        filter_ingesters_by_sources=_filter_ingesters_by_sources,
        get_default_ingesters=get_default_ingesters,
        render_iteration_timeline=_render_iteration_timeline,
        summarize_costs=summarize_costs,
        format_usd=format_usd,
        is_interactive_terminal=is_interactive_terminal,
        help_lines_provider=_tui_help_lines,
        ralph_enabled=ralph_enabled,
        ralph_running=ralph_running,
    )


def _build_tui_dashboard(
    all_cursor_workspaces: bool = False,
    include_banner_header: bool = True,
    slash_status: str | None = None,
    slash_output: list[str] | None = None,
    view: str = "overview",
    ralph_agent_transport: str = "pipe",
    show_slash_console: bool = True,
) -> Group:
    """Build the live TUI dashboard renderable."""
    try:
        tui_config = _read_tui_config(get_files())
    except Exception:  # noqa: BLE001
        tui_config = {}
    banner_size = "normal"
    if isinstance(tui_config, dict):
        raw_banner = tui_config.get("banner_size")
        if isinstance(raw_banner, str):
            banner_size = raw_banner.strip().lower()
    return build_tui_dashboard(
        _dashboard_render_context(),
        all_cursor_workspaces=all_cursor_workspaces,
        include_banner_header=include_banner_header,
        banner_size=banner_size,
        slash_status=slash_status,
        slash_output=slash_output,
        view=view,
        ralph_agent_transport=ralph_agent_transport,
        show_slash_console=show_slash_console,
        widget_visibility=tui_config.get("widget_visibility")
        if isinstance(tui_config, dict)
        else None,
    )


def _run_onboarding_setup(force: bool, quick: bool) -> None:
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    interactive = is_interactive_terminal() and not quick
    if not interactive and not quick:
        console.print("[dim]No interactive terminal detected; applying saved defaults.[/dim]")

    changed = ensure_repo_onboarding(
        files,
        console,
        force=force,
        interactive=interactive,
    )
    if not changed:
        console.print("[dim]Onboarding already complete for this repository.[/dim]")


def _run_setup_from_payload(payload: dict[str, object]) -> tuple[bool, list[str]]:
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    selected_sources_raw = payload.get("selected_sources")
    selected_sources = (
        [source for source in selected_sources_raw if isinstance(source, str)]
        if isinstance(selected_sources_raw, list)
        else []
    )

    temp_output = io.StringIO()
    capture_console = Console(
        file=temp_output, theme=_theme_manager.get_theme(), force_terminal=False
    )
    temperature_raw = payload.get("temperature", 0.3)
    if isinstance(temperature_raw, int | float | str):
        try:
            temperature = float(temperature_raw)
        except ValueError:
            temperature = 0.3
    else:
        temperature = 0.3

    max_tokens_raw = payload.get("max_tokens", 4096)
    if isinstance(max_tokens_raw, int | float | str):
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError:
            max_tokens = 4096
    else:
        max_tokens = 4096

    configure_coding_agent = bool(payload.get("configure_coding_agent", False))
    coding_cli_value = payload.get("coding_cli")
    coding_cli = (
        str(coding_cli_value).strip()
        if isinstance(coding_cli_value, str) and str(coding_cli_value).strip()
        else None
    )
    cli_model_value = payload.get("cli_model")
    cli_model = (
        str(cli_model_value).strip()
        if isinstance(cli_model_value, str) and str(cli_model_value).strip()
        else None
    )
    ralph_enabled_raw = payload.get("ralph_enabled")
    ralph_enabled = bool(ralph_enabled_raw) if isinstance(ralph_enabled_raw, bool) else None
    if configure_coding_agent and ralph_enabled is None:
        ralph_enabled = bool(coding_cli)

    changed = apply_repo_setup(
        files,
        capture_console,
        force=bool(payload.get("force", False)),
        repository_verified=bool(payload.get("repository_verified", False)),
        selected_sources=selected_sources,
        provider=str(payload.get("provider", "")).strip(),
        model=str(payload.get("model", "")).strip(),
        base_url=(
            None
            if str(payload.get("base_url", "")).strip().lower() in {"", "none", "null"}
            else str(payload.get("base_url", "")).strip()
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        validate=bool(payload.get("validate", False)),
        api_key=(
            str(payload.get("api_key", "")).strip()
            if isinstance(payload.get("api_key"), str)
            else None
        ),
        configure_coding_agent=configure_coding_agent,
        coding_cli=coding_cli,
        cli_model=cli_model,
        ralph_enabled=ralph_enabled,
    )

    lines = [line.strip() for line in temp_output.getvalue().splitlines() if line.strip()]
    return changed, lines[-12:]


def _run_model_config(
    provider: str | None,
    model: str | None,
    base_url: str | None,
    temperature: float | None,
    max_tokens: int | None,
    validate: bool,
    output_console: Console | None = None,
) -> dict[str, object]:
    active_console = output_console or console
    _get_theme_manager()
    ensure_initialized()
    inject_stored_api_keys()
    files = get_files()

    config_dict = files.read_config()
    llm_config = dict(config_dict.get("llm", {}))

    if provider:
        llm_config["provider"] = provider
    if model:
        llm_config["model"] = model
    if base_url:
        llm_config["base_url"] = base_url
    if temperature is not None:
        llm_config["temperature"] = float(temperature)
    if max_tokens is not None:
        llm_config["max_tokens"] = int(max_tokens)

    try:
        parsed = LLMConfig(**llm_config)
    except Exception as exc:  # noqa: BLE001
        active_console.print(f"[error]Invalid LLM config: {exc}[/error]")
        raise typer.Exit(1) from None

    dependency_ok, dependency_message = ensure_provider_dependency(
        parsed.provider,
        auto_install=True,
    )
    if not dependency_ok:
        active_console.print(
            f"[error]Provider dependency setup failed: {dependency_message}[/error]"
        )
        raise typer.Exit(1)
    if dependency_message:
        active_console.print(f"[dim]{dependency_message}[/dim]")

    if validate and (
        provider or model or base_url or temperature is not None or max_tokens is not None
    ):
        valid, message = validate_provider_config(parsed)
        if not valid:
            active_console.print(f"[warning]Warning: {message}[/warning]")
        else:
            try:
                llm = create_llm_provider(parsed)
                if output_console is None:
                    success, validation_message = run_with_spinner(
                        "Validating provider connection...",
                        llm.validate,
                    )
                else:
                    success, validation_message = llm.validate()
                if success:
                    active_console.print(f"[success]✓ {validation_message}[/success]")
                else:
                    active_console.print(f"[warning]Warning: {validation_message}[/warning]")
            except Exception as exc:  # noqa: BLE001
                active_console.print(
                    f"[warning]Warning: Could not validate provider: {exc}[/warning]"
                )

    config_dict["llm"] = llm_config
    files.write_config(config_dict)

    base_url_display = llm_config.get("base_url") or "default"
    active_console.print(
        Panel.fit(
            f"Provider: {llm_config.get('provider', 'not set')}\n"
            f"Model: {llm_config.get('model', 'not set')}\n"
            f"Base URL: {base_url_display}\n"
            f"Temperature: {llm_config.get('temperature', 0.3)}\n"
            f"Max tokens: {llm_config.get('max_tokens', 4096)}",
            title="LLM Configuration Updated",
        )
    )
    return llm_config


def _run_model_config_for_tui(
    provider: str | None,
    model: str | None,
    base_url: str | None,
    temperature: float | None,
    max_tokens: int | None,
    validate: bool,
) -> list[str]:
    temp_output = io.StringIO()
    capture_console = Console(
        file=temp_output, theme=_theme_manager.get_theme(), force_terminal=False
    )
    _run_model_config(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        validate=validate,
        output_console=capture_console,
    )
    return [line.strip() for line in temp_output.getvalue().splitlines() if line.strip()][-12:]


def _list_prd_items_for_tui_picker() -> dict[str, Any]:
    """Load PRD items and Ralph config for the TUI PRD selection modal."""
    prd_path = get_default_prd_path()
    items = load_prd_items(prd_path)
    files = get_files()
    ralph_config = read_ralph_config(files)
    selected_value = ralph_config.get("selected_prd_ids")
    selected_ids: list[str] = []
    if isinstance(selected_value, list):
        selected_ids = [str(x) for x in selected_value if x]
    max_iterations = int(ralph_config.get("max_iterations") or 10)
    prepared: list[dict[str, Any]] = []
    for it in items:
        item_id = str(it.get("id") or "")
        if not item_id:
            continue
        prepared.append(
            {
                "id": item_id,
                "title": str(it.get("title") or "Untitled"),
                "priority": int(it.get("priority") or 0),
                "passes": bool(it.get("passes")),
            }
        )
    return {
        "items": prepared,
        "selected_ids": selected_ids,
        "max_iterations": max_iterations,
    }


def _list_sessions_for_tui_picker(
    max_sessions: int = 200,
    *,
    all_cursor_workspaces: bool = False,
) -> list[dict[str, object]]:
    ensure_initialized()
    storage = get_storage()
    files = get_files()
    selected_sources = _get_repo_selected_sources(files)
    ingesters = _filter_ingesters_by_sources(
        get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces),
        selected_sources,
    )
    auto_sync = AutoSync(storage, files, llm=None, ingesters=ingesters)
    results = auto_sync.list_sessions(max_sessions=max_sessions)
    sessions = results.get("sessions", [])

    prepared: list[dict[str, object]] = []
    for row in sessions:
        title = str(row.get("title") or "").strip() or "Untitled conversation"
        prepared.append(
            {
                "source": str(row.get("source") or ""),
                "session_id": str(row.get("session_id") or ""),
                "title": title,
                "started": _format_session_time(row.get("started_at")),
                "message_count": int(row.get("message_count", 0)),
                "processed": bool(row.get("processed")),
            }
        )
    return prepared


def _get_sources_and_sessions_for_tui(
    max_sessions: int = 200,
    *,
    all_cursor_workspaces: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ensure_initialized()
    storage = get_storage()
    files = get_files()
    selected_sources = _get_repo_selected_sources(files)
    ingesters = _filter_ingesters_by_sources(
        get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces),
        selected_sources,
    )

    # Gather sources status
    sources_data: list[dict[str, object]] = []
    for ingester in ingesters:
        try:
            sessions = ingester.discover_sessions()
            available = len(sessions) > 0
            if available:
                location = str(sessions[0].resolve())
            else:
                hint = resolve_source_location_hint(ingester)
                location = str(hint) if hint else "Unknown"
            sources_data.append(
                {
                    "name": ingester.source_name,
                    "available": available,
                    "location": location,
                    "count": len(sessions),
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            sources_data.append(
                {
                    "name": ingester.source_name,
                    "available": False,
                    "location": None,
                    "count": 0,
                    "error": str(exc),
                }
            )

    auto_sync = AutoSync(storage, files, llm=None, ingesters=ingesters)
    results = auto_sync.list_sessions(max_sessions=max_sessions)
    sessions = results.get("sessions", [])

    sessions_data: list[dict[str, object]] = []
    for row in sessions:
        title = str(row.get("title") or "").strip() or "Untitled conversation"
        sessions_data.append(
            {
                "source": str(row.get("source") or ""),
                "session_id": str(row.get("session_id") or ""),
                "title": title,
                "started": _format_session_time(row.get("started_at")),
                "message_count": int(row.get("message_count", 0)),
                "processed": bool(row.get("processed")),
            }
        )
    return sources_data, sessions_data


def _get_session_detail_for_tui(
    source: str,
    session_id: str,
    all_cursor_workspaces: bool = False,
) -> Any | None:
    ensure_initialized()
    files = get_files()
    selected_sources = _get_repo_selected_sources(files)
    ingesters = _filter_ingesters_by_sources(
        get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces),
        selected_sources,
    )
    for ingester in ingesters:
        if str(ingester.source_name) == str(source):
            try:
                paths = ingester.discover_sessions()
                for path in paths:
                    try:
                        extracted = ingester.get_session_id(path)
                        if str(extracted) == str(session_id):
                            return ingester.parse_session(path)
                    except Exception:
                        pass
            except Exception:
                pass
    return None


def _get_theme_defaults_for_tui() -> tuple[list[str], str]:
    current_theme = DEFAULT_THEME
    if AGENT_DIR.exists():
        try:
            config_dict = get_files().read_config()
            current_theme = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        except Exception:  # noqa: BLE001
            pass
    return ThemeManager.get_available_themes(), current_theme


def _read_tui_config(files: FileStorage) -> dict[str, Any]:
    config_dict = files.read_config()
    tui_config = config_dict.get("tui", {}) if isinstance(config_dict, dict) else {}
    return tui_config if isinstance(tui_config, dict) else {}


def _write_tui_config(files: FileStorage, updates: dict[str, object]) -> dict[str, object]:
    config_dict = files.read_config()
    if "tui" not in config_dict or not isinstance(config_dict.get("tui"), dict):
        config_dict["tui"] = {}
    tui_config = config_dict["tui"]
    if isinstance(tui_config, dict):
        tui_config.update(updates)
    config_dict["tui"] = tui_config
    files.write_config(config_dict)
    return config_dict


@app.command()
def tui(
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Include Cursor sessions from all workspaces.",
    ),
    show_terminal: bool = typer.Option(
        False,
        "--show-terminal",
        help="Show the embedded terminal panel when supported.",
    ),
    iterations: int | None = typer.Option(
        None,
        "--iterations",
        hidden=True,
        help="Number of refresh loops (for tests).",
    ),
    no_splash: bool = typer.Option(
        False,
        "--no-splash",
        help="Skip the initial splash animation.",
    ),
    splash_delay: float = typer.Option(
        0.012,
        "--splash-delay",
        min=0.0,
        max=0.1,
        help="Animation delay for splash screen.",
    ),
    onboarding: bool = typer.Option(
        True,
        "--onboarding/--no-onboarding",
        help="Run onboarding setup before launching the dashboard.",
    ),
    force_onboarding: bool = typer.Option(
        False,
        "--force-onboarding",
        help="Run onboarding even if this repository is already configured.",
    ),
    no_delta_setup: bool = typer.Option(
        False,
        "--no-delta-setup",
        help="Skip first-launch delta diff renderer setup prompt.",
    ),
    force_delta_setup: bool = typer.Option(
        False,
        "--force-delta-setup",
        help="Reset delta setup: clear cached binary and show the download prompt again.",
    ),
):
    """Start a live terminal UI dashboard for agent-recall."""
    _get_theme_manager()
    ensure_initialized()
    if force_delta_setup:
        from agent_recall.cli.tui.delta import reset_delta_setup

        reset_delta_setup()
    inject_stored_api_keys()
    files = get_files()
    interactive_shell = is_interactive_terminal()

    if iterations is not None and iterations < 1:
        console.print("[error]--iterations must be >= 1[/error]")
        raise typer.Exit(1)

    onboarding_complete = is_repo_onboarding_complete(files)
    onboarding_required = False
    if onboarding:
        onboarding_required = force_onboarding or not onboarding_complete
    elif not onboarding_complete:
        console.print(
            "[warning]Onboarding is incomplete for this repository. "
            "Run 'agent-recall open' to complete setup in the TUI.[/warning]"
        )

    # Show splash animation first (unless disabled)
    if not no_splash:
        console.clear()
        print_banner(console, _theme_manager, animated=True, delay=splash_delay)
        time.sleep(0.8)  # Pause to admire the banner

    if iterations is not None:
        for index in range(iterations):
            console.print(
                _build_tui_dashboard(
                    all_cursor_workspaces=all_cursor_workspaces,
                    include_banner_header=True,
                    view="overview",
                    show_slash_console=False,
                )
            )
            if index < iterations - 1:
                time.sleep(0.5)
        return

    if not interactive_shell:
        console.print(
            "[error]The Textual TUI requires an interactive terminal. "
            "Use a terminal session or run with --iterations for non-interactive checks.[/error]"
        )
        raise typer.Exit(1)

    try:
        from agent_recall.cli.tui import AgentRecallTextualApp
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Textual TUI unavailable: {exc}[/error]")
        raise typer.Exit(1) from None

    try:
        tui_config = _read_tui_config(files)
        if show_terminal:
            _write_tui_config(files, {"terminal_panel_visible": True})
            tui_config = _read_tui_config(files)

        app_instance = AgentRecallTextualApp(
            render_dashboard=_build_tui_dashboard,
            dashboard_context=_dashboard_render_context(),
            execute_command=lambda raw, width, height: tui_router.execute_tui_slash_command(
                raw,
                app=app,
                slash_runner=_slash_runner,
                get_help_lines=_tui_help_lines,
                run_onboarding_setup=_run_onboarding_setup,
                terminal_width=width,
                terminal_height=max(height * 2, 200),
            ),
            list_sessions_for_picker=(
                lambda max_items, include_all_cursor: _list_sessions_for_tui_picker(
                    max_items,
                    all_cursor_workspaces=include_all_cursor,
                )
            ),
            get_sources_and_sessions_for_tui=(
                lambda max_items, include_all_cursor: _get_sources_and_sessions_for_tui(
                    max_items,
                    all_cursor_workspaces=include_all_cursor,
                )
            ),
            get_session_detail_for_tui=(
                lambda source, session_id, include_all_cursor: _get_session_detail_for_tui(
                    source,
                    session_id,
                    all_cursor_workspaces=include_all_cursor,
                )
            ),
            list_prd_items_for_picker=_list_prd_items_for_tui_picker,
            run_setup_payload=_run_setup_from_payload,
            run_model_config=_run_model_config_for_tui,
            theme_defaults_provider=_get_theme_defaults_for_tui,
            theme_runtime_provider=lambda: (
                _theme_manager.get_theme_name(),
                _theme_manager.get_theme(),
            ),
            theme_resolve_provider=lambda theme_name: (
                Theme(
                    ThemeManager.get_theme_colors(theme_name),
                    inherit=True,
                )
                if ThemeManager.is_valid_theme(theme_name)
                else None
            ),
            model_defaults_provider=lambda: get_files().read_config().get("llm", {}),
            setup_defaults_provider=lambda: get_onboarding_defaults(get_files()),
            discover_models=lambda provider, base_url, api_key_env: discover_provider_models(
                provider,
                base_url=base_url,
                api_key_env=api_key_env,
                timeout_seconds=4.0,
            ),
            discover_coding_models=lambda cli: discover_coding_cli_models(cli, timeout_seconds=8.0),
            providers=get_available_providers(),
            cli_commands=_collect_registered_cli_command_paths(),
            rich_theme=_theme_manager.get_theme(),
            initial_view=str(tui_config.get("default_view", "overview")),
            all_cursor_workspaces=bool(
                tui_config.get("all_cursor_workspaces", all_cursor_workspaces)
            ),
            ralph_agent_transport=str(tui_config.get("ralph_agent_transport", "pipe")),
            onboarding_required=onboarding_required,
            terminal_panel_visible=bool(tui_config.get("terminal_panel_visible", False)),
            terminal_supported=show_terminal,
            no_delta_setup=no_delta_setup,
        )
        app_instance.run()
    except KeyboardInterrupt:
        console.print("\n[dim]TUI closed.[/dim]")


@app.command("open")
def open_dashboard(
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Include Cursor sessions from all workspaces.",
    ),
    iterations: int | None = typer.Option(
        None,
        "--iterations",
        hidden=True,
        help="Number of refresh loops (for tests).",
    ),
    no_splash: bool = typer.Option(
        False,
        "--no-splash",
        help="Skip the initial splash animation.",
    ),
    splash_delay: float = typer.Option(
        0.012,
        "--splash-delay",
        min=0.0,
        max=0.1,
        help="Animation delay for splash screen.",
    ),
    show_terminal: bool = typer.Option(
        False,
        "--show-terminal",
        help="Show the embedded terminal panel when supported.",
    ),
    force_onboarding: bool = typer.Option(
        False,
        "--force-onboarding",
        help="Force setup flow even if this repository is already configured.",
    ),
):
    """Open the TUI dashboard (recommended entrypoint)."""
    tui(
        all_cursor_workspaces=all_cursor_workspaces,
        iterations=iterations,
        no_splash=no_splash,
        splash_delay=splash_delay,
        show_terminal=show_terminal,
        onboarding=True,
        force_onboarding=force_onboarding,
    )


@sync_app.command("reset")
def reset_sync(
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help=f"Clear only one source: {SOURCE_CHOICES_TEXT}",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Clear only one exact processed source session ID",
    ),
):
    """Reset processed-session markers so sessions can be re-synced for testing."""
    _get_theme_manager()
    storage = get_storage()

    if source:
        normalized = normalize_source_name(source)
        if normalized not in VALID_SOURCE_NAMES:
            console.print(f"[error]Invalid source. Use one of: {SOURCE_CHOICES_TEXT}[/error]")
            raise typer.Exit(1)
        source = normalized

    removed = storage.clear_processed_sessions(source=source, source_session_id=session_id)

    if session_id:
        scope = f"session_id={session_id}"
    elif source:
        scope = f"source={source}"
    else:
        scope = "all sources"

    console.print(
        Panel.fit(
            f"[success]✓ Reset sync markers[/success]\n\n"
            f"Scope: {scope}\n"
            f"Cleared processed session markers: {removed}\n\n"
            "[dim]Note: log entries/chunks are unchanged.[/dim]",
            title="Reset Complete",
        )
    )


@app.command("reset-learnings")
def reset_learnings():
    """Reset repository learning ingestion state to allow full re-ingest."""
    _get_theme_manager()
    storage = get_storage()

    removed_processed = storage.clear_processed_sessions()
    removed_checkpoints = storage.clear_session_checkpoints()

    console.print(
        Panel.fit(
            "[success]✓ Reset learning ingestion state[/success]\n\n"
            "Scope: repository\n"
            f"Cleared processed session markers: {removed_processed}\n"
            f"Cleared session checkpoints: {removed_checkpoints}\n\n"
            "[dim]Note: extracted log entries/chunks are unchanged.[/dim]",
            title="Reset Complete",
        )
    )


@config_app.command("setup")
def config_setup(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Run setup even if this repository is already configured.",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Apply setup defaults without interactive prompts.",
    ),
):
    """Run repository setup and credential onboarding."""
    _run_onboarding_setup(force=force, quick=quick)


@config_app.command("model")
def config_model(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help=(
            "LLM provider: anthropic, openai, openrouter, mistral, google, "
            "ollama, vllm, lmstudio, openai-compatible"
        ),
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        "-u",
        help="API base URL (for local/custom providers)",
    ),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature (0.0 to 2.0)",
        min=0.0,
        max=2.0,
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Maximum output tokens (>0)",
        min=1,
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate configuration after setting",
    ),
):
    """Configure model/provider settings."""
    _run_model_config(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        validate=validate,
    )


@config_app.command("adapters")
def config_adapters(
    enabled: bool | None = typer.Option(
        None,
        "--enabled/--disabled",
        help="Enable or disable automatic context adapter payloads",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory where adapter payloads are written",
    ),
    token_budget: int | None = typer.Option(
        None,
        "--token-budget",
        min=1,
        help="Global token budget for adapter payload context",
    ),
    per_adapter_token_budget: str | None = typer.Option(
        None,
        "--per-adapter-token-budget",
        help="Per-adapter token budgets as name=tokens pairs",
    ),
    per_provider_token_budget: str | None = typer.Option(
        None,
        "--per-provider-token-budget",
        help="Per-provider token budgets as provider=tokens pairs",
    ),
    per_model_token_budget: str | None = typer.Option(
        None,
        "--per-model-token-budget",
        help="Per-model token budgets as model=tokens pairs",
    ),
):
    """Configure automatic context adapter payloads."""
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    updates: dict[str, object] = {}
    if enabled is not None:
        updates["enabled"] = enabled
    if output_dir is not None:
        updates["output_dir"] = str(output_dir)
    if token_budget is not None:
        updates["default_token_budget"] = token_budget
    if per_adapter_token_budget is not None:
        updates["per_adapter_token_budget"] = _parse_named_token_budget_map(
            per_adapter_token_budget,
            label="per-adapter",
        )
    if per_provider_token_budget is not None:
        updates["per_provider_token_budget"] = _parse_named_token_budget_map(
            per_provider_token_budget,
            label="per-provider",
        )
    if per_model_token_budget is not None:
        updates["per_model_token_budget"] = _parse_named_token_budget_map(
            per_model_token_budget,
            label="per-model",
        )

    if not updates:
        adapter_config = _read_adapter_config(files)
        lines = [
            "[bold]Context Adapters:[/bold]",
            f"  Enabled: {'yes' if bool(adapter_config.get('enabled')) else 'no'}",
            f"  Output dir: {adapter_config.get('output_dir') or '.agent/context'}",
            f"  Token budget: {_format_default_adapter_token_budget(adapter_config)}",
            f"  Per-adapter budgets: {_format_per_adapter_token_budgets(adapter_config)}",
            "  Per-provider budgets: "
            f"{_format_named_token_budget_map(adapter_config.get('per_provider_token_budget'))}",
            "  Per-model budgets: "
            f"{_format_named_token_budget_map(adapter_config.get('per_model_token_budget'))}",
        ]
        console.print(Panel.fit("\n".join(lines), title="Context Adapters"))
        return

    _write_adapter_config(files, updates)
    adapter_config = _read_adapter_config(files)
    lines = [
        "[success]✓ Adapter settings updated[/success]",
        f"  Enabled: {'yes' if bool(adapter_config.get('enabled')) else 'no'}",
        f"  Output dir: {adapter_config.get('output_dir') or '.agent/context'}",
        f"  Token budget: {_format_default_adapter_token_budget(adapter_config)}",
        f"  Per-adapter budgets: {_format_per_adapter_token_budgets(adapter_config)}",
        "  Per-provider budgets: "
        f"{_format_named_token_budget_map(adapter_config.get('per_provider_token_budget'))}",
        "  Per-model budgets: "
        f"{_format_named_token_budget_map(adapter_config.get('per_model_token_budget'))}",
    ]
    console.print(Panel.fit("\n".join(lines), title="Context Adapters"))


@app.command()
def providers():
    """List available LLM providers and their configuration hints."""
    _get_theme_manager()
    available = set(get_available_providers())

    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="table_header")
    table.add_column("Type")
    table.add_column("API Key Env")
    table.add_column("Default Base URL")

    providers_info = [
        ("anthropic", "Cloud", "ANTHROPIC_API_KEY", "-"),
        ("openai", "Cloud", "OPENAI_API_KEY", "-"),
        ("openrouter", "Cloud", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        ("mistral", "Cloud", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
        ("google", "Cloud", "GOOGLE_API_KEY", "-"),
        ("ollama", "Local", "-", "http://localhost:11434/v1"),
        ("vllm", "Local", "- (optional)", "http://localhost:8000/v1"),
        ("lmstudio", "Local", "-", "http://localhost:1234/v1"),
        ("openai-compatible", "Custom", "- (optional)", "(required)"),
    ]

    for name, provider_type, key_env, default_base_url in providers_info:
        if name in available:
            table.add_row(name, provider_type, key_env, default_base_url)

    console.print(table)
    console.print()
    console.print("[bold]Examples:[/bold]")
    console.print("  export ANTHROPIC_API_KEY=sk-...")
    console.print(
        "  agent-recall config model --provider anthropic --model claude-sonnet-4-20250514"
    )
    console.print("  agent-recall config model --provider ollama --model llama3.1")
    console.print("  agent-recall config model --provider openrouter --model openai/gpt-5.2")
    console.print("  agent-recall config model --provider mistral --model mistral-large-latest")
    console.print("  agent-recall config model --provider lmstudio --model local-model")
    console.print(
        "  agent-recall config model --provider openai-compatible --base-url "
        "http://localhost:8080/v1 --model my-model"
    )


@app.command()
def test_llm():
    """Test the configured LLM provider with a small generation call."""
    _get_theme_manager()
    ensure_initialized()

    try:
        llm = get_llm()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Failed to initialize LLM: {exc}[/error]")
        raise typer.Exit(1) from None

    success, validation_message = run_with_spinner(
        f"Validating {llm.provider_name} ({llm.model_name})...",
        llm.validate,
    )
    if not success:
        console.print(f"[error]✗ {validation_message}[/error]")
        raise typer.Exit(1)
    console.print(f"[success]✓ {validation_message}[/success]")

    async def _run() -> str:
        response = await llm.generate(
            [
                Message(
                    role="user",
                    content="Say 'Hello from agent-recall!' in exactly those words.",
                )
            ],
            max_tokens=50,
        )
        return response.content

    try:
        content = run_with_spinner("Running test generation...", lambda: asyncio.run(_run()))
        console.print("[success]✓ Generation successful[/success]")
        console.print(f"[dim]Response: {content[:120]}[/dim]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]✗ Generation failed: {exc}[/error]")
        raise typer.Exit(1) from None


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        "-k",
        min=1,
        help="Maximum results (default from config)",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Override retrieval backend (fts5 or hybrid)",
    ),
    fusion_k: int | None = typer.Option(
        None,
        "--fusion-k",
        min=1,
        help="Override hybrid fusion constant (default from config)",
    ),
    rerank: bool | None = typer.Option(
        None,
        "--rerank/--no-rerank",
        help="Override reranking behavior (default from config)",
    ),
    rerank_candidate_k: int | None = typer.Option(
        None,
        "--rerank-candidate-k",
        min=1,
        help="Override rerank candidate pool size (default from config)",
    ),
):
    """Retrieve relevant memory chunks."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()
    try:
        retriever, retrieval_cfg = _build_retriever(
            storage,
            files,
            backend=backend,
            fusion_k=fusion_k,
            rerank=rerank,
            rerank_candidate_k=rerank_candidate_k,
        )
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    _get_theme_manager()  # Ensure theme is loaded
    effective_top_k = top_k if top_k is not None else retrieval_cfg.top_k
    chunks = retriever.search(query=query, top_k=effective_top_k)
    if not chunks:
        console.print("[warning]No matching chunks found.[/warning]")
        raise typer.Exit(0)

    for idx, chunk in enumerate(chunks, 1):
        tags = f" [{', '.join(chunk.tags)}]" if chunk.tags else ""
        console.print(f"{idx}. ({chunk.label.value}) {chunk.content}{tags}")


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    source_session_id: str | None = typer.Option(
        None,
        "--source-session-id",
        help="Native session ID",
    ),
):
    """Ingest a JSONL transcript into log entries."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()
    telemetry = PipelineTelemetry.from_config(agent_dir=files.agent_dir, config=files.read_config())
    run_id = telemetry.create_run_id("ingest")
    started = time.perf_counter()
    ingestor = TranscriptIngestor(storage)

    try:
        count = run_with_spinner(
            "Ingesting transcript...",
            lambda: ingestor.ingest_jsonl(path=path, source_session_id=source_session_id),
        )
    except Exception as exc:  # noqa: BLE001
        telemetry.record_event(
            run_id=run_id,
            stage=PipelineStage.INGEST,
            action=PipelineEventAction.ERROR,
            success=False,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"source_session_id": source_session_id, "error": str(exc)},
        )
        raise

    telemetry.record_event(
        run_id=run_id,
        stage=PipelineStage.INGEST,
        action=PipelineEventAction.COMPLETE,
        success=True,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        metadata={"source_session_id": source_session_id, "entries_ingested": int(count)},
    )
    console.print(f"[success]✓ Ingested {count} transcript entries[/success]")


theme_app = typer.Typer(help="Manage CLI themes")
metrics_app = typer.Typer(help="Inspect pipeline telemetry metrics")
curation_app = typer.Typer(help="Review and approve extracted learnings")
feedback_app = typer.Typer(help="Capture and inspect retrieval relevance feedback")
topic_threads_app = typer.Typer(help="Build and inspect cross-session topic threads")
rule_confidence_app = typer.Typer(help="Manage rule confidence decay and pruning")
memory_pack_app = typer.Typer(help="Import/export versioned memory packs")
attribution_app = typer.Typer(help="Inspect attribution metadata by agent/provider")
memory_app = typer.Typer(help="Manage pluggable memory backends")
external_compaction_app = typer.Typer(
    help="Run external conversation compaction and optional MCP server tools"
)
external_compaction_queue_app = typer.Typer(help="Review queued external compaction notes")


@theme_app.command("list")
def theme_list():
    """List all available themes."""
    # Get current theme from config if available
    current_theme = DEFAULT_THEME
    if AGENT_DIR.exists():
        try:
            files = FileStorage(AGENT_DIR)
            config_dict = files.read_config()
            current_theme = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        except Exception:  # noqa: BLE001
            pass

    themes = ThemeManager.get_available_themes()

    table = Table(title="Available Themes")
    table.add_column("Name", style="table_header")
    table.add_column("Status")

    for theme_name in themes:
        status = "[success]✓ Current[/success]" if theme_name == current_theme else ""
        table.add_row(theme_name, status)

    console.print(table)


@theme_app.command("set")
def theme_set(name: str = typer.Argument(..., help="Theme name to set")):
    """Set the active theme."""
    global console
    if not ThemeManager.is_valid_theme(name):
        available = ", ".join(ThemeManager.get_available_themes())
        console.print(f"[error]Invalid theme: {name}[/error]")
        console.print(f"[dim]Available themes: {available}[/dim]")
        raise typer.Exit(1)

    if not AGENT_DIR.exists():
        console.print("[error]Not initialized. Run 'agent-recall init' first.[/error]")
        raise typer.Exit(1)

    try:
        files = get_files()
        config_dict = files.read_config()
        if "theme" not in config_dict:
            config_dict["theme"] = {}
        config_dict["theme"]["name"] = name
        files.write_config(config_dict)

        # Update current theme manager
        _theme_manager.set_theme(name)
        console = Console(theme=_theme_manager.get_theme())

        console.print(f"[success]✓ Theme set to '{name}'[/success]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Failed to set theme: {exc}[/error]")
        raise typer.Exit(1) from None


@theme_app.command("show")
def theme_show():
    """Show the current theme."""
    current_theme = DEFAULT_THEME
    if AGENT_DIR.exists():
        try:
            files = FileStorage(AGENT_DIR)
            config_dict = files.read_config()
            current_theme = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        except Exception:  # noqa: BLE001
            pass

    console.print(f"Current theme: [accent]{current_theme}[/accent]")


@metrics_app.command("report")
def metrics_report(
    limit: int = typer.Option(5, "--limit", "-n", min=1, help="Number of recent runs to summarize"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Report pipeline telemetry counters and recent run summaries."""
    _get_theme_manager()
    files = get_files()
    telemetry = PipelineTelemetry.from_config(agent_dir=files.agent_dir, config=files.read_config())
    snapshot = telemetry.read_snapshot()
    runs = telemetry.list_recent_runs(limit=limit)

    output_format = _resolve_output_format(format)
    payload = {
        "snapshot": snapshot,
        "recent_runs": runs,
    }
    if output_format == "json":
        print_json({"status": "ok", "data": payload, "exit_code": 0})
        return

    counters = snapshot.get("counters", {})
    events_total = int(counters.get("events_total", 0))
    updated_at = snapshot.get("updated_at") or "-"
    console.print(
        Panel.fit(
            f"Total events: {events_total}\nUpdated at: {updated_at}\nRecent runs: {len(runs)}",
            title="metrics report",
        )
    )

    by_stage = counters.get("by_stage", {})
    if isinstance(by_stage, dict) and by_stage:
        stage_table = Table(title="Counters by Stage", box=box.SIMPLE)
        stage_table.add_column("Stage")
        stage_table.add_column("Start", justify="right")
        stage_table.add_column("Complete", justify="right")
        stage_table.add_column("Error", justify="right")
        stage_table.add_column("Success", justify="right")
        stage_table.add_column("Failure", justify="right")
        for stage, values in sorted(by_stage.items()):
            if not isinstance(values, dict):
                continue
            stage_table.add_row(
                str(stage),
                str(int(values.get("start", 0))),
                str(int(values.get("complete", 0))),
                str(int(values.get("error", 0))),
                str(int(values.get("success", 0))),
                str(int(values.get("failure", 0))),
            )
        console.print(stage_table)

    if not runs:
        console.print("[dim]No telemetry runs recorded yet.[/dim]")
        return

    run_table = Table(title="Recent Pipeline Runs", box=box.SIMPLE)
    run_table.add_column("Run ID", overflow="fold")
    run_table.add_column("Started", overflow="fold")
    run_table.add_column("Duration (ms)", justify="right")
    run_table.add_column("Events", justify="right")
    for run in runs:
        run_table.add_row(
            str(run.get("run_id", "")),
            str(run.get("started_at", "-")),
            f"{float(run.get('duration_ms', 0.0)):.1f}",
            str(int(run.get("events_total", 0))),
        )
    console.print(run_table)


@external_compaction_app.command("list")
def external_compaction_list(
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Maximum conversations"),
    pending_only: bool = typer.Option(
        True,
        "--pending-only/--all",
        help="Show only pending conversations (default) or all tracked conversations",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """List imported conversations eligible for external compaction."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)

    conversations = service.list_imported_conversations(limit=limit, pending_only=pending_only)

    output_format = _resolve_output_format(format)
    if output_format == "json":
        print_json(
            {
                "status": "ok",
                "data": {"conversations": conversations},
                "exit_code": 0,
            }
        )
        return

    if not conversations:
        console.print("[dim]No imported conversations found for external compaction.[/dim]")
        return

    table = Table(title="External Compaction Queue", box=box.SQUARE)
    table.add_column("Source Session ID", overflow="fold")
    table.add_column("Last Seen", overflow="fold")
    table.add_column("Entries", justify="right")
    table.add_column("Pending", justify="center")
    for row in conversations:
        table.add_row(
            str(row.get("source_session_id", "")),
            str(row.get("last_timestamp") or "-"),
            str(int(row.get("entry_count", 0))),
            "[warning]yes[/warning]" if bool(row.get("pending")) else "[success]no[/success]",
        )
    console.print(table)


@external_compaction_app.command("export")
def external_compaction_export(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output file path (prints JSON to stdout when omitted)",
    ),
    session_id: list[str] = typer.Option(
        None,
        "--session-id",
        help="Specific source session ID(s); defaults to pending conversations",
    ),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Max conversations to include"),
    entry_limit: int = typer.Option(300, "--entry-limit", min=1, help="Max entries per session"),
    pending_only: bool = typer.Option(
        True,
        "--pending-only/--all",
        help="When no --session-id is provided, include pending only by default",
    ),
    write_target: str | None = typer.Option(
        None,
        "--write-target",
        help="Tier write target: runtime or templates (defaults from config)",
    ),
):
    """Export compaction payload JSON for an external agent."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)

    try:
        target = _resolve_external_write_target_override(files, write_target)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    payload = service.build_payload(
        source_session_ids=[item for item in session_id if item] if session_id else None,
        limit=limit,
        pending_only=pending_only,
        entry_limit=entry_limit,
        write_target=target,
    )

    rendered = json.dumps(payload, indent=2)
    if output is None:
        typer.echo(rendered)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    console.print(f"[success]✓ Exported external compaction payload to {output}[/success]")


@external_compaction_app.command("apply")
def external_compaction_apply(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="JSON notes payload from external agent",
    ),
    write_target: str | None = typer.Option(
        None,
        "--write-target",
        help="Tier write target: runtime or templates (defaults from config)",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        "-d/-w",
        help="Preview changes only by default; writes require --commit",
    ),
    commit: bool = typer.Option(
        False,
        "--commit",
        help="Apply writes (required for non-dry-run execution)",
    ),
    mark_processed: bool = typer.Option(
        True,
        "--mark-processed/--no-mark-processed",
        help="Update external compaction state for referenced source sessions",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Apply external compaction notes to GUARDRAILS/STYLE/RECENT tiers."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)
    exit_ok = 0
    exit_no_changes = 10
    exit_invalid_input = 20

    try:
        target = _resolve_external_write_target_override(files, write_target)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(exit_invalid_input) from None

    effective_dry_run = dry_run
    if commit:
        effective_dry_run = False
    elif not dry_run:
        console.print("[error]Writes require --commit. Re-run with --commit.[/error]")
        raise typer.Exit(exit_invalid_input)

    try:
        payload = json.loads(input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[error]Invalid JSON input: {exc}[/error]")
        raise typer.Exit(exit_invalid_input) from None

    try:
        result = service.apply_notes_payload(
            payload,
            write_target=target,
            dry_run=effective_dry_run,
            mark_processed=mark_processed,
        )
    except ExternalNotesValidationError as exc:
        details = exc.to_dict()
        if format.strip().lower() == "json":
            details["exit_code"] = exit_invalid_input
            typer.echo(json.dumps(details, indent=2))
        else:
            console.print(f"[error]{details['message']}[/error]")
            for issue in details["errors"]:
                console.print(f"[dim]- {issue}[/dim]")
            console.print("[dim]Example payload:[/dim]")
            console.print(json.dumps(details["example"], indent=2))
        raise typer.Exit(exit_invalid_input) from None
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(exit_invalid_input) from None

    changed = result.get("tiers_changed", {})
    changed_lines = ", ".join(f"{tier}={count}" for tier, count in changed.items()) or "none"
    notes_applied = int(result.get("notes_applied", 0))
    code = exit_ok if notes_applied > 0 else exit_no_changes
    output_format = _resolve_output_format(format)
    if output_format == "json":
        payload_out = dict(result)
        payload_out["status"] = "ok" if code == exit_ok else "no_changes"
        print_json({"status": payload_out["status"], "data": payload_out, "exit_code": code})
        raise typer.Exit(code)
    console.print(
        Panel.fit(
            f"[success]✓ External notes processed[/success]\n\n"
            f"Write target: {result.get('write_target')}\n"
            f"Dry run: {result.get('dry_run')}\n"
            f"Notes received: {result.get('notes_received', 0)}\n"
            f"Notes applied: {notes_applied}\n"
            f"Tiers changed: {changed_lines}\n"
            f"Sessions marked: {result.get('sessions_marked', 0)}\n"
            f"Conflicts: {len(result.get('conflicts', []))}\n"
            f"Backlinks written: {result.get('evidence_backlinks_written', 0)}\n"
            f"Exit code: {code}",
            title="external-compaction apply",
        )
    )
    raise typer.Exit(code)


@external_compaction_app.command("patch-preview")
def external_compaction_patch_preview(
    queue_id: list[int] = typer.Option(
        None,
        "--queue-id",
        help="Optional queue ID filter(s)",
    ),
    state: list[str] = typer.Option(
        None,
        "--state",
        help="Queue state filter(s): pending, approved, rejected, applied",
    ),
    write_target: str | None = typer.Option(
        None,
        "--write-target",
        help="Tier write target: runtime or templates (defaults from config)",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Render before/after patch preview by tier for queued notes."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)
    try:
        target = _resolve_external_write_target_override(files, write_target)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    try:
        preview = service.patch_preview(
            queue_ids=[item for item in queue_id if item > 0] if queue_id else None,
            states=[item for item in state if item] if state else None,
            write_target=target,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    output_format = _resolve_output_format(format)
    if output_format == "json":
        print_json({"status": "ok", "data": preview, "exit_code": 0})
        return

    diff_by_tier = preview.get("diff_by_tier", {})
    if not isinstance(diff_by_tier, dict) or not diff_by_tier:
        console.print("[dim]No patch differences for selected queued notes.[/dim]")
        return

    for tier, diff_text in diff_by_tier.items():
        console.print(Panel.fit(str(diff_text), title=f"patch-preview {tier}"))


@external_compaction_app.command("apply-approved")
def external_compaction_apply_approved(
    actor: str = typer.Option("system", "--actor", help="Actor name for queue audit trail"),
    write_target: str | None = typer.Option(
        None,
        "--write-target",
        help="Tier write target: runtime or templates (defaults from config)",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Preview queued approved notes by default; writes require --commit",
    ),
    commit: bool = typer.Option(False, "--commit", help="Apply approved notes to tiers"),
    mark_processed: bool = typer.Option(
        True,
        "--mark-processed/--no-mark-processed",
        help="Update external compaction state for referenced source sessions",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Apply approved queued notes with idempotency checks."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)

    try:
        target = _resolve_external_write_target_override(files, write_target)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    effective_dry_run = dry_run
    if commit:
        effective_dry_run = False
    elif not dry_run:
        console.print("[error]Writes require --commit. Re-run with --commit.[/error]")
        raise typer.Exit(1)

    try:
        result = service.apply_approved_queue(
            actor=actor,
            write_target=target,
            mark_processed=mark_processed,
            dry_run=effective_dry_run,
        )
    except (RuntimeError, ValueError, ExternalNotesValidationError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    output_format = _resolve_output_format(format)
    if output_format == "json":
        print_json({"status": "ok", "data": result, "exit_code": 0})
        return

    changed = result.get("tiers_changed", {})
    changed_lines = ", ".join(f"{tier}={count}" for tier, count in changed.items()) or "none"
    console.print(
        Panel.fit(
            f"Queue items considered: {result.get('queue_items_considered', 0)}\n"
            f"Queue items applied: {result.get('queue_items_applied', 0)}\n"
            f"Notes applied: {result.get('notes_applied', 0)}\n"
            f"Tiers changed: {changed_lines}\n"
            f"Dry run: {result.get('dry_run')}\n"
            f"Conflicts: {len(result.get('conflicts', []))}\n"
            f"Backlinks written: {result.get('evidence_backlinks_written', 0)}\n"
            f"Write target: {result.get('write_target')}",
            title="external-compaction apply-approved",
        )
    )


@external_compaction_app.command("mcp-server")
def external_compaction_mcp_server(
    write_target: str | None = typer.Option(
        None,
        "--write-target",
        help="Tier write target exposed by MCP tools: runtime or templates",
    ),
):
    """Run MCP tools for external conversation compaction."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)

    try:
        target = _resolve_external_write_target_override(files, write_target)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    try:
        run_external_compaction_mcp(service, write_target=target)
    except RuntimeError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None


@external_compaction_app.command("cleanup-state")
def external_compaction_cleanup_state(
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Clean stale/invalid external compaction state entries."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)

    result = service.cleanup_state()
    output_format = _resolve_output_format(format)
    if output_format == "json":
        print_json({"status": "ok", "data": result, "exit_code": 0})
        return

    console.print(
        Panel.fit(
            f"Removed: {result.get('removed', 0)}\n"
            f"Removed invalid: {result.get('removed_invalid', 0)}\n"
            f"Removed stale: {result.get('removed_stale', 0)}",
            title="external-compaction cleanup-state",
        )
    )


@external_compaction_queue_app.command("add")
def external_compaction_queue_add(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="JSON notes payload from external agent",
    ),
    actor: str = typer.Option("system", "--actor", help="Actor name for queue audit trail"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Enqueue external notes for review."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)

    try:
        payload = json.loads(input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[error]Invalid JSON input: {exc}[/error]")
        raise typer.Exit(1) from None

    try:
        result = service.queue_notes_payload(payload, actor=actor)
    except (RuntimeError, ExternalNotesValidationError, ValueError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    output_format = _resolve_output_format(format)
    if output_format == "json":
        print_json({"status": "ok", "data": result, "exit_code": 0})
        return
    console.print(
        Panel.fit(
            f"Queued: {result.get('queued', 0)}",
            title="external-compaction queue add",
        )
    )


@external_compaction_queue_app.command("list")
def external_compaction_queue_list(
    state: list[str] = typer.Option(
        None,
        "--state",
        help="Optional state filter(s): pending, approved, rejected, applied",
    ),
    limit: int = typer.Option(100, "--limit", "-n", min=1, help="Maximum queue items"),
    with_attribution: bool = typer.Option(
        False,
        "--with-attribution",
        help="Include attribution summary inferred from source sessions",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """List queued external compaction notes."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)
    adapter = ExternalCompactionQueueAdapter(service)
    request = QueueListRequest(
        states=list(state) if state else None,
        limit=limit,
        with_attribution=with_attribution,
    )

    try:
        rows = load_queue_rows(
            adapter,
            request=request,
            resolve_attribution=(
                (lambda source_ids: _format_source_session_attribution(storage, source_ids))
                if with_attribution
                else None
            ),
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    output_format = _resolve_output_format(format)
    if output_format == "json":
        print_json({"status": "ok", "data": {"queue": rows}, "exit_code": 0})
        return
    if not rows:
        console.print("[dim]No queued notes found.[/dim]")
        return

    console.print(render_queue_table(rows, with_attribution=with_attribution))


@external_compaction_queue_app.command("approve")
def external_compaction_queue_approve(
    id: list[int] = typer.Option(..., "--id", help="Queue item ID(s) to approve"),
    actor: str = typer.Option("system", "--actor", help="Actor name for queue audit trail"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Approve queued notes."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)
    adapter = ExternalCompactionQueueAdapter(service)
    try:
        result = transition_queue_rows(
            adapter,
            request=QueueTransitionRequest(
                target_state="approved",
                ids=list(id),
                actor=actor,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    output_format = _resolve_output_format(format)
    if output_format == "json":
        print_json({"status": "ok", "data": result, "exit_code": 0})
        return
    console.print(
        Panel.fit(
            f"Updated: {result.get('updated', 0)}\nSkipped: {result.get('skipped', 0)}",
            title="external-compaction queue approve",
        )
    )


@external_compaction_queue_app.command("reject")
def external_compaction_queue_reject(
    id: list[int] = typer.Option(..., "--id", help="Queue item ID(s) to reject"),
    actor: str = typer.Option("system", "--actor", help="Actor name for queue audit trail"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Reject queued notes."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    service = _build_external_compaction_service(storage, files)
    adapter = ExternalCompactionQueueAdapter(service)
    try:
        result = transition_queue_rows(
            adapter,
            request=QueueTransitionRequest(
                target_state="rejected",
                ids=list(id),
                actor=actor,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    output_format = _resolve_output_format(format)
    if output_format == "json":
        print_json({"status": "ok", "data": result, "exit_code": 0})
        return
    console.print(
        Panel.fit(
            f"Updated: {result.get('updated', 0)}\nSkipped: {result.get('skipped', 0)}",
            title="external-compaction queue reject",
        )
    )


def _parse_feedback_score(value: str) -> int:
    normalized = value.strip().lower()
    mapping = {
        "1": 1,
        "+1": 1,
        "up": 1,
        "upvote": 1,
        "positive": 1,
        "-1": -1,
        "down": -1,
        "downvote": -1,
        "negative": -1,
    }
    parsed = mapping.get(normalized)
    if parsed is None:
        raise ValueError("score must be one of: up, down, 1, -1")
    return parsed


@feedback_app.command("add")
def feedback_add(
    query: str = typer.Option(..., "--query", help="Query used during retrieval"),
    chunk_id: str = typer.Option(..., "--chunk-id", help="Retrieved chunk UUID"),
    score: str = typer.Option(..., "--score", help="Feedback score: up|down|1|-1"),
    actor: str = typer.Option("user", "--actor", help="Actor recording feedback"),
    source: str = typer.Option("cli", "--source", help="Feedback source channel"),
    metadata_json: str | None = typer.Option(
        None,
        "--metadata-json",
        help="Optional JSON object metadata",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Record feedback for a retrieved chunk."""
    _get_theme_manager()
    storage = get_storage()
    _require_storage_capability(
        storage,
        "retrieval_feedback",
        message="Active storage backend does not support retrieval feedback.",
    )

    try:
        parsed_chunk_id = UUID(chunk_id)
    except ValueError:
        console.print("[error]Invalid --chunk-id. Expected UUID.[/error]")
        raise typer.Exit(1) from None

    try:
        parsed_score = _parse_feedback_score(score)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    metadata: dict[str, Any] | None = None
    if metadata_json:
        try:
            parsed_metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            console.print(f"[error]Invalid --metadata-json: {exc}[/error]")
            raise typer.Exit(1) from None
        if not isinstance(parsed_metadata, dict):
            console.print("[error]--metadata-json must decode to an object.[/error]")
            raise typer.Exit(1)
        metadata = parsed_metadata

    try:
        result = storage.record_retrieval_feedback(
            query=query,
            chunk_id=parsed_chunk_id,
            score=parsed_score,
            actor=actor,
            source=source,
            metadata=metadata,
        )
    except (UnsupportedStorageCapabilityError, ValueError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(result, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)

    label = "upvote" if int(result.get("score", 0)) > 0 else "downvote"
    console.print(
        Panel.fit(
            f"Feedback saved\nQuery: {result.get('query_text')}\n"
            f"Chunk: {result.get('chunk_id')}\nScore: {label}",
            title="feedback add",
        )
    )


@feedback_app.command("list")
def feedback_list(
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum feedback rows"),
    query: str | None = typer.Option(None, "--query", help="Filter by exact query text"),
    chunk_id: str | None = typer.Option(None, "--chunk-id", help="Filter by chunk UUID"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """List retrieval feedback rows."""
    _get_theme_manager()
    storage = get_storage()
    _require_storage_capability(
        storage,
        "retrieval_feedback",
        message="Active storage backend does not support retrieval feedback.",
    )

    parsed_chunk_id: UUID | None = None
    if chunk_id:
        try:
            parsed_chunk_id = UUID(chunk_id)
        except ValueError:
            console.print("[error]Invalid --chunk-id. Expected UUID.[/error]")
            raise typer.Exit(1) from None

    rows = storage.list_retrieval_feedback(
        limit=limit,
        query=query,
        chunk_id=parsed_chunk_id,
    )

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps({"feedback": rows}, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)
    if not rows:
        console.print("[warning]No feedback rows found.[/warning]")
        return

    table = Table(title="Retrieval Feedback", box=box.SIMPLE)
    table.add_column("Created")
    table.add_column("Score")
    table.add_column("Query")
    table.add_column("Chunk")
    table.add_column("Actor")
    table.add_column("Source")
    for row in rows:
        raw_query = str(row.get("query_text", "")).strip()
        preview_query = textwrap.shorten(raw_query, width=36, placeholder="...")
        score_value = int(row.get("score", 0))
        score_label = "↑" if score_value > 0 else "↓"
        table.add_row(
            str(row.get("created_at", ""))[:19],
            score_label,
            preview_query,
            str(row.get("chunk_id", ""))[:8],
            str(row.get("actor", "")),
            str(row.get("source", "")),
        )
    console.print(table)


@feedback_app.command("evaluate")
def feedback_evaluate(
    top_k: int = typer.Option(10, "--top-k", min=1, help="Ranking depth"),
    min_labels_per_query: int = typer.Option(
        2,
        "--min-labels-per-query",
        min=1,
        help="Minimum labels required per query",
    ),
    feedback_limit: int = typer.Option(2000, "--feedback-limit", min=1, help="Rows to inspect"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Evaluate ranking quality before and after applying feedback signals."""
    _get_theme_manager()
    storage = get_storage()
    _require_storage_capability(
        storage,
        "retrieval_feedback",
        message="Active storage backend does not support retrieval feedback.",
    )
    report = evaluate_feedback_impact(
        storage,
        top_k=top_k,
        min_labels_per_query=min_labels_per_query,
        feedback_limit=feedback_limit,
    )

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(report.to_dict(), indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)

    summary = Table(title="Feedback Evaluation", box=box.SIMPLE)
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Queries evaluated", str(report.queries_evaluated))
    summary.add_row("Mean baseline score", f"{report.mean_baseline_score:.4f}")
    summary.add_row("Mean feedback score", f"{report.mean_feedback_score:.4f}")
    summary.add_row("Mean delta", f"{report.mean_delta:.4f}")
    summary.add_row("Improved", str(report.improved_queries))
    summary.add_row("Regressed", str(report.regressed_queries))
    summary.add_row("Unchanged", str(report.unchanged_queries))
    console.print(summary)


@attribution_app.command("summary")
def attribution_summary(
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum grouped rows per table"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Summarize captured attribution metadata by agent/provider/model."""
    _get_theme_manager()
    storage = get_storage()
    entries = _collect_entries_for_attribution(storage, limit_per_status=500)
    by_agent: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for entry in entries:
        agent_source, provider, model = _entry_attribution_fields(entry)
        by_agent[agent_source] = by_agent.get(agent_source, 0) + 1
        by_provider[provider] = by_provider.get(provider, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1

    agent_rows = sorted(by_agent.items(), key=lambda item: (-item[1], item[0]))[:limit]
    provider_rows = sorted(by_provider.items(), key=lambda item: (-item[1], item[0]))[:limit]
    model_rows = sorted(by_model.items(), key=lambda item: (-item[1], item[0]))[:limit]

    payload = {
        "total_entries": len(entries),
        "by_agent": agent_rows,
        "by_provider": provider_rows,
        "by_model": model_rows,
    }

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)

    console.print(f"[bold]Attributed entries:[/bold] {payload['total_entries']}")
    for title, rows in (
        ("By Agent", agent_rows),
        ("By Provider", provider_rows),
        ("By Model", model_rows),
    ):
        table = Table(title=title, box=box.SIMPLE)
        table.add_column("Name")
        table.add_column("Count", justify="right")
        for name, count in rows:
            table.add_row(str(name), str(count))
        console.print(table)


@attribution_app.command("list")
def attribution_list(
    agent: str | None = typer.Option(None, "--agent", help="Filter by agent source"),
    provider: str | None = typer.Option(None, "--provider", help="Filter by provider"),
    model: str | None = typer.Option(None, "--model", help="Filter by model"),
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum rows"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """List entries with attribution metadata and optional filters."""
    _get_theme_manager()
    storage = get_storage()
    entries = _collect_entries_for_attribution(storage, limit_per_status=max(limit, 200))
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        agent_source, provider_name, model_name = _entry_attribution_fields(entry)
        if agent and agent_source != agent:
            continue
        if provider and provider_name != provider:
            continue
        if model and model_name != model:
            continue
        filtered.append(
            {
                "id": str(entry.id),
                "timestamp": entry.timestamp.isoformat(),
                "label": entry.label.value,
                "agent_source": agent_source,
                "provider": provider_name,
                "model": model_name,
                "content": entry.content,
            }
        )
        if len(filtered) >= limit:
            break

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps({"entries": filtered}, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)
    if not filtered:
        console.print("[warning]No attribution rows matched filters.[/warning]")
        return

    table = Table(title="Attribution Entries", box=box.SIMPLE)
    table.add_column("Time")
    table.add_column("Label")
    table.add_column("Agent")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Content")
    for row in filtered:
        table.add_row(
            str(row["timestamp"])[:19],
            str(row["label"]),
            str(row["agent_source"]),
            str(row["provider"]),
            str(row["model"]),
            textwrap.shorten(str(row["content"]), width=56, placeholder="..."),
        )
    console.print(table)


@topic_threads_app.command("rebuild")
def topic_threads_rebuild(
    min_cluster_size: int = typer.Option(
        2,
        "--min-cluster-size",
        min=1,
        help="Minimum chunk count required to create a thread",
    ),
    max_threads: int = typer.Option(25, "--max-threads", min=1, help="Maximum threads to keep"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute threads without persisting"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Rebuild cross-session topic threads from indexed chunks."""
    _get_theme_manager()
    storage = get_storage()
    threads = build_topic_threads(
        storage,
        min_cluster_size=min_cluster_size,
        max_threads=max_threads,
    )
    persisted = 0
    if not dry_run:
        _require_storage_capability(
            storage,
            "topic_threads",
            message="Active storage backend does not support topic threads.",
        )
        persisted = storage.replace_topic_threads(threads)
    output = {
        "threads_generated": len(threads),
        "threads_persisted": persisted,
        "dry_run": dry_run,
        "threads": threads,
    }

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(output, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)

    table = Table(title="Topic Threads Rebuild", box=box.SIMPLE)
    table.add_column("Thread")
    table.add_column("Score", justify="right")
    table.add_column("Entries", justify="right")
    table.add_column("Summary")
    for thread in threads[:15]:
        summary = textwrap.shorten(str(thread.get("summary", "")), width=60, placeholder="...")
        table.add_row(
            str(thread.get("title", "Untitled")),
            f"{float(thread.get('score', 0.0)):.2f}",
            str(thread.get("entry_count", 0)),
            summary,
        )
    console.print(table)
    console.print(
        f"[dim]Generated {len(threads)} thread(s); persisted {persisted} (dry_run={dry_run}).[/dim]"
    )


@topic_threads_app.command("list")
def topic_threads_list(
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum threads to show"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """List persisted topic thread metadata."""
    _get_theme_manager()
    storage = get_storage()
    _require_storage_capability(
        storage,
        "topic_threads",
        message="Active storage backend does not support topic threads.",
    )
    threads = storage.list_topic_threads(limit=limit)

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps({"threads": threads}, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)
    if not threads:
        console.print(
            "[warning]No topic threads available. Run topic-threads rebuild first.[/warning]"
        )
        return

    table = Table(title="Topic Threads", box=box.SIMPLE)
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Score", justify="right")
    table.add_column("Entries", justify="right")
    table.add_column("Source sessions", justify="right")
    table.add_column("Last seen")
    for thread in threads:
        table.add_row(
            str(thread.get("thread_id", ""))[:8],
            str(thread.get("title", "")),
            f"{float(thread.get('score', 0.0)):.2f}",
            str(thread.get("entry_count", 0)),
            str(thread.get("source_session_count", 0)),
            str(thread.get("last_seen_at", ""))[:19],
        )
    console.print(table)


@topic_threads_app.command("show")
def topic_threads_show(
    thread_id: str = typer.Argument(..., help="Thread ID"),
    limit_links: int = typer.Option(20, "--limit-links", min=1, help="Max linked items to show"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Inspect one topic thread and linked entries/chunks."""
    _get_theme_manager()
    storage = get_storage()
    _require_storage_capability(
        storage,
        "topic_threads",
        message="Active storage backend does not support topic threads.",
    )
    thread = storage.get_topic_thread(thread_id, limit_links=limit_links)
    if thread is None:
        console.print(f"[warning]Thread not found: {thread_id}[/warning]")
        raise typer.Exit(1)

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(thread, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)

    summary = textwrap.shorten(str(thread.get("summary", "")), width=120, placeholder="...")
    console.print(
        Panel.fit(
            f"Title: {thread.get('title')}\n"
            f"Score: {float(thread.get('score', 0.0)):.2f}\n"
            f"Entries: {thread.get('entry_count')}\n"
            f"Source sessions: {thread.get('source_session_count')}\n"
            f"Summary: {summary}",
            title=f"topic-thread {thread.get('thread_id')}",
        )
    )
    links = thread.get("links", [])
    if not isinstance(links, list) or not links:
        return
    table = Table(title="Linked Items", box=box.SIMPLE)
    table.add_column("Entry")
    table.add_column("Chunk")
    table.add_column("Source session")
    table.add_column("Snippet")
    for item in links:
        if not isinstance(item, dict):
            continue
        snippet_source = item.get("entry_content") or item.get("chunk_content") or ""
        snippet = textwrap.shorten(str(snippet_source), width=60, placeholder="...")
        table.add_row(
            str(item.get("entry_id") or "-")[:8],
            str(item.get("chunk_id") or "-")[:8],
            str(item.get("source_session_id") or "-"),
            snippet,
        )
    console.print(table)


@topic_threads_app.command("summary")
def topic_threads_summary(
    limit: int = typer.Option(5, "--limit", min=1, help="Top threads to summarize"),
):
    """Summarize top active topic threads."""
    _get_theme_manager()
    storage = get_storage()
    _require_storage_capability(
        storage,
        "topic_threads",
        message="Active storage backend does not support topic threads.",
    )
    threads = storage.list_topic_threads(limit=limit)
    if not threads:
        console.print(
            "[warning]No topic threads available. Run topic-threads rebuild first.[/warning]"
        )
        return

    lines = ["[bold]Top Active Topic Threads[/bold]"]
    for index, thread in enumerate(threads, start=1):
        lines.append(
            f"{index}. {thread.get('title')} "
            f"(entries={thread.get('entry_count')}, score={float(thread.get('score', 0.0)):.2f})"
        )
        lines.append(textwrap.shorten(str(thread.get("summary", "")), width=120, placeholder="..."))
    console.print(Panel.fit("\n".join(lines), title="topic-threads summary"))


def _prune_rules_from_tier_files(
    files: FileStorage,
    candidates: list[dict[str, Any]],
) -> tuple[int, Path | None]:
    tier_map = {
        "GUARDRAILS": KnowledgeTier.GUARDRAILS,
        "STYLE": KnowledgeTier.STYLE,
    }
    rules_by_tier: dict[KnowledgeTier, set[str]] = {}
    for candidate in candidates:
        tier_name = str(candidate.get("tier", "")).strip().upper()
        line = str(candidate.get("line", "")).strip()
        tier = tier_map.get(tier_name)
        if tier is None or not line:
            continue
        rules_by_tier.setdefault(tier, set()).add(line)

    removed = 0
    archived_lines: list[str] = []
    for tier, lines_to_remove in rules_by_tier.items():
        content = files.read_tier(tier)
        kept_lines: list[str] = []
        for raw in content.splitlines():
            stripped = raw.strip()
            if stripped.startswith("-"):
                candidate_line = stripped[1:].strip()
                if candidate_line in lines_to_remove:
                    removed += 1
                    archived_lines.append(f"- [{tier.value}] {candidate_line}")
                    continue
            kept_lines.append(raw)
        if content and not content.endswith("\n"):
            kept_lines_text = "\n".join(kept_lines)
        else:
            kept_lines_text = "\n".join(kept_lines) + ("\n" if kept_lines else "")
        files.write_tier(tier, kept_lines_text)

    if removed == 0:
        return 0, None

    archive_dir = files.agent_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"rule-prune-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.md"
    archive_lines = [
        "# Rule Prune Archive",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        f"Removed rules: {removed}",
        "",
        "## Removed",
        *archived_lines,
        "",
    ]
    archive_path.write_text("\n".join(archive_lines), encoding="utf-8")
    return removed, archive_path


@rule_confidence_app.command("refresh")
def rule_confidence_refresh(
    default_confidence: float = typer.Option(
        0.6,
        "--default-confidence",
        min=0.0,
        max=1.0,
        help="Initial confidence for newly tracked rules",
    ),
    reinforcement_factor: float = typer.Option(
        0.15,
        "--reinforcement-factor",
        min=0.0,
        max=1.0,
        help="Confidence increase factor for re-observed rules",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Scan tier files and reinforce tracked rules."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    _require_storage_capability(
        storage,
        "rule_confidence",
        message="Active storage backend does not support rule confidence.",
    )
    rules = snapshot_rules(files)
    result = storage.sync_rule_confidence(
        rules,
        default_confidence=default_confidence,
        reinforcement_factor=reinforcement_factor,
    )

    payload = {"rules_scanned": len(rules), **result}
    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)
    console.print(
        Panel.fit(
            f"Rules scanned: {payload['rules_scanned']}\n"
            f"Inserted: {payload.get('inserted', 0)}\n"
            f"Updated: {payload.get('updated', 0)}",
            title="rule-confidence refresh",
        )
    )


@rule_confidence_app.command("list")
def rule_confidence_list(
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum rules to list"),
    stale_only: bool = typer.Option(False, "--stale-only", help="Show only stale rules"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """List tracked rule confidence rows."""
    _get_theme_manager()
    storage = get_storage()
    _require_storage_capability(
        storage,
        "rule_confidence",
        message="Active storage backend does not support rule confidence.",
    )
    rows = storage.list_rule_confidence(limit=limit, stale_only=stale_only)

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps({"rules": rows}, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)
    if not rows:
        console.print("[warning]No tracked rule confidence rows.[/warning]")
        return

    table = Table(title="Rule Confidence", box=box.SIMPLE)
    table.add_column("Tier")
    table.add_column("Confidence", justify="right")
    table.add_column("Stale")
    table.add_column("Reinforce", justify="right")
    table.add_column("Rule")
    for row in rows:
        table.add_row(
            str(row.get("tier", "")),
            f"{float(row.get('confidence', 0.0)):.2f}",
            "yes" if bool(row.get("is_stale")) else "no",
            str(row.get("reinforcement_count", 0)),
            textwrap.shorten(str(row.get("line", "")), width=72, placeholder="..."),
        )
    console.print(table)


@rule_confidence_app.command("decay")
def rule_confidence_decay(
    half_life_days: float = typer.Option(
        45.0,
        "--half-life-days",
        min=1.0,
        help="Half-life used for confidence decay",
    ),
    stale_after_days: float = typer.Option(
        60.0,
        "--stale-after-days",
        min=1.0,
        help="Mark rules stale when inactive for this many days",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Apply confidence decay and stale tagging to tracked rules."""
    _get_theme_manager()
    storage = get_storage()
    _require_storage_capability(
        storage,
        "rule_confidence",
        message="Active storage backend does not support rule confidence.",
    )
    result = storage.decay_rule_confidence(
        half_life_days=half_life_days,
        stale_after_days=stale_after_days,
    )

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(result, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)
    console.print(
        Panel.fit(
            f"Rows decayed: {result.get('decayed', 0)}\n"
            f"Marked stale: {result.get('stale_marked', 0)}",
            title="rule-confidence decay",
        )
    )


@rule_confidence_app.command("prune")
def rule_confidence_prune(
    max_confidence: float = typer.Option(
        0.35,
        "--max-confidence",
        min=0.0,
        max=1.0,
        help="Prune rules at or below this confidence",
    ),
    stale_only: bool = typer.Option(
        True,
        "--stale-only/--include-fresh",
        help="Restrict pruning to stale rules",
    ),
    commit: bool = typer.Option(
        False,
        "--commit",
        help="Apply prune and write tier updates (default is dry-run)",
    ),
    limit: int = typer.Option(500, "--limit", min=1, help="Maximum candidates"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Archive and prune low-confidence rules, with optional tier-file removal."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    _require_storage_capability(
        storage,
        "rule_confidence",
        message="Active storage backend does not support rule confidence.",
    )
    candidates = storage.archive_and_prune_rule_confidence(
        max_confidence=max_confidence,
        stale_only=stale_only,
        dry_run=not commit,
        limit=limit,
    )

    removed_from_tiers = 0
    archive_path: Path | None = None
    if commit and candidates:
        removed_from_tiers, archive_path = _prune_rules_from_tier_files(files, candidates)

    payload = {
        "candidates": candidates,
        "candidate_count": len(candidates),
        "commit": commit,
        "removed_from_tiers": removed_from_tiers,
        "archive_path": str(archive_path) if archive_path else None,
    }
    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)
    if not candidates:
        console.print("[success]No rules matched prune criteria.[/success]")
        return
    table = Table(title="Rule Prune Candidates", box=box.SIMPLE)
    table.add_column("Tier")
    table.add_column("Confidence", justify="right")
    table.add_column("Stale")
    table.add_column("Rule")
    for row in candidates[:25]:
        table.add_row(
            str(row.get("tier", "")),
            f"{float(row.get('confidence', 0.0)):.2f}",
            "yes" if bool(row.get("is_stale")) else "no",
            textwrap.shorten(str(row.get("line", "")), width=72, placeholder="..."),
        )
    console.print(table)
    if commit:
        console.print(
            f"[dim]Pruned {len(candidates)} rows; removed {removed_from_tiers} tier lines.[/dim]"
        )
        if archive_path:
            console.print(f"[dim]Archive: {archive_path}[/dim]")
    else:
        console.print("[dim]Dry-run only. Re-run with --commit to apply changes.[/dim]")


@memory_pack_app.command("export")
def memory_pack_export(
    output: Path = typer.Option(
        Path(".agent/export/memory-pack.json"),
        "--output",
        "-o",
        help="Output memory pack path",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Export tiers/chunks/metadata into a versioned memory pack file."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    pack = build_memory_pack(storage, files, policy=_memory_policy(files))
    write_memory_pack(output, pack)
    policy_report = pack.metadata.get("policy", {}) if isinstance(pack.metadata, dict) else {}
    payload = {
        "path": str(output),
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "chunk_count": len(pack.chunks),
        "policy": policy_report,
    }

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)
    warnings_count = 0
    if isinstance(policy_report, dict):
        warnings_value = policy_report.get("warnings", [])
        if isinstance(warnings_value, list):
            warnings_count = len(warnings_value)
    lines = [
        f"Path: {payload['path']}",
        f"Format: {payload['format']}",
        f"Version: {payload['version']}",
        f"Chunks: {payload['chunk_count']}",
        f"Policy warnings: {warnings_count}",
    ]
    console.print(
        Panel.fit(
            "\n".join(lines),
            title="memory-pack export",
        )
    )


@memory_pack_app.command("validate")
def memory_pack_validate(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        help="Memory pack JSON file",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Validate pack compatibility and schema expectations."""
    _get_theme_manager()
    try:
        pack = read_memory_pack(input_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Failed to parse memory pack: {exc}[/error]")
        raise typer.Exit(1) from None
    validation = validate_memory_pack(pack)
    payload = {
        "path": str(input_path),
        **validation,
    }

    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)

    lines = [
        f"Path: {input_path}",
        f"Valid: {'yes' if validation['valid'] else 'no'}",
        f"Chunks: {validation['chunk_count']}",
    ]
    warnings = validation.get("warnings", [])
    errors = validation.get("errors", [])
    if warnings:
        lines.append(f"Warnings: {len(warnings)}")
        lines.extend(f"- {warning}" for warning in warnings)
    if errors:
        lines.append(f"Errors: {len(errors)}")
        lines.extend(f"- {error}" for error in errors)
    console.print(Panel.fit("\n".join(lines), title="memory-pack validate"))
    if not validation["valid"]:
        raise typer.Exit(1)


@memory_pack_app.command("import")
def memory_pack_import_cmd(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        help="Memory pack JSON file",
    ),
    strategy: str = typer.Option(
        "append",
        "--strategy",
        help="Merge strategy: skip, append, overwrite",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Import a memory pack with configurable merge conflict strategy."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy not in {"skip", "append", "overwrite"}:
        console.print("[error]Invalid --strategy. Use skip, append, or overwrite.[/error]")
        raise typer.Exit(1)
    merge_strategy = cast(MergeStrategy, normalized_strategy)

    try:
        pack = read_memory_pack(input_path)
        report = import_memory_pack(
            storage,
            files,
            pack,
            strategy=merge_strategy,
            policy=_memory_policy(files),
        )
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Memory pack import failed: {exc}[/error]")
        raise typer.Exit(1) from None

    payload = {
        "path": str(input_path),
        "strategy": normalized_strategy,
        **report,
    }
    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)

    lines = [
        f"Path: {input_path}",
        f"Strategy: {normalized_strategy}",
        f"Tier updates: {report['tier_updates']}",
        f"Chunks written: {report['chunks_written']}",
        f"Chunks skipped: {report['chunks_skipped']}",
    ]
    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append(f"Warnings: {len(warnings)}")
        lines.extend(f"- {warning}" for warning in warnings)
    console.print(Panel.fit("\n".join(lines), title="memory-pack import"))


@memory_app.command("mode")
def memory_mode(
    set_mode: str | None = typer.Option(
        None,
        "--set",
        help="Set memory.mode to markdown, hybrid, or vector_primary",
    ),
):
    """Show or update memory mode feature flag."""
    _get_theme_manager()
    files = get_files()
    config = files.read_config()
    memory_cfg = config.get("memory", {}) if isinstance(config, dict) else {}
    if not isinstance(memory_cfg, dict):
        memory_cfg = {}

    if set_mode is None:
        mode = str(memory_cfg.get("mode", "markdown"))
        console.print(f"[bold]memory.mode[/bold] = {mode}")
        return

    normalized = set_mode.strip().lower()
    if normalized not in {"markdown", "hybrid", "vector_primary"}:
        console.print("[error]Invalid mode. Use markdown, hybrid, or vector_primary.[/error]")
        raise typer.Exit(1)
    memory_cfg["mode"] = normalized
    config["memory"] = memory_cfg
    files.write_config(config)
    console.print(f"[success]Updated memory.mode to {normalized}[/success]")


@memory_app.command("migrate-vectors")
def memory_migrate_vectors(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview migration without writing"),
    max_records: int | None = typer.Option(
        None,
        "--max-records",
        min=1,
        help="Optional cap on number of records to migrate",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="Override embedding batch size",
    ),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
):
    """Migrate tier markdown + chunks into vector records."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()
    memory_cfg = _read_memory_config(files)
    config = load_config(AGENT_DIR)
    service = VectorMigrationService(
        storage=storage,
        files=files,
        memory_cfg=memory_cfg,
        policy=_memory_policy(files),
        tenant_id=config.storage.shared.tenant_id,
        project_id=config.storage.shared.project_id,
        embedding_dimensions=config.retrieval.embedding_dimensions,
        vector_db_path=AGENT_DIR / "vector.db",
    )
    try:
        payload = service.migrate(
            VectorMigrationRequest(
                dry_run=dry_run,
                max_records=max_records,
                batch_size=batch_size,
            )
        )
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]memory migrate-vectors failed: {exc}[/error]")
        raise typer.Exit(1) from None
    output_format = format.strip().lower()
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)
    console.print(
        Panel.fit(
            f"Rows discovered: {payload['rows_discovered']}\n"
            f"Rows normalized: {payload['rows_normalized']}\n"
            f"Rows migrated: {payload['rows_migrated']}\n"
            f"Rows written: {payload['rows_written']}\n"
            f"Redacted rows: {payload['redacted_rows']}\n"
            f"Provider/backend: {payload['embedding_provider']}/{payload['vector_backend']}\n"
            f"Estimated tokens: {payload['estimated_tokens']}\n"
            f"Estimated cost USD: {payload['estimated_cost_usd']:.4f}\n"
            f"Dry run: {payload['dry_run']}",
            title="memory migrate-vectors",
        )
    )


@memory_app.command("prune-vectors")
def memory_prune_vectors(
    retention_days: int | None = typer.Option(
        None,
        "--retention-days",
        min=1,
        help="Override memory.privacy.retention_days",
    ),
):
    """Apply vector retention policy for local vector backend."""
    _get_theme_manager()
    files = get_files()
    config = load_config(AGENT_DIR)
    memory_cfg = _read_memory_config(files)
    backend = str(memory_cfg.get("vector_backend", "local")).strip().lower()
    if backend != "local":
        console.print("[error]prune-vectors currently supports local vector backend only.[/error]")
        raise typer.Exit(1)
    policy = _memory_policy(files)
    effective_days = policy.resolve_retention_days(retention_days)
    vector_store = LocalVectorStore(
        AGENT_DIR / "vector.db",
        tenant_id=config.storage.shared.tenant_id,
        project_id=config.storage.shared.project_id,
    )
    removed = vector_store.prune_older_than(retention_days=effective_days)
    console.print(
        Panel.fit(
            f"Retention days: {effective_days}\nRemoved records: {removed}",
            title="memory prune-vectors",
        )
    )


external_compaction_app.add_typer(external_compaction_queue_app, name="queue")
tiers_app.add_typer(tiers_write_app, name="write")


app.add_typer(theme_app, name="theme")
app.add_typer(metrics_app, name="metrics")
app.add_typer(context_app, name="context")
app.add_typer(sync_app, name="sync")
app.add_typer(feedback_app, name="feedback")
app.add_typer(attribution_app, name="attribution")
app.add_typer(topic_threads_app, name="topic-threads")
app.add_typer(rule_confidence_app, name="rule-confidence")
app.add_typer(memory_pack_app, name="memory-pack")
app.add_typer(memory_app, name="memory")
app.add_typer(tiers_app, name="tiers")
app.add_typer(config_app, name="config")
app.add_typer(curation_app, name="curation")
app.add_typer(ralph_app, name="ralph")
app.add_typer(embedding_app, name="embedding")
app.add_typer(external_compaction_app, name="external-compaction")


@curation_app.command("list")
def curation_list(
    status: str = typer.Option("pending", "--status", "-s", help="pending, approved, rejected"),
    limit: int = typer.Option(50, "--limit", "-l", min=1, help="Maximum entries"),
):
    """List log entries by curation status."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    try:
        status_enum = CurationStatus(status.strip().lower())
    except ValueError:
        valid = ", ".join(item.value for item in CurationStatus)
        console.print(f"[error]Invalid status '{status}'. Valid: {valid}[/error]")
        raise typer.Exit(1) from None

    entries = storage.list_entries_by_curation_status(status_enum, limit=limit)
    if not entries:
        console.print(f"[warning]No entries found for status '{status_enum.value}'.[/warning]")
        raise typer.Exit(0)

    table = Table(title=f"Curation Queue ({status_enum.value})", box=box.SIMPLE)
    table.add_column("ID", style="dim")
    table.add_column("Label")
    table.add_column("Confidence", justify="right")
    table.add_column("Status")
    table.add_column("Content")
    for entry in entries:
        preview = textwrap.shorten(entry.content, width=80, placeholder="...")
        table.add_row(
            str(entry.id),
            entry.label.value,
            f"{entry.confidence:.2f}",
            entry.curation_status.value,
            preview,
        )
    console.print(table)


@curation_app.command("approve")
def curation_approve(
    entry_id: str = typer.Argument(..., help="Entry UUID to approve"),
):
    """Approve a pending log entry."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    try:
        entry_uuid = UUID(entry_id)
    except ValueError:
        console.print(f"[error]Invalid entry id '{entry_id}'. Expected UUID.[/error]")
        raise typer.Exit(1) from None

    updated = storage.update_entry_curation_status(entry_uuid, CurationStatus.APPROVED)
    if updated is None:
        console.print(f"[warning]Entry not found: {entry_id}[/warning]")
        raise typer.Exit(1)
    console.print(f"[success]✓ Approved entry {entry_id}[/success]")


@curation_app.command("reject")
def curation_reject(
    entry_id: str = typer.Argument(..., help="Entry UUID to reject"),
):
    """Reject a pending log entry."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    try:
        entry_uuid = UUID(entry_id)
    except ValueError:
        console.print(f"[error]Invalid entry id '{entry_id}'. Expected UUID.[/error]")
        raise typer.Exit(1) from None

    updated = storage.update_entry_curation_status(entry_uuid, CurationStatus.REJECTED)
    if updated is None:
        console.print(f"[warning]Entry not found: {entry_id}[/warning]")
        raise typer.Exit(1)
    console.print(f"[success]✓ Rejected entry {entry_id}[/success]")


@tiers_write_app.command("guardrails")
def write_guardrails(
    iteration: int = typer.Option(..., "--iteration", "-i", help="Iteration number"),
    item_id: str = typer.Option(..., "--item-id", help="PRD item ID"),
    item_title: str = typer.Option(..., "--item-title", help="PRD item title"),
    reason: str = typer.Option("progressed", "--reason", "-r", help="Reason for entry"),
    validation_hint: str = typer.Option("", "--validation-hint", help="Validation signal hint"),
    mode: str = typer.Option("append", "--mode", "-m", help="Write mode: append, replace-section"),
    skip_duplicate: bool = typer.Option(
        True, "--skip-duplicate/--no-skip-duplicate", help="Skip if duplicate"
    ),
):
    """Write a structured entry to GUARDRAILS.md."""
    ensure_initialized()
    files = get_files()

    policy = WritePolicy(
        mode=WriteMode(mode),
        deduplicate=skip_duplicate,
    )
    writer = TierWriter(files, policy)

    try:
        written = writer.write_guardrails_entry(
            iteration=iteration,
            item_id=item_id,
            item_title=item_title,
            reason=reason,
            validation_hint=validation_hint,
        )
        if written:
            console.print(f"[success]Wrote guardrails entry for iteration {iteration}[/success]")
        else:
            console.print(f"[info]Skipped duplicate entry for iteration {iteration}[/info]")
    except TierValidationError as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        raise typer.Exit(1)


@tiers_write_app.command("guardrails-failure")
def write_guardrails_failure(
    iteration: int = typer.Option(..., "--iteration", "-i", help="Iteration number"),
    item_id: str = typer.Option(..., "--item-id", help="PRD item ID"),
    item_title: str = typer.Option(..., "--item-title", help="PRD item title"),
    validation_errors: list[str] = typer.Option([], "--error", help="Validation error messages"),
    validation_hint: str = typer.Option("", "--validation-hint", help="Primary validation signal"),
):
    """Write a hard failure entry to GUARDRAILS.md."""
    ensure_initialized()
    files = get_files()

    writer = TierWriter(files)

    try:
        written = writer.write_guardrails_hard_failure(
            iteration=iteration,
            item_id=item_id,
            item_title=item_title,
            validation_errors=validation_errors,
            validation_hint=validation_hint,
        )
        if written:
            console.print(
                f"[success]Wrote hard failure guardrails entry for iteration {iteration}[/success]"
            )
        else:
            console.print("[info]Skipped duplicate entry[/info]")
    except TierValidationError as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        raise typer.Exit(1)


@tiers_write_app.command("style")
def write_style(
    iteration: int = typer.Option(..., "--iteration", "-i", help="Iteration number"),
    item_id: str = typer.Option(..., "--item-id", help="PRD item ID"),
    validation_hint: str = typer.Option("", "--validation-hint", help="Validation signal hint"),
):
    """Write a structured entry to STYLE.md."""
    ensure_initialized()
    files = get_files()

    writer = TierWriter(files)

    try:
        written = writer.write_style_entry(
            iteration=iteration,
            item_id=item_id,
            validation_hint=validation_hint,
        )
        if written:
            console.print(f"[success]Wrote style entry for iteration {iteration}[/success]")
        else:
            console.print("[info]Skipped duplicate entry[/info]")
    except TierValidationError as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        raise typer.Exit(1)


@tiers_write_app.command("recent")
def write_recent(
    iteration: int = typer.Option(..., "--iteration", "-i", help="Iteration number"),
    item_id: str = typer.Option(..., "--item-id", help="PRD item ID"),
    item_title: str = typer.Option(..., "--item-title", help="PRD item title"),
    work_mode: str = typer.Option("feature", "--mode", "-m", help="Work mode"),
    agent_exit: int = typer.Option(0, "--agent-exit", help="Agent exit code"),
    validate_status: str = typer.Option("passed", "--validate-status", help="Validation status"),
    outcome: str = typer.Option("progressed", "--outcome", "-o", help="Outcome"),
    validation_hint: str = typer.Option("", "--validation-hint", help="Validation signal hint"),
):
    """Write a structured entry to RECENT.md."""
    ensure_initialized()
    files = get_files()

    writer = TierWriter(files)

    try:
        written = writer.write_recent_entry(
            iteration=iteration,
            item_id=item_id,
            item_title=item_title,
            work_mode=work_mode,
            agent_exit=agent_exit,
            validate_status=validate_status,
            outcome=outcome,
            validation_hint=validation_hint,
        )
        if written:
            console.print(f"[success]Wrote recent entry for iteration {iteration}[/success]")
        else:
            console.print("[info]Skipped duplicate entry[/info]")
    except TierValidationError as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        raise typer.Exit(1)


@tiers_app.command("lint")
def lint_tiers(
    tier: str = typer.Option(
        "all", "--tier", "-t", help="Tier to lint: guardrails, style, recent, all"
    ),
    strict: bool = typer.Option(False, "--strict", "-s", help="Treat warnings as errors"),
):
    """Lint tier files for formatting issues and validation errors."""
    ensure_initialized()
    files = get_files()

    tiers_to_lint: list[KnowledgeTier] = []
    if tier == "all":
        tiers_to_lint = [KnowledgeTier.GUARDRAILS, KnowledgeTier.STYLE, KnowledgeTier.RECENT]
    else:
        tier_map = {
            "guardrails": KnowledgeTier.GUARDRAILS,
            "style": KnowledgeTier.STYLE,
            "recent": KnowledgeTier.RECENT,
        }
        if tier not in tier_map:
            console.print(f"[error]Unknown tier: {tier}[/error]")
            raise typer.Exit(1)
        tiers_to_lint = [tier_map[tier]]

    all_passed = True
    for kt in tiers_to_lint:
        content = files.read_tier(kt)
        errors, warnings = lint_tier_file(kt, content, strict=strict)

        console.print(f"\n[accent]{kt.value}.md[/accent]")

        if errors:
            console.print(f"  [error]Errors ({len(errors)}):[/error]")
            for error in errors:
                console.print(f"    • {error}")
            all_passed = False
        else:
            console.print("  [success]No errors[/success]")

        if warnings:
            console.print(f"  [warning]Warnings ({len(warnings)}):[/warning]")
            for warning in warnings:
                console.print(f"    • {warning}")
            if strict:
                all_passed = False

    if all_passed:
        console.print("\n[success]All tier files passed linting[/success]")
        raise typer.Exit(0)
    else:
        console.print("\n[error]Some tier files have issues[/error]")
        raise typer.Exit(1)


@tiers_app.command("stats")
def tier_stats():
    """Show statistics for all tier files."""
    ensure_initialized()
    files = get_files()

    table = Table(title="Tier File Statistics")
    table.add_column("Tier", style="accent")
    table.add_column("Entries", justify="right")
    table.add_column("Size (chars)", justify="right")
    table.add_column("Lines", justify="right")
    table.add_column("Date Range")

    for tier in [KnowledgeTier.GUARDRAILS, KnowledgeTier.STYLE, KnowledgeTier.RECENT]:
        content = files.read_tier(tier)
        stats = get_tier_statistics(content)

        date_range = "N/A"
        if stats["date_range"]["earliest"] and stats["date_range"]["latest"]:
            earliest = stats["date_range"]["earliest"][:10]  # Just the date part
            latest = stats["date_range"]["latest"][:10]
            if earliest == latest:
                date_range = earliest
            else:
                date_range = f"{earliest} to {latest}"

        table.add_row(
            tier.value,
            str(stats["entry_count"]),
            f"{stats['content_size']:,}",
            str(stats["line_count"]),
            date_range,
        )

    console.print(table)


@tiers_app.command("compact")
def compact_tiers(
    max_entries: int | None = typer.Option(
        None,
        "--max-entries",
        "-n",
        min=10,
        max=200,
        help="Maximum entries per tier (overrides config)",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Enable strict deduplication",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Show what would be changed without writing",
    ),
):
    """Compact tier files: normalize, deduplicate, and apply size budgets."""
    ensure_initialized()
    files = get_files()

    # Load config
    config_dict = files.read_config()
    config = TierCompactionConfig.from_config(config_dict)

    # Apply CLI overrides
    if max_entries is not None:
        config.max_entries_per_tier = max_entries
    if strict:
        config.strict_deduplication = True

    hook = TierCompactionHook(files, config)

    if dry_run:
        console.print("[dim]Dry run mode - no changes will be written[/dim]")

    summary = hook.compact_all()

    # Build result display
    lines = ["[success]Tier compaction complete[/success]"]
    lines.append("")

    for result in summary.results:
        tier_name = result.tier.value.upper()
        lines.append(f"[accent]{tier_name}:[/accent]")
        lines.append(f"  Entries: {result.entries_before} → {result.entries_after}")
        lines.append(f"  Size: {result.bytes_before:,} → {result.bytes_after:,} bytes")
        if result.duplicates_removed:
            lines.append(f"  Duplicates removed: {result.duplicates_removed}")
        if result.entries_summarized:
            lines.append(f"  Entries summarized: {result.entries_summarized}")
        lines.append("")

    lines.append("[bold]Total:[/bold]")
    lines.append(f"  Entries: {summary.total_entries_before} → {summary.total_entries_after}")
    lines.append(f"  Size: {summary.total_bytes_before:,} → {summary.total_bytes_after:,} bytes")
    if summary.total_duplicates_removed:
        lines.append(f"  Duplicates removed: {summary.total_duplicates_removed}")
    if summary.total_entries_summarized:
        lines.append(f"  Entries summarized: {summary.total_entries_summarized}")

    console.print(Panel.fit("\n".join(lines), title="tiers compact"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
