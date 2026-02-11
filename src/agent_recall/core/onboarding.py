from __future__ import annotations

import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_recall.ingest import get_default_ingesters
from agent_recall.llm import (
    create_llm_provider,
    ensure_provider_dependency,
    get_available_providers,
)
from agent_recall.storage.files import FileStorage
from agent_recall.storage.models import LLMConfig

VALID_AGENT_SOURCES = ("cursor", "claude-code")
LOCAL_PROVIDERS = {"ollama", "vllm", "lmstudio", "openai-compatible", "custom"}

API_KEY_ENV_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4.1",
    "google": "gemini-2.5-flash",
    "ollama": "llama3.1",
    "vllm": "default",
    "lmstudio": "local-model",
    "openai-compatible": "gpt-4.1-mini",
}

DEFAULT_BASE_URLS = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai-compatible": "http://localhost:8080/v1",
}


def default_agent_recall_home() -> Path:
    override = os.environ.get("AGENT_RECALL_HOME")
    if override:
        return Path(override).expanduser()

    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/Application Support/agent-recall"
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData/Roaming"
        return base / "agent-recall"

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "agent-recall"
    return Path.home() / ".config" / "agent-recall"


def is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _set_private_permissions(path: Path, mode: int) -> None:
    if os.name == "nt":
        return
    try:
        os.chmod(path, mode)
    except OSError:
        return


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _set_private_permissions(path, 0o700)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def _write_yaml_mapping(path: Path, payload: dict[str, Any]) -> None:
    _ensure_private_dir(path.parent)
    path.write_text(yaml.dump(payload, default_flow_style=False, sort_keys=False))
    _set_private_permissions(path, 0o600)


class LocalSecretsStore:
    def __init__(self, home_dir: Path | None = None):
        self.home_dir = home_dir or default_agent_recall_home()
        self.path = self.home_dir / "secrets.yaml"

    def _load_raw(self) -> dict[str, Any]:
        return _read_yaml_mapping(self.path)

    def load(self) -> dict[str, str]:
        raw = self._load_raw()
        api_keys = raw.get("api_keys")
        if not isinstance(api_keys, dict):
            return {}

        parsed: dict[str, str] = {}
        for key, value in api_keys.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                parsed[key] = value
        return parsed

    def get_api_key(self, env_var: str) -> str | None:
        return self.load().get(env_var)

    def set_api_key(self, env_var: str, value: str) -> None:
        keys = self.load()
        keys[env_var] = value
        _write_yaml_mapping(self.path, {"api_keys": keys})


class LocalSettingsStore:
    def __init__(self, home_dir: Path | None = None):
        self.home_dir = home_dir or default_agent_recall_home()
        self.path = self.home_dir / "settings.yaml"

    def load(self) -> dict[str, Any]:
        return _read_yaml_mapping(self.path)

    def save(self, payload: dict[str, Any]) -> None:
        _write_yaml_mapping(self.path, payload)


def inject_stored_api_keys(secrets_store: LocalSecretsStore | None = None) -> int:
    store = secrets_store or LocalSecretsStore()
    injected = 0
    for env_var, value in store.load().items():
        if not os.environ.get(env_var):
            os.environ[env_var] = value
            injected += 1
    return injected


def _resolve_repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def _normalize_source_name(value: str) -> str:
    lowered = value.strip().lower().replace("_", "-")
    if lowered in {"claude", "claudecode"}:
        return "claude-code"
    return lowered


def _normalize_source_values(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = []

    normalized: list[str] = []
    for item in raw_values:
        if not isinstance(item, str):
            continue
        source = _normalize_source_name(item)
        if source in VALID_AGENT_SOURCES and source not in normalized:
            normalized.append(source)
    return normalized


def get_repo_preferred_sources(files: FileStorage) -> list[str] | None:
    config_dict = files.read_config()
    onboarding = config_dict.get("onboarding")
    if not isinstance(onboarding, dict):
        return None
    selected = onboarding.get("selected_agents")
    normalized = _normalize_source_values(selected)
    return normalized or None


def _repo_matches(config_repo_path: Any, current_repo_path: Path) -> bool:
    if not isinstance(config_repo_path, str) or not config_repo_path.strip():
        return False
    try:
        saved_path = Path(config_repo_path).expanduser().resolve()
    except OSError:
        return False
    return saved_path == current_repo_path.resolve()


def is_repo_onboarding_complete(files: FileStorage) -> bool:
    config_dict = files.read_config()
    onboarding = config_dict.get("onboarding")
    if not isinstance(onboarding, dict):
        return False

    completed_at = onboarding.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at.strip():
        return False

    selected_agents = _normalize_source_values(onboarding.get("selected_agents"))
    if not selected_agents:
        return False

    return _repo_matches(onboarding.get("repository_path"), _resolve_repository_root())


def _default_model_for_provider(provider: str) -> str:
    return DEFAULT_MODELS.get(provider, "gpt-4.1-mini")


def _default_base_url_for_provider(provider: str) -> str | None:
    return DEFAULT_BASE_URLS.get(provider)


def _safe_temperature(value: Any, fallback: float = 0.3) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if parsed < 0.0 or parsed > 2.0:
        return fallback
    return parsed


def _safe_max_tokens(value: Any, fallback: int = 4096) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    if parsed <= 0:
        return fallback
    return parsed


def _prompt_temperature(console: Console, default: float, provider: str) -> float:
    if provider in LOCAL_PROVIDERS:
        console.print(
            "[dim]Tip: local models often behave better with lower "
            "temperature (e.g. 0.1-0.4).[/dim]"
        )
    while True:
        value = typer.prompt(
            "Temperature (0.0-2.0)",
            default=round(default, 3),
            type=float,
        )
        if 0.0 <= value <= 2.0:
            return float(value)
        console.print("[warning]Temperature must be between 0.0 and 2.0.[/warning]")


def _prompt_max_tokens(console: Console, default: int) -> int:
    while True:
        value = typer.prompt("Max tokens (>0)", default=default, type=int)
        if value > 0:
            return int(value)
        console.print("[warning]Max tokens must be greater than 0.[/warning]")


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 8.0,
) -> Any:
    req = request.Request(url, headers=headers or {}, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            detail = ""
        message = f"HTTP {exc.code}"
        if detail:
            message = f"{message}: {detail[:180]}"
        raise RuntimeError(message) from exc
    except urlerror.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    except TimeoutError as exc:
        raise RuntimeError("request timed out") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid JSON response") from exc


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for model in models:
        cleaned = model.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _extract_openai_models(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str):
            models.append(model_id)
    return _dedupe_models(models)


def _extract_ollama_models(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("models")
    if not isinstance(data, list):
        return []

    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str):
            models.append(name)
    return _dedupe_models(models)


def _extract_anthropic_models(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str):
            models.append(model_id)
    return _dedupe_models(models)


def _extract_google_models(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("models")
    if not isinstance(data, list):
        return []

    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        methods = item.get("supportedGenerationMethods")
        if isinstance(methods, list) and "generateContent" not in methods:
            continue

        name = item.get("name")
        if not isinstance(name, str):
            continue
        models.append(name.replace("models/", "", 1))

    return _dedupe_models(models)


def _openai_like_models_url(base_url: str | None, provider: str) -> str:
    default_base = "https://api.openai.com/v1" if provider == "openai" else ""
    root = (base_url or default_base).rstrip("/")
    if root.endswith("/models"):
        return root
    return f"{root}/models"


def _ollama_tags_url(base_url: str | None) -> str:
    root = (base_url or _default_base_url_for_provider("ollama") or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return f"{root}/api/tags"


def discover_provider_models(
    provider: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: float = 8.0,
) -> tuple[list[str], str | None]:
    normalized = provider.lower()
    env_var = api_key_env or API_KEY_ENV_BY_PROVIDER.get(normalized)
    api_key = os.environ.get(env_var) if env_var else None

    try:
        if normalized in {"openai", "vllm", "lmstudio", "openai-compatible", "custom"}:
            headers = {"Accept": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = _http_get_json(
                _openai_like_models_url(base_url, "openai" if normalized == "openai" else "custom"),
                headers=headers,
                timeout_seconds=timeout_seconds,
            )
            models = _extract_openai_models(payload)
            if models:
                return models, None
            return [], "provider returned no models"

        if normalized == "ollama":
            openai_headers = {"Accept": "application/json"}
            try:
                payload = _http_get_json(
                    _openai_like_models_url(base_url, "custom"),
                    headers=openai_headers,
                    timeout_seconds=timeout_seconds,
                )
                models = _extract_openai_models(payload)
                if models:
                    return models, None
            except RuntimeError:
                pass

            payload = _http_get_json(
                _ollama_tags_url(base_url),
                headers={"Accept": "application/json"},
                timeout_seconds=timeout_seconds,
            )
            models = _extract_ollama_models(payload)
            if models:
                return models, None
            return [], "provider returned no models"

        if normalized == "anthropic":
            if not api_key:
                return [], "missing API key (ANTHROPIC_API_KEY)"
            payload = _http_get_json(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Accept": "application/json",
                },
                timeout_seconds=timeout_seconds,
            )
            models = _extract_anthropic_models(payload)
            if models:
                return models, None
            return [], "provider returned no models"

        if normalized == "google":
            if not api_key:
                return [], "missing API key (GOOGLE_API_KEY)"
            query = parse.urlencode({"key": api_key})
            payload = _http_get_json(
                f"https://generativelanguage.googleapis.com/v1beta/models?{query}",
                headers={"Accept": "application/json"},
                timeout_seconds=timeout_seconds,
            )
            models = _extract_google_models(payload)
            if models:
                return models, None
            return [], "provider returned no models"
    except RuntimeError as exc:
        message = str(exc)
        lowered = message.lower()
        if "certificate_verify_failed" in lowered or "certificate verify failed" in lowered:
            return (
                [],
                (
                    "TLS certificate verification failed. "
                    "Fix your local CA cert store "
                    "(macOS Python.org build: run 'Install Certificates.command')."
                ),
            )
        return [], message

    return [], f"model discovery not implemented for provider: {normalized}"


def _prompt_model_with_picker(
    console: Console,
    *,
    provider: str,
    default_model: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> str:
    models, error_message = discover_provider_models(
        provider,
        base_url=base_url,
        api_key_env=api_key_env,
    )

    if not models:
        if error_message:
            console.print(f"[warning]Live model discovery unavailable: {error_message}[/warning]")
        return typer.prompt("Model", default=default_model).strip()

    display_limit = 30
    displayed = models[:display_limit]
    table = Table(title=f"Available Models ({provider})")
    table.add_column("#", justify="right")
    table.add_column("Model")
    for index, model_name in enumerate(displayed, start=1):
        table.add_row(str(index), model_name)
    table.add_row("0", "Manual entry")
    console.print(table)

    default_choice = "0"
    if default_model in displayed:
        default_choice = str(displayed.index(default_model) + 1)

    while True:
        selection = typer.prompt("Model selection", default=default_choice).strip()
        if selection.isdigit():
            index = int(selection)
            if index == 0:
                break
            if 1 <= index <= len(displayed):
                return displayed[index - 1]
        if selection in models:
            return selection
        console.print(
            "[warning]Invalid model selection. Choose a number or model id from the list.[/warning]"
        )

    return typer.prompt("Model", default=default_model).strip()


def _prompt_agent_sources(console: Console, defaults: list[str]) -> list[str]:
    default_choice = "3"
    default_set = set(defaults)
    if default_set == {"cursor"}:
        default_choice = "1"
    elif default_set == {"claude-code"}:
        default_choice = "2"

    console.print("Choose which agent transcripts should be enabled for this repository:")
    console.print("  1) Cursor")
    console.print("  2) Claude Code")
    console.print("  3) Both")

    while True:
        selected = typer.prompt("Agent selection", default=default_choice).strip().lower()
        if selected in {"1", "cursor"}:
            return ["cursor"]
        if selected in {"2", "claude", "claude-code", "claudecode"}:
            return ["claude-code"]
        if selected in {"3", "both", "all", "cursor,claude-code", "claude-code,cursor"}:
            return ["cursor", "claude-code"]
        console.print("[warning]Invalid choice. Use 1, 2, or 3.[/warning]")


def _prompt_provider(console: Console, defaults: str) -> str:
    providers = get_available_providers()
    provider_map = {str(index): name for index, name in enumerate(providers, start=1)}
    default_index = next(
        (index for index, name in provider_map.items() if name == defaults),
        "1",
    )

    table = Table(title="LLM Providers")
    table.add_column("#", justify="right")
    table.add_column("Provider")
    table.add_column("Default Model")

    for index, provider in provider_map.items():
        table.add_row(index, provider, _default_model_for_provider(provider))
    console.print(table)

    while True:
        value = typer.prompt("Provider", default=default_index).strip().lower()
        if value in provider_map:
            return provider_map[value]
        if value in providers:
            return value
        console.print(
            "[warning]Unknown provider. Choose a listed number or provider name.[/warning]"
        )


def _maybe_capture_api_key(
    provider: str,
    secrets_store: LocalSecretsStore,
    interactive: bool,
    console: Console,
) -> None:
    env_var = API_KEY_ENV_BY_PROVIDER.get(provider)
    if not env_var:
        return

    existing_env_value = os.environ.get(env_var)
    if existing_env_value:
        console.print(f"[success]✓ Found {env_var} in environment[/success]")
        return

    stored_value = secrets_store.get_api_key(env_var)
    if stored_value:
        os.environ[env_var] = stored_value
        console.print(f"[success]✓ Loaded {env_var} from local secrets store[/success]")
        if not interactive:
            return
        if not typer.confirm("Update this API key now?", default=False):
            return

    if not interactive:
        console.print(
            f"[warning]No API key available for {provider}. "
            f"Set {env_var} or run interactive onboarding.[/warning]"
        )
        return

    entered_key = typer.prompt(
        f"API key for {provider} ({env_var}) - leave blank to skip",
        default="",
        hide_input=True,
        show_default=False,
    ).strip()
    if not entered_key:
        console.print(f"[warning]Skipped API key setup for {provider}.[/warning]")
        return

    secrets_store.set_api_key(env_var, entered_key)
    os.environ[env_var] = entered_key
    console.print(f"[success]✓ Stored {env_var} in {secrets_store.path}[/success]")


def _discover_source_counts(selected_agents: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ingester in get_default_ingesters():
        if ingester.source_name not in selected_agents:
            continue
        try:
            counts[ingester.source_name] = len(ingester.discover_sessions())
        except Exception:  # noqa: BLE001
            counts[ingester.source_name] = -1
    return counts


def _normalize_defaults(defaults: Any) -> dict[str, Any]:
    return defaults if isinstance(defaults, dict) else {}


def get_onboarding_defaults(files: FileStorage) -> dict[str, Any]:
    config_dict = files.read_config()
    llm_config = dict(config_dict.get("llm", {}))
    settings = LocalSettingsStore().load()
    defaults = _normalize_defaults(settings.get("defaults"))

    provider = str(llm_config.get("provider") or defaults.get("provider") or "anthropic").lower()
    available_providers = get_available_providers()
    if provider not in available_providers:
        provider = "anthropic" if "anthropic" in available_providers else available_providers[0]

    selected_agents = _normalize_source_values(
        config_dict.get("onboarding", {}).get("selected_agents")
        if isinstance(config_dict.get("onboarding"), dict)
        else None
    )
    if not selected_agents:
        selected_agents = _normalize_source_values(defaults.get("selected_agents"))
    if not selected_agents:
        selected_agents = ["cursor", "claude-code"]

    base_url = str(
        llm_config.get("base_url")
        or defaults.get("base_url")
        or _default_base_url_for_provider(provider)
        or ""
    ).strip()

    return {
        "repository_path": str(_resolve_repository_root()),
        "repository_verified": True,
        "force": False,
        "provider": provider,
        "model": str(
            llm_config.get("model")
            or defaults.get("model")
            or _default_model_for_provider(provider)
        ),
        "base_url": base_url,
        "temperature": _safe_temperature(
            llm_config.get("temperature", defaults.get("temperature", 0.3)),
            fallback=0.3,
        ),
        "max_tokens": _safe_max_tokens(
            llm_config.get("max_tokens", defaults.get("max_tokens", 4096)),
            fallback=4096,
        ),
        "selected_agents": selected_agents,
        "validate": False,
        "api_key_env": API_KEY_ENV_BY_PROVIDER.get(provider),
    }


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def apply_repo_setup(
    files: FileStorage,
    console: Console,
    *,
    force: bool,
    repository_verified: bool,
    selected_agents: list[str],
    provider: str,
    model: str,
    base_url: str | None,
    temperature: float,
    max_tokens: int,
    validate: bool = False,
    api_key: str | None = None,
) -> bool:
    repo_path = _resolve_repository_root()
    if not repository_verified:
        raise ValueError("Repository verification is required before setup.")

    normalized_provider = provider.strip().lower()
    available_providers = get_available_providers()
    if normalized_provider not in available_providers:
        raise ValueError(
            f"Unknown provider '{normalized_provider}'. "
            f"Expected one of: {', '.join(available_providers)}"
        )

    normalized_agents = _normalize_source_values(selected_agents)
    if not normalized_agents:
        raise ValueError("At least one agent source must be selected.")

    cleaned_model = model.strip()
    if not cleaned_model:
        raise ValueError("Model is required.")

    cleaned_base_url = (base_url or "").strip() or None
    if normalized_provider == "openai-compatible" and not cleaned_base_url:
        raise ValueError("Base URL is required for openai-compatible provider.")

    if temperature < 0.0 or temperature > 2.0:
        raise ValueError("Temperature must be between 0.0 and 2.0.")
    if max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0.")

    secrets_store = LocalSecretsStore()
    inject_stored_api_keys(secrets_store)
    if not force and is_repo_onboarding_complete(files):
        return False

    cloud_key_env = API_KEY_ENV_BY_PROVIDER.get(normalized_provider)
    if cloud_key_env:
        maybe_key = (api_key or "").strip()
        if maybe_key:
            secrets_store.set_api_key(cloud_key_env, maybe_key)
            os.environ[cloud_key_env] = maybe_key
        else:
            stored_value = secrets_store.get_api_key(cloud_key_env)
            if stored_value and not os.environ.get(cloud_key_env):
                os.environ[cloud_key_env] = stored_value

    config_dict = files.read_config()
    llm_payload = dict(config_dict.get("llm", {}))
    llm_payload["provider"] = normalized_provider
    llm_payload["model"] = cleaned_model
    llm_payload["temperature"] = float(temperature)
    llm_payload["max_tokens"] = int(max_tokens)
    if cleaned_base_url:
        llm_payload["base_url"] = cleaned_base_url
    else:
        llm_payload.pop("base_url", None)

    if cloud_key_env:
        llm_payload["api_key_env"] = cloud_key_env
    elif normalized_provider != "openai-compatible":
        llm_payload.pop("api_key_env", None)

    try:
        parsed = LLMConfig(**llm_payload)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid LLM config: {exc}") from None

    dependency_ok, dependency_message = ensure_provider_dependency(
        normalized_provider,
        auto_install=True,
    )
    if not dependency_ok:
        raise ValueError(dependency_message or "Failed installing provider dependency.")
    if dependency_message:
        console.print(f"[dim]{dependency_message}[/dim]")

    if validate:
        try:
            provider_client = create_llm_provider(parsed)
            ok, message = provider_client.validate()
            style = "success" if ok else "warning"
            symbol = "✓" if ok else "!"
            console.print(f"[{style}]{symbol} {message}[/{style}]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[warning]Validation skipped: {exc}[/warning]")

    discovery = _discover_source_counts(normalized_agents)

    now_iso = _now_iso()
    config_dict["llm"] = llm_payload
    config_dict["onboarding"] = {
        "version": 1,
        "completed_at": now_iso,
        "repository_path": str(repo_path),
        "repository_verified": True,
        "selected_agents": normalized_agents,
        "source_discovery": discovery,
    }
    files.write_config(config_dict)

    settings_store = LocalSettingsStore()
    settings = settings_store.load()
    defaults = _normalize_defaults(settings.get("defaults"))
    defaults["provider"] = normalized_provider
    defaults["model"] = cleaned_model
    defaults["temperature"] = float(temperature)
    defaults["max_tokens"] = int(max_tokens)
    defaults["selected_agents"] = normalized_agents
    if cleaned_base_url:
        defaults["base_url"] = cleaned_base_url
    else:
        defaults.pop("base_url", None)

    user_data = settings.get("user")
    if not isinstance(user_data, dict):
        user_data = {}
    if "onboarded_at" not in user_data:
        user_data["onboarded_at"] = now_iso
    user_data["last_onboarded_at"] = now_iso

    repositories = settings.get("repositories")
    if not isinstance(repositories, dict):
        repositories = {}
    repositories[str(repo_path)] = {
        "last_onboarded_at": now_iso,
        "provider": normalized_provider,
        "model": cleaned_model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "selected_agents": normalized_agents,
    }

    settings["defaults"] = defaults
    settings["user"] = user_data
    settings["repositories"] = repositories
    settings_store.save(settings)

    discovered_rows: list[str] = []
    for source_name in normalized_agents:
        count = discovery.get(source_name, -1)
        if count >= 0:
            discovered_rows.append(f"  {source_name}: {count} session(s) discovered")
        else:
            discovered_rows.append(f"  {source_name}: discovery failed")

    panel_body = (
        "[success]✓ Onboarding complete[/success]\n\n"
        f"Repository: {repo_path}\n"
        f"Agents: {', '.join(normalized_agents)}\n"
        f"Provider: {normalized_provider}\n"
        f"Model: {cleaned_model}\n"
        f"Temperature: {temperature}\n"
        f"Max tokens: {max_tokens}\n"
        f"Secrets path: {secrets_store.path}\n\n"
        "Source discovery:\n" + "\n".join(discovered_rows)
    )
    console.print(Panel.fit(panel_body, title="Onboarding Summary"))

    return True


def ensure_repo_onboarding(
    files: FileStorage,
    console: Console,
    *,
    force: bool = False,
    interactive: bool = True,
) -> bool:
    repo_path = _resolve_repository_root()
    config_dict = files.read_config()
    llm_config = dict(config_dict.get("llm", {}))

    secrets_store = LocalSecretsStore()
    settings = LocalSettingsStore().load()
    defaults = _normalize_defaults(settings.get("defaults"))

    if not force and is_repo_onboarding_complete(files):
        inject_stored_api_keys(secrets_store)
        return False

    onboarding_required = "first-time setup" if not settings.get("user") else "new repository setup"

    if interactive:
        console.print(
            Panel.fit(
                f"Running {onboarding_required} for:\n{repo_path}",
                title="Onboarding",
            )
        )
        if not typer.confirm("Use this repository for agent-recall?", default=True):
            console.print("[error]Onboarding cancelled.[/error]")
            raise typer.Exit(1)

    default_provider = str(
        llm_config.get("provider") or defaults.get("provider") or "anthropic"
    ).lower()
    available_providers = get_available_providers()
    if default_provider not in available_providers:
        default_provider = (
            "anthropic" if "anthropic" in available_providers else available_providers[0]
        )

    selected_agents = _normalize_source_values(
        config_dict.get("onboarding", {}).get("selected_agents")
        if isinstance(config_dict.get("onboarding"), dict)
        else None
    )
    if not selected_agents:
        selected_agents = _normalize_source_values(defaults.get("selected_agents"))
    if not selected_agents:
        selected_agents = ["cursor", "claude-code"]

    provider = _prompt_provider(console, default_provider) if interactive else default_provider

    base_url_default = str(
        llm_config.get("base_url")
        or defaults.get("base_url")
        or _default_base_url_for_provider(provider)
        or ""
    ).strip()

    base_url: str | None
    if provider in {"openai-compatible", "ollama", "vllm", "lmstudio"}:
        if interactive:
            prompt = (
                "Base URL (required for openai-compatible)"
                if provider == "openai-compatible"
                else "Base URL"
            )
            entered_base_url = typer.prompt(
                prompt,
                default=base_url_default,
                show_default=bool(base_url_default),
            ).strip()
            base_url = entered_base_url or None
        else:
            base_url = base_url_default or _default_base_url_for_provider(provider)
    else:
        base_url = None

    if interactive:
        selected_agents = _prompt_agent_sources(console, selected_agents)

    _maybe_capture_api_key(provider, secrets_store, interactive, console)

    model_default = str(
        llm_config.get("model") or defaults.get("model") or _default_model_for_provider(provider)
    )
    if interactive:
        configured_key_env = llm_config.get("api_key_env")
        api_key_env = configured_key_env if isinstance(configured_key_env, str) else None
        model = _prompt_model_with_picker(
            console,
            provider=provider,
            default_model=model_default,
            base_url=base_url,
            api_key_env=api_key_env,
        )
    else:
        model = model_default

    temperature_default = _safe_temperature(
        llm_config.get("temperature", defaults.get("temperature", 0.3)),
        fallback=0.3,
    )
    max_tokens_default = _safe_max_tokens(
        llm_config.get("max_tokens", defaults.get("max_tokens", 4096)),
        fallback=4096,
    )

    if interactive:
        temperature = _prompt_temperature(
            console,
            default=temperature_default,
            provider=provider,
        )
        max_tokens = _prompt_max_tokens(console, default=max_tokens_default)
    else:
        temperature = temperature_default
        max_tokens = max_tokens_default

    should_validate = interactive and typer.confirm(
        "Validate provider connection now?", default=False
    )
    try:
        return apply_repo_setup(
            files,
            console,
            force=force,
            repository_verified=True,
            selected_agents=selected_agents,
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            validate=should_validate,
        )
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None
