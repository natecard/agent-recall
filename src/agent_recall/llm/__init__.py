from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from agent_recall.llm.base import (
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    Message,
)

if TYPE_CHECKING:
    from agent_recall.storage.models import LLMConfig

__all__ = [
    "LLMProvider",
    "Message",
    "LLMResponse",
    "LLMError",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "create_llm_provider",
    "get_available_providers",
    "ensure_provider_dependency",
    "validate_provider_config",
]

_PROVIDER_DEPENDENCIES: dict[str, tuple[str, str]] = {
    "anthropic": ("anthropic", "anthropic"),
    "google": ("google.genai", "google-genai"),
    "openai": ("openai", "openai"),
    "ollama": ("openai", "openai"),
    "vllm": ("openai", "openai"),
    "lmstudio": ("openai", "openai"),
    "openai-compatible": ("openai", "openai"),
    "custom": ("openai", "openai"),
}


def _normalize_provider_name(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in {"openai_compatible"}:
        return "openai-compatible"
    return normalized


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def ensure_provider_dependency(
    provider: str,
    *,
    auto_install: bool = True,
) -> tuple[bool, str | None]:
    """Ensure the selected provider dependency is importable."""
    normalized = _normalize_provider_name(provider)
    dependency = _PROVIDER_DEPENDENCIES.get(normalized)
    if dependency is None:
        return True, None

    module_name, package_name = dependency
    if _module_available(module_name):
        return True, None

    if not auto_install:
        return (
            False,
            f"Missing dependency '{package_name}'. Install with: pip install {package_name}",
        )

    completed = _install_dependency(package_name)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        tail = stderr or stdout or "unknown pip error"
        return (
            False,
            f"Failed installing '{package_name}' for provider '{normalized}': {tail}",
        )

    if not _module_available(module_name):
        return (
            False,
            f"Installed '{package_name}' but module '{module_name}' is still unavailable.",
        )

    return True, f"Installed provider dependency: {package_name}"


def _install_dependency(package_name: str) -> subprocess.CompletedProcess[str]:
    pip_install = [sys.executable, "-m", "pip", "install", package_name]
    completed = subprocess.run(  # noqa: S603
        pip_install,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return completed

    lowered = f"{completed.stderr}\n{completed.stdout}".lower()
    missing_pip = "no module named pip" in lowered

    if missing_pip:
        ensurepip = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            capture_output=True,
            text=True,
            check=False,
        )
        if ensurepip.returncode == 0:
            retry = subprocess.run(  # noqa: S603
                pip_install,
                capture_output=True,
                text=True,
                check=False,
            )
            if retry.returncode == 0:
                return retry
            completed = retry

    if shutil.which("uv"):
        uv_completed = subprocess.run(  # noqa: S603
            ["uv", "pip", "install", "--python", sys.executable, package_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if uv_completed.returncode == 0:
            return uv_completed
        combined_stderr = "\n".join(
            part
            for part in [completed.stderr.strip(), uv_completed.stderr.strip()]
            if part
        )
        combined_stdout = "\n".join(
            part
            for part in [completed.stdout.strip(), uv_completed.stdout.strip()]
            if part
        )
        return subprocess.CompletedProcess(
            args=uv_completed.args,
            returncode=uv_completed.returncode,
            stdout=combined_stdout,
            stderr=combined_stderr,
        )

    return completed


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider from config."""
    provider = _normalize_provider_name(config.provider)
    ok, message = ensure_provider_dependency(provider, auto_install=True)
    if not ok:
        raise LLMConfigError(message or f"Provider dependency check failed for '{provider}'.")

    if provider == "anthropic":
        from agent_recall.llm.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=config.model,
            api_key_env=config.api_key_env or "ANTHROPIC_API_KEY",
        )

    if provider == "google":
        from agent_recall.llm.google import GoogleProvider

        return GoogleProvider(
            model=config.model,
            api_key_env=config.api_key_env or "GOOGLE_API_KEY",
        )

    if provider == "openai":
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env or "OPENAI_API_KEY",
            provider_type="openai",
            timeout=config.timeout,
        )

    if provider == "ollama":
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url or "http://localhost:11434/v1",
            provider_type="ollama",
            timeout=config.timeout,
        )

    if provider == "vllm":
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url or "http://localhost:8000/v1",
            api_key_env=config.api_key_env,
            provider_type="vllm",
            timeout=config.timeout,
        )

    if provider == "lmstudio":
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url or "http://localhost:1234/v1",
            provider_type="lmstudio",
            timeout=config.timeout,
        )

    if provider in {"openai-compatible", "openai_compatible", "custom"}:
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        if not config.base_url:
            raise LLMConfigError(
                "base_url is required for openai-compatible provider. "
                "Specify the URL of your OpenAI-compatible server."
            )
        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            provider_type="custom",
            timeout=config.timeout,
        )

    available = ", ".join(get_available_providers())
    raise LLMConfigError(
        f"Unknown LLM provider: '{provider}'. Available providers: {available}"
    )


def get_available_providers() -> list[str]:
    """Return available provider names."""
    return [
        "anthropic",
        "openai",
        "google",
        "ollama",
        "vllm",
        "lmstudio",
        "openai-compatible",
    ]


def validate_provider_config(config: LLMConfig) -> tuple[bool, str]:
    """Validate LLM config structure without calling provider APIs."""
    provider = config.provider.lower()

    if provider not in get_available_providers():
        return False, f"Unknown provider: {provider}"

    cloud_providers = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    if provider in cloud_providers:
        env_var = config.api_key_env or cloud_providers[provider]
        if not os.environ.get(env_var):
            return False, f"API key not set. Set {env_var} environment variable."

    if provider in {"openai-compatible", "custom"} and not config.base_url:
        return False, "base_url is required for openai-compatible provider."

    return True, f"Configuration valid for {provider}"
