from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console

import agent_recall.core.onboarding as onboarding
from agent_recall.core.onboarding import (
    LocalSecretsStore,
    LocalSettingsStore,
    apply_repo_setup,
    discover_provider_models,
    get_onboarding_defaults,
    get_repo_preferred_sources,
    inject_stored_api_keys,
    is_repo_onboarding_complete,
)
from agent_recall.storage.files import FileStorage


def _make_agent_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("llm:\n  provider: anthropic\n")
    return path


def test_secrets_store_injects_missing_api_keys(monkeypatch, tmp_path: Path) -> None:
    store = LocalSecretsStore(home_dir=tmp_path / "app-home")
    store.set_api_key("OPENAI_API_KEY", "test-key")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    injected = inject_stored_api_keys(store)

    assert injected == 1
    assert os.environ["OPENAI_API_KEY"] == "test-key"
    assert store.path.exists()
    assert store.path.parent.exists()


def test_repo_preferred_sources_from_onboarding(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path / ".agent")
    files = FileStorage(agent_dir)
    files.write_config(
        {
            "llm": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            "onboarding": {"selected_agents": ["cursor", "claude-code"]},
        }
    )

    assert get_repo_preferred_sources(files) == ["cursor", "claude-code"]


def test_repo_onboarding_complete_requires_matching_repo(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    agent_dir = _make_agent_dir(repo_path / ".agent")
    files = FileStorage(agent_dir)

    files.write_config(
        {
            "llm": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            "onboarding": {
                "completed_at": "2026-02-01T00:00:00+00:00",
                "repository_path": str(repo_path.resolve()),
                "selected_agents": ["cursor"],
            },
        }
    )

    monkeypatch.chdir(repo_path)
    assert is_repo_onboarding_complete(files) is True

    files.write_config(
        {
            "llm": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            "onboarding": {
                "completed_at": "2026-02-01T00:00:00+00:00",
                "repository_path": str((tmp_path / "other-repo").resolve()),
                "selected_agents": ["cursor"],
            },
        }
    )
    assert is_repo_onboarding_complete(files) is False


def test_discover_provider_models_openai(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_http_get_json(url: str, *, headers=None, timeout_seconds=8.0):
        _ = timeout_seconds
        captured["url"] = url
        captured["headers"] = headers or {}
        return {"data": [{"id": "gpt-4.1"}, {"id": "gpt-4.1-mini"}]}

    monkeypatch.setattr(onboarding, "_http_get_json", fake_http_get_json)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    models, error = discover_provider_models("openai")

    assert error is None
    assert models == ["gpt-4.1", "gpt-4.1-mini"]
    assert str(captured["url"]).endswith("/v1/models")
    assert captured["headers"] == {
        "Accept": "application/json",
        "Authorization": "Bearer openai-test-key",
    }


def test_discover_provider_models_google_filters_generation(monkeypatch) -> None:
    def fake_http_get_json(url: str, *, headers=None, timeout_seconds=8.0):
        _ = (url, headers, timeout_seconds)
        return {
            "models": [
                {
                    "name": "models/gemini-2.5-flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/embedding-001",
                    "supportedGenerationMethods": ["embedContent"],
                },
            ]
        }

    monkeypatch.setattr(onboarding, "_http_get_json", fake_http_get_json)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")

    models, error = discover_provider_models("google")

    assert error is None
    assert models == ["gemini-2.5-flash"]


def test_discover_provider_models_ollama_falls_back_to_tags(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_http_get_json(url: str, *, headers=None, timeout_seconds=8.0):
        _ = (headers, timeout_seconds)
        calls["count"] += 1
        if calls["count"] == 1:
            assert url.endswith("/v1/models")
            raise RuntimeError("not supported")
        assert url.endswith("/api/tags")
        return {"models": [{"name": "llama3.2"}, {"name": "qwen2.5-coder"}]}

    monkeypatch.setattr(onboarding, "_http_get_json", fake_http_get_json)

    models, error = discover_provider_models(
        "ollama",
        base_url="http://localhost:11434/v1",
    )

    assert error is None
    assert models == ["llama3.2", "qwen2.5-coder"]


def test_discover_provider_models_reports_tls_certificate_hint(monkeypatch) -> None:
    def fake_http_get_json(url: str, *, headers=None, timeout_seconds=8.0):
        _ = (url, headers, timeout_seconds)
        raise RuntimeError("SSL: CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(onboarding, "_http_get_json", fake_http_get_json)

    models, error = discover_provider_models("openai")

    assert models == []
    assert error is not None
    assert "certificate" in error.lower()
    assert "Install Certificates.command" in error


def test_apply_repo_setup_persists_modal_values(monkeypatch, tmp_path: Path) -> None:
    class FakeIngester:
        def __init__(self, source_name: str, count: int):
            self.source_name = source_name
            self.count = count

        def discover_sessions(self):
            return [Path(f"{self.source_name}-{index}") for index in range(self.count)]

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()
    agent_dir = _make_agent_dir(repo_path / ".agent")
    files = FileStorage(agent_dir)

    monkeypatch.chdir(repo_path)
    monkeypatch.setattr(
        onboarding,
        "get_default_ingesters",
        lambda **_kwargs: [FakeIngester("cursor", 2), FakeIngester("claude-code", 1)],
    )

    changed = apply_repo_setup(
        files,
        Console(record=True),
        force=True,
        repository_verified=True,
        selected_agents=["cursor"],
        provider="ollama",
        model="qwen3:4b",
        base_url="http://localhost:11434/v1",
        temperature=0.4,
        max_tokens=8192,
        validate=False,
    )

    assert changed is True

    config = files.read_config()
    assert config["llm"]["provider"] == "ollama"
    assert config["llm"]["model"] == "qwen3:4b"
    assert config["llm"]["temperature"] == 0.4
    assert config["llm"]["max_tokens"] == 8192
    assert config["llm"]["base_url"] == "http://localhost:11434/v1"
    assert config["onboarding"]["repository_verified"] is True
    assert config["onboarding"]["selected_agents"] == ["cursor"]
    assert config["onboarding"]["source_discovery"]["cursor"] == 2

    settings = LocalSettingsStore().load()
    assert settings["defaults"]["provider"] == "ollama"
    assert settings["defaults"]["model"] == "qwen3:4b"
    assert settings["defaults"]["selected_agents"] == ["cursor"]


def test_apply_repo_setup_requires_repository_verification(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()
    files = FileStorage(_make_agent_dir(repo_path / ".agent"))

    try:
        apply_repo_setup(
            files,
            Console(record=True),
            force=True,
            repository_verified=False,
            selected_agents=["cursor"],
            provider="ollama",
            model="qwen3:4b",
            base_url="http://localhost:11434/v1",
            temperature=0.2,
            max_tokens=2048,
            validate=False,
        )
    except ValueError as exc:
        assert "Repository verification is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError when repository_verified is False")


def test_get_onboarding_defaults_uses_settings_fallback(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()
    files = FileStorage(_make_agent_dir(repo_path / ".agent"))

    settings_store = LocalSettingsStore()
    settings_store.save(
        {
            "defaults": {
                "provider": "ollama",
                "model": "llama3.1",
                "temperature": 0.25,
                "max_tokens": 5000,
                "selected_agents": ["cursor"],
                "base_url": "http://localhost:11434/v1",
            }
        }
    )

    defaults = get_onboarding_defaults(files)
    assert defaults["provider"] == "anthropic"
    assert defaults["model"] == "llama3.1"
    assert defaults["temperature"] == 0.25
    assert defaults["max_tokens"] == 5000
    assert defaults["selected_agents"] == ["cursor"]
    assert defaults["base_url"] == "http://localhost:11434/v1"
