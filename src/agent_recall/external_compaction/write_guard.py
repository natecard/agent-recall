from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from agent_recall.storage.files import KnowledgeTier

WriteTarget = Literal["runtime", "templates"]

_DEFAULT_TEMPLATE_ALLOWLIST: dict[KnowledgeTier, Path] = {
    KnowledgeTier.GUARDRAILS: Path("src/agent_recall/templates/GUARDRAILS.md"),
    KnowledgeTier.STYLE: Path("src/agent_recall/templates/STYLE.md"),
    KnowledgeTier.RECENT: Path("src/agent_recall/templates/RECENT.md"),
}


class ExternalWritePolicyError(ValueError):
    """Raised when an external compaction write target/path violates policy."""


class ExternalWriteScopeGuard:
    """Centralized write-scope policy checks for external compaction operations."""

    def __init__(
        self,
        *,
        repo_root: Path,
        default_target: WriteTarget = "runtime",
        allow_template_writes: bool = False,
        template_allowlist: dict[KnowledgeTier, Path] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.default_target = default_target
        self.allow_template_writes = allow_template_writes
        raw_allowlist = template_allowlist or _DEFAULT_TEMPLATE_ALLOWLIST
        self._template_allowlist = {
            tier: self._normalize_relative(path) for tier, path in raw_allowlist.items()
        }

    @classmethod
    def from_config(
        cls,
        *,
        repo_root: Path,
        config: dict[str, Any] | None,
    ) -> ExternalWriteScopeGuard:
        compaction_cfg = config.get("compaction") if isinstance(config, dict) else None
        external_cfg = compaction_cfg.get("external") if isinstance(compaction_cfg, dict) else None

        target = "runtime"
        allow_template_writes = False
        if isinstance(external_cfg, dict):
            raw_target = str(external_cfg.get("write_target", "runtime")).strip().lower()
            target = "templates" if raw_target == "templates" else "runtime"
            allow_template_writes = bool(external_cfg.get("allow_template_writes", False))

        return cls(
            repo_root=repo_root,
            default_target=target,
            allow_template_writes=allow_template_writes,
        )

    def resolve_target(self, override: str | None = None) -> WriteTarget:
        if override is None:
            target = self.default_target
        else:
            target = self._normalize_target(override)

        if target == "templates" and not self.allow_template_writes:
            raise ExternalWritePolicyError(
                "Template writes are disabled. Set compaction.external.allow_template_writes=true "
                "or use --write-target runtime."
            )
        return target

    def template_path_for_tier(self, tier: KnowledgeTier) -> Path:
        relative = self._template_allowlist[tier]
        return self.resolve_template_relative_path(relative)

    def resolve_template_relative_path(self, relative_path: str | Path) -> Path:
        normalized = self._normalize_relative(relative_path)
        if normalized not in self._template_allowlist.values():
            raise ExternalWritePolicyError(
                f"Template path is not in writable allowlist: {normalized.as_posix()}"
            )

        candidate = self.repo_root / normalized
        self._ensure_no_symlink_components(candidate)

        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise ExternalWritePolicyError(f"Template path escapes repo root: {resolved}") from exc
        return candidate

    @staticmethod
    def _normalize_target(value: str) -> WriteTarget:
        normalized = str(value).strip().lower()
        if normalized not in {"runtime", "templates"}:
            raise ExternalWritePolicyError("write-target must be 'runtime' or 'templates'")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _normalize_relative(value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ExternalWritePolicyError("Template path must be relative to repo root.")
        if any(part == ".." for part in path.parts):
            raise ExternalWritePolicyError(
                "Template path must not contain parent traversal ('..')."
            )
        cleaned_parts = [part for part in path.parts if part not in {"", "."}]
        if not cleaned_parts:
            raise ExternalWritePolicyError("Template path cannot be empty.")
        return Path(*cleaned_parts)

    def _ensure_no_symlink_components(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.repo_root)
        except ValueError as exc:
            raise ExternalWritePolicyError(f"Template path escapes repo root: {path}") from exc

        current = self.repo_root
        for part in relative.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise ExternalWritePolicyError(f"Template path cannot traverse symlinks: {current}")
