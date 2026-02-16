from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_recall.core.tier_compaction import compact_if_over_tokens
from agent_recall.core.tier_format import (
    EntryFormat,
    ParsedEntry,
    merge_tier_content,
    parse_bullet_entry,
    parse_tier_content,
)
from agent_recall.core.tier_writer import TIER_HEADERS
from agent_recall.llm import LLMProvider, Message
from agent_recall.ralph.iteration_store import IterationReport, IterationReportStore
from agent_recall.storage.files import FileStorage, KnowledgeTier

GUARDRAILS_SYNTHESIS_PROMPT = (
    "You are distilling Ralph iteration learnings into guardrails. "
    "Given CURRENT guardrails and candidate notes, output only NEW concise bullet rules "
    "(1 line each), max {max_entries} bullets. Do not repeat or paraphrase existing "
    "rules. Keep them actionable.\n\n"
    "Current guardrails:\n{current_content}\n\n"
    "Candidates:\n{candidates}"
)

STYLE_SYNTHESIS_PROMPT = (
    "You are distilling Ralph iteration learnings into a coding style guide. "
    "Given CURRENT style guide and candidate notes, output only NEW concise bullet "
    "patterns (1 line each), max {max_entries} bullets. Do not repeat or paraphrase "
    "existing patterns. Keep them actionable.\n\n"
    "Current style guide:\n{current_content}\n\n"
    "Candidates:\n{candidates}"
)


@dataclass
class SynthesisConfig:
    max_guardrails: int = 30
    max_style: int = 30
    auto_after_loop: bool = True


class ClimateSynthesizer:
    def __init__(
        self,
        ralph_dir: Path,
        files: FileStorage,
        llm: LLMProvider | None,
        config: SynthesisConfig | None = None,
    ) -> None:
        self.ralph_dir = ralph_dir
        self.files = files
        self.llm = llm
        self.config = config or SynthesisConfig()
        self.state_path = self.ralph_dir / "synthesis_state.json"

    def _extract_guardrail_candidates(self, reports: list[IterationReport]) -> list[str]:
        candidates: list[str] = []
        for report in reports:
            if report.failure_reason:
                candidates.append(report.failure_reason)
            if report.gotcha_discovered:
                candidates.append(report.gotcha_discovered)
        return candidates

    def _extract_style_candidates(self, reports: list[IterationReport]) -> list[str]:
        candidates: list[str] = []
        for report in reports:
            if report.pattern_that_worked:
                candidates.append(report.pattern_that_worked)
        return candidates

    def _deduplicate_candidates(self, candidates: list[str]) -> list[tuple[str, int]]:
        normalized: dict[str, tuple[str, int]] = {}
        for entry in candidates:
            cleaned = entry.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in normalized:
                original, count = normalized[key]
                normalized[key] = (original, count + 1)
            else:
                normalized[key] = (cleaned, 1)
        ranked = sorted(normalized.values(), key=lambda pair: (-pair[1], pair[0]))
        return ranked

    def _normalize_entry(self, text: str) -> str:
        return " ".join(text.split()).strip().lower()

    def _extract_existing_bullet_texts(self, content: str) -> set[str]:
        parsed = parse_tier_content(content)
        existing: set[str] = set()
        for entry in parsed.bullet_entries:
            if entry.text:
                existing.add(self._normalize_entry(entry.text))
        # Back-compat for legacy plain markdown bullets (e.g. "- item"),
        # but only from non-Ralph sections to avoid false dedup against
        # iteration details inside Ralph blocks.
        for line in [*parsed.preamble, *parsed.unknown_lines]:
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            bullet = parse_bullet_entry(stripped)
            if bullet is not None:
                _kind, text = bullet
                existing.add(self._normalize_entry(text))
                continue
            plain_text = stripped[2:].strip()
            if plain_text:
                existing.add(self._normalize_entry(plain_text))
        return existing

    def _extract_synthesized_lines(
        self,
        content: str,
        *,
        kind: str,
        existing: set[str],
    ) -> list[str]:
        extracted: list[str] = []
        seen = set(existing)
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            bullet = parse_bullet_entry(line)
            if bullet is not None:
                _existing_kind, text = bullet
            else:
                text = line[2:].strip()
            normalized = self._normalize_entry(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            extracted.append(f"- [{kind}] {text}")
        return extracted

    def _merge_bullets(self, current_content: str, new_bullets: list[str]) -> str:
        if not new_bullets:
            return current_content
        parsed = parse_tier_content(current_content)
        for line in new_bullets:
            bullet = parse_bullet_entry(line)
            if bullet is None:
                continue
            kind, text = bullet
            parsed.bullet_entries.append(
                ParsedEntry(
                    format=EntryFormat.BULLET,
                    raw_content=line,
                    kind=kind,
                    text=text,
                )
            )
        return merge_tier_content(parsed)

    def _fallback_bullets_from_candidates(
        self,
        candidates: list[str],
        *,
        kind: str,
        existing: set[str],
        max_entries: int,
    ) -> list[str]:
        fallback: list[str] = []
        seen = set(existing)
        for candidate in candidates:
            normalized = self._normalize_entry(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            fallback.append(f"- [{kind}] {candidate.strip()}")
            if len(fallback) >= max_entries:
                break
        return fallback

    async def synthesize_guardrails(self, candidates: list[str]) -> str:
        current_content = self.files.read_tier(KnowledgeTier.GUARDRAILS)
        had_existing_content = bool(current_content.strip())
        if not had_existing_content:
            current_content = "# Guardrails\n"
        if not candidates:
            if had_existing_content:
                return current_content
            return "# Guardrails\n\n- No guardrails synthesized yet."
        existing = self._extract_existing_bullet_texts(current_content)
        novel_candidates = [
            item for item in candidates if self._normalize_entry(item) not in existing
        ]
        if not novel_candidates:
            if had_existing_content:
                return current_content
            return "# Guardrails\n\n- No guardrails synthesized yet."
        fallback = self._fallback_bullets_from_candidates(
            novel_candidates,
            kind="GOTCHA",
            existing=existing,
            max_entries=self.config.max_guardrails,
        )
        synthesized: list[str] = []
        if self.llm is not None:
            prompt = GUARDRAILS_SYNTHESIS_PROMPT.format(
                current_content=current_content.strip() or "(empty)",
                candidates="\n".join(f"- {item}" for item in novel_candidates),
                max_entries=self.config.max_guardrails,
            )
            try:
                response = await self.llm.generate(
                    [Message(role="user", content=prompt)],
                    temperature=0.2,
                    max_tokens=800,
                )
                synthesized = self._extract_synthesized_lines(
                    response.content,
                    kind="GOTCHA",
                    existing=existing,
                )
            except Exception:  # noqa: BLE001
                synthesized = []
        if not synthesized:
            synthesized = fallback
        merged = self._merge_bullets(current_content, synthesized)
        if merged.strip():
            return merged
        if synthesized:
            return "# Guardrails\n\n" + "\n".join(synthesized) + "\n"
        return "# Guardrails\n\n- No guardrails synthesized yet."

    async def synthesize_style(self, candidates: list[str]) -> str:
        current_content = self.files.read_tier(KnowledgeTier.STYLE)
        had_existing_content = bool(current_content.strip())
        if not had_existing_content:
            current_content = "# Style Guide\n"
        if not candidates:
            if had_existing_content:
                return current_content
            return "# Style Guide\n\n- No style patterns synthesized yet."
        existing = self._extract_existing_bullet_texts(current_content)
        novel_candidates = [
            item for item in candidates if self._normalize_entry(item) not in existing
        ]
        if not novel_candidates:
            if had_existing_content:
                return current_content
            return "# Style Guide\n\n- No style patterns synthesized yet."
        fallback = self._fallback_bullets_from_candidates(
            novel_candidates,
            kind="PATTERN",
            existing=existing,
            max_entries=self.config.max_style,
        )
        synthesized: list[str] = []
        if self.llm is not None:
            prompt = STYLE_SYNTHESIS_PROMPT.format(
                current_content=current_content.strip() or "(empty)",
                candidates="\n".join(f"- {item}" for item in novel_candidates),
                max_entries=self.config.max_style,
            )
            try:
                response = await self.llm.generate(
                    [Message(role="user", content=prompt)],
                    temperature=0.2,
                    max_tokens=800,
                )
                synthesized = self._extract_synthesized_lines(
                    response.content,
                    kind="PATTERN",
                    existing=existing,
                )
            except Exception:  # noqa: BLE001
                synthesized = []
        if not synthesized:
            synthesized = fallback
        merged = self._merge_bullets(current_content, synthesized)
        if merged.strip():
            return merged
        if synthesized:
            return "# Style Guide\n\n" + "\n".join(synthesized) + "\n"
        return "# Style Guide\n\n- No style patterns synthesized yet."

    def should_synthesize(self) -> bool:
        if not self.state_path.exists():
            return True
        try:
            payload = json_load(self.state_path)
        except (OSError, ValueError):
            return True
        if not isinstance(payload, dict):
            return True
        stored_count = payload.get("iteration_count")
        if not isinstance(stored_count, int):
            return True
        current_count = self._iteration_count()
        return current_count != stored_count

    async def synthesize(self) -> dict[str, int]:
        store = IterationReportStore(self.ralph_dir)
        reports = store.load_all()
        guardrail_candidates = [
            entry
            for entry, _count in self._deduplicate_candidates(
                self._extract_guardrail_candidates(reports)
            )
        ]
        style_candidates = [
            entry
            for entry, _count in self._deduplicate_candidates(
                self._extract_style_candidates(reports)
            )
        ]
        guardrails_content = await self.synthesize_guardrails(guardrail_candidates)
        style_content = await self.synthesize_style(style_candidates)
        self.files.write_tier(KnowledgeTier.GUARDRAILS, guardrails_content)
        self.files.write_tier(KnowledgeTier.STYLE, style_content)
        if compact_if_over_tokens(
            files=self.files,
            tier=KnowledgeTier.GUARDRAILS,
            content=guardrails_content,
        ):
            updated = self.files.read_tier(KnowledgeTier.GUARDRAILS)
            if updated.strip():
                guardrails_content = updated
            else:
                guardrails_content = TIER_HEADERS[KnowledgeTier.GUARDRAILS]
                self.files.write_tier(KnowledgeTier.GUARDRAILS, guardrails_content)
        if compact_if_over_tokens(
            files=self.files,
            tier=KnowledgeTier.STYLE,
            content=style_content,
        ):
            updated = self.files.read_tier(KnowledgeTier.STYLE)
            if updated.strip():
                style_content = updated
            else:
                style_content = TIER_HEADERS[KnowledgeTier.STYLE]
                self.files.write_tier(KnowledgeTier.STYLE, style_content)
        self._write_state(len(reports))
        return {
            "guardrails": len(guardrail_candidates),
            "style": len(style_candidates),
        }

    def _iteration_count(self) -> int:
        store = IterationReportStore(self.ralph_dir)
        return len(store.load_all())

    def _write_state(self, iteration_count: int) -> None:
        payload = {
            "last_synthesized_at": datetime.now(UTC).isoformat(),
            "iteration_count": iteration_count,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json_dump(payload), encoding="utf-8")


def json_load(path: Path) -> dict[str, Any]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid JSON payload")
    return data


def json_dump(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2)
