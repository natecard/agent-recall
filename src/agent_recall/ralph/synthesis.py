from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_recall.llm import LLMProvider, Message
from agent_recall.ralph.iteration_store import IterationReport, IterationReportStore
from agent_recall.storage.files import FileStorage, KnowledgeTier

GUARDRAILS_SYNTHESIS_PROMPT = (
    "You are distilling Ralph iteration learnings into guardrails. "
    "Given the candidate notes below, output concise bullet rules (1 line each), "
    "max {max_entries} bullets. Avoid duplication and keep them actionable.\n\n"
    "Candidates:\n{candidates}"
)

STYLE_SYNTHESIS_PROMPT = (
    "You are distilling Ralph iteration learnings into a coding style guide. "
    "Given the candidate notes below, output concise bullet patterns (1 line each), "
    "max {max_entries} bullets. Avoid duplication and keep them actionable.\n\n"
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
        llm: LLMProvider,
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

    async def synthesize_guardrails(self, candidates: list[str]) -> str:
        if not candidates:
            return "# Guardrails\n\n- No guardrails synthesized yet."
        prompt = GUARDRAILS_SYNTHESIS_PROMPT.format(
            candidates="\n".join(f"- {item}" for item in candidates),
            max_entries=self.config.max_guardrails,
        )
        response = await self.llm.generate(
            [Message(role="user", content=prompt)],
            temperature=0.2,
            max_tokens=800,
        )
        content = response.content.strip()
        if content.startswith("#"):
            return content
        return "# Guardrails\n\n" + content

    async def synthesize_style(self, candidates: list[str]) -> str:
        if not candidates:
            return "# Style Guide\n\n- No style patterns synthesized yet."
        prompt = STYLE_SYNTHESIS_PROMPT.format(
            candidates="\n".join(f"- {item}" for item in candidates),
            max_entries=self.config.max_style,
        )
        response = await self.llm.generate(
            [Message(role="user", content=prompt)],
            temperature=0.2,
            max_tokens=800,
        )
        content = response.content.strip()
        if content.startswith("#"):
            return content
        return "# Style Guide\n\n" + content

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
