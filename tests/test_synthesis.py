from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from agent_recall.llm.base import LLMProvider, LLMResponse, Message
from agent_recall.ralph.iteration_store import (
    IterationReport,
    IterationReportStore,
)
from agent_recall.ralph.synthesis import ClimateSynthesizer, SynthesisConfig
from agent_recall.storage.files import FileStorage


def _write_report(store: IterationReportStore, report: IterationReport) -> None:
    store.iterations_dir.mkdir(parents=True, exist_ok=True)
    path = store.iterations_dir / f"{report.iteration:03d}.json"
    path.write_text(_json_dumps(report), encoding="utf-8")


def _json_dumps(report: IterationReport) -> str:
    import json

    return json.dumps(report.to_dict(), indent=2)


class StubLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        content = (
            "# Guardrails\n\n- synthesized"
            if "guardrails" in messages[0].content.lower()
            else "# Style Guide\n\n- synthesized"
        )
        return LLMResponse(content=content, model=self.model_name)

    def validate(self) -> tuple[bool, str]:
        return (True, "ok")


def test_extract_candidates() -> None:
    reports = [
        IterationReport(
            iteration=1,
            item_id="WM-005",
            item_title="Synthesis",
            started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
            failure_reason="failed",
            gotcha_discovered="watch this",
            pattern_that_worked="keep it small",
        ),
        IterationReport(
            iteration=2,
            item_id="WM-005",
            item_title="Synthesis",
            started_at=datetime(2026, 2, 14, 12, 10, 0, tzinfo=UTC),
        ),
    ]
    synth = ClimateSynthesizer(Path("."), FileStorage(Path(".")), llm=StubLLM())

    guardrails = synth._extract_guardrail_candidates(reports)
    styles = synth._extract_style_candidates(reports)

    assert guardrails == ["failed", "watch this"]
    assert styles == ["keep it small"]


def test_deduplicate_candidates_orders_by_frequency(tmp_path: Path) -> None:
    synth = ClimateSynthesizer(tmp_path, FileStorage(tmp_path), llm=StubLLM())

    ranked = synth._deduplicate_candidates(["Alpha", "alpha", "beta", "Beta", "beta"])

    assert ranked[0] == ("beta", 3)
    assert ranked[1] == ("Alpha", 2)


def test_synthesize_writes_tiers_and_state(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    report = IterationReport(
        iteration=1,
        item_id="WM-005",
        item_title="Synthesis",
        started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
        failure_reason="failed",
        pattern_that_worked="keep it small",
    )
    _write_report(store, report)
    files = FileStorage(tmp_path)
    llm = StubLLM()
    synth = ClimateSynthesizer(
        ralph_dir,
        files,
        llm=llm,
        config=SynthesisConfig(max_guardrails=10, max_style=10),
    )

    results = asyncio.run(synth.synthesize())

    assert results == {"guardrails": 1, "style": 1}
    assert (tmp_path / "GUARDRAILS.md").exists()
    assert (tmp_path / "STYLE.md").exists()
    assert (ralph_dir / "synthesis_state.json").exists()
    assert llm.calls
    assert all(call["temperature"] == 0.2 for call in llm.calls)
    assert all(call["max_tokens"] == 800 for call in llm.calls)


def test_synthesize_empty_candidates_writes_defaults(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    files = FileStorage(tmp_path)
    llm = StubLLM()
    synth = ClimateSynthesizer(ralph_dir, files, llm=llm)

    results = asyncio.run(synth.synthesize())

    guardrails = (tmp_path / "GUARDRAILS.md").read_text()
    style = (tmp_path / "STYLE.md").read_text()
    assert results == {"guardrails": 0, "style": 0}
    assert "No guardrails synthesized yet" in guardrails
    assert "No style patterns synthesized yet" in style
    assert llm.calls == []


def test_should_synthesize_checks_iteration_count(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    files = FileStorage(tmp_path)
    llm = StubLLM()
    synth = ClimateSynthesizer(ralph_dir, files, llm=llm)

    assert synth.should_synthesize() is True
    store = IterationReportStore(ralph_dir)
    _write_report(
        store,
        IterationReport(
            iteration=1,
            item_id="WM-005",
            item_title="Synthesis",
            started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
        ),
    )
    asyncio.run(synth.synthesize())
    assert synth.should_synthesize() is False
    _write_report(
        store,
        IterationReport(
            iteration=2,
            item_id="WM-005",
            item_title="Synthesis",
            started_at=datetime(2026, 2, 14, 12, 10, 0, tzinfo=UTC),
        ),
    )
    assert synth.should_synthesize() is True


def test_synthesized_tiers_consumable_by_context_assembly(tmp_path: Path) -> None:
    """Synthesized tier writes are readable by ContextAssembler used in refresh."""
    from agent_recall.core.context import ContextAssembler
    from agent_recall.storage.files import KnowledgeTier
    from agent_recall.storage.sqlite import SQLiteStorage

    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    report = IterationReport(
        iteration=1,
        item_id="WM-005",
        item_title="Synthesis",
        started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
        failure_reason="avoid race conditions",
        pattern_that_worked="use small commits",
    )
    _write_report(store, report)
    files = FileStorage(tmp_path)
    llm = StubLLM()
    synth = ClimateSynthesizer(
        ralph_dir,
        files,
        llm=llm,
        config=SynthesisConfig(max_guardrails=10, max_style=10),
    )

    asyncio.run(synth.synthesize())

    storage = SQLiteStorage(tmp_path / "state.db")
    assembler = ContextAssembler(storage=storage, files=files, retriever=None)
    context = assembler.assemble(task=None, include_retrieval=False)

    assert "synthesized" in context or "Guardrails" in context
    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    style = files.read_tier(KnowledgeTier.STYLE)
    assert "synthesized" in guardrails or "Guardrails" in guardrails
    assert "synthesized" in style or "Style" in style
