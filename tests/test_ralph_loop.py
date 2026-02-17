from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "agent_recall"
        / "scripts"
        / "ralph-agent-recall-loop.sh"
    )


def _write_default_repo_layout(repo_root: Path) -> None:
    ralph_dir = repo_root / "agent_recall" / "ralph"
    agent_dir = repo_root / ".agent"

    ralph_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)

    (ralph_dir / "prd.json").write_text(
        """{
  "project": "Ralph Test",
  "items": [
    {
      "id": "RLPH-001",
      "priority": 1,
      "title": "Test item",
      "user_story": "As a maintainer, I want loop context tests.",
      "passes": false
    }
  ]
}
""",
    )
    (ralph_dir / "progress.txt").write_text("# Agent Recall Ralph Progress Log\n")
    (ralph_dir / "agent-prompt.md").write_text("# Agent Recall Ralph Task\n")

    (agent_dir / "GUARDRAILS.md").write_text("# Guardrails\n")
    (agent_dir / "STYLE.md").write_text("# Style\n")
    (agent_dir / "RECENT.md").write_text("# Recent\n")


def _run_loop(
    repo_root: Path,
    *extra_args: str,
    agent_cmd: str = "true",
) -> subprocess.CompletedProcess[str]:
    args = [
        "bash",
        str(_script_path()),
        "--agent-cmd",
        agent_cmd,
        "--max-iterations",
        "1",
        "--sleep-seconds",
        "0",
        "--compact-mode",
        "off",
        *extra_args,
    ]
    return subprocess.run(
        args,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ralph_loop_injects_iteration_memory_and_agent_context(tmp_path: Path) -> None:
    """Weather Model: rebuild-forecast overwrites RECENT from iteration reports;
    tier files are read-only for agent."""
    _write_default_repo_layout(tmp_path)

    result = _run_loop(tmp_path)

    assert result.returncode == 2
    prompt_path = tmp_path / "agent_recall" / "ralph" / ".runtime" / "prompt-1.md"
    assert prompt_path.exists()
    prompt = prompt_path.read_text()

    assert "## Agent Recall Memory Context" in prompt
    assert "## Agent Recall Directives" in prompt
    assert "### GUARDRAILS.md" in prompt
    assert "### STYLE.md" in prompt
    assert "### RECENT.md" in prompt

    # Weather Model: RECENT.md rebuilt by rebuild-forecast from iteration reports
    recent = (tmp_path / ".agent" / "RECENT.md").read_text()
    assert "# Current Situation" in recent
    assert "## Trajectory" in recent


def test_ralph_loop_prd_ids_filters_items(tmp_path: Path) -> None:
    """When --prd-ids is set, loop processes only those PRD items."""
    ralph_dir = tmp_path / "agent_recall" / "ralph"
    agent_dir = tmp_path / ".agent"
    ralph_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)

    (ralph_dir / "prd.json").write_text(
        """{
  "project": "Ralph Test",
  "items": [
    {"id": "RLPH-001", "priority": 1, "title": "First", "passes": false},
    {"id": "RLPH-002", "priority": 2, "title": "Second", "passes": false},
    {"id": "RLPH-003", "priority": 3, "title": "Third", "passes": false}
  ]
}
"""
    )
    (ralph_dir / "progress.txt").write_text("# Progress\n")
    (ralph_dir / "agent-prompt.md").write_text("# Task\n")
    (agent_dir / "GUARDRAILS.md").write_text("# Guardrails\n")
    (agent_dir / "STYLE.md").write_text("# Style\n")
    (agent_dir / "RECENT.md").write_text("# Recent\n")

    result = _run_loop(
        tmp_path,
        "--prd-file",
        str(ralph_dir / "prd.json"),
        "--prd-ids",
        "RLPH-002",
    )

    assert result.returncode == 2
    prompt_path = tmp_path / "agent_recall" / "ralph" / ".runtime" / "prompt-1.md"
    assert prompt_path.exists()
    prompt = prompt_path.read_text()
    assert "RLPH-002" in prompt
    assert "RLPH-001" not in prompt or "RLPH-002" in prompt
    recent = (agent_dir / "RECENT.md").read_text()
    assert "# Current Situation" in recent
    assert "001" in recent


def test_ralph_loop_supports_external_repo_layout_with_custom_paths(tmp_path: Path) -> None:
    spec_dir = tmp_path / "spec"
    logs_dir = tmp_path / "logs"
    prompts_dir = tmp_path / "prompts"
    context_dir = tmp_path / "memory"

    spec_dir.mkdir()
    logs_dir.mkdir()
    prompts_dir.mkdir()
    context_dir.mkdir()

    (spec_dir / "prd.json").write_text(
        """{
  "project": "External Repo Ralph Test",
  "items": [
    {"id": "EXT-001", "priority": 1, "title": "External item", "passes": false}
  ]
}
""",
    )
    (logs_dir / "progress.txt").write_text("# Progress\n")
    (prompts_dir / "agent.md").write_text("# External Prompt\n")

    result = _run_loop(
        tmp_path,
        "--prd-file",
        "spec/prd.json",
        "--progress-file",
        "logs/progress.txt",
        "--prompt-template",
        "prompts/agent.md",
        "--memory-dir",
        "memory",
    )

    assert result.returncode == 2
    prompt_path = tmp_path / "agent_recall" / "ralph" / ".runtime" / "prompt-1.md"
    assert prompt_path.exists()
    prompt = prompt_path.read_text()

    assert "External item" in prompt
    assert "### GUARDRAILS.md" in prompt
    assert "### STYLE.md" in prompt
    assert "### RECENT.md" in prompt
    assert (context_dir / "GUARDRAILS.md").exists()
    assert (context_dir / "STYLE.md").exists()
    assert (context_dir / "RECENT.md").exists()


def test_ralph_loop_stream_json_completion_marker_exits_success(tmp_path: Path) -> None:
    _write_default_repo_layout(tmp_path)

    json_agent_cmd = (
        "printf '%s\\n' "
        '\'{"type":"assistant","message":{"content":['
        '{"type":"text","text":"done"}]}}\' '
        '\'{"type":"result","result":"<promise>COMPLETE</promise>"}\''
    )
    result = _run_loop(
        tmp_path,
        "--agent-output-mode",
        "stream-json",
        agent_cmd=json_agent_cmd,
    )

    assert result.returncode == 0
    assert "Completion marker seen and validation green. Exiting early." in result.stdout


def test_ralph_loop_runtime_validation_signal_enriches_memory_files(tmp_path: Path) -> None:
    """Weather Model: extract-iteration captures validation hint;
    rebuild-forecast puts it in RECENT."""
    _write_default_repo_layout(tmp_path)

    noisy_validate_cmd = (
        "printf '%s\\n' "
        "'============================= test session starts ==============================' "
        "'platform darwin -- Python 3.12.0' "
        "'E   AssertionError: expected 2 == 3'; "
        "exit 1"
    )
    result = _run_loop(
        tmp_path,
        "--validate-cmd",
        noisy_validate_cmd,
    )

    assert result.returncode == 2

    # Weather Model: validation hint flows into iteration report, rebuild-forecast puts it in RECENT
    recent = (tmp_path / ".agent" / "RECENT.md").read_text()
    assert "E AssertionError: expected 2 == 3" in recent


def test_ralph_loop_compaction_and_synthesis_before_refresh_context(tmp_path: Path) -> None:
    """Compaction and synthesis run before refresh-context
    so next-iteration prompt uses optimized memory."""
    _write_default_repo_layout(tmp_path)
    compact_marker = "- [COMPACT-MARKER] compaction-ran-before-refresh"
    compact_cmd = f"echo '{compact_marker}' >> .agent/RECENT.md"

    result = _run_loop(
        tmp_path,
        "--max-iterations",
        "2",
        "--compact-mode",
        "always",
        "--compact-cmd",
        compact_cmd,
    )

    assert result.returncode == 2
    prompt_path = tmp_path / "agent_recall" / "ralph" / ".runtime" / "prompt-2.md"
    assert prompt_path.exists(), "Second iteration prompt should exist"
    prompt_2 = prompt_path.read_text()
    assert compact_marker in prompt_2, (
        "Iteration 2 prompt must include compacted content; "
        "compaction runs before refresh-context so next iteration sees optimized memory"
    )


def test_ralph_loop_reports_agent_transport_marker(tmp_path: Path) -> None:
    _write_default_repo_layout(tmp_path)

    result = _run_loop(tmp_path, agent_cmd="printf 'agent output\\n'")

    assert result.returncode == 2
    if sys.platform == "darwin":
        assert "Agent transport: pty(script)" in result.stdout
    else:
        assert "Agent transport: legacy(pipe)" in result.stdout
