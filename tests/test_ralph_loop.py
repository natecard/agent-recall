from __future__ import annotations

import subprocess
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

    guardrails = (tmp_path / ".agent" / "GUARDRAILS.md").read_text()
    style = (tmp_path / ".agent" / "STYLE.md").read_text()
    recent = (tmp_path / ".agent" / "RECENT.md").read_text()
    assert "Iteration 1 (RLPH-001)" in guardrails
    assert "Iteration 1 (RLPH-001)" in style
    assert "Iteration 1" in recent
    assert "Outcome: progressed" in recent


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
        "'{\"type\":\"assistant\",\"message\":{\"content\":["
        "{\"type\":\"text\",\"text\":\"done\"}]}}' "
        "'{\"type\":\"result\",\"result\":\"<promise>COMPLETE</promise>\"}'"
    )
    result = _run_loop(
        tmp_path,
        "--agent-output-mode",
        "stream-json",
        agent_cmd=json_agent_cmd,
    )

    assert result.returncode == 0
    assert "Completion marker seen and validation green. Exiting early." in result.stdout
