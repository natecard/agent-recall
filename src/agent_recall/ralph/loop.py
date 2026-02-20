from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent_recall.ralph.costs import budget_exceeded, summarize_costs
from agent_recall.ralph.iteration_store import IterationReportStore
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage
from agent_recall.storage.models import RalphNotificationEvent

# CLI binary lookup for Ralph-supported coding agents.
_CODING_CLI_BINARIES: dict[str, str] = {
    "claude-code": "claude",
    "codex": "codex",
    "opencode": "opencode",
}


class RalphStatus(StrEnum):
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"


@dataclass
class RalphState:
    status: RalphStatus = RalphStatus.DISABLED
    total_iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "total_iterations": self.total_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RalphState:
        status_raw = str(data.get("status", RalphStatus.DISABLED.value))
        try:
            status = RalphStatus(status_raw)
        except ValueError:
            status = RalphStatus.DISABLED
        total_iterations = int(data.get("total_iterations") or 0)
        return cls(status=status, total_iterations=total_iterations)


class RalphStateManager:
    def __init__(self, agent_dir: Path) -> None:
        self.agent_dir = agent_dir
        self.state_path = agent_dir / "ralph" / "ralph_state.json"

    def load(self) -> RalphState:
        if not self.state_path.exists():
            return RalphState()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return RalphState()
        if not isinstance(payload, dict):
            return RalphState()
        return RalphState.from_dict(payload)

    def save(self, state: RalphState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state.to_dict(), indent=2),
            encoding="utf-8",
        )


class RalphLoop:
    def __init__(self, agent_dir: Path, storage: Storage, files: FileStorage) -> None:
        self.agent_dir = agent_dir
        self.storage = storage
        self.files = files
        self.state = RalphStateManager(agent_dir)

    def enable(self) -> RalphState:
        state = self.state.load()
        state.status = RalphStatus.ENABLED
        self.state.save(state)
        return state

    def disable(self) -> RalphState:
        state = self.state.load()
        state.status = RalphStatus.DISABLED
        self.state.save(state)
        return state

    def initialize_from_prd(self, prd_path: Path) -> int:
        data = json.loads(prd_path.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else None
        item_count = len(items) if isinstance(items, list) else 0
        self.enable()
        return item_count

    @staticmethod
    def _resolve_prd_path(agent_dir: Path) -> Path:
        candidates = [
            agent_dir / "ralph" / "prd.json",
            Path("agent_recall/ralph/prd.json"),
            Path("prd.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _load_prd_items(prd_path: Path) -> list[dict[str, Any]]:
        data = json.loads(prd_path.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else None
        return list(items) if isinstance(items, list) else []

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[dict[str, Any]], None] | None,
        payload: dict[str, Any],
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(payload)
        except Exception:  # noqa: BLE001
            return

    def _emit_notification(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        event: RalphNotificationEvent,
        *,
        iteration: int | None = None,
    ) -> None:
        self._emit_progress(
            progress_callback,
            {
                "event": "notification",
                "notification_event": event.value,
                "iteration": iteration,
            },
        )

    def _emit_cost_budget_exceeded(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        *,
        total_cost: float,
        budget: float,
        iteration: int | None = None,
    ) -> None:
        self._emit_progress(
            progress_callback,
            {
                "event": "budget_exceeded",
                "budget_type": "cost_usd",
                "total_cost_usd": total_cost,
                "budget_usd": budget,
                "iteration": iteration,
            },
        )

    @staticmethod
    def _extract_codex_display_lines(raw_line: str) -> list[str]:
        stripped = raw_line.strip()
        if not stripped:
            return []
        if not stripped.startswith("{"):
            return [raw_line]

        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            return [raw_line]
        if not isinstance(payload, dict):
            return []

        event_type = str(payload.get("type") or "")
        if event_type == "assistant":
            message = payload.get("message")
            if not isinstance(message, dict):
                return []
            content = message.get("content")
            if not isinstance(content, list):
                return []
            lines: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if str(block.get("type") or "") != "text":
                    continue
                text = str(block.get("text") or "").strip()
                if text:
                    lines.append(text)
            return lines

        if event_type == "result":
            result = payload.get("result")
            if isinstance(result, str):
                result_text = result.strip()
                if result_text:
                    return [result_text]
            return []

        return []

    async def _run_agent_subprocess(
        self,
        *,
        coding_cli: str,
        cli_model: str | None,
        item_title: str,
        iteration: int,
        item_id: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> int:
        """Spawn the coding CLI and stream output lines via progress_callback."""
        binary = _CODING_CLI_BINARIES.get(coding_cli)
        if not binary:
            self._emit_progress(
                progress_callback,
                {
                    "event": "output_line",
                    "line": f"Unknown coding CLI: {coding_cli}",
                    "iteration": iteration,
                    "item_id": item_id,
                },
            )
            return 1

        resolved = shutil.which(binary)
        if resolved is None:
            self._emit_progress(
                progress_callback,
                {
                    "event": "output_line",
                    "line": f"CLI binary not found on PATH: {binary}",
                    "iteration": iteration,
                    "item_id": item_id,
                },
            )
            return 1

        prompt = f"Work on PRD item {item_id}: {item_title}"
        if coding_cli == "codex":
            cmd = [
                resolved,
                "--ask-for-approval",
                "never",
                "exec",
                "--sandbox",
                "danger-full-access",
            ]
            if cli_model:
                cmd.extend(["--model", cli_model])
            cmd.append("--json")
            cmd.append(prompt)
        else:
            cmd = [resolved, "--print", prompt]
            if cli_model:
                cmd.extend(["--model", cli_model])

        self._emit_progress(
            progress_callback,
            {
                "event": "output_line",
                "line": f"$ {' '.join(cmd)}",
                "iteration": iteration,
                "item_id": item_id,
            },
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            self._emit_progress(
                progress_callback,
                {
                    "event": "output_line",
                    "line": f"Failed to start {binary}: {exc}",
                    "iteration": iteration,
                    "item_id": item_id,
                },
            )
            return 1

        assert proc.stdout is not None  # noqa: S101
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            display_lines = (
                self._extract_codex_display_lines(line) if coding_cli == "codex" else [line]
            )
            for display_line in display_lines:
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "output_line",
                        "line": display_line,
                        "iteration": iteration,
                        "item_id": item_id,
                    },
                )

        exit_code = await proc.wait()
        return exit_code

    def _read_state_payload(self) -> dict[str, Any]:
        if not self.state.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_state_payload(self, payload: dict[str, Any]) -> None:
        self.state.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state.state_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    async def run_loop(
        self,
        *,
        max_iterations: int | None = None,
        item_id: str | None = None,
        selected_prd_ids: list[str] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        coding_cli: str | None = None,
        cli_model: str | None = None,
    ) -> dict[str, int]:
        prd_path = self._resolve_prd_path(self.agent_dir)
        if not prd_path.exists():
            raise FileNotFoundError(f"PRD file not found: {prd_path}")

        items = self._load_prd_items(prd_path)
        pending_items = [
            item for item in items if isinstance(item, dict) and item.get("passes") is not True
        ]

        candidates = pending_items
        if item_id:
            candidates = [
                item
                for item in items
                if isinstance(item, dict) and str(item.get("id") or "") == item_id
            ]
            if not candidates:
                raise ValueError(f"PRD item not found: {item_id}")
        elif selected_prd_ids:
            wanted = {str(value).strip() for value in selected_prd_ids if str(value).strip()}
            if wanted:
                candidates = [item for item in candidates if str(item.get("id") or "") in wanted]

        def _priority_key(entry: dict[str, Any]) -> int:
            raw = entry.get("priority")
            return int(raw) if isinstance(raw, int) else 999999

        if not item_id:
            candidates = sorted(candidates, key=_priority_key)

        if max_iterations is not None:
            candidates = candidates[:max_iterations]

        state = self.state.load()
        state_payload = self._read_state_payload()
        total_iterations = int(state_payload.get("total_iterations") or state.total_iterations)

        config_dict = self.files.read_config()
        ralph_cfg = config_dict.get("ralph") if isinstance(config_dict, dict) else None
        if not isinstance(ralph_cfg, dict):
            ralph_cfg = {}
        cost_budget_value = ralph_cfg.get("cost_budget_usd")
        cost_budget = (
            float(cost_budget_value)
            if isinstance(cost_budget_value, int | float) and cost_budget_value >= 0
            else None
        )

        passed = 0
        failed = 0
        processed_ids: set[str] = set()
        failed_ids: set[str] = set()

        for index, item in enumerate(candidates, start=1):
            item_id_value = str(item.get("id") or "")
            title = str(item.get("title") or "")
            processed_ids.add(item_id_value)
            start_time = time.monotonic()
            self._emit_progress(
                progress_callback,
                {
                    "event": "iteration_started",
                    "iteration": index,
                    "item_id": item_id_value,
                    "title": title,
                },
            )

            if coding_cli:
                exit_code = await self._run_agent_subprocess(
                    coding_cli=coding_cli,
                    cli_model=cli_model,
                    item_title=title,
                    iteration=index,
                    item_id=item_id_value,
                    progress_callback=progress_callback,
                )
            else:
                exit_code = 0
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "output_line",
                        "line": "No coding CLI configured; skipping agent execution.",
                        "iteration": index,
                        "item_id": item_id_value,
                    },
                )
            self._emit_progress(
                progress_callback,
                {
                    "event": "agent_complete",
                    "iteration": index,
                    "item_id": item_id_value,
                    "exit_code": exit_code,
                },
            )

            validation_success = True
            validation_hint = ""
            self._emit_progress(
                progress_callback,
                {
                    "event": "validation_complete",
                    "iteration": index,
                    "item_id": item_id_value,
                    "success": validation_success,
                    "hint": validation_hint,
                },
            )
            if not validation_success:
                self._emit_notification(
                    progress_callback,
                    RalphNotificationEvent.VALIDATION_FAILED,
                    iteration=index,
                )

            duration_seconds = time.monotonic() - start_time
            if exit_code == 0 and validation_success:
                outcome = "passed"
                passed += 1
            else:
                outcome = "failed"
                failed += 1
                failed_ids.add(item_id_value)
            self._emit_progress(
                progress_callback,
                {
                    "event": "iteration_complete",
                    "iteration": index,
                    "item_id": item_id_value,
                    "outcome": outcome,
                    "duration_seconds": duration_seconds,
                },
            )
            self._emit_notification(
                progress_callback,
                RalphNotificationEvent.ITERATION_COMPLETE,
                iteration=index,
            )

            if cost_budget is not None:
                cost_summary = summarize_costs(
                    IterationReportStore(self.agent_dir / "ralph").load_all()
                )
                if budget_exceeded(cost_summary.total_cost_usd, cost_budget):
                    self._emit_cost_budget_exceeded(
                        progress_callback,
                        total_cost=cost_summary.total_cost_usd,
                        budget=cost_budget,
                        iteration=index,
                    )
                    self._emit_notification(
                        progress_callback,
                        RalphNotificationEvent.BUDGET_EXCEEDED,
                        iteration=index,
                    )
                    break

        total_iterations += len(candidates)
        now = datetime.now(UTC).isoformat()
        last_outcome = "passed" if failed == 0 else "failed"
        state_payload.update(
            {
                "status": state.status.value,
                "current_iteration": len(candidates),
                "total_iterations": total_iterations,
                "successful_iterations": passed,
                "failed_iterations": failed,
                "last_run_at": now,
                "last_outcome": last_outcome,
                "prd_path": str(prd_path),
            }
        )

        item_entries: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry_id = str(item.get("id") or "")
            title = str(item.get("title") or "")
            iterations = 1 if entry_id in processed_ids else 0
            if entry_id in failed_ids:
                status = "blocked"
            elif item.get("passes") is True or entry_id in processed_ids:
                status = "completed"
            else:
                status = "pending"
            item_entries.append(
                {
                    "id": entry_id,
                    "title": title,
                    "status": status,
                    "iterations": iterations,
                }
            )
        state_payload["items"] = item_entries
        self._write_state_payload(state_payload)

        state.total_iterations = total_iterations
        self.state.save(state)

        self._emit_notification(
            progress_callback,
            RalphNotificationEvent.LOOP_FINISHED,
        )

        return {
            "total_iterations": len(candidates),
            "passed": passed,
            "failed": failed,
        }
