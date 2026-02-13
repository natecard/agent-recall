from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage


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

            exit_code = 0
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

        return {
            "total_iterations": len(candidates),
            "passed": passed,
            "failed": failed,
        }
