from __future__ import annotations

from typing import cast

import pytest

from agent_recall.core.background_sync import BackgroundSyncManager
from agent_recall.core.sync import AutoSync


class FlakyAutoSync:
    def __init__(self, fail_attempts: int):
        self.fail_attempts = fail_attempts
        self.calls = 0

    async def sync(self, sources=None, max_sessions=None):
        _ = (sources, max_sessions)
        self.calls += 1
        if self.calls <= self.fail_attempts:
            raise RuntimeError("temporary transport failure")
        return {"sessions_processed": 2, "learnings_extracted": 5}

    async def sync_and_compact(self, sources=None, max_sessions=None):
        return await self.sync(sources=sources, max_sessions=max_sessions)


@pytest.mark.asyncio
async def test_background_sync_retries_then_succeeds(storage, files) -> None:
    auto_sync = FlakyAutoSync(fail_attempts=2)
    manager = BackgroundSyncManager(
        storage=storage,
        files=files,
        auto_sync=cast(AutoSync, auto_sync),
        retry_attempts=3,
        retry_backoff_seconds=0.0,
    )

    result = await manager.run_sync(compact=True)
    assert result.success is True
    assert result.attempts == 3
    assert auto_sync.calls == 3
    assert len(result.diagnostics) == 2
    assert "Attempt 1/3 failed" in result.diagnostics[0]
    assert manager.is_sync_running() is False

    status = storage.get_background_sync_status()
    assert status.is_running is False
    assert status.sessions_processed == 2
    assert status.learnings_extracted == 5
    assert status.error_message is None


@pytest.mark.asyncio
async def test_background_sync_reports_final_error_after_retry_exhaustion(storage, files) -> None:
    auto_sync = FlakyAutoSync(fail_attempts=10)
    manager = BackgroundSyncManager(
        storage=storage,
        files=files,
        auto_sync=cast(AutoSync, auto_sync),
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    )

    result = await manager.run_sync(sources=["cursor"], max_sessions=1, compact=False)
    assert result.success is False
    assert result.attempts == 2
    assert auto_sync.calls == 2
    assert len(result.diagnostics) == 2
    assert "Attempt 2/2 failed" in result.diagnostics[-1]
    assert "sources=cursor" in result.diagnostics[-1]
    assert result.error_message is not None
    assert "Background sync failed after 2 attempt(s)." in result.error_message
    assert manager.is_sync_running() is False

    status = storage.get_background_sync_status()
    assert status.is_running is False
    assert status.error_message == result.error_message
