"""Background sync functionality with safe locking.

Provides file-based locking to prevent duplicate sync operations
and tracks sync status persistently.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_recall.core.sync import AutoSync
from agent_recall.storage.files import FileStorage
from agent_recall.storage.models import BackgroundSyncStatus
from agent_recall.storage.sqlite import SQLiteStorage


@dataclass
class BackgroundSyncResult:
    """Result of a background sync operation."""

    success: bool
    sessions_processed: int = 0
    learnings_extracted: int = 0
    error_message: str | None = None
    was_already_running: bool = False


class BackgroundSyncLock:
    """File-based lock for background sync operations.

    Uses flock on Unix systems to ensure only one sync runs at a time.
    Automatically cleans up stale locks from dead processes.
    """

    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if acquired, False if already held."""
        try:
            self._fd = os.open(str(self.lock_file), os.O_CREAT | os.O_RDWR)
            # Try to acquire exclusive lock without blocking
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write PID to file for debugging/stale detection
            os.write(self._fd, str(os.getpid()).encode())
            os.ftruncate(self._fd, 0)
            os.write(self._fd, str(os.getpid()).encode())
            return True
        except OSError:
            # Lock is held by another process
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            finally:
                self._fd = None

    def is_held_by_another_process(self) -> bool:
        """Check if lock is held by another (potentially dead) process."""
        if not self.lock_file.exists():
            return False

        try:
            fd = os.open(str(self.lock_file), os.O_RDWR)
            try:
                # Try non-blocking lock
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # We got the lock, so it wasn't held
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                return False
            except OSError:
                os.close(fd)
                # Lock is held, check if process is alive
                try:
                    with open(self.lock_file) as f:
                        pid_str = f.read().strip()
                        if pid_str:
                            pid = int(pid_str)
                            # Check if process exists
                            os.kill(pid, 0)
                            return True  # Process exists, lock is valid
                except (ValueError, OSError, ProcessLookupError):
                    # Process doesn't exist, lock is stale
                    pass
                return False
        except OSError:
            return False

    def cleanup_stale(self) -> bool:
        """Remove stale lock file if the holding process is dead."""
        if not self.lock_file.exists():
            return True

        try:
            with open(self.lock_file) as f:
                pid_str = f.read().strip()
                if pid_str:
                    try:
                        pid = int(pid_str)
                        os.kill(pid, 0)
                        # Process exists, don't cleanup
                        return False
                    except (OSError, ProcessLookupError):
                        # Process is dead, remove stale lock
                        pass
            self.lock_file.unlink(missing_ok=True)
            return True
        except (ValueError, OSError):
            return False

    def __enter__(self) -> BackgroundSyncLock:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class BackgroundSyncManager:
    """Manages background sync operations with safe locking and status tracking."""

    LOCK_FILENAME = ".background_sync.lock"

    def __init__(
        self,
        storage: SQLiteStorage,
        files: FileStorage,
        auto_sync: AutoSync | None,
        lock_file: Path | None = None,
    ):
        self.storage = storage
        self.files = files
        self.auto_sync = auto_sync
        self.lock_file = lock_file or (files.agent_dir / self.LOCK_FILENAME)
        self._lock = BackgroundSyncLock(self.lock_file)

    def get_status(self) -> BackgroundSyncStatus:
        """Get current background sync status."""
        status = self.storage.get_background_sync_status()

        # Check if status shows running but lock is not held
        if status.is_running:
            if not self._lock.is_held_by_another_process():
                # Sync process died without updating status
                status = self.storage.complete_background_sync(
                    sessions_processed=status.sessions_processed,
                    learnings_extracted=status.learnings_extracted,
                    error_message="Sync process terminated unexpectedly",
                )

        return status

    def is_sync_running(self) -> bool:
        """Check if a background sync is currently running."""
        return self._lock.is_held_by_another_process()

    def can_start_sync(self) -> tuple[bool, str | None]:
        """Check if sync can be started. Returns (can_start, reason)."""
        if self.is_sync_running():
            status = self.get_status()
            return False, f"Sync already running (PID: {status.pid})"
        return True, None

    async def run_sync(
        self,
        sources: list[str] | None = None,
        max_sessions: int | None = None,
        compact: bool = True,
    ) -> BackgroundSyncResult:
        """Run sync in background with proper locking.

        Args:
            sources: Optional list of source names to sync
            max_sessions: Optional maximum number of sessions to process
            compact: Whether to run compaction after sync

        Returns:
            BackgroundSyncResult with outcome details
        """
        can_start, reason = self.can_start_sync()
        if not can_start:
            return BackgroundSyncResult(
                success=False,
                was_already_running=True,
                error_message=reason,
            )

        if self.auto_sync is None:
            return BackgroundSyncResult(
                success=False,
                error_message="AutoSync not initialized",
            )

        pid = os.getpid()
        self.storage.start_background_sync(pid)

        try:
            # Run the actual sync
            if compact:
                results = await self.auto_sync.sync_and_compact(
                    sources=sources,
                    max_sessions=max_sessions,
                )
            else:
                results = await self.auto_sync.sync(
                    sources=sources,
                    max_sessions=max_sessions,
                )

            sessions_processed = int(results.get("sessions_processed", 0))
            learnings_extracted = int(results.get("learnings_extracted", 0))

            self.storage.complete_background_sync(
                sessions_processed=sessions_processed,
                learnings_extracted=learnings_extracted,
            )

            return BackgroundSyncResult(
                success=True,
                sessions_processed=sessions_processed,
                learnings_extracted=learnings_extracted,
            )

        except Exception as exc:
            error_msg = str(exc)
            self.storage.complete_background_sync(
                sessions_processed=0,
                learnings_extracted=0,
                error_message=error_msg,
            )
            return BackgroundSyncResult(
                success=False,
                error_message=error_msg,
            )

    def cleanup_stale_lock(self) -> bool:
        """Clean up a stale lock file if the holding process is dead."""
        return self._lock.cleanup_stale()
