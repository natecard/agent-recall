import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_recall.storage.base import SharedBackendUnavailableError
from agent_recall.storage.models import SharedStorageConfig
from agent_recall.storage.remote import RemoteStorage


class TestRemoteResilience:
    @pytest.fixture
    def config(self):
        return SharedStorageConfig(base_url="file:///tmp/shared.db", retry_attempts=3)

    @pytest.fixture
    def local_path(self, tmp_path):
        return tmp_path / "local.db"

    def test_retry_success(self, config, local_path):
        """Test that operations retry on transient errors and eventually succeed."""
        with patch("agent_recall.storage.remote.SQLiteStorage"):
            # Mock resolve_shared_db_path to return a Path
            with patch(
                "agent_recall.storage.remote.resolve_shared_db_path",
                return_value=Path("/tmp/shared.db"),
            ):
                storage = RemoteStorage(config, local_db_path=local_path)

                # Replace delegates with mocks
                mock_delegate = MagicMock()
                mock_local = MagicMock()
                storage._delegate = mock_delegate
                storage._local = mock_local

                # Mock delegate method to fail twice then succeed
                method = MagicMock(
                    side_effect=[
                        sqlite3.OperationalError("locked"),
                        sqlite3.OperationalError("locked"),
                        "success",
                    ]
                )
                # We mock getattr behavior by setting the method on the mock
                mock_delegate.some_method = method

                # Call _execute
                # We pass "some_method" as string
                result = storage._execute("some_method", "arg1")

                assert result == "success"
                assert method.call_count == 3
                # Local should not be called
                assert len(mock_local.method_calls) == 0

    def test_fallback_success(self, config, local_path):
        """Test that operations fall back to local storage after retries are exhausted."""
        with patch("agent_recall.storage.remote.SQLiteStorage"):
            with patch(
                "agent_recall.storage.remote.resolve_shared_db_path",
                return_value=Path("/tmp/shared.db"),
            ):
                storage = RemoteStorage(config, local_db_path=local_path)
                storage._delegate = MagicMock()
                storage._local = MagicMock()

                # Mock delegate to fail always
                storage._delegate.some_method.side_effect = sqlite3.OperationalError("locked")

                # Mock local to succeed
                storage._local.some_method.return_value = "local_success"

                result = storage._execute("some_method", "arg1")

                assert result == "local_success"
                # Should have tried delegate 'retry_attempts' times (3)
                assert storage._delegate.some_method.call_count == 3
                # Should have called local once
                assert storage._local.some_method.call_count == 1

    def test_failure_no_fallback(self, config):
        """Test that operations raise SharedBackendUnavailableError if no fallback is configured."""
        with patch("agent_recall.storage.remote.SQLiteStorage"):
            with patch(
                "agent_recall.storage.remote.resolve_shared_db_path",
                return_value=Path("/tmp/shared.db"),
            ):
                # No local path
                storage = RemoteStorage(config, local_db_path=None)
                storage._delegate = MagicMock()

                storage._delegate.some_method.side_effect = sqlite3.OperationalError("fatal")

                with pytest.raises(SharedBackendUnavailableError) as exc:
                    storage._execute("some_method")

                assert "failed after 3 attempts" in str(exc.value)

    def test_failure_fallback_fails(self, config, local_path):
        """Test that operations raise SharedBackendUnavailableError if fallback also fails."""
        with patch("agent_recall.storage.remote.SQLiteStorage"):
            with patch(
                "agent_recall.storage.remote.resolve_shared_db_path",
                return_value=Path("/tmp/shared.db"),
            ):
                storage = RemoteStorage(config, local_db_path=local_path)
                storage._delegate = MagicMock()
                storage._local = MagicMock()

                storage._delegate.some_method.side_effect = sqlite3.OperationalError(
                    "primary fail"
                )
                storage._local.some_method.side_effect = Exception("local fail")

                with pytest.raises(SharedBackendUnavailableError) as exc:
                    storage._execute("some_method")

                assert "failed after 3 attempts" in str(exc.value)
