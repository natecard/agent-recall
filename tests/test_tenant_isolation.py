"""Tests for tenant isolation and namespace safety (AR-010)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from agent_recall.storage.base import NamespaceValidationError, validate_shared_namespace
from agent_recall.storage.models import (
    Chunk,
    ChunkSource,
    LogEntry,
    LogSource,
    SemanticLabel,
    Session,
    SessionCheckpoint,
)
from agent_recall.storage.remote import RemoteStorage
from agent_recall.storage.sqlite import SQLiteStorage


class TestNamespaceValidation:
    """Test namespace validation utilities."""

    def test_validate_shared_namespace_accepts_valid(self):
        """Should accept valid non-default tenant/project IDs."""
        # Should not raise
        validate_shared_namespace("org-123", "repo-abc")
        validate_shared_namespace("acme-corp", "project-x")

    def test_validate_shared_namespace_rejects_default_tenant(self):
        """Should reject default tenant ID."""
        with pytest.raises(NamespaceValidationError) as exc_info:
            validate_shared_namespace("default", "repo-abc")
        assert "tenant_id" in str(exc_info.value)
        assert "default" in str(exc_info.value)

    def test_validate_shared_namespace_rejects_default_project(self):
        """Should reject default project ID."""
        with pytest.raises(NamespaceValidationError) as exc_info:
            validate_shared_namespace("org-123", "default")
        assert "project_id" in str(exc_info.value)
        assert "default" in str(exc_info.value)

    def test_validate_shared_namespace_rejects_empty_tenant(self):
        """Should reject empty tenant ID."""
        with pytest.raises(NamespaceValidationError) as exc_info:
            validate_shared_namespace("", "repo-abc")
        assert "tenant_id" in str(exc_info.value)

    def test_validate_shared_namespace_rejects_empty_project(self):
        """Should reject empty project ID."""
        with pytest.raises(NamespaceValidationError) as exc_info:
            validate_shared_namespace("org-123", "")
        assert "project_id" in str(exc_info.value)

    def test_validate_shared_namespace_rejects_whitespace_only(self):
        """Should reject whitespace-only IDs."""
        with pytest.raises(NamespaceValidationError):
            validate_shared_namespace("   ", "repo-abc")
        with pytest.raises(NamespaceValidationError):
            validate_shared_namespace("org-123", "   ")


class TestSQLiteStorageNamespaceIsolation:
    """Test that SQLiteStorage properly isolates data by namespace."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test.db"

    def test_local_mode_allows_default_namespace(self, db_path: Path):
        """Local mode should work with default namespace."""
        storage = SQLiteStorage(db_path, tenant_id="default", project_id="default")
        session = Session(task="test task")
        storage.create_session(session)

        retrieved = storage.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    def test_strict_mode_rejects_default_tenant(self, db_path: Path):
        """Strict mode should reject default tenant."""
        storage = SQLiteStorage(
            db_path,
            tenant_id="default",
            project_id="my-project",
            strict_namespace_validation=True,
        )
        session = Session(task="test task")

        with pytest.raises(NamespaceValidationError) as exc_info:
            storage.create_session(session)
        assert "tenant_id" in str(exc_info.value)

    def test_strict_mode_rejects_default_project(self, db_path: Path):
        """Strict mode should reject default project."""
        storage = SQLiteStorage(
            db_path,
            tenant_id="my-tenant",
            project_id="default",
            strict_namespace_validation=True,
        )
        session = Session(task="test task")

        with pytest.raises(NamespaceValidationError) as exc_info:
            storage.create_session(session)
        assert "project_id" in str(exc_info.value)

    def test_strict_mode_accepts_explicit_namespace(self, db_path: Path):
        """Strict mode should accept explicit namespace."""
        storage = SQLiteStorage(
            db_path,
            tenant_id="org-123",
            project_id="repo-abc",
            strict_namespace_validation=True,
        )
        session = Session(task="test task")
        storage.create_session(session)

        retrieved = storage.get_session(session.id)
        assert retrieved is not None
        assert retrieved.tenant_id == "org-123"
        assert retrieved.project_id == "repo-abc"

    def test_cross_tenant_isolation_sessions(self, db_path: Path):
        """Sessions from one tenant should not be visible to another."""
        tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="project-1")
        tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="project-1")

        session_a = Session(task="task for tenant a")
        tenant_a.create_session(session_a)

        # Tenant B should not see Tenant A's session
        assert tenant_b.get_session(session_a.id) is None

        # Tenant A should see their own session
        retrieved = tenant_a.get_session(session_a.id)
        assert retrieved is not None
        assert retrieved.id == session_a.id

    def test_cross_project_isolation_sessions(self, db_path: Path):
        """Sessions from one project should not be visible to another."""
        project_a = SQLiteStorage(db_path, tenant_id="tenant-1", project_id="project-a")
        project_b = SQLiteStorage(db_path, tenant_id="tenant-1", project_id="project-b")

        session_a = Session(task="task for project a")
        project_a.create_session(session_a)

        # Project B should not see Project A's session
        assert project_b.get_session(session_a.id) is None

        # Project A should see their own session
        retrieved = project_a.get_session(session_a.id)
        assert retrieved is not None
        assert retrieved.id == session_a.id

    def test_cross_tenant_isolation_entries(self, db_path: Path):
        """Log entries from one tenant should not be visible to another."""
        tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="project-1")
        tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="project-1")

        # Create session for tenant A
        session_a = Session(task="task for tenant a")
        tenant_a.create_session(session_a)

        # Add entry to tenant A's session
        entry = LogEntry(
            session_id=session_a.id,
            source=LogSource.EXPLICIT,
            content="Secret entry for tenant A",
            label=SemanticLabel.PATTERN,
        )
        tenant_a.append_entry(entry)

        # Tenant B should not see the entry
        tenant_b_entries = tenant_b.get_entries(session_a.id)
        assert len(tenant_b_entries) == 0

        # Tenant A should see their entry
        tenant_a_entries = tenant_a.get_entries(session_a.id)
        assert len(tenant_a_entries) == 1
        assert tenant_a_entries[0].content == "Secret entry for tenant A"

    def test_cross_tenant_isolation_chunks(self, db_path: Path):
        """Chunks from one tenant should not be visible to another."""
        tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="project-1")
        tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="project-1")

        # Create chunk for tenant A
        chunk = Chunk(
            source=ChunkSource.LOG_ENTRY,
            source_ids=[uuid4()],
            content="Secret knowledge for tenant A",
            label=SemanticLabel.PATTERN,
        )
        tenant_a.store_chunk(chunk)

        # Tenant B should not see the chunk in search
        tenant_b_results = tenant_b.search_chunks_fts("Secret knowledge")
        assert len(tenant_b_results) == 0

        # Tenant A should see their chunk
        tenant_a_results = tenant_a.search_chunks_fts("Secret knowledge")
        assert len(tenant_a_results) == 1
        assert tenant_a_results[0].content == "Secret knowledge for tenant A"

    def test_cross_tenant_isolation_checkpoints(self, db_path: Path):
        """Checkpoints from one tenant should not be visible to another."""
        tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="project-1")
        tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="project-1")

        source_session_id = "source-session-123"

        # Create checkpoint for tenant A
        checkpoint_a = SessionCheckpoint(source_session_id=source_session_id)
        tenant_a.save_session_checkpoint(checkpoint_a)

        # Tenant B should not see the checkpoint
        assert tenant_b.get_session_checkpoint(source_session_id) is None

        # Tenant A should see their checkpoint
        retrieved = tenant_a.get_session_checkpoint(source_session_id)
        assert retrieved is not None
        assert retrieved.tenant_id == "tenant-a"

    def test_cross_tenant_isolation_processed_sessions(self, db_path: Path):
        """Processed session markers from one tenant should not affect another."""
        tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="project-1")
        tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="project-1")

        source_session_id = "source-session-456"

        # Mark as processed for tenant A
        tenant_a.mark_session_processed(source_session_id)

        # Tenant B should not see it as processed
        assert not tenant_b.is_session_processed(source_session_id)

        # Tenant A should see it as processed
        assert tenant_a.is_session_processed(source_session_id)

    def test_cross_tenant_isolation_background_sync(self, db_path: Path):
        """Background sync status from one tenant should not affect another."""
        tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="project-1")
        tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="project-1")

        # Start sync for tenant A
        tenant_a.start_background_sync(pid=1234)

        # Tenant B should not see running sync
        status_b = tenant_b.get_background_sync_status()
        assert not status_b.is_running

        # Tenant A should see running sync
        status_a = tenant_a.get_background_sync_status()
        assert status_a.is_running
        assert status_a.pid == 1234


class TestRemoteStorageNamespaceValidation:
    """Test RemoteStorage namespace validation."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "local.db"

    @pytest.fixture
    def shared_db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "shared" / "state.db"

    def test_remote_storage_rejects_default_tenant(self, db_path: Path, shared_db_path: Path):
        """RemoteStorage should reject default tenant ID."""
        from agent_recall.storage.models import SharedStorageConfig

        config = SharedStorageConfig(
            base_url=f"file://{shared_db_path.parent}",
            tenant_id="default",
            project_id="my-project",
        )

        with pytest.raises(NamespaceValidationError) as exc_info:
            RemoteStorage(config, local_db_path=db_path)
        assert "tenant_id" in str(exc_info.value)

    def test_remote_storage_rejects_default_project(self, db_path: Path, shared_db_path: Path):
        """RemoteStorage should reject default project ID."""
        from agent_recall.storage.models import SharedStorageConfig

        config = SharedStorageConfig(
            base_url=f"file://{shared_db_path.parent}",
            tenant_id="my-tenant",
            project_id="default",
        )

        with pytest.raises(NamespaceValidationError) as exc_info:
            RemoteStorage(config, local_db_path=db_path)
        assert "project_id" in str(exc_info.value)

    def test_remote_storage_accepts_explicit_namespace(self, db_path: Path, shared_db_path: Path):
        """RemoteStorage should accept explicit namespace."""
        from agent_recall.storage.models import SharedStorageConfig

        shared_db_path.parent.mkdir(parents=True, exist_ok=True)
        config = SharedStorageConfig(
            base_url=f"file://{shared_db_path.parent}",
            tenant_id="org-123",
            project_id="repo-abc",
        )

        # Should not raise
        storage = RemoteStorage(config, local_db_path=db_path)

        # Should be able to write
        session = Session(task="test task")
        storage.create_session(session)

        # Should be able to read back
        retrieved = storage.get_session(session.id)
        assert retrieved is not None
        assert retrieved.tenant_id == "org-123"
        assert retrieved.project_id == "repo-abc"

    def test_remote_storage_isolation(self, db_path: Path, shared_db_path: Path):
        """RemoteStorage should maintain tenant isolation."""
        from agent_recall.storage.models import SharedStorageConfig

        shared_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create two storages with different namespaces
        config_a = SharedStorageConfig(
            base_url=f"file://{shared_db_path.parent}",
            tenant_id="tenant-a",
            project_id="project-1",
        )
        storage_a = RemoteStorage(config_a, local_db_path=None)

        config_b = SharedStorageConfig(
            base_url=f"file://{shared_db_path.parent}",
            tenant_id="tenant-b",
            project_id="project-1",
        )
        storage_b = RemoteStorage(config_b, local_db_path=None)

        # Create session in tenant A
        session = Session(task="secret task")
        storage_a.create_session(session)

        # Tenant B should not see it
        assert storage_b.get_session(session.id) is None

        # Tenant A should see it
        retrieved = storage_a.get_session(session.id)
        assert retrieved is not None


class TestNamespaceLeakageNegativeAssertions:
    """Negative assertions to prove namespace boundaries are enforced."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test.db"

    def test_explicit_cross_tenant_data_leakage_fails(self, db_path: Path):
        """Explicitly test that data does not leak across tenants."""
        tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="shared-project")
        tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="shared-project")

        # Create data in tenant A
        session = Session(task="sensitive task")
        tenant_a.create_session(session)

        entry = LogEntry(
            session_id=session.id,
            source=LogSource.EXPLICIT,
            content="sensitive information",
            label=SemanticLabel.PATTERN,
        )
        tenant_a.append_entry(entry)

        chunk = Chunk(
            source=ChunkSource.LOG_ENTRY,
            source_ids=[entry.id],
            content="sensitive knowledge",
            label=SemanticLabel.PATTERN,
        )
        tenant_a.store_chunk(chunk)

        # Negative assertions - these MUST fail (return None/empty)
        assert tenant_b.get_session(session.id) is None, "Session leaked across tenants"
        assert len(tenant_b.get_entries(session.id)) == 0, "Entries leaked across tenants"
        assert len(tenant_b.search_chunks_fts("sensitive")) == 0, "Chunks leaked across tenants"
        assert tenant_b.count_log_entries() == 0, "Log entry count leaked across tenants"
        assert tenant_b.count_chunks() == 0, "Chunk count leaked across tenants"

    def test_explicit_cross_project_data_leakage_fails(self, db_path: Path):
        """Explicitly test that data does not leak across projects."""
        project_a = SQLiteStorage(db_path, tenant_id="shared-tenant", project_id="project-a")
        project_b = SQLiteStorage(db_path, tenant_id="shared-tenant", project_id="project-b")

        # Create data in project A
        session = Session(task="project-specific task")
        project_a.create_session(session)

        entry = LogEntry(
            session_id=session.id,
            source=LogSource.EXPLICIT,
            content="project-specific information",
            label=SemanticLabel.PATTERN,
        )
        project_a.append_entry(entry)

        chunk = Chunk(
            source=ChunkSource.LOG_ENTRY,
            source_ids=[entry.id],
            content="project-specific knowledge",
            label=SemanticLabel.PATTERN,
        )
        project_a.store_chunk(chunk)

        # Negative assertions - these MUST fail (return None/empty)
        assert project_b.get_session(session.id) is None, "Session leaked across projects"
        assert len(project_b.get_entries(session.id)) == 0, "Entries leaked across projects"
        assert len(project_b.search_chunks_fts("project-specific")) == 0, (
            "Chunks leaked across projects"
        )

    def test_list_sessions_isolated(self, db_path: Path):
        """list_sessions must not return sessions from other tenants."""
        tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="project-1")
        tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="project-1")

        # Create sessions for both tenants
        for i in range(3):
            tenant_a.create_session(Session(task=f"Tenant A Task {i}"))
            tenant_b.create_session(Session(task=f"Tenant B Task {i}"))

        # Each tenant should only see their own sessions
        assert len(tenant_a.list_sessions()) == 3
        assert len(tenant_b.list_sessions()) == 3

        # Verify no cross-contamination
        for session in tenant_a.list_sessions():
            assert session.tenant_id == "tenant-a"
        for session in tenant_b.list_sessions():
            assert session.tenant_id == "tenant-b"

    def test_get_entries_by_label_isolated(self, db_path: Path):
        """get_entries_by_label must not return entries from other tenants."""
        tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="project-1")
        tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="project-1")

        # Create sessions
        session_a = Session(task="Task A")
        session_b = Session(task="Task B")
        tenant_a.create_session(session_a)
        tenant_b.create_session(session_b)

        # Add entries with same label
        tenant_a.append_entry(
            LogEntry(
                session_id=session_a.id,
                source=LogSource.EXPLICIT,
                content="Tenant A pattern",
                label=SemanticLabel.PATTERN,
            )
        )
        tenant_b.append_entry(
            LogEntry(
                session_id=session_b.id,
                source=LogSource.EXPLICIT,
                content="Tenant B pattern",
                label=SemanticLabel.PATTERN,
            )
        )

        # Each tenant should only see their own entries
        entries_a = tenant_a.get_entries_by_label([SemanticLabel.PATTERN])
        entries_b = tenant_b.get_entries_by_label([SemanticLabel.PATTERN])

        assert len(entries_a) == 1
        assert len(entries_b) == 1
        assert entries_a[0].content == "Tenant A pattern"
        assert entries_b[0].content == "Tenant B pattern"
