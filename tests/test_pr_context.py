from __future__ import annotations

from uuid import UUID

from agent_recall.core.pr_context import filter_chunks_for_scope, parse_name_status_lines
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def test_parse_name_status_lines_handles_rename_and_move() -> None:
    scope = parse_name_status_lines(
        lines=[
            "R100\tsrc/old_name.py\tsrc/new_name.py",
            "M\tsrc/core/retrieve.py",
            "A\tdocs/notes.md",
        ],
        base_ref="origin/main",
        head_ref="HEAD",
    )
    assert ("src/old_name.py", "src/new_name.py") in scope.renamed
    assert "src/new_name.py" in scope.files
    assert "src/core/retrieve.py" in scope.files
    assert scope.added == 1
    assert scope.modified == 2


def test_parse_name_status_lines_truncates_large_diff() -> None:
    lines = [f"M\tsrc/module_{index}.py" for index in range(300)]
    scope = parse_name_status_lines(
        lines=lines,
        base_ref="origin/main",
        head_ref="HEAD",
        max_files=40,
    )
    assert len(scope.files) == 40
    assert scope.truncated is True


def test_filter_chunks_for_scope_uses_files_and_modules() -> None:
    matching = Chunk(
        id=UUID("00000000-0000-0000-0000-000000000051"),
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="update src/api client timeout",
        label=SemanticLabel.PATTERN,
        tags=["src/api/client.py"],
    )
    non_matching = Chunk(
        id=UUID("00000000-0000-0000-0000-000000000052"),
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="database transaction note",
        label=SemanticLabel.PATTERN,
        tags=["db"],
    )
    scope = parse_name_status_lines(
        lines=["M\tsrc/api/client.py"],
        base_ref="origin/main",
        head_ref="HEAD",
    )
    filtered = filter_chunks_for_scope([matching, non_matching], scope)
    assert [chunk.id for chunk in filtered] == [matching.id]
