"""Tests for delta diff renderer module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent_recall.cli.tui.delta import (
    _platform_asset_suffix,
    get_delta_data_dir,
    get_delta_path,
    is_delta_available,
    is_delta_setup_declined,
    write_delta_setup_declined,
)


def test_get_delta_data_dir_returns_path() -> None:
    """get_delta_data_dir returns a Path with bin subdir."""
    data_dir = get_delta_data_dir()
    assert isinstance(data_dir, Path)
    assert (data_dir / "bin").exists()


@pytest.mark.parametrize(
    ("system", "machine", "expected_suffix"),
    [
        ("Darwin", "arm64", "aarch64-apple-darwin.tar.gz"),
        ("Darwin", "aarch64", "aarch64-apple-darwin.tar.gz"),
        ("Darwin", "x86_64", "x86_64-apple-darwin.tar.gz"),
        ("Linux", "x86_64", "x86_64-unknown-linux-gnu.tar.gz"),
        ("Linux", "aarch64", "aarch64-unknown-linux-gnu.tar.gz"),
        ("Windows", "AMD64", "x86_64-pc-windows-msvc.zip"),
    ],
)
def test_platform_asset_suffix(system: str, machine: str, expected_suffix: str) -> None:
    """_platform_asset_suffix maps platform to correct asset."""
    with (
        patch("platform.system", return_value=system),
        patch("platform.machine", return_value=machine),
    ):
        assert _platform_asset_suffix() == expected_suffix


def test_get_delta_path_uses_which_when_in_path() -> None:
    """get_delta_path returns path from shutil.which when delta in PATH."""
    with patch("shutil.which", return_value="/usr/bin/delta"):
        path = get_delta_path()
        assert path is not None
        assert str(path) == "/usr/bin/delta"


def test_get_delta_path_returns_none_when_not_found() -> None:
    """get_delta_path returns None when delta not in PATH or cached."""
    with (
        patch("shutil.which", return_value=None),
        patch("agent_recall.cli.tui.delta._cached_delta_path", return_value=None),
    ):
        assert get_delta_path() is None


def test_is_delta_available() -> None:
    """is_delta_available reflects get_delta_path result."""
    with patch("agent_recall.cli.tui.delta.get_delta_path", return_value=None):
        assert not is_delta_available()
    with patch(
        "agent_recall.cli.tui.delta.get_delta_path",
        return_value=Path("/usr/bin/delta"),
    ):
        assert is_delta_available()


def test_setup_declined_roundtrip(tmp_path: Path) -> None:
    """write_delta_setup_declined and is_delta_setup_declined roundtrip."""
    with patch(
        "agent_recall.cli.tui.delta.get_delta_data_dir",
        return_value=tmp_path,
    ):
        assert not is_delta_setup_declined()
        write_delta_setup_declined()
        assert is_delta_setup_declined()
