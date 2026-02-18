"""Delta diff renderer: detection, download, and path resolution."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
from platformdirs import user_data_dir

DELTA_RELEASES_URL = "https://api.github.com/repos/dandavison/delta/releases/latest"
DELTA_MANUAL_INSTALL_URL = "https://github.com/dandavison/delta/releases"
APP_NAME = "agent-recall"
APP_AUTHOR = "natecard"


def get_delta_data_dir() -> Path:
    """Return the app data directory for delta binary and config."""
    data_dir = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    (data_dir / "bin").mkdir(parents=True, exist_ok=True)
    return data_dir


def _read_setup_status() -> str | None:
    """Read delta_setup status from config.json."""
    data_dir = get_delta_data_dir()
    config_path = data_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text())
        return data.get("delta_setup")
    except (json.JSONDecodeError, OSError):
        return None


def _write_setup_status(status: str) -> None:
    """Persist delta_setup status to config.json."""
    data_dir = get_delta_data_dir()
    config_path = data_dir / "config.json"
    data: dict[str, object] = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    data["delta_setup"] = status
    config_path.write_text(json.dumps(data, indent=2))


def _cached_delta_path() -> Path | None:
    """Return path to cached delta binary if it exists and is executable."""
    data_dir = get_delta_data_dir()
    bin_dir = data_dir / "bin"
    # Windows uses delta.exe
    if platform.system() == "Windows":
        delta_path = bin_dir / "delta.exe"
    else:
        delta_path = bin_dir / "delta"
    if delta_path.exists() and os.access(delta_path, os.X_OK):
        return delta_path
    return None


def get_delta_path() -> Path | None:
    """Return path to delta if found (system PATH or cached binary)."""
    which_path = shutil.which("delta")
    if which_path:
        return Path(which_path)
    return _cached_delta_path()


def is_delta_available() -> bool:
    """Return True if delta is available (system or cached)."""
    return get_delta_path() is not None


def is_delta_setup_declined() -> bool:
    """Return True if user previously declined delta setup."""
    return _read_setup_status() == "declined"


def write_delta_setup_declined() -> None:
    """Persist that user declined delta setup."""
    _write_setup_status("declined")


def reset_delta_setup() -> None:
    """Clear delta setup status and cached binary so the first-launch prompt shows again."""
    data_dir = get_delta_data_dir()
    bin_dir = data_dir / "bin"
    config_path = data_dir / "config.json"

    # Remove cached binary
    for name in ("delta", "delta.exe"):
        path = bin_dir / name
        if path.exists():
            path.unlink()

    # Clear delta_setup from config
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            data.pop("delta_setup", None)
            if data:
                config_path.write_text(json.dumps(data, indent=2))
            else:
                config_path.unlink()
        except (json.JSONDecodeError, OSError):
            config_path.unlink(missing_ok=True)


def _platform_asset_suffix() -> str | None:
    """Map platform to delta release asset filename suffix. Returns None if unsupported."""
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return "aarch64-apple-darwin.tar.gz"
        if machine == "x86_64":
            return "x86_64-apple-darwin.tar.gz"
    elif system == "Linux":
        if machine == "x86_64":
            return "x86_64-unknown-linux-gnu.tar.gz"
        if machine in ("aarch64", "arm64"):
            return "aarch64-unknown-linux-gnu.tar.gz"
        if machine == "armv7l":
            return "arm-unknown-linux-gnueabihf.tar.gz"
        if machine == "i686":
            return "i686-unknown-linux-gnu.tar.gz"
    elif system == "Windows":
        if machine == "amd64":
            return "x86_64-pc-windows-msvc.zip"
        if machine == "x86_64":
            return "x86_64-pc-windows-msvc.zip"

    return None


def _verify_checksum(data: bytes, expected_sha256: str) -> bool:
    """Verify SHA256 of downloaded data."""
    digest = hashlib.sha256(data).hexdigest()
    return digest.lower() == expected_sha256.lower()


def _extract_binary(
    archive_path: Path,
    dest_dir: Path,
    progress_callback: Callable[[float, str], None] | None,
) -> Path:
    """Extract delta binary from tar.gz or zip to dest_dir. Returns path to binary."""
    is_windows = platform.system() == "Windows"
    binary_name = "delta.exe" if is_windows else "delta"
    final_path = dest_dir / binary_name

    def move_to_final(extracted: Path) -> Path:
        if extracted != final_path:
            final_path.unlink(missing_ok=True)
            extracted.rename(final_path)
        return final_path

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            for i, member in enumerate(names):
                if progress_callback and len(names) > 1:
                    progress_callback((i + 1) / len(names), "Extracting...")
                if member.endswith(binary_name):
                    zf.extract(member, dest_dir)
                    return move_to_final(dest_dir / member)
        raise RuntimeError(f"delta binary not found in {archive_path}")
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            members = tf.getnames()
            for i, member in enumerate(members):
                if progress_callback and len(members) > 1:
                    progress_callback((i + 1) / len(members), "Extracting...")
                if member.endswith(binary_name):
                    tf.extract(member, dest_dir)
                    return move_to_final(dest_dir / member)
        raise RuntimeError(f"delta binary not found in {archive_path}")


class DeltaDownloadError(Exception):
    """Raised when delta download or setup fails."""

    pass


def check_network() -> bool:
    """Return True if network appears reachable (GitHub)."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get("https://api.github.com")
            return resp.status_code < 500
    except Exception:
        return False


def download_delta(
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    """
    Download delta binary for current platform to app data dir.

    Returns path to delta binary on success.
    Raises DeltaDownloadError on failure.
    """
    suffix = _platform_asset_suffix()
    if not suffix:
        raise DeltaDownloadError(
            f"Unsupported platform: {platform.system()} / {platform.machine()}"
        )

    data_dir = get_delta_data_dir()
    bin_dir = data_dir / "bin"
    archive_ext = ".zip" if suffix.endswith(".zip") else ".tar.gz"
    tmp_path = bin_dir / f"delta_download{archive_ext}.tmp"

    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            resp = client.get(DELTA_RELEASES_URL)
            resp.raise_for_status()
            release = resp.json()

        version = release.get("tag_name") or release.get("name", "0.18.2")
        if version.startswith("v"):
            version = version[1:]
        asset_name = f"delta-{version}-{suffix}"

        download_url: str | None = None
        for asset in release.get("assets", []):
            if asset.get("name") == asset_name:
                download_url = asset.get("browser_download_url")
                break

        if not download_url:
            raise DeltaDownloadError(
                f"No release asset for {asset_name}. Install manually: {DELTA_MANUAL_INSTALL_URL}"
            )

        # Try to fetch checksums (checksums.txt format: hash  filename)
        base_url = f"https://github.com/dandavison/delta/releases/download/{version}"
        checksums_url = f"{base_url}/checksums.txt"
        expected_sha256: str | None = None
        try:
            with httpx.Client(follow_redirects=True, timeout=15.0) as client:
                cs_resp = client.get(checksums_url)
                if cs_resp.status_code == 200 and asset_name in cs_resp.text:
                    for line in cs_resp.text.splitlines():
                        if asset_name not in line:
                            continue
                        for part in line.split():
                            if len(part) == 64 and all(
                                c in "0123456789abcdef" for c in part.lower()
                            ):
                                expected_sha256 = part
                                break
        except Exception:
            pass

        archive_path = bin_dir / f"delta_download{archive_ext}"

        if progress_callback:
            progress_callback(0.0, "Downloading...")
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            with client.stream("GET", download_url) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total > 0:
                            progress_callback(downloaded / total, "Downloading...")

        data = tmp_path.read_bytes()
        if expected_sha256 and not _verify_checksum(data, expected_sha256):
            tmp_path.unlink(missing_ok=True)
            raise DeltaDownloadError("Checksum verification failed")

        tmp_path.rename(archive_path)
        try:
            delta_path = _extract_binary(archive_path, bin_dir, progress_callback)
            os.chmod(delta_path, 0o755)
            _write_setup_status("completed")
            return delta_path
        finally:
            archive_path.unlink(missing_ok=True)

    except httpx.HTTPStatusError as e:
        tmp_path.unlink(missing_ok=True)
        raise DeltaDownloadError(
            f"Download failed: {e.response.status_code}. "
            f"Install manually: {DELTA_MANUAL_INSTALL_URL}"
        ) from e
    except httpx.RequestError as e:
        tmp_path.unlink(missing_ok=True)
        raise DeltaDownloadError(
            f"Network error: {e}. Install manually: {DELTA_MANUAL_INSTALL_URL}"
        ) from e
