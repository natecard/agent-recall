from __future__ import annotations

import sys
import time
from pathlib import Path

from agent_recall.cli.stream_pipeline import run_streaming_command, stream_debug_dir


def test_stream_pipeline_emits_partial_without_newline(tmp_path: Path) -> None:
    fragments: list[tuple[float, str]] = []
    started = time.monotonic()
    cmd = [
        sys.executable,
        "-c",
        "import sys,time;"
        "sys.stdout.write('partial');"
        "sys.stdout.flush();"
        "time.sleep(0.65);"
        "sys.stdout.write(' done\\n');"
        "sys.stdout.flush()",
    ]

    exit_code = run_streaming_command(
        cmd,
        cwd=tmp_path,
        on_emit=lambda fragment: fragments.append((time.monotonic() - started, fragment)),
        context="test_partial_emit",
        partial_flush_ms=100,
    )

    assert exit_code == 0
    first_partial = next((t for t, chunk in fragments if "partial" in chunk), None)
    assert first_partial is not None
    assert first_partial < 0.55
    assert "".join(chunk for _, chunk in fragments).endswith(" done\n")


def test_stream_pipeline_preserves_carriage_return_order(tmp_path: Path) -> None:
    emitted: list[str] = []
    cmd = [
        sys.executable,
        "-c",
        "import sys,time;"
        "sys.stdout.write('spin-1\\r');"
        "sys.stdout.flush();"
        "time.sleep(0.05);"
        "sys.stdout.write('spin-2\\r');"
        "sys.stdout.flush();"
        "time.sleep(0.05);"
        "sys.stdout.write('done\\n');"
        "sys.stdout.flush()",
    ]

    exit_code = run_streaming_command(
        cmd,
        cwd=tmp_path,
        on_emit=emitted.append,
        context="test_cr_order",
        partial_flush_ms=250,
    )

    assert exit_code == 0
    assert "".join(emitted) == "spin-1\rspin-2\rdone\n"


def test_stream_pipeline_handles_split_utf8_bytes(tmp_path: Path) -> None:
    emitted: list[str] = []
    cmd = [
        sys.executable,
        "-c",
        "import sys,time;"
        "sys.stdout.buffer.write(b'caf\\xc3');"
        "sys.stdout.flush();"
        "time.sleep(0.05);"
        "sys.stdout.buffer.write(b'\\xa9\\n');"
        "sys.stdout.flush()",
    ]

    exit_code = run_streaming_command(
        cmd,
        cwd=tmp_path,
        on_emit=emitted.append,
        context="test_utf8_split",
    )

    assert exit_code == 0
    assert "".join(emitted) == "café\n"


def test_stream_pipeline_flushes_final_fragment_at_eof(tmp_path: Path) -> None:
    emitted: list[str] = []
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('tail-fragment'); sys.stdout.flush()",
    ]

    exit_code = run_streaming_command(
        cmd,
        cwd=tmp_path,
        on_emit=emitted.append,
        context="test_eof_flush",
    )

    assert exit_code == 0
    assert "".join(emitted) == "tail-fragment"


def test_stream_pipeline_preserves_non_zero_exit_code(tmp_path: Path) -> None:
    emitted: list[str] = []
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('x'); sys.stdout.flush(); sys.exit(7)",
    ]

    exit_code = run_streaming_command(
        cmd,
        cwd=tmp_path,
        on_emit=emitted.append,
        context="test_exit_code",
    )

    assert exit_code == 7
    assert "".join(emitted) == "x"


def test_stream_pipeline_writes_debug_artifacts_and_prunes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_RECALL_RALPH_STREAM_DEBUG", "1")
    monkeypatch.setenv("AGENT_RECALL_RALPH_STREAM_DEBUG_MAX_MB", "0.001")
    out_dir = stream_debug_dir(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(30):
        stem = f"stream-old-{index:03d}"
        (out_dir / f"{stem}.jsonl").write_text("{}", encoding="utf-8")
        (out_dir / f"{stem}.raw.log").write_text("old", encoding="utf-8")

    emitted: list[str] = []
    cmd = [
        sys.executable,
        "-c",
        "import sys;sys.stdout.write('x' * 4000);sys.stdout.flush()",
    ]

    exit_code = run_streaming_command(
        cmd,
        cwd=tmp_path,
        on_emit=emitted.append,
        context="test_debug_artifacts",
        partial_flush_ms=50,
    )

    assert exit_code == 0
    bundles = sorted(out_dir.glob("stream-*.jsonl"))
    assert len(bundles) <= 25
    newest = max(bundles, key=lambda path: path.stat().st_mtime)
    raw_path = out_dir / f"{newest.stem}.raw.log"
    assert raw_path.exists()
    # 0.001 MiB ~= 1048 bytes; allow a small filesystem rounding margin.
    assert raw_path.stat().st_size <= 1200
    latest_lines = newest.read_text(encoding="utf-8").splitlines()
    assert latest_lines
    assert '"event": "complete"' in latest_lines[-1]
