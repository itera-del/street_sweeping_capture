from __future__ import annotations

import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, List, Mapping, Tuple

_FFMPEG_STATUS_TIME = re.compile(r"\btime=(\d+):(\d+):(\d+(?:\.\d+)?)\b")


def _frac_to_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except (ZeroDivisionError, ValueError, TypeError):
        return None


def parse_probe_json(data: Mapping[str, Any]) -> Tuple[float, float, bool]:
    """Parse ffprobe JSON output.

    Returns:
        duration_sec: from format.duration
        fps: first video stream, prefer r_frame_rate then avg_frame_rate, else 25.0
        has_audio: True if any stream has codec_type == 'audio'
    """
    fmt = data.get("format") or {}
    duration = float(fmt.get("duration", 0) or 0)

    streams = list(data.get("streams") or [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    fps = 25.0
    for s in streams:
        if s.get("codec_type") != "video":
            continue
        fps_candidate = _frac_to_float(str(s.get("r_frame_rate", "")))
        if fps_candidate and fps_candidate > 0:
            fps = fps_candidate
            break
        fps_candidate = _frac_to_float(str(s.get("avg_frame_rate", "")))
        if fps_candidate and fps_candidate > 0:
            fps = fps_candidate
            break

    return duration, float(fps), has_audio


def check_binaries() -> tuple[bool, str]:
    if not shutil.which("ffmpeg"):
        return False, "未在 PATH 中找到 ffmpeg。请安装 FFmpeg 或先执行 conda activate video_process。"
    if not shutil.which("ffprobe"):
        return False, "未在 PATH 中找到 ffprobe。请安装 FFmpeg 或先执行 conda activate video_process。"
    return True, ""


def build_ffmpeg_argv(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    *,
    has_audio: bool,
) -> List[str]:
    if end <= start:
        raise ValueError("end 必须大于 start")
    duration = end - start
    argv: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(src),
        "-t",
        str(duration),
        "-map",
        "0:v:0",
    ]
    if has_audio:
        argv += ["-map", "0:a:0"]
    argv += ["-c", "copy", str(dst)]
    return argv


def probe_media(path: Path) -> tuple[float, float, bool]:
    """Run ffprobe and return (duration, fps, has_audio)."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe 失败")
    data = json.loads(proc.stdout)
    return parse_probe_json(data)


def run_ffmpeg(argv: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _ffmpeg_status_time_sec(fragment: str) -> float | None:
    """Parse ``time=HH:MM:SS.xx`` from a ffmpeg status fragment (stderr)."""
    m = _FFMPEG_STATUS_TIME.search(fragment)
    if not m:
        return None
    h, m_, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600.0 + m_ * 60.0 + s


def run_ffmpeg_with_progress(
    argv: List[str],
    *,
    clip_duration_sec: float,
    on_progress: Callable[[float], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg, read stderr for ``time=`` and report approximate progress 0..1.

    ``clip_duration_sec`` should match the segment length (e.g. end - start) so
    progress maps to output timeline. Safe for ``-c copy`` (time still advances).
    """
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stderr_parts: list[str] = []
    duration = max(float(clip_duration_sec), 1e-9)

    try:
        assert proc.stderr is not None
        buffer = ""
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            stderr_parts.append(chunk)
            buffer += chunk
            while True:
                if "\r" in buffer:
                    line, buffer = buffer.split("\r", 1)
                elif "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                else:
                    break
                if on_progress is None:
                    continue
                t = _ffmpeg_status_time_sec(line)
                if t is None:
                    continue
                on_progress(min(1.0, max(0.0, t / duration)))
    finally:
        if proc.stderr is not None:
            proc.stderr.close()

    rc = proc.wait()
    stderr_full = "".join(stderr_parts)
    return subprocess.CompletedProcess(argv, rc, None, stderr_full)
