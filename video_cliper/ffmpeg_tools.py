from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, List, Mapping, Tuple


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
