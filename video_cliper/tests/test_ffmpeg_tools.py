from pathlib import Path
from unittest.mock import patch

import pytest

import ffmpeg_tools


def test_parse_probe_json_duration_fps_audio():
    raw = {
        "format": {"duration": "12.5"},
        "streams": [
            {"codec_type": "video", "r_frame_rate": "25/1"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    duration, fps, has_audio = ffmpeg_tools.parse_probe_json(raw)
    assert duration == pytest.approx(12.5)
    assert fps == pytest.approx(25.0)
    assert has_audio is True


def test_parse_probe_json_no_audio_defaults_fps_from_avg_frame_rate():
    raw = {
        "format": {"duration": "3"},
        "streams": [
            {
                "codec_type": "video",
                "r_frame_rate": "0/0",
                "avg_frame_rate": "30000/1001",
            },
        ],
    }
    duration, fps, has_audio = ffmpeg_tools.parse_probe_json(raw)
    assert duration == pytest.approx(3.0)
    assert fps == pytest.approx(30000 / 1001)
    assert has_audio is False


def test_build_ffmpeg_argv_with_audio():
    src = Path("C:/in/movie.mp4")
    dst = Path("D:/out/movie_clip.mp4")
    argv = ffmpeg_tools.build_ffmpeg_argv(src, dst, start=1.5, end=4.0, has_audio=True)
    assert argv[:6] == ["ffmpeg", "-hide_banner", "-y", "-ss", "1.5", "-i"]
    assert Path(argv[6]) == src
    assert argv[7:9] == ["-t", "2.5"]
    assert "-map" in argv
    vi = argv.index("-map")
    assert argv[vi : vi + 4] == ["-map", "0:v:0", "-map", "0:a:0"]
    # 末尾为 -c copy <dst>，最后两项是 "copy" 与输出路径，不是 "-c" 与 "copy"
    assert argv[-3:-1] == ["-c", "copy"]
    assert Path(argv[-1]) == dst


def test_build_ffmpeg_argv_without_audio():
    src = Path("C:/in/nosound.mov")
    dst = Path("C:/in/nosound_clip.mov")
    argv = ffmpeg_tools.build_ffmpeg_argv(src, dst, 0.0, 10.0, has_audio=False)
    joined = " ".join(argv)
    assert "0:a:0" not in joined
    assert argv.count("-map") == 1
    assert argv[argv.index("-map") + 1] == "0:v:0"


@patch("ffmpeg_tools.shutil.which")
def test_check_binaries_ok(mock_which):
    mock_which.side_effect = lambda name: f"C:/fake/{name}.exe"
    ok, msg = ffmpeg_tools.check_binaries()
    assert ok is True
    assert msg == ""


@patch("ffmpeg_tools.shutil.which")
def test_check_binaries_missing(mock_which):
    mock_which.return_value = None
    ok, msg = ffmpeg_tools.check_binaries()
    assert ok is False
    assert "ffmpeg" in msg.lower() or "ffprobe" in msg.lower()
