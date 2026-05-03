# video_cliper GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在仓库 `video_cliper/` 下实现基于 tkinter 的单段视频裁剪桌面应用（静音 OpenCV 预览、PATH 上 ffmpeg/ffprobe stream copy 导出），行为对齐 `docs/superpowers/specs/2026-05-03-video-cliper-design.md`。

**Architecture:** 纯 Python 模块拆分：`ffmpeg_tools` 负责 PATH 检测、ffprobe JSON 解析与 ffmpeg 参数表；`config_store` 读写 `%USERPROFILE%\.video_cliper\config.json`；`preview` 封装 `cv2.VideoCapture` 的 seek/帧步进；`ui_main` 组合 tkinter 控件并在后台线程跑导出，经 `after` 回写 UI。

**Tech Stack:** Python 3.11+、tkinter、opencv-python、pytest（仅单元测试）、子进程调用系统 `ffmpeg`/`ffprobe`。

**工作目录约定：** 在 Git worktree `E:\Projects\.worktrees\video-cliper` 上检出分支 `feature/video-cliper`；本计划涉及路径均以仓库内 **`video_cliper/`** 子目录为根（即 `video_cliper/ffmpeg_tools.py` 指 `E:\Projects\.worktrees\video-cliper\video_cliper\ffmpeg_tools.py`）。

---

## 文件结构（将创建 / 修改）

| 文件 | 职责 |
|------|------|
| `video_cliper/requirements.txt` | 运行时依赖：`opencv-python` |
| `video_cliper/requirements-dev.txt` | 开发依赖：`pytest` |
| `video_cliper/pytest.ini` | `pythonpath = .` 以便 `import ffmpeg_tools` |
| `video_cliper/ffmpeg_tools.py` | `check_binaries`、`probe_media`（子进程 ffprobe）、`parse_probe_json`、`build_ffmpeg_argv`、`run_ffmpeg` |
| `video_cliper/config_store.py` | `config_path()`、`load_config`、`save_last_export_dir`、`get_last_export_dir` |
| `video_cliper/preview.py` | `VideoPreview`：`open`、`close`、`seek_seconds`、`step_frames`、`read_frame_bgr`、`effective_fps`、`duration_sec` |
| `video_cliper/ui_main.py` | `Application(tk.Tk)`：工具栏、Canvas、Scale、只读标签、状态栏、导出线程入口 |
| `video_cliper/main.py` | `main()`：建根窗、`Application`、启动 `mainloop` |
| `video_cliper/__main__.py` | `python -m video_cliper` 时调用 `main.main()`（需包结构，见 Task 1 调整） |
| `video_cliper/tests/test_ffmpeg_tools.py` | 解析与 argv 单元测试 |
| `video_cliper/tests/test_config_store.py` | 配置读写测试（临时目录 monkeypatch） |

**包导入说明：** 为支持 `python -m video_cliper` 与清晰导入，Task 1 采用在 `video_cliper/` 下增加空包目录 `video_cliper/video_cliper_pkg/` 易与文件夹重名混淆。**改为扁平方案：** 所有模块放在 `video_cliper/` 根目录（`ffmpeg_tools.py` 等与文件夹同名冲突——仓库根是 `street_sweeping_capture` 下的 `video_cliper` 子文件夹，其内部文件名为 `ffmpeg_tools.py` 不与外层目录冲突）。入口使用 `video_cliper/main.py`，运行方式 `python main.py` 自 `video_cliper` 目录，或从仓库根 `python -m video_cliper.main` 需在 `video_cliper` 旁加 `pyproject` 包——**最终采用：** 在 `video_cliper` 目录内放 `main.py`，文档写 `cd video_cliper` 后 `python main.py`。**不添加** `__main__.py` 包入口以降低复杂度（YAGNI）。

---

### Task 1: 依赖与 pytest 脚手架

**Files:**
- Create: `video_cliper/requirements.txt`
- Create: `video_cliper/requirements-dev.txt`
- Create: `video_cliper/pytest.ini`
- Create: `video_cliper/tests/__init__.py`（空文件）

- [ ] **Step 1.1: 写入 requirements.txt**

```text
opencv-python>=4.8.0,<5
```

- [ ] **Step 1.2: 写入 requirements-dev.txt**

```text
pytest>=7.4.0,<9
```

- [ ] **Step 1.3: 写入 pytest.ini**

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 1.4: 创建空 tests/__init__.py**

```python

```

- [ ] **Step 1.5: 安装并验证 pytest**

Run（在已激活的 conda `video_process` 或等价环境中，先 `cd` 到本计划 `video_cliper` 目录）:

```powershell
Set-Location E:\Projects\.worktrees\video-cliper\video_cliper
pip install -r requirements.txt -r requirements-dev.txt
pytest --collect-only
```

Expected: `collected 0 items`（尚无测试）或已存在测试则显示收集数量；无 ImportError。

- [ ] **Step 1.6: Commit**

```bash
git add video_cliper/requirements.txt video_cliper/requirements-dev.txt video_cliper/pytest.ini video_cliper/tests/__init__.py
git commit -m "chore(video_cliper): add deps and pytest scaffold"
```

---

### Task 2: ffprobe 解析（纯函数 + TDD）

**Files:**
- Create: `video_cliper/ffmpeg_tools.py`（先只含解析函数）
- Create: `video_cliper/tests/test_ffmpeg_tools.py`

- [ ] **Step 2.1: 写入失败测试 `tests/test_ffmpeg_tools.py`**

```python
import json

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
```

- [ ] **Step 2.2: 运行测试确认失败**

Run:

```powershell
Set-Location E:\Projects\.worktrees\video-cliper\video_cliper
pytest tests/test_ffmpeg_tools.py -v
```

Expected: `ImportError` 或 `AttributeError: module 'ffmpeg_tools' has no attribute 'parse_probe_json'`.

- [ ] **Step 2.3: 在 `ffmpeg_tools.py` 实现 `parse_probe_json`**

```python
from __future__ import annotations

import json
from fractions import Fraction
from typing import Any, Mapping, Tuple


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
```

- [ ] **Step 2.4: 运行测试通过**

Run:

```powershell
pytest tests/test_ffmpeg_tools.py::test_parse_probe_json_duration_fps_audio tests/test_ffmpeg_tools.py::test_parse_probe_json_no_audio_defaults_fps_from_avg_frame_rate -v
```

Expected: `2 passed`.

- [ ] **Step 2.5: Commit**

```bash
git add video_cliper/ffmpeg_tools.py video_cliper/tests/test_ffmpeg_tools.py
git commit -m "feat(video_cliper): add ffprobe JSON parse helper with tests"
```

---

### Task 3: `check_binaries` 与 `build_ffmpeg_argv`

**Files:**
- Modify: `video_cliper/ffmpeg_tools.py`（追加函数）
- Modify: `video_cliper/tests/test_ffmpeg_tools.py`（追加测试）

- [ ] **Step 3.1: 追加测试**

在 `tests/test_ffmpeg_tools.py` 末尾追加（若文件顶部尚无 `Path` / `patch`，在文件顶部补充 `from pathlib import Path` 与 `from unittest.mock import patch`，勿重复 import `pytest` / `ffmpeg_tools`）：

```python
from pathlib import Path
from unittest.mock import patch


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
    assert argv[-2:] == ["-c", "copy"]
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
```

- [ ] **Step 3.2: 运行新测试失败**

Run: `pytest tests/test_ffmpeg_tools.py::test_build_ffmpeg_argv_with_audio -v`  
Expected: `AttributeError: module 'ffmpeg_tools' has no attribute 'build_ffmpeg_argv'`.

- [ ] **Step 3.3: 在 `ffmpeg_tools.py` 追加实现**

在文件顶部增加 `import shutil`，在 `parse_probe_json` 之后追加：

```python
import shutil
from pathlib import Path
from typing import List


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
```

同时在文件顶部确保已有 `import json` 与 `import subprocess`（与 `parse_probe_json` 同文件合并 import，避免重复）。

- [ ] **Step 3.4: 运行 `pytest tests/test_ffmpeg_tools.py -v`**

Expected: 全部通过（含 Task 2 测试）。

- [ ] **Step 3.5: Commit**

```bash
git add video_cliper/ffmpeg_tools.py video_cliper/tests/test_ffmpeg_tools.py
git commit -m "feat(video_cliper): binary check, ffprobe run, ffmpeg argv builder"
```

---

### Task 4: `config_store`（TDD）

**Files:**
- Create: `video_cliper/config_store.py`
- Create: `video_cliper/tests/test_config_store.py`

- [ ] **Step 4.1: 写入测试**

```python
import json
from pathlib import Path

import pytest

import config_store


def test_save_and_get_last_export_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config_store, "config_path", lambda: tmp_path / "cfg.json")
    config_store.save_last_export_dir(str(tmp_path / "outdir"))
    assert config_store.get_last_export_dir() == str(tmp_path / "outdir")


def test_get_last_export_dir_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config_store, "config_path", lambda: tmp_path / "nope.json")
    assert config_store.get_last_export_dir() is None
```

- [ ] **Step 4.2: pytest 失败**

Run: `pytest tests/test_config_store.py -v` → 缺模块失败。

- [ ] **Step 4.3: 实现 `config_store.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def config_path() -> Path:
    return Path.home() / ".video_cliper" / "config.json"


def _read_all() -> Dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def get_last_export_dir() -> Optional[str]:
    value = _read_all().get("last_export_dir")
    if isinstance(value, str) and value:
        return value
    return None


def save_last_export_dir(directory: str) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_all()
    data["last_export_dir"] = directory
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4.4: pytest 通过**

Run: `pytest tests/test_config_store.py -v` → `2 passed`.

- [ ] **Step 4.5: Commit**

```bash
git add video_cliper/config_store.py video_cliper/tests/test_config_store.py
git commit -m "feat(video_cliper): persist last export directory"
```

---

### Task 5: `preview.VideoPreview`

**Files:**
- Create: `video_cliper/preview.py`
- Create: `video_cliper/tests/test_preview.py`

- [ ] **Step 5.1: 测试（mock cv2.VideoCapture）**

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import preview


@patch("preview.cv2.VideoCapture")
def test_seek_seconds_sets_pos_msec(mock_cap_cls):
    instance = MagicMock()
    instance.isOpened.return_value = True
    instance.read.side_effect = [
        (True, np.zeros((2, 2, 3), dtype=np.uint8)),
    ]
    mock_cap_cls.return_value = instance

    vp = preview.VideoPreview()
    vp.open("C:/fake/video.mp4")
    vp.seek_seconds(1.25)
    instance.set.assert_called()
    # CAP_PROP_POS_MSEC = 0
    assert any(c[0][0] == 0 and abs(c[0][1] - 1250.0) < 1 for c in instance.set.call_args_list)
```

**说明：** OpenCV 常量 `cv2.CAP_PROP_POS_MSEC` 在运行时为 `0`。若实现用命名更清晰，测试中断言 `call_args` 与实现一致即可。

- [ ] **Step 5.2: 实现 `preview.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoPreview:
    cap: Optional[cv2.VideoCapture] = None
    duration_sec: float = 0.0
    effective_fps: float = 25.0

    def open(self, path: str, *, duration_sec: float, fps: float) -> bool:
        self.close()
        self.duration_sec = float(duration_sec)
        self.effective_fps = float(fps) if fps > 0 else 25.0
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            return False
        cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if cap_fps > 0:
            self.effective_fps = cap_fps
        return True

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def seek_seconds(self, t: float) -> None:
        if self.cap is None:
            return
        t = max(0.0, min(t, self.duration_sec))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)

    def read_frame_bgr(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def step_frames(self, delta_frames: int) -> None:
        if self.cap is None or delta_frames == 0:
            return
        dt = delta_frames / self.effective_fps
        # 近似：用当前毫秒位置 + dt
        cur_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        self.seek_seconds(cur_ms / 1000.0 + dt)
```

**测试调整：** `VideoPreview` 为 dataclass 时 `vp = VideoPreview()` 然后 `open`；测试中 `instance.set` 应对 `cv2.CAP_PROP_POS_MSEC`（值为 0）。将 Step 5.1 中断言改为：

```python
    vp = preview.VideoPreview()
    assert vp.open("C:/fake/video.mp4", duration_sec=10.0, fps=30.0) is True
    vp.seek_seconds(1.25)
    instance.set.assert_called()
```

并在 `instance.set` 的调用里查找 `(0, pytest.approx(1250.0))`（第一个参数为 CAP_PROP_POS_MSEC）。

- [ ] **Step 5.3: pytest `tests/test_preview.py`**

Expected: `1 passed`（若常量不匹配则修正测试与实现对齐）。

- [ ] **Step 5.4: Commit**

```bash
git add video_cliper/preview.py video_cliper/tests/test_preview.py
git commit -m "feat(video_cliper): OpenCV preview wrapper"
```

---

### Task 6: `ui_main.Application` 骨架与二进制检测

**Files:**
- Create: `video_cliper/ui_main.py`
- Create: `video_cliper/main.py`

- [ ] **Step 6.1: `ui_main.py` 骨架（类与启动警告）**

```python
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import ffmpeg_tools
import config_store
from preview import VideoPreview


class Application:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("video_cliper")
        self.preview = VideoPreview()
        self.src_path: Path | None = None
        self.duration_sec: float = 0.0
        self.fps: float = 25.0
        self.has_audio: bool = False
        self.current_sec: float = 0.0
        self.mark_in: float | None = None
        self.mark_out: float | None = None
        self._play_after_id: str | None = None
        self._build_ui()
        self._startup_binary_warning()

    def _startup_binary_warning(self) -> None:
        ok, msg = ffmpeg_tools.check_binaries()
        if not ok:
            messagebox.showwarning("FFmpeg 不可用", msg)

    def _build_ui(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(bar, text="打开视频…", command=self._on_open).pack(side=tk.LEFT, padx=2)
        self.status = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status, anchor=tk.W).pack(fill=tk.X, padx=6, pady=2)

    def _require_binaries_or_warn(self) -> bool:
        ok, msg = ffmpeg_tools.check_binaries()
        if not ok:
            messagebox.showerror("无法继续", msg)
            return False
        return True

    def _on_open(self) -> None:
        if not self._require_binaries_or_warn():
            return
        path = filedialog.askopenfilename(title="选择视频", filetypes=[("视频", "*.mp4 *.mov *.mkv *.*"), ("全部", "*.*")])
        if not path:
            return
        self.status.set("正在分析…")
        self.root.update_idletasks()
        try:
            duration, fps, has_audio = ffmpeg_tools.probe_media(Path(path))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("ffprobe 失败", str(exc))
            self.status.set("就绪")
            return
        opened = self.preview.open(path, duration_sec=duration, fps=fps)
        if not opened:
            messagebox.showerror("预览失败", "OpenCV 无法打开该文件。")
            self.status.set("就绪")
            return
        self.src_path = Path(path)
        self.duration_sec = duration
        self.fps = fps
        self.has_audio = has_audio
        self.current_sec = 0.0
        self.preview.seek_seconds(0.0)
        self.status.set(f"已打开: {self.src_path.name}")
```

后续 Task 继续在同一文件追加 Canvas、Scale、按钮与 `_on_export`；本步先保证能弹警告与打开文件探测。

- [ ] **Step 6.2: `main.py`**

```python
import tkinter as tk

from ui_main import Application


def main() -> None:
    root = tk.Tk()
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.3: 手工验证**

Run（需 PATH 含 ffmpeg/ffprobe，且 conda 已激活）:

```powershell
Set-Location E:\Projects\.worktrees\video-cliper\video_cliper
python main.py
```

Expected: 窗口出现；若未配置 FFmpeg 则弹出警告；「打开视频」可选文件并在 ffprobe 成功时状态栏显示已打开。

- [ ] **Step 6.4: Commit**

```bash
git add video_cliper/ui_main.py video_cliper/main.py
git commit -m "feat(video_cliper): app shell and open+probe flow"
```

---

### Task 7: 预览 Canvas、缩放、Scale 与只读时间

**Files:**
- Modify: `video_cliper/ui_main.py`

- [ ] **Step 7.1: 增加依赖 `from PIL import Image, ImageTk`**

在 `requirements.txt` 追加一行：`Pillow>=10.0.0,<12`（tkinter 显示用）。

- [ ] **Step 7.2: 在 `Application._build_ui` 中于 `bar` 之下加入：**

- `self.canvas`（`tk.Canvas`，高度约 360，背景黑）
- `self.time_var`（`tk.StringVar`）绑定 `ttk.Label` 显示 `当前 / 总长 / 入 / 出`
- `self.pos_scale`（`tk.Scale`，`from_=0`, `to=1`, `resolution=0.001`, `orient=tk.HORIZONTAL`, `command=self._on_scale`) —— **注意：** 初始 `to=1`，在打开文件后设为 `duration`。

实现辅助方法：

```python
    def _format_time(self, sec: float) -> str:
        sec = max(0.0, sec)
        ms = int(round((sec - int(sec)) * 1000))
        s = int(sec) % 60
        m = (int(sec) // 60) % 60
        h = int(sec) // 3600
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _refresh_time_label(self) -> None:
        dur = self.duration_sec
        mi = self.mark_in if self.mark_in is not None else float("nan")
        mo = self.mark_out if self.mark_out is not None else float("nan")
        if self.mark_in is None:
            mi_s = "--"
        else:
            mi_s = self._format_time(self.mark_in)
        if self.mark_out is None:
            mo_s = "--"
        else:
            mo_s = self._format_time(self.mark_out)
        self.time_var.set(
            f"当前 {self._format_time(self.current_sec)} | 总长 {self._format_time(dur)} | 入 {mi_s} | 出 {mo_s}"
        )

    def _on_scale(self, value: str) -> None:
        self._stop_play()
        self.current_sec = float(value)
        self.preview.seek_seconds(self.current_sec)
        self._show_frame()

    def _show_frame(self) -> None:
        ok, frame = self.preview.read_frame_bgr()
        if not ok or frame is None:
            return
        # BGR->RGB，按 canvas 宽度缩放，ImageTk.PhotoImage 保持引用 self._tk_image
        ...
        self._refresh_time_label()
```

**` _show_frame` 完整实现（粘贴进类中）:**

```python
    def _show_frame(self) -> None:
        import cv2
        from PIL import Image, ImageTk

        ok, frame = self.preview.read_frame_bgr()
        if not ok or frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cw = max(self.canvas.winfo_width(), 2)
        ch = max(self.canvas.winfo_height(), 2)
        h, w, _ = rgb.shape
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR)
        self._tk_image = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self._tk_image)
        self._refresh_time_label()
```

在 `_on_open` 成功末尾：配置 `self.pos_scale.configure(to=self.duration_sec)`、`self.pos_scale.set(0)`、调用 `_show_frame()`。

- [ ] **Step 7.3: 手工拖动 Scale 验证画面变化**

- [ ] **Step 7.4: Commit**

```bash
git add video_cliper/ui_main.py video_cliper/requirements.txt
git commit -m "feat(video_cliper): canvas preview and timeline scale"
```

---

### Task 8: 入点/出点、逐帧、播放/暂停、导出按钮状态

**Files:**
- Modify: `video_cliper/ui_main.py`

- [ ] **Step 8.1: 工具栏按钮与逻辑（文字对齐设计）**

在 `bar` 上依次 `pack`：`设入点` `设出点` `跳到入点` `跳到出点` `上一帧` `下一帧` `播放/暂停` `导出片段…`，`command` 分别绑定 `_mark_in` `_mark_out` `_goto_in` `_goto_out` `_step_prev` `_step_next` `_toggle_play` `_on_export`。

规则：

- `_mark_in`：`self.mark_in = self.current_sec`；若 `mark_out` 不为 None且 `<= mark_in`，清空 `mark_out`；`_refresh_export_button_state()`。
- `_mark_out`：`self.mark_out = self.current_sec`；若 `mark_in` 不为 None且 `>= mark_out`，状态栏提示 `出点应大于入点` 并 return。
- `_goto_in/_goto_out`：若对应 mark 为 None 则 return；`self.pos_scale.set(t)` 会触发 `_on_scale`。
- `_step_prev/_step_next`：`_stop_play()`；`delta = ±1` 帧，`self.current_sec = clamp(self.current_sec + delta/self.fps, 0, duration)`；**注意：** 规格里 fps 优先 ffprobe；此处用 `self.fps`（打开时从 probe 写入）与 `preview.effective_fps` 一致即可在 `_on_open` 设置 `self.fps = self.preview.effective_fps`。
- `_toggle_play`：`after` 每 `max(int(1000/self.fps),1)` ms 调 `_play_tick`；`_play_tick`：`self.current_sec += 1/self.fps`；若 `>= duration`：停止并 `seek duration`；更新 scale 与 `_show_frame`。
- `_stop_play`：若有 `after` id 则 `after_cancel`。
- `_refresh_export_button_state`：`state=tk.NORMAL` 当 `mark_in is not None and mark_out is not None and mark_out > mark_in` 否则 `DISABLED`。

- [ ] **Step 8.2: 手工验证出点≤入点时导出灰显**

- [ ] **Step 8.3: Commit**

```bash
git add video_cliper/ui_main.py
git commit -m "feat(video_cliper): marks, frame step, play/pause, export gating"
```

---

### Task 9: 导出线程与另存为

**Files:**
- Modify: `video_cliper/ui_main.py`

- [ ] **Step 9.1: 实现 `_on_export`**

逻辑要点：

1. `if not self._require_binaries_or_warn(): return`
2. 若导出按钮 disabled（双保险）：`mark_in` / `mark_out` 无效则 return。
3. `initialdir = config_store.get_last_export_dir() or (self.src_path.parent if self.src_path else Path.home())`
4. `default = f"{self.src_path.stem}_clip{self.src_path.suffix}"`
5. `dst = filedialog.asksaveasfilename(..., initialdir=str(initialdir), initialfile=default, defaultextension=self.src_path.suffix)`
6. 若无路径 return。
7. `argv = ffmpeg_tools.build_ffmpeg_argv(self.src_path, Path(dst), self.mark_in, self.mark_out, has_audio=self.has_audio)` —— **类型：** `mark_in`/`mark_out` 已断言非 None。
8. `threading.Thread(target=self._export_worker, args=(argv, Path(dst)), daemon=True).start()`；`self.status.set("正在导出…")`。

`_export_worker`：

```python
    def _export_worker(self, argv: list[str], dst: Path) -> None:
        proc = ffmpeg_tools.run_ffmpeg(argv)

        def _done() -> None:
            if proc.returncode == 0:
                self.status.set(f"导出完成: {dst.name}")
                config_store.save_last_export_dir(str(dst.parent))
            else:
                err = (proc.stderr or "").strip()
                tail = "\n".join(err.splitlines()[-40:])
                messagebox.showerror("ffmpeg 失败", tail or f"退出码 {proc.returncode}")
                self.status.set("导出失败")

        self.root.after(0, _done)
```

- [ ] **Step 9.2: 手工导出一段 MP4 并用播放器检查时长（约等于 end-start）**

- [ ] **Step 9.3: Commit**

```bash
git add video_cliper/ui_main.py
git commit -m "feat(video_cliper): threaded export and save dialog"
```

---

### Task 10: 收尾与自测矩阵

- [ ] **Step 10.1: 全量 pytest**

Run:

```powershell
Set-Location E:\Projects\.worktrees\video-cliper\video_cliper
pytest -q
```

Expected: 全部 green。

- [ ] **Step 10.2: 手工矩阵（记录结果于 PR 描述或本地笔记）**

- H.264 + AAC 的 MP4：打开、拖条、入出点、导出、播放。
- 无音轨 MOV（若有样例）：确认 `build_ffmpeg_argv` 无 `0:a:0` 且可播放。
- 临时从 PATH 移除 ffmpeg：启动警告 + 打开/导出被拦截。

- [ ] **Step 10.3: 推送功能分支**

```powershell
Set-Location E:\Projects\.worktrees\video-cliper
git push -u origin feature/video-cliper
```

- [ ] **Step 10.4: Commit 若有文档小修**

仅当修正计划与代码不一致时提交。

---

## 计划自检（已对 spec 覆盖）

| 设计章节 | 对应 Task |
|----------|-----------|
| PATH 检测与阻断 | Task 3、6、9 |
| ffprobe 与 OpenCV 打开 | Task 2–3、5–6 |
| `[start,end)` 与 argv | Task 3、9 |
| 静音预览与缩放 | Task 5、7 |
| 入出点仅按钮、无手填 | Task 8 |
| 配置 `last_export_dir` | Task 4、9 |
| 导出线程 + stderr | Task 9 |
| 非目标（不重编码 fallback） | 未实现（符合 YAGNI） |

**占位符扫描：** 本计划不含 TBD/TODO 式步骤；测试代码已给完整片段。

**类型一致性：** `VideoPreview.open(path, duration_sec=..., fps=...)` 与测试、UI 调用需一致；若重命名方法，同步修改 Task 5 测试与 Task 6/7 调用。

---

**计划已保存到：** `video_cliper/docs/superpowers/plans/2026-05-03-video-cliper-gui.md`（物理路径：`E:\Projects\.worktrees\video-cliper\video_cliper\docs\superpowers\plans\2026-05-03-video-cliper-gui.md`）。

**执行方式二选一：**

1. **Subagent-Driven（推荐）** — 每个 Task 派生子代理执行，任务间人工/代理复核，迭代快。  
2. **Inline Execution** — 在本会话用 executing-plans 按 Task 批量执行并设检查点。

你要用 **1** 还是 **2**？
