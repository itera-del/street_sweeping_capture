from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import config_store
import ffmpeg_tools
from preview import VideoPreview


class Application:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("video_cliper")
        root.minsize(880, 620)

        self.preview = VideoPreview()
        self.src_path: Path | None = None
        self.duration_sec: float = 0.0
        self.fps: float = 25.0
        self.has_audio: bool = False
        self.current_sec: float = 0.0
        self.mark_in: float | None = None
        self.mark_out: float | None = None
        self._play_after_id: str | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self._play_button_text = tk.StringVar(value="播放")

        self.timeline_var = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="就绪")
        self.time_var = tk.StringVar(value="当前 00:00:00.000 | 总长 -- | 入 -- | 出 --")

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
        ttk.Button(bar, text="设入点", command=self._mark_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="设出点", command=self._mark_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="跳到入点", command=self._goto_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="跳到出点", command=self._goto_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="上一帧", command=self._step_prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="下一帧", command=self._step_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, textvariable=self._play_button_text, command=self._toggle_play).pack(
            side=tk.LEFT, padx=2
        )
        self.export_btn = ttk.Button(bar, text="导出片段…", command=self._on_export, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=2)

        self.canvas = tk.Canvas(self.root, height=400, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.pos_scale = tk.Scale(
            self.root,
            from_=0.0,
            to=1.0,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.timeline_var,
            command=self._on_scale,
            showvalue=0,
        )
        self.pos_scale.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(self.root, textvariable=self.time_var, anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(self.root, textvariable=self.status, anchor=tk.W).pack(fill=tk.X, padx=6, pady=4)

        self._draw_placeholder()

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        if self.src_path is not None:
            self._show_frame()

    def _require_binaries_or_warn(self) -> bool:
        ok, msg = ffmpeg_tools.check_binaries()
        if not ok:
            messagebox.showerror("无法继续", msg)
            return False
        return True

    def _format_time(self, sec: float) -> str:
        sec = max(0.0, sec)
        ms = int(round((sec - int(sec)) * 1000)) % 1000
        s = int(sec) % 60
        m = (int(sec) // 60) % 60
        h = int(sec) // 3600
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _refresh_time_label(self) -> None:
        dur = self.duration_sec
        mi_s = "--" if self.mark_in is None else self._format_time(self.mark_in)
        mo_s = "--" if self.mark_out is None else self._format_time(self.mark_out)
        self.time_var.set(
            f"当前 {self._format_time(self.current_sec)} | 总长 {self._format_time(dur)} | 入 {mi_s} | 出 {mo_s}"
        )

    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width(), 2)
        ch = max(self.canvas.winfo_height(), 2)
        self.canvas.create_text(
            cw // 2,
            ch // 2,
            text="请打开视频文件",
            fill="white",
            font=("Segoe UI", 14),
        )
        self._refresh_time_label()

    def _on_scale(self, value: str) -> None:
        self._stop_play()
        self.current_sec = float(value)
        if self.duration_sec > 0:
            self.current_sec = max(0.0, min(self.current_sec, self.duration_sec))
        self.preview.seek_seconds(self.current_sec)
        self._show_frame()

    def _show_frame(self) -> None:
        if self.preview.cap is None:
            self._draw_placeholder()
            return

        ok, frame = self.preview.read_frame_bgr()
        if not ok or frame is None:
            self._draw_placeholder()
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

    def _on_open(self) -> None:
        if not self._require_binaries_or_warn():
            return
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频", "*.mp4 *.mov *.mkv *.avi *.webm"), ("全部", "*.*")],
        )
        if not path:
            return
        self._stop_play()
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
        self.has_audio = has_audio
        self.fps = max(self.preview.effective_fps, 1e-6)
        self.mark_in = None
        self.mark_out = None
        self.current_sec = 0.0
        self.preview.seek_seconds(0.0)
        self.timeline_var.set(0.0)
        self.pos_scale.configure(to=max(self.duration_sec, 0.001))
        self._refresh_export_button_state()
        self._show_frame()
        self.status.set(f"已打开: {self.src_path.name}")

    def _mark_in(self) -> None:
        if self.src_path is None:
            return
        self.mark_in = self.current_sec
        if self.mark_out is not None and self.mark_out <= self.mark_in:
            self.mark_out = None
        self._refresh_export_button_state()
        self._refresh_time_label()

    def _mark_out(self) -> None:
        if self.src_path is None:
            return
        if self.mark_in is not None and self.current_sec <= self.mark_in:
            self.status.set("出点应大于入点")
            self._refresh_time_label()
            return
        self.mark_out = self.current_sec
        self._refresh_export_button_state()
        self._refresh_time_label()

    def _goto_in(self) -> None:
        if self.mark_in is None:
            return
        self._stop_play()
        self.timeline_var.set(self.mark_in)
        self._on_scale(str(self.mark_in))

    def _goto_out(self) -> None:
        if self.mark_out is None:
            return
        self._stop_play()
        self.timeline_var.set(self.mark_out)
        self._on_scale(str(self.mark_out))

    def _step_prev(self) -> None:
        if self.src_path is None:
            return
        self._stop_play()
        self.preview.step_frames(-1)
        cur_ms = float(self.preview.cap.get(cv2.CAP_PROP_POS_MSEC)) if self.preview.cap else 0.0
        self.current_sec = max(0.0, min(cur_ms / 1000.0, self.duration_sec))
        self.timeline_var.set(self.current_sec)
        self._show_frame()

    def _step_next(self) -> None:
        if self.src_path is None:
            return
        self._stop_play()
        self.preview.step_frames(1)
        cur_ms = float(self.preview.cap.get(cv2.CAP_PROP_POS_MSEC)) if self.preview.cap else 0.0
        self.current_sec = max(0.0, min(cur_ms / 1000.0, self.duration_sec))
        self.timeline_var.set(self.current_sec)
        self._show_frame()

    def _stop_play(self) -> None:
        if self._play_after_id is not None:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None
        self._play_button_text.set("播放")

    def _toggle_play(self) -> None:
        if self.src_path is None:
            return
        if self._play_after_id is not None:
            self._stop_play()
            return
        self._play_button_text.set("暂停")
        interval = max(int(1000 / self.fps), 1)
        self._play_after_id = self.root.after(interval, self._play_tick)

    def _play_tick(self) -> None:
        if self._play_after_id is None:
            return
        self.current_sec += 1.0 / self.fps
        if self.current_sec >= self.duration_sec:
            self._stop_play()
            self.current_sec = max(0.0, self.duration_sec - 1e-3)
            self.preview.seek_seconds(self.current_sec)
            self.timeline_var.set(self.current_sec)
            self._show_frame()
            return
        self.preview.seek_seconds(self.current_sec)
        self.timeline_var.set(self.current_sec)
        self._show_frame()
        interval = max(int(1000 / self.fps), 1)
        self._play_after_id = self.root.after(interval, self._play_tick)

    def _refresh_export_button_state(self) -> None:
        ok = (
            self.mark_in is not None
            and self.mark_out is not None
            and self.mark_out > self.mark_in
        )
        self.export_btn.configure(state=tk.NORMAL if ok else tk.DISABLED)

    def _on_export(self) -> None:
        if not self._require_binaries_or_warn():
            return
        if self.src_path is None or self.mark_in is None or self.mark_out is None:
            return
        if self.mark_out <= self.mark_in:
            return

        initialdir = config_store.get_last_export_dir()
        if not initialdir:
            initialdir = str(self.src_path.parent)
        default_name = f"{self.src_path.stem}_clip{self.src_path.suffix}"
        dst = filedialog.asksaveasfilename(
            title="导出为",
            initialdir=initialdir,
            initialfile=default_name,
            defaultextension=self.src_path.suffix or ".mp4",
            filetypes=[("视频", "*.mp4 *.mov *.mkv *.avi"), ("全部", "*.*")],
        )
        if not dst:
            return

        dst_path = Path(dst)
        argv = ffmpeg_tools.build_ffmpeg_argv(
            self.src_path,
            dst_path,
            self.mark_in,
            self.mark_out,
            has_audio=self.has_audio,
        )
        self.status.set("正在导出…")
        threading.Thread(
            target=self._export_worker,
            args=(argv, dst_path),
            daemon=True,
        ).start()

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
