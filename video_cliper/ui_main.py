from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Literal

import cv2
from PIL import Image, ImageTk

import config_store
import ffmpeg_tools
from preview import VideoPreview

DragMode = Literal["start", "end", "playhead"] | None


class Application:
    """主界面：主预览 → 合并条（抽帧 + 裁剪范围 + 播放头）。"""

    _PAD = 12
    _HANDLE = 12
    _MIN_GAP_SEC = 0.05
    _STRIP_TOP = 34
    _THUMB_H = 86
    _STRIP_H = 128

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("video_cliper")
        root.minsize(880, 700)

        self.preview = VideoPreview()
        self.src_path: Path | None = None
        self.duration_sec: float = 0.0
        self.fps: float = 25.0
        self.has_audio: bool = False
        self.current_sec: float = 0.0
        self.range_start_sec: float = 0.0
        self.range_end_sec: float = 0.0

        self._play_after_id: str | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self._strip_photos: list[ImageTk.PhotoImage] = []
        self._strip_building = False
        self._strip_rebuild_after: str | None = None
        self._native_w: int = 1920
        self._native_h: int = 1080
        self._strip_thumb_w: int = 0
        self._strip_thumb_h: int = 0
        self._play_button_text = tk.StringVar(value="播放")

        self._drag: DragMode = None
        self._drag_offset_sec: float = 0.0

        self.status = tk.StringVar(value="就绪")
        self.time_var = tk.StringVar(
            value="播放头 -- | 开始 -- | 结束 -- | 总长 --（缩略图与视频同宽高比；青/玫手柄=起止；白线=播放头；蓝框=选中段）"
        )

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
        ttk.Button(bar, text="上一帧", command=self._step_prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="下一帧", command=self._step_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, textvariable=self._play_button_text, command=self._toggle_play).pack(
            side=tk.LEFT, padx=2
        )
        self.export_btn = ttk.Button(bar, text="导出片段…", command=self._on_export, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=2)

        self.canvas = tk.Canvas(self.root, height=360, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.canvas.bind("<Configure>", self._on_main_canvas_configure)

        ttk.Label(
            self.root,
            text="裁剪条（缩略图与视频同比例；拖动青/玫菱形手柄；点条上其它位置移动播放头；播放仅在蓝框内循环）",
        ).pack(anchor=tk.W, padx=10, pady=(4, 0))

        self.range_strip = tk.Canvas(
            self.root,
            height=self._STRIP_H,
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self.range_strip.pack(fill=tk.X, padx=10, pady=4)
        self.range_strip.bind("<Configure>", self._on_range_strip_configure)
        for seq in ("<Button-1>", "<B1-Motion>", "<ButtonRelease-1>"):
            self.range_strip.bind(seq, self._make_strip_handler(seq))

        ttk.Label(self.root, textvariable=self.time_var, anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(self.root, textvariable=self.status, anchor=tk.W).pack(fill=tk.X, padx=6, pady=4)

        self._draw_placeholder()

    def _on_range_strip_configure(self, _event: tk.Event) -> None:
        self._paint_range_strip()
        if self.src_path is None or self._strip_building:
            return
        if self._strip_rebuild_after is not None:
            self.root.after_cancel(self._strip_rebuild_after)
        self._strip_rebuild_after = self.root.after(350, self._debounced_strip_rebuild)

    def _debounced_strip_rebuild(self) -> None:
        self._strip_rebuild_after = None
        self._schedule_strip_rebuild()

    def _make_strip_handler(self, seq: str):
        def handler(e: tk.Event) -> None:
            if seq == "<Button-1>":
                self._strip_press(e)
            elif seq == "<B1-Motion>":
                self._strip_motion(e)
            else:
                self._strip_release(e)

        return handler

    def _on_main_canvas_configure(self, _event: tk.Event) -> None:
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

    def _usable_strip_width(self) -> float:
        w = max(self.range_strip.winfo_width(), 2 * self._PAD + 4)
        return max(1.0, w - 2 * self._PAD)

    def _x_to_time(self, x: float) -> float:
        if self.duration_sec <= 0:
            return 0.0
        u = (x - self._PAD) / self._usable_strip_width()
        u = max(0.0, min(1.0, u))
        return u * self.duration_sec

    def _time_to_x(self, t: float) -> float:
        if self.duration_sec <= 0:
            return self._PAD
        u = max(0.0, min(1.0, t / self.duration_sec))
        return self._PAD + u * self._usable_strip_width()

    def _thumb_y(self) -> int:
        return self._STRIP_TOP

    def _thumb_display_h(self) -> int:
        return self._strip_thumb_h if self._strip_thumb_h > 0 else self._THUMB_H

    def _draw_trim_handle(self, c: tk.Canvas, x: float, kind: str) -> None:
        """起止标记：菱形 + 细柄，避免夸张箭头。"""
        ty = self._thumb_y()
        cy = ty - 14
        r = 8
        sx, sy = 1, 1
        shadow = (x + sx, cy - r + sy, x + r + sx, cy + sy, x + sx, cy + r + sy, x - r + sx, cy + sy)
        body = (x, cy - r, x + r, cy, x, cy + r, x - r, cy)
        if kind == "start":
            fill, outline = "#2dd4bf", "#0d9488"
        else:
            fill, outline = "#fb7185", "#e11d48"
        c.create_polygon(*shadow, fill="#0b0b0b", outline="")
        c.create_polygon(*body, fill=fill, outline=outline, width=2)
        c.create_line(x, cy - r + 3, x, ty - 2, fill=outline, width=3, capstyle=tk.ROUND)

    def _hit_test(self, x: float) -> DragMode:
        if self.duration_sec <= 0:
            return None
        xs = self._time_to_x(self.range_start_sec)
        xe = self._time_to_x(self.range_end_sec)
        xp = self._time_to_x(self.current_sec)
        h = self._HANDLE
        if abs(x - xs) <= h:
            return "start"
        if abs(x - xe) <= h:
            return "end"
        if abs(x - xp) <= h:
            return "playhead"
        return None

    def _strip_press(self, e: tk.Event) -> None:
        self._stop_play()
        mode = self._hit_test(float(e.x))
        if mode is None:
            self.current_sec = self._x_to_time(float(e.x))
            self.preview.seek_seconds(self.current_sec)
            self._show_frame()
            mode = "playhead"
        self._drag = mode
        if mode == "start":
            self._drag_offset_sec = self.range_start_sec - self._x_to_time(float(e.x))
        elif mode == "end":
            self._drag_offset_sec = self.range_end_sec - self._x_to_time(float(e.x))
        else:
            self._drag_offset_sec = self.current_sec - self._x_to_time(float(e.x))
        self._paint_range_strip()

    def _strip_motion(self, e: tk.Event) -> None:
        if self._drag is None or self.duration_sec <= 0:
            return
        t = self._x_to_time(float(e.x)) + self._drag_offset_sec
        t = max(0.0, min(self.duration_sec, t))
        if self._drag == "start":
            self.range_start_sec = min(t, self.range_end_sec - self._MIN_GAP_SEC)
            self.range_start_sec = max(0.0, self.range_start_sec)
        elif self._drag == "end":
            self.range_end_sec = max(t, self.range_start_sec + self._MIN_GAP_SEC)
            self.range_end_sec = min(self.duration_sec, self.range_end_sec)
        else:
            self.current_sec = t
            self.preview.seek_seconds(self.current_sec)
            self._show_frame()
        self._paint_range_strip()
        self._refresh_time_label()
        self._refresh_export_button_state()

    def _strip_release(self, _e: tk.Event) -> None:
        self._drag = None
        self._drag_offset_sec = 0.0

    def _paint_range_strip(self) -> None:
        c = self.range_strip
        c.delete("all")
        w = max(c.winfo_width(), 4)
        h = c.winfo_height()
        c.create_rectangle(0, 0, w, h, fill="#1e1e1e", outline="")

        if self.duration_sec <= 0:
            c.create_text(w // 2, h // 2, text="打开视频后显示抽帧与裁剪范围", fill="#aaa")
            return

        ty = self._thumb_y()
        th = self._thumb_display_h()
        x0s = self._time_to_x(self.range_start_sec)
        x1s = self._time_to_x(self.range_end_sec)

        c.create_line(self._PAD, ty - 4, self._PAD + self._usable_strip_width(), ty - 4, fill="#3f3f46", width=1)

        # 左右非选中区压暗（与缩略图同一坐标系）
        c.create_rectangle(self._PAD, ty, x0s, ty + th, fill="#0a0a0a", outline="")
        c.create_rectangle(x1s, ty, self._PAD + self._usable_strip_width(), ty + th, fill="#0a0a0a", outline="")

        if self._strip_photos:
            n = len(self._strip_photos)
            for i, ph in enumerate(self._strip_photos):
                t_c = (i + 0.5) * self.duration_sec / max(n, 1)
                xc = self._time_to_x(t_c)
                x_img = int(xc - ph.width() // 2)
                c.create_image(x_img, ty, anchor=tk.NW, image=ph)
            c.create_rectangle(x0s, ty, x1s, ty + th, outline="#7dd3fc", width=2)
        else:
            c.create_text(
                w // 2,
                ty + th // 2,
                text="正在生成缩略图…" if self._strip_building else "缩略图未就绪",
                fill="#888",
            )

        xp = self._time_to_x(self.current_sec)
        c.create_line(xp, 6, xp, ty + th + 2, fill="#f8fafc", width=2)
        c.create_oval(xp - 4, 4, xp + 4, 12, fill="#f8fafc", outline="#94a3b8")

        self._draw_trim_handle(c, x0s, "start")
        self._draw_trim_handle(c, x1s, "end")

    def _refresh_time_label(self) -> None:
        dur = self.duration_sec
        self.time_var.set(
            f"播放头 {self._format_time(self.current_sec)} | 开始 {self._format_time(self.range_start_sec)} | "
            f"结束 {self._format_time(self.range_end_sec)} | 总长 {self._format_time(dur)}"
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
        self.duration_sec = max(duration, 0.0)
        self.has_audio = has_audio
        self.fps = max(self.preview.effective_fps, 1e-6)
        self.current_sec = 0.0
        self.range_start_sec = 0.0
        self.range_end_sec = max(self.duration_sec, self._MIN_GAP_SEC)
        if self.range_end_sec <= self.range_start_sec:
            self.range_end_sec = self.range_start_sec + self._MIN_GAP_SEC
        self.preview.seek_seconds(0.0)
        self._native_w = int(self.preview.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._native_h = int(self.preview.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if self._native_w <= 0 or self._native_h <= 0:
            ok_f, fr = self.preview.read_frame_bgr()
            if ok_f and fr is not None:
                self._native_h, self._native_w = fr.shape[:2]
            else:
                self._native_w, self._native_h = 1920, 1080
            self.preview.seek_seconds(0.0)
        self._strip_photos.clear()
        self._strip_thumb_w = 0
        self._strip_thumb_h = 0
        self._refresh_export_button_state()
        self._show_frame()
        self._paint_range_strip()
        self._schedule_strip_rebuild()
        self.status.set(f"已打开: {self.src_path.name}")

    def _schedule_strip_rebuild(self) -> None:
        if self.src_path is None or self.duration_sec <= 0:
            self.range_strip.delete("all")
            return
        if self._strip_building:
            return
        self._strip_building = True
        self.status.set("正在生成缩略图…")
        w = max(self.range_strip.winfo_width(), 400)
        pad = self._PAD
        usable = max(1.0, w - 2 * pad)
        nat_w = max(1, self._native_w)
        nat_h = max(1, self._native_h)
        thumb_h = self._THUMB_H
        thumb_w = max(1, int(round(thumb_h * float(nat_w) / float(nat_h))))
        gap = 4
        n = max(4, min(96, int(usable // max(1, thumb_w + gap))))
        path = str(self.src_path)
        duration = float(self.duration_sec)

        def worker() -> None:
            thumbs: list[Image.Image] = []
            cap = cv2.VideoCapture(path)
            try:
                if not cap.isOpened():
                    return
                for i in range(n):
                    t = (i + 0.5) * duration / max(n, 1)
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        continue
                    fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(fr_rgb).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                    thumbs.append(img)
            finally:
                cap.release()

            def apply() -> None:
                self._strip_building = False
                self._strip_photos.clear()
                self._strip_thumb_w = thumb_w
                self._strip_thumb_h = thumb_h
                for im in thumbs:
                    self._strip_photos.append(ImageTk.PhotoImage(image=im))
                if not self._strip_photos:
                    self.status.set("缩略图生成失败")
                else:
                    self.status.set(f"已打开: {self.src_path.name}（缩略 {thumb_w}×{thumb_h}，与视频 {nat_w}×{nat_h} 同比例）")
                self._paint_range_strip()

            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _step_prev(self) -> None:
        if self.src_path is None:
            return
        self._stop_play()
        self.preview.step_frames(-1)
        cur_ms = float(self.preview.cap.get(cv2.CAP_PROP_POS_MSEC)) if self.preview.cap else 0.0
        self.current_sec = max(0.0, min(cur_ms / 1000.0, self.duration_sec))
        self._show_frame()
        self._paint_range_strip()

    def _step_next(self) -> None:
        if self.src_path is None:
            return
        self._stop_play()
        self.preview.step_frames(1)
        cur_ms = float(self.preview.cap.get(cv2.CAP_PROP_POS_MSEC)) if self.preview.cap else 0.0
        self.current_sec = max(0.0, min(cur_ms / 1000.0, self.duration_sec))
        self._show_frame()
        self._paint_range_strip()

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
        rs, re = self.range_start_sec, self.range_end_sec
        if re <= rs + 1e-6:
            self.status.set("裁剪范围无效：结束应大于开始")
            return
        if self.current_sec < rs or self.current_sec >= re:
            self.current_sec = rs
            self.preview.seek_seconds(self.current_sec)
            self._show_frame()
        self._play_button_text.set("暂停")
        interval = max(int(1000 / self.fps), 1)
        self._play_after_id = self.root.after(interval, self._play_tick)

    def _play_tick(self) -> None:
        if self._play_after_id is None:
            return
        rs, re = self.range_start_sec, self.range_end_sec
        self.current_sec += 1.0 / self.fps
        if self.current_sec >= re:
            self.current_sec = rs
        self.preview.seek_seconds(self.current_sec)
        self._show_frame()
        self._paint_range_strip()
        interval = max(int(1000 / self.fps), 1)
        self._play_after_id = self.root.after(interval, self._play_tick)

    def _refresh_export_button_state(self) -> None:
        ok = self.duration_sec > 0 and self.range_end_sec > self.range_start_sec
        self.export_btn.configure(state=tk.NORMAL if ok else tk.DISABLED)

    def _on_export(self) -> None:
        if not self._require_binaries_or_warn():
            return
        if self.src_path is None:
            return
        if self.range_end_sec <= self.range_start_sec:
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
            self.range_start_sec,
            self.range_end_sec,
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
