"""分頁三：影片匯出"""
import logging
import os
import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from config import DEFAULT_VIDEO_RESOLUTION, OUTPUT_DIR
from core.subtitle_generator import generate_srt, save_srt
from core.video_generator import generate_full_video
from ui.widgets import ProgressSection

logger = logging.getLogger(__name__)

_RESOLUTIONS = {
    "1920x1080 (1080p)": (1920, 1080),
    "1280x720 (720p)": (1280, 720),
    "2560x1440 (1440p)": (2560, 1440),
}


class ExportTab:
    """匯出影片分頁"""

    def __init__(self, parent: ctk.CTkFrame, shared_state, app):
        self.parent = parent
        self.state = shared_state
        self.app = app
        self._is_exporting = False

        self._build_ui()

    def _build_ui(self) -> None:
        # ===== 影片設定 =====
        settings = ctk.CTkFrame(self.parent)
        settings.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            settings, text="影片設定",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        # 解析度
        res_row = ctk.CTkFrame(settings, fg_color="transparent")
        res_row.pack(fill="x", padx=10, pady=3)

        ctk.CTkLabel(res_row, text="解析度:").pack(side="left", padx=(0, 5))
        self._res_var = ctk.StringVar(value="1920x1080 (1080p)")
        ctk.CTkOptionMenu(
            res_row, variable=self._res_var,
            values=list(_RESOLUTIONS.keys()),
            width=200,
        ).pack(side="left")

        # ===== 字幕設定 =====
        sub_section = ctk.CTkFrame(self.parent)
        sub_section.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            sub_section, text="字幕選項",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        self._gen_srt_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            sub_section, text="產生 SRT 字幕檔",
            variable=self._gen_srt_var,
        ).pack(anchor="w", padx=15, pady=2)

        self._burn_srt_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            sub_section, text="將字幕燒錄進影片",
            variable=self._burn_srt_var,
        ).pack(anchor="w", padx=15, pady=2)

        # 字型設定
        font_row = ctk.CTkFrame(sub_section, fg_color="transparent")
        font_row.pack(fill="x", padx=15, pady=(2, 8))

        ctk.CTkLabel(font_row, text="字幕字型:").pack(side="left", padx=(0, 5))
        self._font_var = ctk.StringVar(value="Microsoft JhengHei")
        ctk.CTkEntry(font_row, textvariable=self._font_var, width=200).pack(side="left", padx=(0, 10))

        ctk.CTkLabel(font_row, text="字型大小:").pack(side="left", padx=(0, 5))
        self._fontsize_var = ctk.IntVar(value=24)
        ctk.CTkEntry(font_row, textvariable=self._fontsize_var, width=50).pack(side="left")

        # ===== 輸出設定 =====
        output_section = ctk.CTkFrame(self.parent)
        output_section.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            output_section, text="輸出設定",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        path_row = ctk.CTkFrame(output_section, fg_color="transparent")
        path_row.pack(fill="x", padx=10, pady=3)

        ctk.CTkLabel(path_row, text="輸出路徑:").pack(side="left", padx=(0, 5))
        self._output_entry = ctk.CTkEntry(path_row, placeholder_text="選擇輸出資料夾...")
        self._output_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._output_entry.insert(0, str(OUTPUT_DIR))

        ctk.CTkButton(
            path_row, text="選擇...", width=80,
            command=self._browse_output,
        ).pack(side="left")

        name_row = ctk.CTkFrame(output_section, fg_color="transparent")
        name_row.pack(fill="x", padx=10, pady=(3, 8))

        ctk.CTkLabel(name_row, text="檔案名稱:").pack(side="left", padx=(0, 5))
        self._filename_entry = ctk.CTkEntry(name_row, width=300)
        self._filename_entry.pack(side="left")
        self._filename_entry.insert(0, "presentation_narrated")

        # ===== 匯出按鈕 =====
        self._export_btn = ctk.CTkButton(
            self.parent, text="開始匯出影片",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=45,
            command=self._start_export,
        )
        self._export_btn.pack(padx=10, pady=10)

        # 進度
        self._progress = ProgressSection(self.parent)
        self._progress.pack(fill="x", padx=10, pady=(0, 5))

        # ===== 完成區域 =====
        self._done_frame = ctk.CTkFrame(self.parent)
        self._done_frame.pack(fill="x", padx=10, pady=5)

        self._done_label = ctk.CTkLabel(
            self._done_frame, text="",
            font=ctk.CTkFont(size=13),
        )
        self._done_label.pack(anchor="w", padx=10, pady=4)

        done_btns = ctk.CTkFrame(self._done_frame, fg_color="transparent")
        done_btns.pack(fill="x", padx=10, pady=(0, 8))

        self._open_folder_btn = ctk.CTkButton(
            done_btns, text="開啟資料夾", width=100,
            state="disabled", command=self._open_folder,
        )
        self._open_folder_btn.pack(side="left", padx=(0, 5))

        self._play_btn = ctk.CTkButton(
            done_btns, text="播放影片", width=100,
            state="disabled", command=self._play_video,
        )
        self._play_btn.pack(side="left")

        self._output_video_path = ""

    # ----- 操作 -----

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="選擇輸出資料夾")
        if folder:
            self._output_entry.delete(0, "end")
            self._output_entry.insert(0, folder)

    def _start_export(self) -> None:
        if self._is_exporting:
            return

        # 驗證前置條件
        if not self.state.slide_images:
            self._progress.set_status("請先在「匯入」分頁匯入簡報")
            return

        if not self.state.page_audios:
            self._progress.set_status("請先在「語音合成」分頁完成合成")
            return

        if not self.state.script:
            self._progress.set_status("請先驗證講稿")
            return

        self._is_exporting = True
        self._export_btn.configure(state="disabled", text="匯出中...")

        thread = threading.Thread(
            target=self._export_worker,
            daemon=True,
        )
        thread.start()

    def _export_worker(self) -> None:
        try:
            output_dir = self._output_entry.get() or str(OUTPUT_DIR)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            filename = self._filename_entry.get() or "presentation_narrated"
            resolution = _RESOLUTIONS.get(self._res_var.get(), DEFAULT_VIDEO_RESOLUTION)

            # 產生 SRT
            srt_path = None
            if self._gen_srt_var.get():
                srt_content = generate_srt(self.state.script)
                self.state.srt_content = srt_content
                srt_path = str(Path(output_dir) / f"{filename}.srt")
                save_srt(srt_content, srt_path)
                self.state.srt_path = srt_path

            # 取得頁面時長
            page_durations = [dur for _, dur in self.state.page_audios]

            # 產生影片
            video_path = str(Path(output_dir) / f"{filename}.mp4")
            generate_full_video(
                slide_images=self.state.slide_images,
                page_audio_paths=self.state.page_audio_paths,
                page_durations=page_durations,
                srt_path=srt_path,
                output_path=video_path,
                burn_srt=self._burn_srt_var.get(),
                resolution=resolution,
                progress_callback=self._thread_safe_progress,
                sample_rate=self.state.sample_rate,
            )

            self._output_video_path = video_path
            self.parent.after(0, self._on_export_complete, video_path, srt_path)

        except Exception as e:
            logger.error("匯出失敗: %s", e)
            self.parent.after(0, self._on_export_error, str(e))

    def _thread_safe_progress(self, current: int, total: int, message: str) -> None:
        self.parent.after(0, self._progress.update_progress, current, total, message)

    def _on_export_complete(self, video_path: str, srt_path) -> None:
        self._is_exporting = False
        self._export_btn.configure(state="normal", text="開始匯出影片")
        self._progress.set_status("匯出完成")

        msg = f"影片: {video_path}"
        if srt_path:
            msg += f"\n字幕: {srt_path}"
        self._done_label.configure(text=msg, text_color="green")

        self._open_folder_btn.configure(state="normal")
        self._play_btn.configure(state="normal")

    def _on_export_error(self, error: str) -> None:
        self._is_exporting = False
        self._export_btn.configure(state="normal", text="開始匯出影片")
        self._progress.set_status(f"匯出失敗: {error[:100]}")
        self._done_label.configure(text=f"匯出失敗: {error}", text_color="red")

    def _open_folder(self) -> None:
        if self._output_video_path:
            folder = str(Path(self._output_video_path).parent)
            os.startfile(folder)

    def _play_video(self) -> None:
        if self._output_video_path and Path(self._output_video_path).exists():
            os.startfile(self._output_video_path)
