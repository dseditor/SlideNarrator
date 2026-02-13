"""分頁三：影片匯出"""
import logging
import os
import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from config import DEFAULT_VIDEO_RESOLUTION, OUTPUT_DIR
from core.subtitle_generator import generate_srt, save_srt
from core.video_generator import (
    EncoderConfig,
    SW_ENCODER,
    detect_available_encoders,
    generate_full_video,
    get_encoder_by_name,
)
from ui.widgets import ProgressSection

logger = logging.getLogger(__name__)

_RESOLUTIONS = {
    "1920x1080 (1080p)": (1920, 1080),
    "1280x720 (720p)": (1280, 720),
    "2560x1440 (1440p)": (2560, 1440),
}

# 硬體類型對應的狀態圖標
_HW_ICONS = {
    "nvidia": "🟢 NVIDIA",
    "intel": "🟢 Intel",
    "amd": "🟢 AMD",
    "sw": "⚪ CPU",
}


class ExportTab:
    """匯出影片分頁"""

    def __init__(self, parent: ctk.CTkFrame, shared_state, app):
        self.parent = parent
        self.state = shared_state
        self.app = app
        self._is_exporting = False

        # 編碼器偵測結果
        self._available_encoders: list[EncoderConfig] = [SW_ENCODER]
        self._encoder_detected = False

        self._build_ui()
        self._start_encoder_detection()

    def _build_ui(self) -> None:
        # ===== 編碼器狀態 =====
        encoder_section = ctk.CTkFrame(self.parent)
        encoder_section.pack(fill="x", padx=10, pady=(10, 5))

        header_row = ctk.CTkFrame(encoder_section, fg_color="transparent")
        header_row.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            header_row, text="影片編碼器",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="left")

        self._detect_status_label = ctk.CTkLabel(
            header_row, text="  ⏳ 偵測中...",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._detect_status_label.pack(side="left", padx=(8, 0))

        # 編碼器選擇
        enc_row = ctk.CTkFrame(encoder_section, fg_color="transparent")
        enc_row.pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkLabel(enc_row, text="編碼器:").pack(side="left", padx=(0, 5))
        self._encoder_var = ctk.StringVar(value=SW_ENCODER.name)
        self._encoder_menu = ctk.CTkOptionMenu(
            enc_row, variable=self._encoder_var,
            values=[SW_ENCODER.name],
            width=280,
        )
        self._encoder_menu.pack(side="left")

        # 硬體加速狀態標籤
        self._hw_status_frame = ctk.CTkFrame(encoder_section, fg_color="transparent")
        self._hw_status_frame.pack(fill="x", padx=10, pady=(0, 8))

        self._hw_status_label = ctk.CTkLabel(
            self._hw_status_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self._hw_status_label.pack(anchor="w")

        # ===== 影片設定 =====
        settings = ctk.CTkFrame(self.parent)
        settings.pack(fill="x", padx=10, pady=5)

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

    # ----- 編碼器偵測 -----

    def _start_encoder_detection(self) -> None:
        """背景執行硬體編碼器偵測（不阻塞 UI）"""
        thread = threading.Thread(
            target=self._detect_encoders_worker,
            daemon=True,
        )
        thread.start()

    def _detect_encoders_worker(self) -> None:
        """背景 worker：偵測可用編碼器"""
        try:
            encoders = detect_available_encoders()
            self.parent.after(0, self._on_detection_complete, encoders)
        except Exception as e:
            logger.error("編碼器偵測失敗: %s", e)
            self.parent.after(0, self._on_detection_complete, [SW_ENCODER])

    def _on_detection_complete(self, encoders: list) -> None:
        """偵測完成後更新 UI（在主執行緒）"""
        self._available_encoders = encoders
        self._encoder_detected = True

        encoder_names = [e.name for e in encoders]
        self._encoder_menu.configure(values=encoder_names)

        # 自動選擇最佳的編碼器（列表第一個，即優先硬體）
        best = encoders[0]
        self._encoder_var.set(best.name)

        # 更新狀態標籤
        hw_encoders = [e for e in encoders if e.hw_type != "sw"]
        if hw_encoders:
            hw_names = ", ".join(_HW_ICONS.get(e.hw_type, e.name) for e in hw_encoders)
            self._detect_status_label.configure(
                text=f"  ✅ 已偵測到硬體加速",
                text_color="#2ecc71",
            )
            self._hw_status_label.configure(
                text=f"可用: {hw_names}  |  {_HW_ICONS['sw']} 軟體備援",
                text_color=None,
            )
        else:
            self._detect_status_label.configure(
                text="  ℹ️ 僅軟體編碼",
                text_color="orange",
            )
            self._hw_status_label.configure(
                text="未偵測到硬體加速（NVIDIA/Intel/AMD），使用 CPU 軟體編碼",
                text_color="gray",
            )

        logger.info("UI 編碼器更新: 選擇 %s", best.name)

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

            # 取得使用者選擇的編碼器
            selected_encoder = get_encoder_by_name(
                self._encoder_var.get(),
                self._available_encoders,
            )
            logger.info("匯出使用編碼器: %s", selected_encoder.name)

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
                encoder=selected_encoder,
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
        self._progress.update_progress(1, 1, "匯出完成")
        self._progress.set_status("匯出完成")

        enc_name = self._encoder_var.get()
        msg = f"影片: {video_path}\n編碼器: {enc_name}"
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
