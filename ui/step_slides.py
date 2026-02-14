"""步驟 1：載入簡報"""
import logging
import threading
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image

from config import DEFAULT_SLIDE_DPI, TEMP_DIR
from core.slide_converter import convert_slides
from ui.widgets import ProgressSection

logger = logging.getLogger(__name__)


class StepSlides:
    """載入簡報 — PDF/PPTX 匯入與縮圖預覽"""

    def __init__(self, parent: ctk.CTkFrame, shared_state, app):
        self.parent = parent
        self.state = shared_state
        self.app = app
        self._build_ui()

    def _build_ui(self) -> None:
        # 標題
        ctk.CTkLabel(
            self.parent, text="📊 載入簡報",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=15, pady=(15, 5))

        ctk.CTkLabel(
            self.parent,
            text="選擇 PDF 或 PPTX 簡報檔案，系統會自動將每頁轉換為圖片。",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(anchor="w", padx=15, pady=(0, 10))

        # 檔案選擇列
        file_row = ctk.CTkFrame(self.parent, fg_color="transparent")
        file_row.pack(fill="x", padx=15, pady=(0, 5))

        self._file_entry = ctk.CTkEntry(
            file_row, placeholder_text="選擇 PDF 或 PPTX 檔案...",
        )
        self._file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ctk.CTkButton(
            file_row, text="瀏覽...", width=80,
            command=self._browse_slide,
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            file_row, text="清除", width=60,
            fg_color="gray", command=self._clear_slides,
        ).pack(side="left")

        # 狀態
        self._slide_status = ctk.CTkLabel(
            self.parent, text="尚未匯入簡報",
            font=ctk.CTkFont(size=12), text_color="gray",
        )
        self._slide_status.pack(anchor="w", padx=15, pady=(0, 5))

        # 縮圖預覽
        self._thumb_frame = ctk.CTkScrollableFrame(
            self.parent, height=130, orientation="horizontal",
        )
        self._thumb_frame.pack(fill="x", padx=15, pady=(0, 8))

        # 進度條
        self._progress = ProgressSection(self.parent)
        self._progress.pack(fill="x", padx=15, pady=(0, 10))

    # ----- 簡報操作 -----

    def _browse_slide(self) -> None:
        filepath = filedialog.askopenfilename(
            title="選擇簡報檔案",
            filetypes=[
                ("簡報檔案", "*.pdf *.pptx *.ppt"),
                ("PDF", "*.pdf"),
                ("PowerPoint", "*.pptx *.ppt"),
            ],
        )
        if not filepath:
            return

        self._file_entry.delete(0, "end")
        self._file_entry.insert(0, filepath)
        self.state.slide_path = filepath

        self._progress.set_status("正在轉換簡報為圖片...")
        thread = threading.Thread(
            target=self._convert_slides_worker,
            args=(filepath,),
            daemon=True,
        )
        thread.start()

    def _convert_slides_worker(self, filepath: str) -> None:
        try:
            output_dir = str(TEMP_DIR / "slides")
            images = convert_slides(filepath, output_dir, DEFAULT_SLIDE_DPI)
            self.state.slide_images = images
            self.parent.after(0, self._on_slides_converted, images)
        except Exception as e:
            logger.error("簡報轉換失敗: %s", e)
            self.parent.after(0, self._on_slides_error, str(e))

    def _on_slides_converted(self, images) -> None:
        self._slide_status.configure(
            text=f"已匯入 {len(images)} 頁簡報",
            text_color="green",
        )
        self._progress.set_status(f"轉換完成：{len(images)} 頁")
        self._show_thumbnails(images)

    def _on_slides_error(self, error: str) -> None:
        self._slide_status.configure(
            text=f"轉換失敗: {error[:60]}",
            text_color="red",
        )
        self._progress.set_status("轉換失敗")

    def _clear_slides(self) -> None:
        self._file_entry.delete(0, "end")
        self.state.slide_images = []
        self.state.slide_path = ""
        self._slide_status.configure(text="尚未匯入簡報", text_color="gray")
        for widget in self._thumb_frame.winfo_children():
            widget.destroy()
        self._progress.reset()

    def _show_thumbnails(self, images) -> None:
        for widget in self._thumb_frame.winfo_children():
            widget.destroy()

        for i, img_path in enumerate(images):
            try:
                img = Image.open(img_path)
                img.thumbnail((150, 100))
                ctk_img = ctk.CTkImage(light_image=img, size=img.size)
                label = ctk.CTkLabel(
                    self._thumb_frame, image=ctk_img, text=f"P{i+1}",
                    compound="top", font=ctk.CTkFont(size=10),
                )
                label.pack(side="left", padx=4, pady=4)
                label._ctk_img = ctk_img
            except Exception:
                ctk.CTkLabel(
                    self._thumb_frame, text=f"P{i+1}\n(預覽失敗)",
                    width=80, height=60,
                ).pack(side="left", padx=4, pady=4)

    def can_proceed(self) -> bool:
        return len(self.state.slide_images) > 0

    def load_from_project(self, slide_images: list) -> None:
        """從專案還原簡報狀態"""
        self.state.slide_images = slide_images
        if slide_images:
            self._file_entry.delete(0, "end")
            self._file_entry.insert(0, "(從專案載入)")
            self._on_slides_converted(slide_images)
        else:
            self._clear_slides()
