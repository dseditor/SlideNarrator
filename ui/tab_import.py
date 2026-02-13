"""分頁一：匯入簡報與講稿"""
import logging
import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image

from config import DEFAULT_SLIDE_DPI, PROMPTS_DIR, TEMP_DIR
from core.script_parser import (
    format_script_preview,
    parse_script,
    validate_script,
)
from core.slide_converter import convert_slides
from ui.widgets import ProgressSection

logger = logging.getLogger(__name__)

# AI 提示詞路徑（使用 config 中的 PROMPTS_DIR，支援打包環境）
_PROMPT_PATH = PROMPTS_DIR / "script_generator.md"


class ImportTab:
    """匯入分頁"""

    def __init__(self, parent: ctk.CTkFrame, shared_state, app):
        self.parent = parent
        self.state = shared_state
        self.app = app

        self._build_ui()

    def _build_ui(self) -> None:
        # ===== 簡報區域 =====
        slide_section = ctk.CTkFrame(self.parent)
        slide_section.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            slide_section, text="簡報檔案",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        file_row = ctk.CTkFrame(slide_section, fg_color="transparent")
        file_row.pack(fill="x", padx=10, pady=(0, 4))

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

        # 簡報狀態
        self._slide_status = ctk.CTkLabel(
            slide_section, text="尚未匯入簡報",
            font=ctk.CTkFont(size=12), text_color="gray",
        )
        self._slide_status.pack(anchor="w", padx=10, pady=(0, 4))

        # 縮圖預覽（可捲動）
        self._thumb_frame = ctk.CTkScrollableFrame(
            slide_section, height=120, orientation="horizontal",
        )
        self._thumb_frame.pack(fill="x", padx=10, pady=(0, 8))

        self._slide_progress = ProgressSection(slide_section)
        self._slide_progress.pack(fill="x", padx=10, pady=(0, 8))

        # ===== 講稿區域 =====
        script_section = ctk.CTkFrame(self.parent)
        script_section.pack(fill="both", expand=True, padx=10, pady=5)

        header_row = ctk.CTkFrame(script_section, fg_color="transparent")
        header_row.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            header_row, text="講稿內容",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="left")

        ctk.CTkLabel(
            header_row,
            text="(支援: Page1: / 第1頁： / 第一頁 等格式，Gemini 單行或手動多行皆可)",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(side="left", padx=(10, 0))

        # 講稿操作按鈕列
        btn_row = ctk.CTkFrame(script_section, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkButton(
            btn_row, text="從檔案匯入", width=100,
            command=self._import_script_file,
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            btn_row, text="驗證講稿", width=100,
            command=self._validate_script,
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            btn_row, text="複製 AI 提示詞", width=120,
            fg_color="#7B68EE",
            command=self._copy_ai_prompt,
        ).pack(side="left")

        # 講稿文字區域 + 預覽區域並排
        content_row = ctk.CTkFrame(script_section, fg_color="transparent")
        content_row.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        # 左側：原始講稿
        left = ctk.CTkFrame(content_row, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        ctk.CTkLabel(
            left, text="原始講稿", font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(anchor="w")

        self._script_text = ctk.CTkTextbox(
            left, font=ctk.CTkFont(size=13),
        )
        self._script_text.pack(fill="both", expand=True)

        # 右側：解析預覽
        right = ctk.CTkFrame(content_row, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))

        ctk.CTkLabel(
            right, text="解析結果預覽", font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(anchor="w")

        self._preview_text = ctk.CTkTextbox(
            right, font=ctk.CTkFont(size=12),
            state="disabled",
        )
        self._preview_text.pack(fill="both", expand=True)

        # 講稿狀態
        self._script_status = ctk.CTkLabel(
            script_section, text="",
            font=ctk.CTkFont(size=12), text_color="gray",
        )
        self._script_status.pack(anchor="w", padx=10, pady=(0, 8))

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

        # 在背景執行緒轉換簡報
        self._slide_progress.set_status("正在轉換簡報為圖片...")
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
        self._slide_progress.set_status(f"轉換完成：{len(images)} 頁")

        # 顯示縮圖
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
                # 保持引用避免 GC
                label._ctk_img = ctk_img
            except Exception:
                ctk.CTkLabel(
                    self._thumb_frame, text=f"P{i+1}\n(預覽失敗)",
                    width=80, height=60,
                ).pack(side="left", padx=4, pady=4)

    def _on_slides_error(self, error: str) -> None:
        self._slide_status.configure(
            text=f"轉換失敗: {error[:60]}",
            text_color="red",
        )
        self._slide_progress.set_status("轉換失敗")

    def _clear_slides(self) -> None:
        self._file_entry.delete(0, "end")
        self.state.slide_images = []
        self.state.slide_path = ""
        self._slide_status.configure(text="尚未匯入簡報", text_color="gray")
        for widget in self._thumb_frame.winfo_children():
            widget.destroy()
        self._slide_progress.reset()

    # ----- 講稿操作 -----

    def _import_script_file(self) -> None:
        filepath = filedialog.askopenfilename(
            title="選擇講稿檔案",
            filetypes=[
                ("文字檔案", "*.txt"),
                ("所有檔案", "*.*"),
            ],
        )
        if not filepath:
            return

        try:
            text = Path(filepath).read_text(encoding="utf-8")
            self._script_text.delete("0.0", "end")
            self._script_text.insert("0.0", text)
            self._script_status.configure(
                text=f"已匯入: {Path(filepath).name}，請按「驗證講稿」確認解析結果",
                text_color="green",
            )
            # 自動驗證
            self._validate_script()
        except UnicodeDecodeError:
            try:
                text = Path(filepath).read_text(encoding="utf-8-sig")
                self._script_text.delete("0.0", "end")
                self._script_text.insert("0.0", text)
                self._validate_script()
            except Exception as e:
                self._script_status.configure(
                    text=f"匯入失敗: {e}", text_color="red",
                )
        except Exception as e:
            self._script_status.configure(
                text=f"匯入失敗: {e}", text_color="red",
            )

    def _validate_script(self) -> None:
        text = self._script_text.get("0.0", "end").strip()
        if not text:
            self._script_status.configure(text="請先輸入講稿", text_color="red")
            return

        script = parse_script(text)
        self.state.script = script

        # 帶入簡報頁數做交叉驗證
        slide_count = len(self.state.slide_images)
        warnings = validate_script(script, slide_count=slide_count)

        # 更新解析預覽
        preview = format_script_preview(script)
        self._preview_text.configure(state="normal")
        self._preview_text.delete("0.0", "end")
        self._preview_text.insert("0.0", preview)
        self._preview_text.configure(state="disabled")

        # 狀態訊息
        if warnings:
            msg = f"頁數: {len(script.pages)}, 句數: {script.total_sentences} | 警告: {'; '.join(warnings)}"
            self._script_status.configure(text=msg, text_color="orange")
        else:
            slide_info = f", 簡報: {slide_count} 頁" if slide_count > 0 else ""
            msg = f"驗證通過 - 講稿: {len(script.pages)} 頁, {script.total_sentences} 句{slide_info}"
            self._script_status.configure(text=msg, text_color="green")

    def _copy_ai_prompt(self) -> None:
        try:
            if _PROMPT_PATH.exists():
                prompt = _PROMPT_PATH.read_text(encoding="utf-8")
            else:
                prompt = (
                    "請根據以下簡報內容，為每一頁生成口語化的繁體中文旁白講稿。\n\n"
                    "格式要求：\n"
                    "1. 每頁以 Page數字: 開頭（例如 Page1:）\n"
                    "2. 所有句子寫在同一行，用空格分隔\n"
                    "3. 全部使用繁體中文\n"
                    "4. 句末不需要加標點符號\n\n"
                    "簡報內容：\n（請將簡報的文字內容貼在這裡）"
                )
            self.app.clipboard_clear()
            self.app.clipboard_append(prompt)
            self._script_status.configure(
                text="AI 提示詞已複製到剪貼簿",
                text_color="green",
            )
        except Exception as e:
            self._script_status.configure(
                text=f"複製失敗: {e}",
                text_color="red",
            )

    def get_script(self):
        """供其他分頁取得最新講稿"""
        text = self._script_text.get("0.0", "end").strip()
        if text:
            script = parse_script(text)
            self.state.script = script
            return script
        return self.state.script
