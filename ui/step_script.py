"""步驟 2：載入講稿"""
import logging
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from config import PROMPTS_DIR
from core.script_parser import format_script_preview, parse_script, validate_script

logger = logging.getLogger(__name__)

_PROMPT_PATH = PROMPTS_DIR / "script_generator.md"


class StepScript:
    """載入講稿 — 文字輸入/匯入 + AI 提示詞"""

    def __init__(self, parent: ctk.CTkFrame, shared_state, app):
        self.parent = parent
        self.state = shared_state
        self.app = app
        self._build_ui()

    def _build_ui(self) -> None:
        # 標題
        ctk.CTkLabel(
            self.parent, text="📝 載入講稿",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=15, pady=(15, 5))

        ctk.CTkLabel(
            self.parent,
            text="(支援: Page1: / 第1頁： / 第一頁 等格式，Gemini 單行或手動多行皆可)",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(anchor="w", padx=15, pady=(0, 8))

        # 操作按鈕列
        btn_row = ctk.CTkFrame(self.parent, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=(0, 5))

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
        content_row = ctk.CTkFrame(self.parent, fg_color="transparent")
        content_row.pack(fill="both", expand=True, padx=15, pady=(0, 5))

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

        # 狀態
        self._script_status = ctk.CTkLabel(
            self.parent, text="",
            font=ctk.CTkFont(size=12), text_color="gray",
        )
        self._script_status.pack(anchor="w", padx=15, pady=(0, 10))

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

        slide_count = len(self.state.slide_images)
        warnings = validate_script(script, slide_count=slide_count)

        # 更新解析預覽
        preview = format_script_preview(script)
        self._preview_text.configure(state="normal")
        self._preview_text.delete("0.0", "end")
        self._preview_text.insert("0.0", preview)
        self._preview_text.configure(state="disabled")

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

    def get_script_text(self) -> str:
        return self._script_text.get("0.0", "end").strip()

    def get_script(self):
        text = self.get_script_text()
        if text:
            script = parse_script(text)
            self.state.script = script
            return script
        return self.state.script

    def can_proceed(self) -> bool:
        if not self.state.script or self.state.script.total_sentences == 0:
            return False
        slide_count = len(self.state.slide_images)
        if slide_count > 0 and len(self.state.script.pages) != slide_count:
            return False
        return True

    def load_from_project(self, script_text: str) -> None:
        """從專案還原講稿狀態"""
        if script_text:
            self._script_text.delete("0.0", "end")
            self._script_text.insert("0.0", script_text)
            self._validate_script()
