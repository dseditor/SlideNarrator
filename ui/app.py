"""CustomTkinter 主視窗 — 步驟引導式流程"""
import logging
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import List, Optional, Tuple

import customtkinter as ctk
import numpy as np

from config import TEMP_DIR
from core.project_manager import load_project, save_project
from core.script_parser import Script, parse_script
from core.tts_engine import TTSEngine
from ui.widgets import StepSidebar

logger = logging.getLogger(__name__)

STEPS = [
    {"name": "歡迎", "icon": "🏠"},
    {"name": "簡報", "icon": "📊"},
    {"name": "講稿", "icon": "📝"},
    {"name": "編輯", "icon": "📋"},
    {"name": "語音", "icon": "🔊"},
    {"name": "匯出", "icon": "🎬"},
]


class SharedState:
    """步驟間共享的應用狀態"""

    def __init__(self):
        self.script: Optional[Script] = None
        self.slide_images: List[str] = []
        self.slide_path: str = ""
        self.page_audios: List[Tuple[np.ndarray, float]] = []
        self.page_audio_paths: List[str] = []
        self.srt_content: str = ""
        self.srt_path: str = ""
        self.tts_engine: Optional[TTSEngine] = None
        self.output_dir: str = ""
        self.sample_rate: int = 48000


class NarratorApp(ctk.CTk):
    """簡報自動旁白應用程式主視窗 — 步驟引導式"""

    def __init__(self, shared_state: SharedState):
        super().__init__()

        self.title("簡報自動旁白系統 - SlideNarrator")
        self.geometry("1100x750")
        self.minsize(900, 650)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.shared_state = shared_state
        self._current_step = 0
        self._step_frames: dict = {}  # step_index -> CTkFrame
        self._step_handlers: dict = {}  # step_index -> handler instance

        self._build_layout()

        # 初始顯示歡迎頁
        self.goto_step(0)

    def _build_layout(self) -> None:
        # ===== 標題列 =====
        title_frame = ctk.CTkFrame(self, height=50)
        title_frame.pack(fill="x", padx=10, pady=(10, 0))
        title_frame.pack_propagate(False)

        ctk.CTkLabel(
            title_frame,
            text="簡報自動旁白系統",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left", padx=15, pady=10)

        engine_status = (
            "TTS 引擎就緒"
            if self.shared_state.tts_engine and self.shared_state.tts_engine.is_ready
            else "TTS 引擎未載入"
        )
        self._status_label = ctk.CTkLabel(
            title_frame,
            text=engine_status,
            font=ctk.CTkFont(size=12),
            text_color="green" if "就緒" in engine_status else "red",
        )
        self._status_label.pack(side="right", padx=15)

        ctk.CTkButton(
            title_frame, text="載入專案", width=90, height=30,
            fg_color="#7B68EE",
            command=self._load_project,
        ).pack(side="right", padx=5)

        ctk.CTkButton(
            title_frame, text="保存專案", width=90, height=30,
            fg_color="#2E8B57",
            command=self._save_project,
        ).pack(side="right", padx=5)

        # ===== 主體區域（側邊欄 + 內容區） =====
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=(5, 0))

        # 左側邊欄
        self._sidebar = StepSidebar(
            body,
            steps=STEPS,
            on_step_click=self.goto_step,
        )
        self._sidebar.pack(side="left", fill="y", padx=(0, 5), pady=5)

        # 右側內容區
        self._content_frame = ctk.CTkFrame(body)
        self._content_frame.pack(side="left", fill="both", expand=True, pady=5)

        # ===== 底部導航列 =====
        nav_frame = ctk.CTkFrame(self, height=50)
        nav_frame.pack(fill="x", padx=10, pady=(0, 10))
        nav_frame.pack_propagate(False)

        self._prev_btn = ctk.CTkButton(
            nav_frame, text="上一步", width=120, height=36,
            command=self.prev_step,
        )
        self._prev_btn.pack(side="left", padx=15, pady=7)

        self._next_btn = ctk.CTkButton(
            nav_frame, text="下一步", width=120, height=36,
            command=self.next_step,
        )
        self._next_btn.pack(side="right", padx=15, pady=7)

    # ----- 步驟切換 -----

    def goto_step(self, index: int) -> None:
        if index < 0 or index >= len(STEPS):
            return

        # 離開當前步驟的回調
        current_handler = self._step_handlers.get(self._current_step)
        if current_handler and hasattr(current_handler, "on_leave"):
            current_handler.on_leave()

        # 隱藏當前步驟
        current_frame = self._step_frames.get(self._current_step)
        if current_frame:
            current_frame.pack_forget()

        self._current_step = index

        # 確保步驟 frame 已建立（lazy init）
        if index not in self._step_frames:
            self._init_step(index)

        # 顯示目標步驟
        target_frame = self._step_frames[index]
        target_frame.pack(fill="both", expand=True)

        # 進入步驟的回調
        target_handler = self._step_handlers.get(index)
        if target_handler and hasattr(target_handler, "on_enter"):
            target_handler.on_enter()

        # 更新側邊欄
        self._sidebar.set_current(index)

        # 更新底部按鈕
        self._update_nav_buttons()

    def _init_step(self, index: int) -> None:
        """延遲初始化步驟"""
        frame = ctk.CTkFrame(self._content_frame)
        self._step_frames[index] = frame

        if index == 0:
            from ui.step_welcome import StepWelcome
            handler = StepWelcome(frame, self.shared_state, self)
        elif index == 1:
            from ui.step_slides import StepSlides
            handler = StepSlides(frame, self.shared_state, self)
        elif index == 2:
            from ui.step_script import StepScript
            handler = StepScript(frame, self.shared_state, self)
        elif index == 3:
            from ui.step_edit import StepEdit
            handler = StepEdit(frame, self.shared_state, self)
        elif index == 4:
            from ui.step_tts import StepTTS
            handler = StepTTS(frame, self.shared_state, self)
        elif index == 5:
            from ui.step_export import StepExport
            handler = StepExport(frame, self.shared_state, self)
        else:
            handler = None

        if handler:
            self._step_handlers[index] = handler

    def next_step(self) -> None:
        if self._current_step >= len(STEPS) - 1:
            return
        if not self._validate_step(self._current_step):
            return
        self.goto_step(self._current_step + 1)

    def prev_step(self) -> None:
        if self._current_step <= 0:
            return
        self.goto_step(self._current_step - 1)

    def _validate_step(self, index: int) -> bool:
        """驗證當前步驟是否可以進入下一步"""
        handler = self._step_handlers.get(index)

        if index == 0:
            # 歡迎頁：直接通過
            return True
        elif index == 1:
            # 載入簡報：需要有圖片
            if handler and not handler.can_proceed():
                messagebox.showwarning("提示", "請先匯入簡報檔案。")
                return False
        elif index == 2:
            # 載入講稿：需要驗證通過
            if handler:
                handler.get_script()
                if not handler.can_proceed():
                    messagebox.showwarning("提示", "請確認講稿已驗證通過且頁數與簡報一致。")
                    return False
        elif index == 3:
            # 編輯講稿：每頁至少 1 句有文字
            if handler:
                handler.on_leave()
                if not handler.can_proceed():
                    messagebox.showwarning("提示", "請確認每頁至少有一句非空白的講稿。")
                    return False
        elif index == 4:
            # 語音合成：需要有音訊
            if handler and not handler.can_proceed():
                messagebox.showwarning("提示", "請先完成語音合成。")
                return False

        return True

    def _update_nav_buttons(self) -> None:
        # 上一步
        if self._current_step == 0:
            self._prev_btn.configure(state="disabled", text="上一步")
        else:
            self._prev_btn.configure(state="normal", text="上一步")

        # 下一步
        if self._current_step >= len(STEPS) - 1:
            self._next_btn.configure(state="disabled", text="已完成")
        else:
            self._next_btn.configure(state="normal", text="下一步")

    # ----- 專案保存/載入 -----

    def _save_project(self) -> None:
        script_text = ""
        script_handler = self._step_handlers.get(2)
        if script_handler and hasattr(script_handler, "get_script_text"):
            script_text = script_handler.get_script_text()

        if not script_text and not self.shared_state.slide_images:
            messagebox.showwarning("保存專案", "目前沒有可保存的資料。\n請先匯入簡報或輸入講稿。")
            return

        filepath = filedialog.asksaveasfilename(
            title="保存專案",
            defaultextension=".zip",
            filetypes=[("專案檔案", "*.zip")],
            initialfile="project.zip",
        )
        if not filepath:
            return

        try:
            sentence_audios = []
            if self.shared_state.script:
                for page in self.shared_state.script.pages:
                    for sent in page.sentences:
                        if sent.audio_path and Path(sent.audio_path).exists():
                            sentence_audios.append({
                                "page": page.page_number,
                                "sent_idx": sent.sentence_index,
                                "path": sent.audio_path,
                            })

            save_project(
                output_path=filepath,
                slide_images=self.shared_state.slide_images,
                script_text=script_text,
                page_audio_paths=self.shared_state.page_audio_paths,
                sentence_audios=sentence_audios,
            )

            messagebox.showinfo("保存專案", f"專案已保存至:\n{filepath}")
        except Exception as e:
            logger.error("保存專案失敗: %s", e)
            messagebox.showerror("保存專案", f"保存失敗: {e}")

    def _load_project(self) -> None:
        filepath = filedialog.askopenfilename(
            title="載入專案",
            filetypes=[("專案檔案", "*.zip")],
        )
        if not filepath:
            return
        self._do_load_project(filepath)

    def load_project_and_jump(self) -> None:
        """從歡迎頁呼叫的載入專案"""
        filepath = filedialog.askopenfilename(
            title="載入專案",
            filetypes=[("專案檔案", "*.zip")],
        )
        if not filepath:
            return
        self._do_load_project(filepath)

    def _do_load_project(self, filepath: str) -> None:
        try:
            extract_dir = str(TEMP_DIR / "project_load")
            data = load_project(filepath, extract_dir)

            # 確保步驟 1-4 已初始化
            for i in range(1, 5):
                if i not in self._step_frames:
                    self._init_step(i)

            # 還原簡報
            slides_handler = self._step_handlers.get(1)
            if slides_handler:
                slides_handler.load_from_project(data["slide_images"])

            # 還原講稿
            script_handler = self._step_handlers.get(2)
            if script_handler and data["script_text"]:
                script_handler.load_from_project(data["script_text"])

            # 還原頁面音訊路徑
            self.shared_state.page_audio_paths = data["page_audio_paths"]

            # 還原 page_audios
            import wave
            page_audios = []
            for audio_path in data["page_audio_paths"]:
                if Path(audio_path).exists():
                    with wave.open(audio_path, "rb") as wf:
                        raw = wf.readframes(wf.getnframes())
                        sr = wf.getframerate()
                        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                        duration = len(audio) / sr
                        page_audios.append((audio, duration))
                else:
                    page_audios.append((np.array([], dtype=np.float32), 0.0))
            self.shared_state.page_audios = page_audios

            # 還原 TTS
            tts_handler = self._step_handlers.get(4)
            if tts_handler and self.shared_state.script:
                tts_handler.load_from_project(
                    self.shared_state.script,
                    data["sentence_audios"],
                )

            # 決定跳到哪個步驟 & 可到達的最遠步驟
            has_audio = len(data["page_audio_paths"]) > 0 and any(
                Path(p).exists() for p in data["page_audio_paths"]
            )
            if has_audio:
                target_step = 4   # 預設顯示語音合成步驟
                max_step = 5      # 有音訊 → 匯出也可用
            elif data["script_text"]:
                target_step = 3
                max_step = 3
            else:
                target_step = 1
                max_step = 1

            # 確保匯出步驟已初始化（編碼器偵測需提早啟動）
            if max_step >= 5 and 5 not in self._step_frames:
                self._init_step(5)

            self._sidebar.set_max_reached(max_step)
            self.goto_step(target_step)

            messagebox.showinfo("載入專案", "專案載入完成")
        except Exception as e:
            logger.error("載入專案失敗: %s", e)
            messagebox.showerror("載入專案", f"載入失敗: {e}")
