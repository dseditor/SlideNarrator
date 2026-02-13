"""CustomTkinter 主視窗"""
import logging
from typing import List, Optional, Tuple

import customtkinter as ctk
import numpy as np

from core.script_parser import Script
from core.tts_engine import TTSEngine

logger = logging.getLogger(__name__)


class SharedState:
    """分頁間共享的應用狀態"""

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
    """簡報自動旁白應用程式主視窗"""

    def __init__(self, shared_state: SharedState):
        super().__init__()

        self.title("簡報自動旁白系統 - SlideNarrator")
        self.geometry("1000x720")
        self.minsize(800, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.shared_state = shared_state

        # 標題列
        title_frame = ctk.CTkFrame(self, height=50)
        title_frame.pack(fill="x", padx=10, pady=(10, 0))
        title_frame.pack_propagate(False)

        ctk.CTkLabel(
            title_frame,
            text="簡報自動旁白系統",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left", padx=15, pady=10)

        engine_status = "TTS 引擎就緒" if shared_state.tts_engine and shared_state.tts_engine.is_ready else "TTS 引擎未載入"
        self._status_label = ctk.CTkLabel(
            title_frame,
            text=engine_status,
            font=ctk.CTkFont(size=12),
            text_color="green" if "就緒" in engine_status else "red",
        )
        self._status_label.pack(side="right", padx=15)

        # 分頁
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # 延遲匯入避免循環依賴
        from ui.tab_import import ImportTab
        from ui.tab_tts import TTSTab
        from ui.tab_export import ExportTab

        tab_import = self.tabview.add("匯入")
        tab_tts = self.tabview.add("語音合成")
        tab_export = self.tabview.add("匯出影片")

        self.import_tab = ImportTab(tab_import, self.shared_state, self)
        self.tts_tab = TTSTab(tab_tts, self.shared_state, self)
        self.export_tab = ExportTab(tab_export, self.shared_state, self)
