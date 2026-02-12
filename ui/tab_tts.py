"""分頁二：TTS 合成與預覽"""
import logging
import threading
import winsound
from pathlib import Path

import customtkinter as ctk

from config import DEFAULT_SPEED, SENTENCE_PAUSE_SEC, TEMP_DIR
from core.audio_processor import process_all_pages, save_wav
from core.script_parser import parse_script
from ui.widgets import ProgressSection, SentenceListItem

logger = logging.getLogger(__name__)


class TTSTab:
    """TTS 合成分頁"""

    def __init__(self, parent: ctk.CTkFrame, shared_state, app):
        self.parent = parent
        self.state = shared_state
        self.app = app
        self._is_synthesizing = False
        self._wav_cache: dict = {}  # sentence_key -> wav_bytes

        self._build_ui()

    def _build_ui(self) -> None:
        # ===== 控制列 =====
        ctrl = ctk.CTkFrame(self.parent)
        ctrl.pack(fill="x", padx=10, pady=(10, 5))

        # 語速
        speed_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        speed_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(speed_frame, text="語速:").pack(side="left", padx=(0, 5))
        self._speed_var = ctk.DoubleVar(value=DEFAULT_SPEED)
        self._speed_slider = ctk.CTkSlider(
            speed_frame, from_=0.5, to=2.0,
            variable=self._speed_var,
            command=self._on_speed_change,
            width=250,
        )
        self._speed_slider.pack(side="left", padx=5)
        self._speed_label = ctk.CTkLabel(speed_frame, text="1.0x", width=40)
        self._speed_label.pack(side="left")

        # 句間停頓
        ctk.CTkLabel(speed_frame, text="    句間停頓:").pack(side="left", padx=(15, 5))
        self._pause_var = ctk.DoubleVar(value=SENTENCE_PAUSE_SEC)
        self._pause_slider = ctk.CTkSlider(
            speed_frame, from_=0.0, to=2.0,
            variable=self._pause_var,
            command=self._on_pause_change,
            width=200,
        )
        self._pause_slider.pack(side="left", padx=5)
        self._pause_label = ctk.CTkLabel(speed_frame, text="0.5s", width=40)
        self._pause_label.pack(side="left")

        # 合成按鈕
        btn_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        self._synth_btn = ctk.CTkButton(
            btn_frame, text="開始合成所有語音",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self._start_synthesis,
        )
        self._synth_btn.pack(side="left", padx=(0, 10))

        self._export_audio_btn = ctk.CTkButton(
            btn_frame, text="匯出所有音訊", width=120,
            fg_color="gray", state="disabled",
            command=self._export_all_audio,
        )
        self._export_audio_btn.pack(side="left")

        # 進度條
        self._progress = ProgressSection(ctrl)
        self._progress.pack(fill="x", padx=10, pady=(0, 8))

        # ===== 預覽列表 =====
        preview_frame = ctk.CTkFrame(self.parent)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ctk.CTkLabel(
            preview_frame, text="語音預覽",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        self._list_frame = ctk.CTkScrollableFrame(preview_frame)
        self._list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # 總時長
        self._total_label = ctk.CTkLabel(
            preview_frame, text="",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._total_label.pack(anchor="w", padx=10, pady=(0, 8))

    # ----- 事件 -----

    def _on_speed_change(self, value) -> None:
        self._speed_label.configure(text=f"{value:.1f}x")

    def _on_pause_change(self, value) -> None:
        self._pause_label.configure(text=f"{value:.1f}s")

    def _start_synthesis(self) -> None:
        if self._is_synthesizing:
            return

        # 確保有講稿
        if hasattr(self.app, 'import_tab'):
            self.app.import_tab.get_script()

        if not self.state.script or self.state.script.total_sentences == 0:
            self._progress.set_status("請先在「匯入」分頁輸入並驗證講稿")
            return

        if not self.state.tts_engine or not self.state.tts_engine.is_ready:
            self._progress.set_status("TTS 引擎未就緒")
            return

        self._is_synthesizing = True
        self._synth_btn.configure(state="disabled", text="合成中...")
        self._wav_cache.clear()

        thread = threading.Thread(
            target=self._synthesis_worker,
            daemon=True,
        )
        thread.start()

    def _synthesis_worker(self) -> None:
        try:
            output_dir = str(TEMP_DIR / "audio")
            speed = self._speed_var.get()
            pause = self._pause_var.get()

            results = process_all_pages(
                script=self.state.script,
                tts_engine=self.state.tts_engine,
                speed=speed,
                pause_sec=pause,
                output_dir=output_dir,
                progress_callback=self._thread_safe_progress,
            )

            self.state.page_audios = results

            # 收集頁面音訊路徑，並從實際 WAV 檔案取得正確的取樣率
            audio_paths = []
            for page in self.state.script.pages:
                page_wav = Path(output_dir) / f"page{page.page_number:03d}_full.wav"
                audio_paths.append(str(page_wav))
            self.state.page_audio_paths = audio_paths

            # 從實際產生的 WAV 檔案讀取取樣率（比引擎宣告值更可靠）
            if audio_paths and Path(audio_paths[0]).exists():
                import wave
                with wave.open(audio_paths[0], "rb") as wf:
                    self.state.sample_rate = wf.getframerate()
                    logger.info("實際音訊取樣率: %d Hz", self.state.sample_rate)
            else:
                self.state.sample_rate = self.state.tts_engine.sample_rate

            self.parent.after(0, self._on_synthesis_complete)
        except Exception as e:
            logger.error("TTS 合成失敗: %s", e)
            self.parent.after(0, self._on_synthesis_error, str(e))

    def _thread_safe_progress(self, current: int, total: int, message: str) -> None:
        self.parent.after(0, self._progress.update_progress, current, total, message)

    def _on_synthesis_complete(self) -> None:
        self._is_synthesizing = False
        self._synth_btn.configure(state="normal", text="開始合成所有語音")
        self._export_audio_btn.configure(state="normal", fg_color=["#3B8ED0", "#1F6AA5"])
        self._progress.set_status("合成完成")

        self._build_preview_list()

        total_dur = self.state.script.total_duration
        minutes = int(total_dur // 60)
        seconds = total_dur % 60
        self._total_label.configure(
            text=f"總時長: {minutes} 分 {seconds:.1f} 秒 ({self.state.script.total_sentences} 句)"
        )

    def _on_synthesis_error(self, error: str) -> None:
        self._is_synthesizing = False
        self._synth_btn.configure(state="normal", text="開始合成所有語音")
        self._progress.set_status(f"合成失敗: {error[:80]}")

    def _build_preview_list(self) -> None:
        """建立語音預覽列表"""
        for widget in self._list_frame.winfo_children():
            widget.destroy()

        if not self.state.script:
            return

        global_idx = 0
        for page in self.state.script.pages:
            # 頁面標題
            page_frame = ctk.CTkFrame(self._list_frame, fg_color=["#E8E8E8", "#2B2B2B"])
            page_frame.pack(fill="x", padx=2, pady=(8, 2))

            page_dur = page.total_duration
            ctk.CTkLabel(
                page_frame,
                text=f"第 {page.page_number} 頁  (小計: {page_dur:.1f}s)",
                font=ctk.CTkFont(size=13, weight="bold"),
            ).pack(anchor="w", padx=10, pady=4)

            # 每句
            for sentence in page.sentences:
                item = SentenceListItem(
                    self._list_frame,
                    index=global_idx,
                    text=sentence.text,
                    duration=sentence.duration_sec,
                    on_play=lambda idx, s=sentence: self._play_sentence(s),
                )
                item.pack(fill="x", padx=4, pady=1)
                global_idx += 1

    def _play_sentence(self, sentence) -> None:
        """播放單句預覽"""
        if not sentence.audio_path:
            return
        try:
            # winsound 可播放 WAV 檔案
            winsound.PlaySound(sentence.audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            logger.error("播放失敗: %s", e)

    def _export_all_audio(self) -> None:
        """匯出所有音訊"""
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="選擇音訊匯出資料夾")
        if not folder:
            return

        try:
            import shutil
            count = 0
            for page in self.state.script.pages:
                for sentence in page.sentences:
                    if sentence.audio_path and Path(sentence.audio_path).exists():
                        dest = Path(folder) / Path(sentence.audio_path).name
                        shutil.copy2(sentence.audio_path, str(dest))
                        count += 1
            self._progress.set_status(f"已匯出 {count} 個音訊檔案到 {folder}")
        except Exception as e:
            self._progress.set_status(f"匯出失敗: {e}")
