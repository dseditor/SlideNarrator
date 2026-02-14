"""步驟 4：語音合成"""
import logging
import threading
import wave
import winsound
from pathlib import Path

import customtkinter as ctk
import numpy as np

from config import DEFAULT_SPEED, SENTENCE_PAUSE_SEC, TEMP_DIR
from core.audio_processor import (
    concatenate_audio,
    get_wav_duration,
    process_all_pages,
    save_wav,
)
from ui.widgets import ProgressSection, SentenceListItem

logger = logging.getLogger(__name__)


class StepTTS:
    """語音合成 — TTS 合成 + 單句重新產生"""

    def __init__(self, parent: ctk.CTkFrame, shared_state, app):
        self.parent = parent
        self.state = shared_state
        self.app = app
        self._is_synthesizing = False
        self._wav_cache: dict = {}
        self._sentence_history: dict = {}
        self._sentence_items: dict = {}
        self._page_labels: dict = {}

        self._build_ui()

    def _build_ui(self) -> None:
        # 標題
        ctk.CTkLabel(
            self.parent, text="🔊 產生音檔",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=15, pady=(15, 5))

        # 控制列
        ctrl = ctk.CTkFrame(self.parent)
        ctrl.pack(fill="x", padx=15, pady=(0, 5))

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

        # 預覽列表
        preview_frame = ctk.CTkFrame(self.parent)
        preview_frame.pack(fill="both", expand=True, padx=15, pady=5)

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

        if not self.state.script or self.state.script.total_sentences == 0:
            self._progress.set_status("請先完成講稿編輯步驟")
            return

        if not self.state.tts_engine or not self.state.tts_engine.is_ready:
            self._progress.set_status("TTS 引擎未就緒")
            return

        self._is_synthesizing = True
        self._synth_btn.configure(state="disabled", text="合成中...")
        self._wav_cache.clear()
        self._sentence_history.clear()

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

            audio_paths = []
            for page in self.state.script.pages:
                page_wav = Path(output_dir) / f"page{page.page_number:03d}_full.wav"
                audio_paths.append(str(page_wav))
            self.state.page_audio_paths = audio_paths

            if audio_paths and Path(audio_paths[0]).exists():
                with wave.open(audio_paths[0], "rb") as wf:
                    self.state.sample_rate = wf.getframerate()
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
        self._update_total_label()

    def _on_synthesis_error(self, error: str) -> None:
        self._is_synthesizing = False
        self._synth_btn.configure(state="normal", text="開始合成所有語音")
        self._progress.set_status(f"合成失敗: {error[:80]}")

    def _update_total_label(self) -> None:
        if not self.state.script:
            return
        total_dur = self.state.script.total_duration
        minutes = int(total_dur // 60)
        seconds = total_dur % 60
        self._total_label.configure(
            text=f"總時長: {minutes} 分 {seconds:.1f} 秒 ({self.state.script.total_sentences} 句)"
        )

    def _build_preview_list(self) -> None:
        for widget in self._list_frame.winfo_children():
            widget.destroy()

        self._sentence_items.clear()
        self._page_labels.clear()

        if not self.state.script:
            return

        has_audio = any(
            s.audio_path
            for p in self.state.script.pages
            for s in p.sentences
        )

        global_idx = 0
        for page in self.state.script.pages:
            page_frame = ctk.CTkFrame(self._list_frame, fg_color=["#E8E8E8", "#2B2B2B"])
            page_frame.pack(fill="x", padx=2, pady=(8, 2))

            page_label = ctk.CTkLabel(
                page_frame,
                text=f"第 {page.page_number} 頁  (小計: {page.total_duration:.1f}s)",
                font=ctk.CTkFont(size=13, weight="bold"),
            )
            page_label.pack(anchor="w", padx=10, pady=4)
            self._page_labels[page.page_index] = page_label

            for sentence in page.sentences:
                key = (sentence.page_index, sentence.sentence_index)
                has_hist = key in self._sentence_history

                item = SentenceListItem(
                    self._list_frame,
                    index=global_idx,
                    text=sentence.text,
                    duration=sentence.duration_sec,
                    on_play=lambda idx, s=sentence: self._play_sentence(s),
                    on_regenerate=lambda idx, txt, pi=sentence.page_index, si=sentence.sentence_index: self._regenerate_sentence(pi, si, txt),
                    on_revert=lambda idx, pi=sentence.page_index, si=sentence.sentence_index: self._revert_sentence(pi, si),
                    editable=has_audio,
                    has_history=has_hist,
                )
                item.pack(fill="x", padx=4, pady=1)
                self._sentence_items[global_idx] = item
                global_idx += 1

    # ----- 單句重新產生 / 復原 -----

    def _get_sentence(self, page_index: int, sentence_index: int):
        for page in self.state.script.pages:
            if page.page_index == page_index:
                for sent in page.sentences:
                    if sent.sentence_index == sentence_index:
                        return sent
        return None

    def _get_global_idx(self, page_index: int, sentence_index: int) -> int:
        idx = 0
        for page in self.state.script.pages:
            for sent in page.sentences:
                if sent.page_index == page_index and sent.sentence_index == sentence_index:
                    return idx
                idx += 1
        return -1

    def _regenerate_sentence(self, page_index: int, sentence_index: int, new_text: str) -> None:
        if self._is_synthesizing:
            return

        if not self.state.tts_engine or not self.state.tts_engine.is_ready:
            self._progress.set_status("TTS 引擎未就緒")
            return

        sentence = self._get_sentence(page_index, sentence_index)
        if sentence is None:
            return

        global_idx = self._get_global_idx(page_index, sentence_index)
        item = self._sentence_items.get(global_idx)
        if item:
            item.set_regenerating(True)

        key = (page_index, sentence_index)
        self._sentence_history[key] = {
            "text": sentence.text,
            "audio_path": sentence.audio_path,
            "duration_sec": sentence.duration_sec,
        }

        self._is_synthesizing = True

        thread = threading.Thread(
            target=self._regen_worker,
            args=(page_index, sentence_index, new_text),
            daemon=True,
        )
        thread.start()

    def _regen_worker(self, page_index: int, sentence_index: int, new_text: str) -> None:
        try:
            sentence = self._get_sentence(page_index, sentence_index)
            speed = self._speed_var.get()

            samples, sr = self.state.tts_engine.synthesize(new_text, speed=speed)

            if sentence.audio_path:
                wav_path = sentence.audio_path
            else:
                output_dir = str(TEMP_DIR / "audio")
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                page_num = page_index + 1
                wav_path = str(
                    Path(output_dir) / f"page{page_num:03d}_sent{sentence_index:03d}.wav"
                )

            save_wav(wav_path, samples, sr)

            sentence.text = new_text
            sentence.audio_path = wav_path
            sentence.duration_sec = len(samples) / sr

            self._rebuild_page_audio(page_index)
            self._recalculate_timeline()

            self.parent.after(0, self._on_regen_complete, page_index, sentence_index)
        except Exception as e:
            logger.error("重新產生失敗: %s", e)
            self.parent.after(0, self._on_regen_error, page_index, sentence_index, str(e))

    def _on_regen_complete(self, page_index: int, sentence_index: int) -> None:
        self._is_synthesizing = False
        sentence = self._get_sentence(page_index, sentence_index)
        global_idx = self._get_global_idx(page_index, sentence_index)
        item = self._sentence_items.get(global_idx)

        if item and sentence:
            item.update_duration(sentence.duration_sec)
            item.set_regenerating(False)
            item.set_revert_enabled(True)

        self._update_page_label(page_index)
        self._update_total_label()
        self._progress.set_status("單句重新產生完成")

    def _on_regen_error(self, page_index: int, sentence_index: int, error: str) -> None:
        self._is_synthesizing = False
        global_idx = self._get_global_idx(page_index, sentence_index)
        item = self._sentence_items.get(global_idx)
        if item:
            item.set_regenerating(False)

        key = (page_index, sentence_index)
        if key in self._sentence_history:
            del self._sentence_history[key]

        self._progress.set_status(f"重新產生失敗: {error[:60]}")

    def _revert_sentence(self, page_index: int, sentence_index: int) -> None:
        key = (page_index, sentence_index)
        backup = self._sentence_history.get(key)
        if not backup:
            return

        sentence = self._get_sentence(page_index, sentence_index)
        if sentence is None:
            return

        sentence.text = backup["text"]
        sentence.audio_path = backup["audio_path"]
        sentence.duration_sec = backup["duration_sec"]

        del self._sentence_history[key]

        self._rebuild_page_audio(page_index)
        self._recalculate_timeline()

        global_idx = self._get_global_idx(page_index, sentence_index)
        item = self._sentence_items.get(global_idx)
        if item:
            item.set_text(sentence.text)
            item.update_duration(sentence.duration_sec)
            item.set_revert_enabled(False)

        self._update_page_label(page_index)
        self._update_total_label()
        self._progress.set_status("已復原")

    def _rebuild_page_audio(self, page_index: int) -> None:
        page = None
        for p in self.state.script.pages:
            if p.page_index == page_index:
                page = p
                break
        if page is None:
            return

        sr = self.state.sample_rate
        pause_sec = self._pause_var.get()
        segments = []

        for sentence in page.sentences:
            if sentence.audio_path and Path(sentence.audio_path).exists():
                with wave.open(sentence.audio_path, "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                    segments.append(audio)
            else:
                segments.append(np.zeros(int(sr), dtype=np.float32))

        if segments:
            combined = concatenate_audio(segments, pause_sec, sr)

            output_dir = str(TEMP_DIR / "audio")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            page_wav = str(Path(output_dir) / f"page{page.page_number:03d}_full.wav")
            save_wav(page_wav, combined, sr)

            page_duration = get_wav_duration(page_wav)

            if page_index < len(self.state.page_audios):
                self.state.page_audios[page_index] = (combined, page_duration)
            if page_index < len(self.state.page_audio_paths):
                self.state.page_audio_paths[page_index] = page_wav

    def _recalculate_timeline(self) -> None:
        if not self.state.script:
            return

        pause_sec = self._pause_var.get()
        global_cursor = 0.0

        for page in self.state.script.pages:
            for i, sentence in enumerate(page.sentences):
                if i > 0:
                    global_cursor += pause_sec
                sentence.start_sec = global_cursor
                global_cursor += sentence.duration_sec

    def _update_page_label(self, page_index: int) -> None:
        label = self._page_labels.get(page_index)
        if label is None:
            return
        for page in self.state.script.pages:
            if page.page_index == page_index:
                label.configure(
                    text=f"第 {page.page_number} 頁  (小計: {page.total_duration:.1f}s)"
                )
                break

    # ----- 播放 -----

    def _play_sentence(self, sentence) -> None:
        if not sentence.audio_path:
            return
        try:
            winsound.PlaySound(sentence.audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            logger.error("播放失敗: %s", e)

    # ----- 匯出 -----

    def _export_all_audio(self) -> None:
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

    def can_proceed(self) -> bool:
        return len(self.state.page_audios) > 0

    # ----- 專案載入 -----

    def load_from_project(self, script, audio_info: list) -> None:
        self._sentence_history.clear()

        audio_map = {}
        for info in audio_info:
            audio_map[(info["page"], info["sent_idx"])] = info["path"]

        for page in script.pages:
            for sentence in page.sentences:
                key = (page.page_number, sentence.sentence_index)
                audio_path = audio_map.get(key)
                if audio_path and Path(audio_path).exists():
                    sentence.audio_path = audio_path
                    sentence.duration_sec = get_wav_duration(audio_path)

        self._recalculate_timeline()

        has_audio = any(
            s.audio_path
            for p in script.pages
            for s in p.sentences
        )
        if has_audio:
            self._export_audio_btn.configure(state="normal", fg_color=["#3B8ED0", "#1F6AA5"])

        for info in audio_info:
            p = Path(info["path"])
            if p.exists():
                with wave.open(str(p), "rb") as wf:
                    self.state.sample_rate = wf.getframerate()
                break

        self._build_preview_list()
        self._update_total_label()
        self._progress.set_status("已從專案還原語音資料")
