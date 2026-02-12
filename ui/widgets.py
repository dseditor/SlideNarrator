"""可複用的 CustomTkinter 元件"""
import customtkinter as ctk


class ProgressSection(ctk.CTkFrame):
    """進度條 + 狀態文字組合元件"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self._status_label = ctk.CTkLabel(
            self, text="就緒", anchor="w",
        )
        self._status_label.pack(fill="x", padx=5, pady=(5, 2))

        self._progress_bar = ctk.CTkProgressBar(self)
        self._progress_bar.pack(fill="x", padx=5, pady=(0, 2))
        self._progress_bar.set(0)

        self._detail_label = ctk.CTkLabel(
            self, text="", anchor="w",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._detail_label.pack(fill="x", padx=5, pady=(0, 5))

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """更新進度"""
        ratio = current / total if total > 0 else 0
        self._progress_bar.set(ratio)
        self._status_label.configure(text=f"進度：{current}/{total} ({ratio:.0%})")
        if message:
            self._detail_label.configure(text=message)

    def set_status(self, text: str) -> None:
        """設定狀態文字"""
        self._status_label.configure(text=text)

    def set_detail(self, text: str) -> None:
        """設定詳細資訊"""
        self._detail_label.configure(text=text)

    def reset(self) -> None:
        """重置"""
        self._progress_bar.set(0)
        self._status_label.configure(text="就緒")
        self._detail_label.configure(text="")


class SentenceListItem(ctk.CTkFrame):
    """單句預覽項目"""

    def __init__(
        self,
        parent,
        index: int,
        text: str,
        duration: float = 0.0,
        on_play=None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        self._index = index
        self._on_play = on_play

        # 序號
        idx_label = ctk.CTkLabel(
            self, text=f"{index + 1}.", width=30,
            font=ctk.CTkFont(size=12),
        )
        idx_label.pack(side="left", padx=(5, 2))

        # 文字
        text_label = ctk.CTkLabel(
            self, text=text, anchor="w",
            font=ctk.CTkFont(size=13),
        )
        text_label.pack(side="left", fill="x", expand=True, padx=2)

        # 時長
        self._duration_label = ctk.CTkLabel(
            self, text=f"{duration:.1f}s" if duration > 0 else "--",
            width=50,
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._duration_label.pack(side="left", padx=2)

        # 播放按鈕
        self._play_btn = ctk.CTkButton(
            self, text="播放", width=50, height=24,
            font=ctk.CTkFont(size=11),
            command=self._handle_play,
            state="disabled" if duration <= 0 else "normal",
        )
        self._play_btn.pack(side="right", padx=5)

    def update_duration(self, duration: float) -> None:
        self._duration_label.configure(text=f"{duration:.1f}s")
        self._play_btn.configure(state="normal")

    def _handle_play(self) -> None:
        if self._on_play:
            self._on_play(self._index)
