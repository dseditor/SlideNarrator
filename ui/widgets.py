"""可複用的 CustomTkinter 元件"""
from typing import Callable, List, Optional

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
    """單句預覽項目（支援編輯、重新產生、復原）"""

    def __init__(
        self,
        parent,
        index: int,
        text: str,
        duration: float = 0.0,
        on_play=None,
        on_regenerate=None,
        on_revert=None,
        editable: bool = False,
        has_history: bool = False,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        self._index = index
        self._on_play = on_play
        self._on_regenerate = on_regenerate
        self._on_revert = on_revert

        # 序號
        idx_label = ctk.CTkLabel(
            self, text=f"{index + 1}.", width=30,
            font=ctk.CTkFont(size=12),
        )
        idx_label.pack(side="left", padx=(5, 2))

        # 文字（可編輯 Entry 或唯讀 Label）
        if editable:
            self._text_entry = ctk.CTkEntry(
                self, font=ctk.CTkFont(size=13),
            )
            self._text_entry.insert(0, text)
            self._text_entry.pack(side="left", fill="x", expand=True, padx=2)
        else:
            self._text_entry = None
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

        # 按鈕區（從右到左排列：復原、重新產生、播放）
        if on_revert is not None:
            self._revert_btn = ctk.CTkButton(
                self, text="復原", width=50, height=24,
                font=ctk.CTkFont(size=11),
                fg_color="gray",
                command=self._handle_revert,
                state="normal" if has_history else "disabled",
            )
            self._revert_btn.pack(side="right", padx=(2, 5))
        else:
            self._revert_btn = None

        if on_regenerate is not None:
            self._regen_btn = ctk.CTkButton(
                self, text="重新產生", width=70, height=24,
                font=ctk.CTkFont(size=11),
                fg_color="#D97706",
                command=self._handle_regenerate,
                state="normal" if editable else "disabled",
            )
            self._regen_btn.pack(side="right", padx=2)
        else:
            self._regen_btn = None

        # 播放按鈕
        self._play_btn = ctk.CTkButton(
            self, text="播放", width=50, height=24,
            font=ctk.CTkFont(size=11),
            command=self._handle_play,
            state="disabled" if duration <= 0 else "normal",
        )
        self._play_btn.pack(side="right", padx=2)

    @property
    def current_text(self) -> str:
        """取得目前 Entry 中的文字"""
        if self._text_entry is not None:
            return self._text_entry.get().strip()
        return ""

    def set_text(self, text: str) -> None:
        """設定 Entry 文字"""
        if self._text_entry is not None:
            self._text_entry.delete(0, "end")
            self._text_entry.insert(0, text)

    def update_duration(self, duration: float) -> None:
        self._duration_label.configure(text=f"{duration:.1f}s")
        self._play_btn.configure(state="normal")

    def set_revert_enabled(self, enabled: bool) -> None:
        if self._revert_btn is not None:
            self._revert_btn.configure(
                state="normal" if enabled else "disabled",
            )

    def set_regenerating(self, busy: bool) -> None:
        """設定重新產生中狀態"""
        if self._regen_btn is not None:
            if busy:
                self._regen_btn.configure(state="disabled", text="產生中...")
            else:
                self._regen_btn.configure(state="normal", text="重新產生")

    def _handle_play(self) -> None:
        if self._on_play:
            self._on_play(self._index)

    def _handle_regenerate(self) -> None:
        if self._on_regenerate and self._text_entry is not None:
            new_text = self._text_entry.get().strip()
            if new_text:
                self._on_regenerate(self._index, new_text)

    def _handle_revert(self) -> None:
        if self._on_revert:
            self._on_revert(self._index)


class StepSidebar(ctk.CTkFrame):
    """左側垂直步驟導航欄"""

    # 每個步驟的主題色（light, dark）
    _STEP_COLORS = [
        ("#4A90D9", "#3A7BC8"),  # 0 歡迎 - 藍
        ("#D97706", "#C06A05"),  # 1 簡報 - 橘
        ("#059669", "#048A5F"),  # 2 講稿 - 綠
        ("#7C3AED", "#6D28D9"),  # 3 編輯 - 紫
        ("#DC2626", "#B91C1C"),  # 4 語音 - 紅
        ("#0891B2", "#0E7490"),  # 5 匯出 - 青
    ]
    # 已完成步驟 — 保留步驟色但降低飽和度
    _DONE_ALPHA = 0.45
    _COLOR_LOCKED = ("#3A3A3A", "#2A2A2A")
    _TEXT_ACTIVE = "white"
    _TEXT_DONE = "#E0E0E0"
    _TEXT_LOCKED = ("#555555", "#505050")

    def __init__(
        self,
        parent,
        steps: List[dict],
        on_step_click: Optional[Callable[[int], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, width=140, **kwargs)
        self.pack_propagate(False)

        self._steps = steps
        self._on_step_click = on_step_click
        self._current = 0
        self._max_reached = 0
        self._buttons: list[ctk.CTkButton] = []

        for i, step in enumerate(steps):
            btn = ctk.CTkButton(
                self,
                text=f" {step['icon']}  {step['name']}",
                font=ctk.CTkFont(size=13),
                anchor="w",
                height=42,
                corner_radius=6,
                command=lambda idx=i: self._handle_click(idx),
            )
            btn.pack(fill="x", padx=6, pady=3)
            self._buttons.append(btn)

        self._refresh()

    def set_current(self, index: int) -> None:
        self._current = index
        if index > self._max_reached:
            self._max_reached = index
        self._refresh()

    def set_max_reached(self, index: int) -> None:
        self._max_reached = index
        self._refresh()

    def _handle_click(self, index: int) -> None:
        if index <= self._max_reached and self._on_step_click:
            self._on_step_click(index)

    @staticmethod
    def _dim_color(hex_color: str, factor: float) -> str:
        """將 hex 顏色混合黑色，factor=0 全黑, factor=1 原色"""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _get_step_color(self, index: int, dimmed: bool = False) -> str:
        """取得步驟顏色（dark mode 用第二個值）"""
        pair = self._STEP_COLORS[index % len(self._STEP_COLORS)]
        color = pair[1]  # dark mode
        if dimmed:
            return self._dim_color(color, self._DONE_ALPHA)
        return color

    def _refresh(self) -> None:
        for i, btn in enumerate(self._buttons):
            if i == self._current:
                fg = self._get_step_color(i)
                hover = fg
                text_clr = self._TEXT_ACTIVE
                state = "normal"
            elif i <= self._max_reached:
                fg = self._get_step_color(i, dimmed=True)
                hover = self._get_step_color(i)
                text_clr = self._TEXT_DONE
                state = "normal"
            else:
                fg = self._COLOR_LOCKED
                hover = fg
                text_clr = self._TEXT_LOCKED
                state = "disabled"

            btn.configure(
                fg_color=fg, hover_color=hover,
                text_color=text_clr, state=state,
            )


class EditableSentenceItem(ctk.CTkFrame):
    """講稿編輯頁用的可編輯句子元件"""

    def __init__(
        self,
        parent,
        index: int,
        text: str,
        on_delete: Optional[Callable[[int], None]] = None,
        on_insert: Optional[Callable[[int], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self._index = index
        self._on_delete = on_delete
        self._on_insert = on_insert

        # 序號
        self._idx_label = ctk.CTkLabel(
            self, text=f"{index + 1}.", width=30,
            font=ctk.CTkFont(size=12),
        )
        self._idx_label.pack(side="left", padx=(5, 2))

        # 文字 Entry
        self._text_entry = ctk.CTkEntry(
            self, font=ctk.CTkFont(size=13),
        )
        self._text_entry.insert(0, text)
        self._text_entry.pack(side="left", fill="x", expand=True, padx=2)

        # 插入按鈕
        if on_insert is not None:
            ctk.CTkButton(
                self, text="+", width=30, height=24,
                font=ctk.CTkFont(size=13),
                fg_color="#2E8B57",
                command=self._handle_insert,
            ).pack(side="left", padx=2)

        # 刪除按鈕
        if on_delete is not None:
            ctk.CTkButton(
                self, text="✕", width=30, height=24,
                font=ctk.CTkFont(size=13),
                fg_color="#C0392B",
                command=self._handle_delete,
            ).pack(side="left", padx=(2, 5))

    @property
    def current_text(self) -> str:
        return self._text_entry.get().strip()

    def set_index(self, index: int) -> None:
        self._index = index
        self._idx_label.configure(text=f"{index + 1}.")

    def _handle_delete(self) -> None:
        if self._on_delete:
            self._on_delete(self._index)

    def _handle_insert(self) -> None:
        if self._on_insert:
            self._on_insert(self._index)
