"""步驟 3：講稿編輯對照"""
import copy
import logging

import customtkinter as ctk
from PIL import Image

from core.script_parser import Sentence
from ui.widgets import EditableSentenceItem

logger = logging.getLogger(__name__)


class StepEdit:
    """講稿編輯 — 左圖右文對照、逐句增刪"""

    def __init__(self, parent: ctk.CTkFrame, shared_state, app):
        self.parent = parent
        self.state = shared_state
        self.app = app

        self._current_page_idx = 0
        self._page_backups: dict = {}  # page_index -> original sentences list
        self._sentence_items: list[EditableSentenceItem] = []

        self._build_ui()

    def _build_ui(self) -> None:
        # 標題
        ctk.CTkLabel(
            self.parent, text="✏️ 編輯講稿",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=15, pady=(15, 5))

        self._page_info_label = ctk.CTkLabel(
            self.parent,
            text="",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        self._page_info_label.pack(anchor="w", padx=15, pady=(0, 8))

        # 主要內容：左圖右文
        content = ctk.CTkFrame(self.parent, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=(0, 5))

        # 左側：投影片圖片
        left = ctk.CTkFrame(content, width=400)
        left.pack(side="left", fill="both", padx=(0, 10))
        left.pack_propagate(False)

        self._slide_image_label = ctk.CTkLabel(
            left, text="（尚無投影片）",
            font=ctk.CTkFont(size=14),
        )
        self._slide_image_label.pack(expand=True)

        # 右側：句子列表
        right = ctk.CTkFrame(content)
        right.pack(side="left", fill="both", expand=True)

        self._sentence_scroll = ctk.CTkScrollableFrame(right)
        self._sentence_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # 復原按鈕
        self._revert_btn = ctk.CTkButton(
            right, text="復原此頁所有變更", width=160,
            fg_color="gray",
            command=self._revert_page,
        )
        self._revert_btn.pack(pady=(5, 8))

        # 底部縮圖列
        thumb_section = ctk.CTkFrame(self.parent)
        thumb_section.pack(fill="x", padx=15, pady=(0, 5))

        self._thumb_scroll = ctk.CTkScrollableFrame(
            thumb_section, height=90, orientation="horizontal",
        )
        self._thumb_scroll.pack(fill="x", padx=5, pady=5)

        # 底部翻頁
        nav = ctk.CTkFrame(self.parent, fg_color="transparent")
        nav.pack(fill="x", padx=15, pady=(0, 8))

        self._prev_page_btn = ctk.CTkButton(
            nav, text="◀ 上一頁", width=100,
            command=self._prev_page,
        )
        self._prev_page_btn.pack(side="left")

        self._page_label = ctk.CTkLabel(
            nav, text="第 0 / 0 頁",
            font=ctk.CTkFont(size=13),
        )
        self._page_label.pack(side="left", expand=True)

        self._next_page_btn = ctk.CTkButton(
            nav, text="下一頁 ▶", width=100,
            command=self._next_page,
        )
        self._next_page_btn.pack(side="right")

        # 縮圖相關
        self._thumb_labels: list = []
        self._thumb_images: list = []  # 防止 GC

    def on_enter(self) -> None:
        """進入此步驟時呼叫，備份並載入"""
        if not self.state.script:
            return

        # 備份每頁原始句子（只在初次進入或尚未備份時）
        for page in self.state.script.pages:
            if page.page_index not in self._page_backups:
                self._page_backups[page.page_index] = [
                    copy.copy(s) for s in page.sentences
                ]

        self._build_thumbnail_strip()
        self._current_page_idx = 0
        self._load_page(0)

    def _load_page(self, page_idx: int) -> None:
        """載入指定頁面"""
        if not self.state.script or page_idx < 0:
            return
        if page_idx >= len(self.state.script.pages):
            return

        self._current_page_idx = page_idx
        page = self.state.script.pages[page_idx]
        total_pages = len(self.state.script.pages)

        # 更新頁面資訊
        self._page_info_label.configure(
            text=f"第 {page.page_number} 頁 (共 {total_pages} 頁, {len(page.sentences)} 句)",
        )
        self._page_label.configure(
            text=f"第 {page_idx + 1} / {total_pages} 頁",
        )

        # 更新翻頁按鈕
        self._prev_page_btn.configure(
            state="normal" if page_idx > 0 else "disabled",
        )
        self._next_page_btn.configure(
            state="normal" if page_idx < total_pages - 1 else "disabled",
        )

        # 載入投影片圖片
        self._load_slide_image(page_idx)

        # 載入句子列表
        self._build_sentence_list(page)

        # 更新縮圖高亮
        self._update_thumbnail_highlight(page_idx)

    def _load_slide_image(self, page_idx: int) -> None:
        """載入投影片圖片"""
        if page_idx < len(self.state.slide_images):
            try:
                img = Image.open(self.state.slide_images[page_idx])
                # 縮放到適當大小
                max_w, max_h = 380, 400
                ratio = min(max_w / img.width, max_h / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img_resized = img.resize(new_size, Image.LANCZOS)

                ctk_img = ctk.CTkImage(
                    light_image=img_resized,
                    size=new_size,
                )
                self._slide_image_label.configure(image=ctk_img, text="")
                self._slide_image_label._ctk_img = ctk_img
            except Exception as e:
                logger.error("載入投影片圖片失敗: %s", e)
                self._slide_image_label.configure(
                    image=None, text=f"P{page_idx + 1}\n(載入失敗)",
                )
        else:
            self._slide_image_label.configure(
                image=None, text="（無對應投影片）",
            )

    def _build_sentence_list(self, page) -> None:
        """建立句子編輯列表"""
        for widget in self._sentence_scroll.winfo_children():
            widget.destroy()
        self._sentence_items.clear()

        for i, sentence in enumerate(page.sentences):
            item = EditableSentenceItem(
                self._sentence_scroll,
                index=i,
                text=sentence.text,
                on_delete=self._delete_sentence,
                on_insert=self._insert_sentence,
            )
            item.pack(fill="x", padx=2, pady=2)
            self._sentence_items.append(item)

    def _save_current_page_edits(self) -> None:
        """儲存當前頁面的編輯到 Script 物件"""
        if not self.state.script:
            return
        if self._current_page_idx >= len(self.state.script.pages):
            return

        page = self.state.script.pages[self._current_page_idx]
        for i, item in enumerate(self._sentence_items):
            if i < len(page.sentences):
                page.sentences[i].text = item.current_text

    def _delete_sentence(self, index: int) -> None:
        """刪除指定句子"""
        if not self.state.script:
            return
        page = self.state.script.pages[self._current_page_idx]

        # 至少保留 1 句
        if len(page.sentences) <= 1:
            return

        # 先儲存所有編輯中的文字
        self._save_current_page_edits()

        # 刪除句子
        page.sentences.pop(index)

        # 重新編號
        for i, s in enumerate(page.sentences):
            s.sentence_index = i

        # 重建 UI
        self._build_sentence_list(page)
        self._page_info_label.configure(
            text=f"第 {page.page_number} 頁 (共 {len(self.state.script.pages)} 頁, {len(page.sentences)} 句)",
        )

    def _insert_sentence(self, after_index: int) -> None:
        """在指定位置後插入新句子"""
        if not self.state.script:
            return
        page = self.state.script.pages[self._current_page_idx]

        # 先儲存所有編輯中的文字
        self._save_current_page_edits()

        # 建立新句子
        new_sentence = Sentence(
            text="",
            page_index=page.page_index,
            sentence_index=after_index + 1,
        )
        page.sentences.insert(after_index + 1, new_sentence)

        # 重新編號
        for i, s in enumerate(page.sentences):
            s.sentence_index = i

        # 重建 UI
        self._build_sentence_list(page)
        self._page_info_label.configure(
            text=f"第 {page.page_number} 頁 (共 {len(self.state.script.pages)} 頁, {len(page.sentences)} 句)",
        )

    def _revert_page(self) -> None:
        """復原此頁到進入此步驟時的狀態"""
        if not self.state.script:
            return

        page = self.state.script.pages[self._current_page_idx]
        backup = self._page_backups.get(page.page_index)
        if not backup:
            return

        page.sentences = [copy.copy(s) for s in backup]
        self._build_sentence_list(page)
        self._page_info_label.configure(
            text=f"第 {page.page_number} 頁 (共 {len(self.state.script.pages)} 頁, {len(page.sentences)} 句)",
        )

    def _prev_page(self) -> None:
        self._save_current_page_edits()
        self._load_page(self._current_page_idx - 1)

    def _next_page(self) -> None:
        self._save_current_page_edits()
        self._load_page(self._current_page_idx + 1)

    def _build_thumbnail_strip(self) -> None:
        """建立底部縮圖列"""
        for widget in self._thumb_scroll.winfo_children():
            widget.destroy()
        self._thumb_labels.clear()
        self._thumb_images.clear()

        for i, img_path in enumerate(self.state.slide_images):
            frame = ctk.CTkFrame(self._thumb_scroll, corner_radius=4)
            frame.pack(side="left", padx=3, pady=3)

            try:
                img = Image.open(img_path)
                img.thumbnail((100, 65))
                ctk_img = ctk.CTkImage(light_image=img, size=img.size)
                self._thumb_images.append(ctk_img)

                label = ctk.CTkLabel(
                    frame, image=ctk_img, text=f"P{i+1}",
                    compound="bottom", font=ctk.CTkFont(size=9),
                    cursor="hand2",
                )
                label.pack(padx=3, pady=3)
                label.bind("<Button-1>", lambda e, idx=i: self._jump_to_page(idx))
            except Exception:
                label = ctk.CTkLabel(
                    frame, text=f"P{i+1}", width=60, height=45,
                    font=ctk.CTkFont(size=10),
                    cursor="hand2",
                )
                label.pack(padx=3, pady=3)
                label.bind("<Button-1>", lambda e, idx=i: self._jump_to_page(idx))

            self._thumb_labels.append(frame)

    def _update_thumbnail_highlight(self, page_idx: int) -> None:
        """高亮當前頁縮圖"""
        for i, frame in enumerate(self._thumb_labels):
            if i == page_idx:
                frame.configure(border_width=2, border_color="#3B8ED0")
            else:
                frame.configure(border_width=0, border_color="transparent")

    def _jump_to_page(self, page_idx: int) -> None:
        """縮圖點擊跳頁"""
        self._save_current_page_edits()
        self._load_page(page_idx)

    def on_leave(self) -> None:
        """離開此步驟時呼叫，同步所有修改"""
        self._save_current_page_edits()
        # 重算 sentence_index
        if self.state.script:
            for page in self.state.script.pages:
                for i, s in enumerate(page.sentences):
                    s.sentence_index = i

    def can_proceed(self) -> bool:
        """至少每頁有 1 句"""
        if not self.state.script:
            return False
        for page in self.state.script.pages:
            if len(page.sentences) == 0:
                return False
            # 確認每句都有文字
            for s in page.sentences:
                if not s.text.strip():
                    return False
        return True
