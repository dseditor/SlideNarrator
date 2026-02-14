"""步驟 0：歡迎頁"""
import customtkinter as ctk


class StepWelcome:
    """歡迎頁 — 新建或載入專案"""

    def __init__(self, parent: ctk.CTkFrame, shared_state, app):
        self.parent = parent
        self.state = shared_state
        self.app = app
        self._build_ui()

    def _build_ui(self) -> None:
        # 置中容器
        center = ctk.CTkFrame(self.parent, fg_color="transparent")
        center.place(relx=0.5, rely=0.4, anchor="center")

        # 標題
        ctk.CTkLabel(
            center,
            text="📊 簡報自動旁白系統",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).pack(pady=(0, 8))

        ctk.CTkLabel(
            center,
            text="快速開始您的簡報旁白製作",
            font=ctk.CTkFont(size=15),
            text_color="gray",
        ).pack(pady=(0, 40))

        # 按鈕列
        btn_row = ctk.CTkFrame(center, fg_color="transparent")
        btn_row.pack()

        # 新建專案
        new_frame = ctk.CTkFrame(btn_row, corner_radius=12)
        new_frame.pack(side="left", padx=20)

        ctk.CTkButton(
            new_frame,
            text="📄 新建專案",
            font=ctk.CTkFont(size=16, weight="bold"),
            width=180,
            height=50,
            command=self._on_new_project,
        ).pack(padx=20, pady=(20, 8))

        ctk.CTkLabel(
            new_frame,
            text="從頭開始建立",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(pady=(0, 16))

        # 載入專案
        load_frame = ctk.CTkFrame(btn_row, corner_radius=12)
        load_frame.pack(side="left", padx=20)

        ctk.CTkButton(
            load_frame,
            text="📂 載入專案",
            font=ctk.CTkFont(size=16, weight="bold"),
            width=180,
            height=50,
            fg_color="#7B68EE",
            command=self._on_load_project,
        ).pack(padx=20, pady=(20, 8))

        ctk.CTkLabel(
            load_frame,
            text="開啟既有專案",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(pady=(0, 16))

    def _on_new_project(self) -> None:
        self.app.goto_step(1)

    def _on_load_project(self) -> None:
        self.app.load_project_and_jump()
