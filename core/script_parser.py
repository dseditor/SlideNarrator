"""講稿解析模組 -- 智慧解析多種格式的簡報講稿

支援的頁面標題格式：
  - Page1: / Page 1: / Page1：
  - 第1頁: / 第1頁： / 第 1 頁：
  - 第一頁: / 第一頁：
  - 【第1頁】 / ---第1頁---

支援的內容格式：
  - 每句獨立一行（scr.txt 風格）
  - 同行多句空格分隔（s2.txt / Gemini 風格）
  - 混合格式
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Sentence:
    """單一句子"""
    text: str
    page_index: int
    sentence_index: int
    audio_path: Optional[str] = None
    duration_sec: float = 0.0
    start_sec: float = 0.0


@dataclass
class Page:
    """單一頁面"""
    page_number: int
    page_index: int
    sentences: List[Sentence] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(s.duration_sec for s in self.sentences)


@dataclass
class Script:
    """完整講稿"""
    pages: List[Page] = field(default_factory=list)

    @property
    def total_sentences(self) -> int:
        return sum(len(p.sentences) for p in self.pages)

    @property
    def total_duration(self) -> float:
        return sum(p.total_duration for p in self.pages)


# ── 中文數字轉阿拉伯數字 ──

_CN_DIGITS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "壹": 1, "貳": 2, "參": 3, "肆": 4, "伍": 5,
    "陸": 6, "柒": 7, "捌": 8, "玖": 9, "拾": 10,
}


def _chinese_num_to_int(cn: str) -> Optional[int]:
    """將簡單中文數字轉為整數，例如 '一' -> 1, '十二' -> 12, '二十三' -> 23"""
    cn = cn.strip()
    if not cn:
        return None

    # 單字
    if cn in _CN_DIGITS:
        return _CN_DIGITS[cn]

    # 十幾
    if len(cn) == 2 and cn[0] == "十" and cn[1] in _CN_DIGITS:
        return 10 + _CN_DIGITS[cn[1]]

    # 幾十 / 幾十幾
    if "十" in cn:
        parts = cn.split("十")
        tens = _CN_DIGITS.get(parts[0], 0) if parts[0] else 1
        ones = _CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones

    return None


# ── 頁面標題匹配 ──

# 英文格式: Page1: / Page 1: / page1：
_PAGE_EN = re.compile(
    r'^Page\s*(\d+)\s*[:：]\s*(.*)',
    re.IGNORECASE,
)

# 中文數字格式: 第1頁: / 第 1 頁： / 第1頁
_PAGE_CN_ARABIC = re.compile(
    r'^[【\-─]*\s*第\s*(\d+)\s*頁\s*[】\-─]*\s*[:：]?\s*(.*)',
)

# 中文大寫數字: 第一頁: / 第二十三頁：
_PAGE_CN_CHAR = re.compile(
    r'^[【\-─]*\s*第\s*([一二三四五六七八九十壹貳參肆伍陸柒捌玖拾]+)\s*頁\s*[】\-─]*\s*[:：]?\s*(.*)',
)


def _match_page_header(line: str) -> Optional[Tuple[int, str]]:
    """
    嘗試匹配頁面標題，回傳 (page_number, remaining_content) 或 None。
    remaining_content 是頁面標題後面的剩餘內容（可能為空）。
    """
    stripped = line.strip()

    # 英文格式
    m = _PAGE_EN.match(stripped)
    if m:
        return int(m.group(1)), m.group(2).strip()

    # 中文阿拉伯數字格式
    m = _PAGE_CN_ARABIC.match(stripped)
    if m:
        return int(m.group(1)), m.group(2).strip()

    # 中文大寫數字格式
    m = _PAGE_CN_CHAR.match(stripped)
    if m:
        num = _chinese_num_to_int(m.group(1))
        if num is not None:
            return num, m.group(2).strip()

    return None


# ── 智慧分句 ──

# 中文標點作為分句依據
_SENTENCE_TERMINATORS = re.compile(r'[。！？；\n]')

# 至少要有幾個字才算一句（過短的忽略或合併）
_MIN_SENTENCE_LEN = 4


def _split_into_sentences(text: str) -> List[str]:
    """
    將一段文字智慧分割為多個句子。

    策略（依優先順序）：
    1. 如果文字中有換行，按換行分割
    2. 如果有中文句號/問號/驚嘆號/分號，按標點分割
    3. 如果以上都沒有，嘗試按空格分割（Gemini 輸出格式）
    4. 如果都不適用，整段作為一句
    """
    text = text.strip()
    if not text:
        return []

    # 策略 1: 按換行分割
    if "\n" in text:
        lines = [l.strip() for l in text.splitlines()]
        return [l for l in lines if l]

    # 策略 2: 按中文標點分割
    if _SENTENCE_TERMINATORS.search(text):
        parts = _SENTENCE_TERMINATORS.split(text)
        sentences = [p.strip() for p in parts if p.strip()]
        # 合併過短的句子
        return _merge_short_sentences(sentences)

    # 策略 3: 按空格分割（Gemini 的 s2.txt 格式）
    # 檢查是否有多個長段落被空格分隔
    space_parts = text.split(" ")
    # 過濾掉空白片段
    space_parts = [p.strip() for p in space_parts if p.strip()]

    if len(space_parts) > 1:
        # 計算每個片段的平均長度，如果夠長就視為獨立句子
        avg_len = sum(len(p) for p in space_parts) / len(space_parts)
        if avg_len >= _MIN_SENTENCE_LEN:
            return _merge_short_sentences(space_parts)

    # 策略 4: 整段作為一句
    return [text]


def _merge_short_sentences(sentences: List[str]) -> List[str]:
    """合併過短的句子到前一句"""
    if not sentences:
        return []

    merged: List[str] = []
    for s in sentences:
        if merged and len(s) < _MIN_SENTENCE_LEN:
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)
    return merged


# ── 主要解析函數 ──

def parse_script(text: str) -> Script:
    """
    智慧解析講稿文字。

    自動識別以下格式：
    - s2.txt 風格: "Page1: 句子1 句子2 句子3"（同行空格分隔）
    - scr.txt 風格: "第1頁：\\n句子1\\n句子2"（每句獨立一行）
    - 混合格式

    頁面標題本身不會被加入句子（不會被 TTS 朗讀）。
    """
    # 移除 BOM
    if text.startswith("\ufeff"):
        text = text[1:]

    lines = text.splitlines()
    script = Script()
    current_page: Optional[Page] = None
    page_index = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # 嘗試匹配頁面標題
        header = _match_page_header(stripped)
        if header is not None:
            page_number, remaining = header

            # 建立新頁面
            current_page = Page(
                page_number=page_number,
                page_index=page_index,
            )
            script.pages.append(current_page)
            page_index += 1

            # 處理標題同一行的剩餘內容（s2.txt 格式的關鍵）
            if remaining:
                for sent_text in _split_into_sentences(remaining):
                    sentence = Sentence(
                        text=sent_text,
                        page_index=current_page.page_index,
                        sentence_index=len(current_page.sentences),
                    )
                    current_page.sentences.append(sentence)
            continue

        # 不是頁面標題，是普通內容行
        if current_page is None:
            # 還沒遇到任何頁面標題，建立預設第 1 頁
            current_page = Page(page_number=1, page_index=0)
            script.pages.append(current_page)
            page_index = 1

        # 分句後加入
        for sent_text in _split_into_sentences(stripped):
            sentence = Sentence(
                text=sent_text,
                page_index=current_page.page_index,
                sentence_index=len(current_page.sentences),
            )
            current_page.sentences.append(sentence)

    return script


def parse_script_file(filepath: str, encoding: str = "utf-8") -> Script:
    """從檔案載入並解析講稿"""
    path = Path(filepath)
    # 嘗試 utf-8，失敗則 utf-8-sig（BOM）
    try:
        text = path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    return parse_script(text)


def validate_script(script: Script, slide_count: int = 0) -> List[str]:
    """
    驗證講稿完整性。
    slide_count: 如果有簡報，傳入簡報頁數進行交叉驗證。
    回傳警告訊息列表。
    """
    warnings: List[str] = []

    if not script.pages:
        warnings.append("講稿中沒有任何頁面")
        return warnings

    for page in script.pages:
        if not page.sentences:
            warnings.append(f"第 {page.page_number} 頁沒有任何句子")

    # 頁碼連續性
    page_numbers = [p.page_number for p in script.pages]
    for i in range(1, len(page_numbers)):
        if page_numbers[i] != page_numbers[i - 1] + 1:
            warnings.append(
                f"頁碼不連續：第 {page_numbers[i-1]} 頁之後是第 {page_numbers[i]} 頁"
            )

    # 與簡報頁數交叉驗證
    if slide_count > 0:
        script_pages = len(script.pages)
        if script_pages != slide_count:
            warnings.append(
                f"講稿有 {script_pages} 頁，但簡報有 {slide_count} 頁，數量不一致"
            )

    return warnings


def format_script_preview(script: Script) -> str:
    """
    產生講稿的結構化預覽文字，讓使用者清楚看到解析結果。
    """
    lines: List[str] = []
    for page in script.pages:
        lines.append(f"═══ 第 {page.page_number} 頁 ({len(page.sentences)} 句) ═══")
        for i, s in enumerate(page.sentences):
            text_preview = s.text[:50] + "..." if len(s.text) > 50 else s.text
            lines.append(f"  {i + 1}. {text_preview}")
        lines.append("")

    total = script.total_sentences
    lines.append(f"合計: {len(script.pages)} 頁, {total} 句")
    return "\n".join(lines)
