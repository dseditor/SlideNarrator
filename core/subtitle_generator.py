"""SRT 字幕生成模組"""
import logging
from pathlib import Path

from core.script_parser import Script

logger = logging.getLogger(__name__)


def format_srt_time(seconds: float) -> str:
    """
    將秒數轉為 SRT 時間格式 HH:MM:SS,mmm
    例如 65.5 -> '00:01:05,500'
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(script: Script) -> str:
    """
    根據 Script 中每個 Sentence 的 start_sec 和 duration_sec 生成 SRT 字幕。

    時間軸邏輯：
    - 直接使用音訊處理階段記錄的精確起始時間 start_sec
    - 每句字幕的結束時間 = start_sec + duration_sec
    - 不再自行估算停頓，完全與實際音訊同步
    """
    lines = []
    index = 1

    for page in script.pages:
        for sentence in page.sentences:
            if sentence.duration_sec <= 0:
                continue

            start = sentence.start_sec
            end = start + sentence.duration_sec

            lines.append(str(index))
            lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
            lines.append(sentence.text)
            lines.append("")

            index += 1

    return "\n".join(lines)


def save_srt(
    srt_content: str,
    output_path: str,
    encoding: str = "utf-8-sig",
) -> None:
    """儲存 SRT 檔案，預設使用 UTF-8-BOM 編碼"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(srt_content, encoding=encoding)
    logger.info("SRT 字幕已儲存: %s", output_path)
