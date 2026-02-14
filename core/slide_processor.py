"""投影片圖片預處理模組 -- 字幕空間擴展"""
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from config import SUBTITLE_SPACE_MULTIPLIER

logger = logging.getLogger(__name__)


def _get_bottom_row_median_color(img: Image.Image) -> Tuple[int, int, int]:
    """取得圖片最底部一列像素的中位數顏色"""
    arr = np.array(img)
    bottom_row = arr[-1, :, :3]  # 最底列，取 RGB
    median_color = tuple(int(v) for v in np.median(bottom_row, axis=0))
    return median_color


def prepare_slides_with_subtitle_space(
    slide_images: List[str],
    target_resolution: Tuple[int, int],
    font_size: int = 24,
    output_dir: str = "",
) -> List[str]:
    """
    為每張投影片圖片加上底部字幕空間。

    邏輯：
    1. 計算字幕區高度 = font_size * SUBTITLE_SPACE_MULTIPLIER
    2. 內容區高度 = target_h - subtitle_h
    3. 取圖片最底部一列像素的中位數顏色作為填充色
    4. 建立 target_w x target_h 畫布，底部填色
    5. 將投影片等比縮放到 target_w x content_h，居中放置於畫布頂部

    回傳處理後的圖片路徑列表。
    """
    target_w, target_h = target_resolution
    subtitle_h = font_size * SUBTITLE_SPACE_MULTIPLIER
    content_h = target_h - subtitle_h

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_paths: List[str] = []

    for i, img_path in enumerate(slide_images):
        img = Image.open(img_path).convert("RGB")

        # 取底部顏色
        fill_color = _get_bottom_row_median_color(img)

        # 建立畫布，整張用填充色
        canvas = Image.new("RGB", (target_w, target_h), fill_color)

        # 等比縮放投影片到 content 區域
        src_w, src_h = img.size
        scale = min(target_w / src_w, content_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        resized = img.resize((new_w, new_h), Image.LANCZOS)

        # 居中放置於畫布頂部（content 區域內）
        x_offset = (target_w - new_w) // 2
        y_offset = (content_h - new_h) // 2
        canvas.paste(resized, (x_offset, y_offset))

        # 儲存
        out_path = out_dir / f"slide_sub_{i + 1:03d}.png"
        canvas.save(str(out_path), "PNG")
        processed_paths.append(str(out_path))

    logger.info(
        "字幕空間處理完成: %d 張圖片, 字幕區高度=%dpx",
        len(processed_paths),
        subtitle_h,
    )
    return processed_paths
