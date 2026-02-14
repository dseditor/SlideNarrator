"""專案保存/載入模組 -- ZIP 打包與解壓"""
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 專案格式版本
_FORMAT_VERSION = "1.0"


def save_project(
    output_path: str,
    slide_images: List[str],
    script_text: str,
    page_audio_paths: List[str],
    sentence_audios: List[dict],
) -> str:
    """
    將目前工作狀態打包為 ZIP。

    Parameters
    ----------
    output_path : str
        輸出 .zip 路徑
    slide_images : list[str]
        投影片圖片路徑列表
    script_text : str
        原始講稿文字
    page_audio_paths : list[str]
        頁面完整音訊 WAV 路徑列表
    sentence_audios : list[dict]
        單句音訊，每個 dict 含 {page, sent_idx, path}

    Returns
    -------
    str
        實際寫入的 ZIP 路徑
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(output), "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. 投影片圖片
        for i, img_path in enumerate(slide_images):
            p = Path(img_path)
            if p.exists():
                arc_name = f"slides/slide_{i + 1:03d}{p.suffix}"
                zf.write(str(p), arc_name)

        # 2. 講稿原文
        zf.writestr("script.txt", script_text)

        # 3. 頁面完整音訊
        for i, audio_path in enumerate(page_audio_paths):
            p = Path(audio_path)
            if p.exists():
                arc_name = f"audio/page{i + 1:03d}_full.wav"
                zf.write(str(p), arc_name)

        # 4. 單句音訊
        for info in sentence_audios:
            p = Path(info["path"])
            if p.exists():
                arc_name = f"audio/page{info['page']:03d}_sent{info['sent_idx']:03d}.wav"
                zf.write(str(p), arc_name)

        # 5. manifest.json
        manifest = {
            "version": _FORMAT_VERSION,
            "slide_count": len(slide_images),
            "page_count": len(page_audio_paths),
            "sentence_count": len(sentence_audios),
            "slides": [
                f"slides/slide_{i + 1:03d}{Path(p).suffix}"
                for i, p in enumerate(slide_images)
                if Path(p).exists()
            ],
            "page_audios": [
                f"audio/page{i + 1:03d}_full.wav"
                for i, p in enumerate(page_audio_paths)
                if Path(p).exists()
            ],
            "sentence_audios": [
                {
                    "page": info["page"],
                    "sent_idx": info["sent_idx"],
                    "arc_name": f"audio/page{info['page']:03d}_sent{info['sent_idx']:03d}.wav",
                }
                for info in sentence_audios
                if Path(info["path"]).exists()
            ],
        }
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    logger.info("專案已保存: %s", output)
    return str(output)


def load_project(
    zip_path: str,
    extract_dir: str,
) -> Dict:
    """
    從 ZIP 解壓並還原專案。

    Parameters
    ----------
    zip_path : str
        .zip 路徑
    extract_dir : str
        解壓目標目錄

    Returns
    -------
    dict
        {"slide_images": [...], "script_text": "...",
         "page_audio_paths": [...], "sentence_audios": [...]}
    """
    extract = Path(extract_dir)
    if extract.exists():
        shutil.rmtree(str(extract))
    extract.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract))

    # 讀取 manifest
    manifest_path = extract / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    # 1. 投影片圖片
    slide_images = []
    slides_dir = extract / "slides"
    if slides_dir.exists():
        for f in sorted(slides_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                slide_images.append(str(f))

    # 2. 講稿
    script_path = extract / "script.txt"
    script_text = ""
    if script_path.exists():
        script_text = script_path.read_text(encoding="utf-8")

    # 3. 頁面完整音訊
    page_audio_paths = []
    audio_dir = extract / "audio"
    if audio_dir.exists():
        for f in sorted(audio_dir.iterdir()):
            if f.name.endswith("_full.wav"):
                page_audio_paths.append(str(f))

    # 4. 單句音訊
    sentence_audios = []
    if "sentence_audios" in manifest:
        for info in manifest["sentence_audios"]:
            arc_path = extract / info["arc_name"]
            if arc_path.exists():
                sentence_audios.append({
                    "page": info["page"],
                    "sent_idx": info["sent_idx"],
                    "path": str(arc_path),
                })
    elif audio_dir and audio_dir.exists():
        # fallback: 從檔名推斷
        import re
        for f in sorted(audio_dir.iterdir()):
            m = re.match(r"page(\d+)_sent(\d+)\.wav", f.name)
            if m:
                sentence_audios.append({
                    "page": int(m.group(1)),
                    "sent_idx": int(m.group(2)),
                    "path": str(f),
                })

    logger.info("專案已載入: %d 張投影片, %d 頁音訊, %d 句音訊",
                len(slide_images), len(page_audio_paths), len(sentence_audios))

    return {
        "slide_images": slide_images,
        "script_text": script_text,
        "page_audio_paths": page_audio_paths,
        "sentence_audios": sentence_audios,
    }
