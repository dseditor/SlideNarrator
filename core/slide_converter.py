"""簡報轉圖片模組 -- 支援 PDF 和 PPTX"""
import logging
from pathlib import Path
from typing import List

from config import DEFAULT_SLIDE_DPI

logger = logging.getLogger(__name__)


def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = DEFAULT_SLIDE_DPI,
) -> List[str]:
    """
    PDF 轉圖片，使用 PyMuPDF (fitz)。
    每頁產生一張 PNG，回傳圖片路徑列表。
    """
    import fitz

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths: List[str] = []

    matrix = fitz.Matrix(dpi / 72, dpi / 72)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        output_path = str(Path(output_dir) / f"slide_{page_num + 1:03d}.png")
        pix.save(output_path)
        image_paths.append(output_path)
        logger.info("轉換第 %d/%d 頁", page_num + 1, len(doc))

    doc.close()
    logger.info("PDF 轉圖片完成: %d 頁", len(image_paths))
    return image_paths


def pptx_to_pdf(
    pptx_path: str,
    output_pdf_path: str,
) -> str:
    """
    PPTX 轉 PDF，使用 comtypes 透過 PowerPoint COM。
    需要系統已安裝 Microsoft PowerPoint。
    """
    import comtypes.client

    pptx_abs = str(Path(pptx_path).resolve())
    pdf_abs = str(Path(output_pdf_path).resolve())

    Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)

    powerpoint = None
    presentation = None
    try:
        powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
        powerpoint.Visible = 1

        presentation = powerpoint.Presentations.Open(pptx_abs)
        # 32 = ppSaveAsPDF
        presentation.SaveAs(pdf_abs, 32)
        logger.info("PPTX 轉 PDF 完成: %s", pdf_abs)
        return pdf_abs
    except OSError as e:
        raise RuntimeError(
            "無法啟動 PowerPoint。請確認已安裝 Microsoft PowerPoint。"
        ) from e
    finally:
        if presentation:
            try:
                presentation.Close()
            except Exception:
                pass
        if powerpoint:
            try:
                powerpoint.Quit()
            except Exception:
                pass


def pptx_to_images(
    pptx_path: str,
    output_dir: str,
    dpi: int = DEFAULT_SLIDE_DPI,
) -> List[str]:
    """PPTX 轉圖片：PPTX -> PDF -> images"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    temp_pdf = str(Path(output_dir) / "_temp_slides.pdf")

    pptx_to_pdf(pptx_path, temp_pdf)
    images = pdf_to_images(temp_pdf, output_dir, dpi)

    # 清理暫存 PDF
    try:
        Path(temp_pdf).unlink()
    except Exception:
        pass

    return images


def convert_slides(
    slide_path: str,
    output_dir: str,
    dpi: int = DEFAULT_SLIDE_DPI,
) -> List[str]:
    """統一入口：根據副檔名自動選擇轉換方式"""
    ext = Path(slide_path).suffix.lower()
    if ext == ".pdf":
        return pdf_to_images(slide_path, output_dir, dpi)
    elif ext in (".pptx", ".ppt"):
        return pptx_to_images(slide_path, output_dir, dpi)
    else:
        raise ValueError(f"不支援的檔案格式: {ext}（僅支援 .pdf, .pptx）")
