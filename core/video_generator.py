"""影片生成模組 -- 使用 ffmpeg subprocess

核心策略：單次合成（single-pass）
  1. 合併所有頁面 WAV 為一個完整音訊
  2. 用 concat demuxer 指定每張投影片的顯示時長
  3. 一次 ffmpeg 指令產生最終影片
  避免了逐頁 AAC 編碼再拼接導致的時間軸累積偏移。
"""
import logging
import subprocess
import wave as wave_module
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import DEFAULT_VIDEO_RESOLUTION, FFMPEG_PATH

logger = logging.getLogger(__name__)

# Windows: 隱藏 console 視窗
_CREATE_NO_WINDOW = 0x08000000


def get_ffmpeg_path() -> str:
    """取得 ffmpeg 執行檔路徑"""
    if FFMPEG_PATH.exists():
        return str(FFMPEG_PATH)
    return "ffmpeg"


def _run_ffmpeg(args: List[str], description: str = "") -> None:
    """執行 ffmpeg 指令"""
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg] + args
    logger.info("執行 ffmpeg: %s", description or " ".join(cmd[:6]))
    logger.debug("完整指令: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=_CREATE_NO_WINDOW,
            timeout=600,
        )
        if result.returncode != 0:
            logger.error("ffmpeg stderr: %s", result.stderr[-500:] if result.stderr else "")
            raise RuntimeError(f"ffmpeg 失敗 ({description}): {result.stderr[-200:]}")
    except FileNotFoundError:
        raise RuntimeError(
            "找不到 ffmpeg。請將 ffmpeg.exe 放入 ffmpeg/ 資料夾，"
            "或確認 ffmpeg 已加入系統 PATH。"
        )


# ── 音訊工具 ──

def _concatenate_wav_files(
    wav_paths: List[str],
    output_path: str,
) -> None:
    """合併多個 WAV 檔案為一個（直接拼接 PCM 資料，零誤差）"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with wave_module.open(output_path, "wb") as out_wav:
        params_set = False
        for wav_path in wav_paths:
            with wave_module.open(wav_path, "rb") as in_wav:
                if not params_set:
                    out_wav.setparams(in_wav.getparams())
                    params_set = True
                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))

    # 驗證合併結果
    with wave_module.open(output_path, "rb") as wf:
        total_sec = wf.getnframes() / wf.getframerate()
    logger.info("合併音訊: %d 個檔案 → %.4f 秒", len(wav_paths), total_sec)


def _generate_silent_wav(
    output_path: str,
    duration_sec: float,
    sample_rate: int = 48000,
) -> None:
    """產生靜音 WAV 檔案"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    num_samples = int(duration_sec * sample_rate)
    silence = np.zeros(num_samples, dtype=np.int16)

    with wave_module.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silence.tobytes())


# ── 字幕 ──

def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_name: str = "Microsoft JhengHei",
    font_size: int = 24,
) -> str:
    """將 SRT 字幕燒錄進影片"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    srt_escaped = str(Path(srt_path).resolve()).replace("\\", "/")
    srt_escaped = srt_escaped.replace(":", "\\:")

    vf = (
        f"subtitles='{srt_escaped}'"
        f":force_style='FontName={font_name},FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'"
    )

    _run_ffmpeg(
        [
            "-i", str(video_path),
            "-vf", vf,
            "-c:a", "copy",
            "-y",
            str(output_path),
        ],
        description="燒錄字幕",
    )
    return output_path


# ── 主要影片生成 ──

def generate_full_video(
    slide_images: List[str],
    page_audio_paths: List[str],
    page_durations: List[float],
    srt_path: Optional[str],
    output_path: str,
    burn_srt: bool = False,
    resolution: Tuple[int, int] = DEFAULT_VIDEO_RESOLUTION,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    sample_rate: int = 48000,
) -> str:
    """
    單次合成完整影片。

    策略：
    1. 合併所有頁面 WAV → 一個完整音訊檔
    2. 用 concat demuxer 指定每張投影片的顯示時長
    3. 一次 ffmpeg 指令產生影片（音訊只編碼一次，無累積偏移）

    slide_images: 每頁簡報圖片路徑
    page_audio_paths: 每頁合併音訊 WAV 路徑
    page_durations: 每頁音訊時長（從 WAV 檔讀回的精確值）
    srt_path: SRT 檔案路徑（可為 None）
    output_path: 最終影片輸出路徑
    burn_srt: 是否燒錄字幕
    sample_rate: 音訊取樣率（用於產生靜音頁面）
    """
    temp_dir = Path(output_path).parent / "_temp_videos"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total = len(slide_images)
    steps = total + 3  # 準備 + 合成 + 字幕

    # ── 步驟 1: 準備每頁音訊 ──
    if progress_callback:
        progress_callback(1, steps, "正在準備音訊...")

    all_audio_paths: List[str] = []
    all_durations: List[float] = []

    # 從第一個有效的頁面音訊讀取實際取樣率（用於產生匹配的靜音 WAV）
    actual_sr = sample_rate
    for p_path, p_dur in zip(page_audio_paths, page_durations):
        if p_dur > 0 and Path(p_path).exists():
            with wave_module.open(p_path, "rb") as wf:
                actual_sr = wf.getframerate()
            logger.info("使用實際音訊取樣率: %d Hz", actual_sr)
            break

    for i in range(total):
        if i < len(page_audio_paths) and i < len(page_durations) and page_durations[i] > 0:
            all_audio_paths.append(page_audio_paths[i])
            all_durations.append(page_durations[i])
        else:
            # 產生靜音 WAV（無旁白的頁面顯示 3 秒），使用與音訊匹配的取樣率
            silent_wav = str(temp_dir / f"silent_{i + 1:03d}.wav")
            _generate_silent_wav(silent_wav, 3.0, actual_sr)
            all_audio_paths.append(silent_wav)
            all_durations.append(3.0)

    # ── 步驟 2: 合併所有 WAV 為一個完整音訊 ──
    if progress_callback:
        progress_callback(2, steps, "正在合併音訊...")

    combined_audio = str(temp_dir / "combined_audio.wav")
    _concatenate_wav_files(all_audio_paths, combined_audio)

    # ── 步驟 3: 建立圖片序列清單 ──
    image_list = str(temp_dir / "_image_list.txt")
    with open(image_list, "w", encoding="utf-8") as f:
        for i in range(total):
            safe_path = str(Path(slide_images[i]).resolve()).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")
            f.write(f"duration {all_durations[i]:.6f}\n")
        # concat demuxer 要求：重複最後一張圖（否則最後一頁時長不正確）
        if slide_images:
            safe_path = str(Path(slide_images[-1]).resolve()).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    # ── 步驟 4: 單次合成影片 ──
    if progress_callback:
        progress_callback(3, steps, "正在合成影片（單次合成）...")

    w, h = resolution
    combined_video = str(temp_dir / "_combined.mp4")

    _run_ffmpeg(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", image_list,
            "-i", combined_audio,
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black",
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-y",
            str(combined_video),
        ],
        description="單次合成完整影片",
    )

    # ── 步驟 5: 燒錄字幕或輸出 ──
    if burn_srt and srt_path and Path(srt_path).exists():
        if progress_callback:
            progress_callback(steps - 1, steps, "正在燒錄字幕...")
        burn_subtitles(combined_video, srt_path, output_path)
        try:
            Path(combined_video).unlink()
        except Exception:
            pass
    else:
        import shutil
        shutil.move(combined_video, output_path)

    # 清理暫存
    try:
        import shutil
        shutil.rmtree(str(temp_dir), ignore_errors=True)
    except Exception:
        pass

    logger.info("影片生成完成: %s", output_path)
    return output_path
