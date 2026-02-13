"""影片生成模組 -- 使用 ffmpeg subprocess

核心策略：單次合成（single-pass）
  1. 合併所有頁面 WAV 為一個完整音訊
  2. 用 concat demuxer 指定每張投影片的顯示時長
  3. 一次 ffmpeg 指令產生最終影片
  避免了逐頁 AAC 編碼再拼接導致的時間軸累積偏移。

支援硬體加速編碼器：NVENC (CUDA)、Intel QSV、AMD AMF。
"""
import logging
import subprocess
import wave as wave_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import DEFAULT_VIDEO_RESOLUTION, FFMPEG_PATH

logger = logging.getLogger(__name__)

# Windows: 隱藏 console 視窗
_CREATE_NO_WINDOW = 0x08000000


# ── 編碼器設定 ──

@dataclass
class EncoderConfig:
    """影片編碼器設定"""
    name: str               # 顯示名稱，例如 "NVIDIA NVENC (H.264)"
    codec: str              # ffmpeg codec 名稱，例如 "h264_nvenc"
    extra_args: List[str] = field(default_factory=list)  # 額外編碼參數
    hw_type: str = "sw"     # "sw" / "nvidia" / "intel" / "amd"


# 軟體編碼器（永遠可用的備援）
SW_ENCODER = EncoderConfig(
    name="軟體編碼 (libx264)",
    codec="libx264",
    extra_args=["-preset", "medium", "-crf", "23", "-tune", "stillimage"],
    hw_type="sw",
)

# 候選硬體編碼器列表（按優先順序嘗試）
_HW_ENCODER_CANDIDATES: List[EncoderConfig] = [
    EncoderConfig(
        name="NVIDIA NVENC (H.264)",
        codec="h264_nvenc",
        extra_args=["-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"],
        hw_type="nvidia",
    ),
    EncoderConfig(
        name="Intel QSV (H.264)",
        codec="h264_qsv",
        extra_args=["-preset", "medium", "-global_quality", "23"],
        hw_type="intel",
    ),
    EncoderConfig(
        name="AMD AMF (H.264)",
        codec="h264_amf",
        extra_args=["-quality", "balanced", "-rc", "cqp", "-qp_i", "23", "-qp_p", "23"],
        hw_type="amd",
    ),
]


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


# ── 硬體編碼器偵測 ──

def _test_encoder(encoder: EncoderConfig) -> bool:
    """
    實際嘗試用指定編碼器編碼一小段畫面，確認硬體是否真正可用。

    比單純查 -encoders 列表更可靠：有些系統列出了編碼器但驅動不支援。
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-f", "lavfi",
        "-i", "color=black:s=64x64:d=0.1:r=25",
        "-c:v", encoder.codec,
        *encoder.extra_args,
        "-frames:v", "3",
        "-f", "null",
        "-y",
        "NUL",  # Windows null device
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=_CREATE_NO_WINDOW,
            timeout=10,
        )
        success = result.returncode == 0
        if success:
            logger.info("編碼器可用: %s (%s)", encoder.name, encoder.codec)
        else:
            logger.debug(
                "編碼器不可用: %s -- %s",
                encoder.codec,
                result.stderr[-200:] if result.stderr else "unknown error",
            )
        return success
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug("編碼器測試失敗: %s -- %s", encoder.codec, e)
        return False


def detect_available_encoders() -> List[EncoderConfig]:
    """
    偵測所有可用的硬體編碼器，回傳可用編碼器列表。

    永遠包含軟體編碼器作為最後的備援選項。
    可用於 UI 啟動時背景偵測，結果可快取。
    """
    available: List[EncoderConfig] = []

    for candidate in _HW_ENCODER_CANDIDATES:
        if _test_encoder(candidate):
            available.append(candidate)

    # 軟體編碼器永遠可用
    available.append(SW_ENCODER)

    logger.info(
        "編碼器偵測完成: %d 個可用 (%s)",
        len(available),
        ", ".join(e.name for e in available),
    )
    return available


def get_encoder_by_name(
    name: str,
    available: Optional[List[EncoderConfig]] = None,
) -> EncoderConfig:
    """根據名稱取得編碼器設定，找不到時回傳軟體編碼器"""
    encoders = available or detect_available_encoders()
    for enc in encoders:
        if enc.name == name:
            return enc
    return SW_ENCODER


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


def _build_video_encode_args(encoder: EncoderConfig) -> List[str]:
    """根據編碼器設定，產生 ffmpeg 影片編碼參數"""
    args = ["-c:v", encoder.codec]
    args.extend(encoder.extra_args)

    # 硬體編碼器不支援 -tune stillimage，但軟體編碼器的 extra_args 已包含
    return args


# ── 字幕 ──

def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_name: str = "Microsoft JhengHei",
    font_size: int = 24,
    encoder: Optional[EncoderConfig] = None,
) -> str:
    """將 SRT 字幕燒錄進影片"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    enc = encoder or SW_ENCODER

    srt_escaped = str(Path(srt_path).resolve()).replace("\\", "/")
    srt_escaped = srt_escaped.replace(":", "\\:")

    vf = (
        f"subtitles='{srt_escaped}'"
        f":force_style='FontName={font_name},FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'"
    )

    encode_args = _build_video_encode_args(enc)

    _run_ffmpeg(
        [
            "-i", str(video_path),
            "-vf", vf,
            *encode_args,
            "-c:a", "copy",
            "-y",
            str(output_path),
        ],
        description=f"燒錄字幕 (編碼器: {enc.name})",
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
    encoder: Optional[EncoderConfig] = None,
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
    encoder: 編碼器設定（None 時使用軟體編碼器）
    """
    enc = encoder or SW_ENCODER
    logger.info("使用編碼器: %s", enc.name)

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
        progress_callback(3, steps, f"正在合成影片（{enc.name}）...")

    w, h = resolution
    combined_video = str(temp_dir / "_combined.mp4")
    encode_args = _build_video_encode_args(enc)

    _run_ffmpeg(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", image_list,
            "-i", combined_audio,
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black",
            *encode_args,
            "-r", "25",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            "-y",
            str(combined_video),
        ],
        description=f"單次合成完整影片 ({enc.name})",
    )

    # ── 步驟 5: 燒錄字幕或輸出 ──
    if burn_srt and srt_path and Path(srt_path).exists():
        if progress_callback:
            progress_callback(steps - 1, steps, "正在燒錄字幕...")
        burn_subtitles(combined_video, srt_path, output_path, encoder=enc)
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

    if progress_callback:
        progress_callback(steps, steps, "影片生成完成")

    logger.info("影片生成完成: %s", output_path)
    return output_path
