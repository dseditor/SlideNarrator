"""音訊處理模組 -- 合併句子音訊、計算時長、匯出 WAV"""
import logging
import wave
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import SENTENCE_PAUSE_SEC
from core.script_parser import Script

logger = logging.getLogger(__name__)


def generate_silence(duration_sec: float, sample_rate: int = 48000) -> np.ndarray:
    """產生指定長度的靜音"""
    num_samples = int(duration_sec * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def calculate_duration(samples: np.ndarray, sample_rate: int) -> float:
    """計算音訊秒數"""
    return len(samples) / sample_rate


def get_wav_duration(filepath: str) -> float:
    """從實際 WAV 檔案讀取精確時長（秒）"""
    with wave.open(filepath, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate


def save_wav(
    filepath: str,
    samples: np.ndarray,
    sample_rate: int = 48000,
) -> None:
    """將 float32 numpy array 儲存為 16-bit PCM WAV"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    int_samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    raw_data = int_samples.tobytes()

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_data)


def concatenate_audio(
    audio_segments: List[np.ndarray],
    pause_between_sec: float = SENTENCE_PAUSE_SEC,
    sample_rate: int = 48000,
) -> np.ndarray:
    """串接多段音訊，段間插入靜音"""
    if not audio_segments:
        return np.array([], dtype=np.float32)

    silence = generate_silence(pause_between_sec, sample_rate)
    parts: List[np.ndarray] = []

    for i, segment in enumerate(audio_segments):
        parts.append(segment)
        if i < len(audio_segments) - 1:
            parts.append(silence)

    return np.concatenate(parts)


def process_all_pages(
    script: Script,
    tts_engine,
    speed: float = 1.0,
    pause_sec: float = SENTENCE_PAUSE_SEC,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Tuple[np.ndarray, float]]:
    """
    處理所有頁面的 TTS 生成與音訊處理。

    progress_callback(current, total, message):
        用於 UI 進度條更新。

    回傳:
        [(page_audio, page_duration), ...] 每頁的合併音訊與總時長。
        同時更新 script 中每個 Sentence 的 duration_sec 和 start_sec。

    時間軸策略（改進版）：
        - 使用單一 global_cursor 追蹤全域時間，消除跨頁面算法累積偏移
        - 所有時長直接基於 WAV 檔案的實際 frames / framerate，精確到 sample 級別
        - 合成時統一使用第一次合成回傳的實際取樣率
    """
    total_sentences = script.total_sentences
    current_sentence = 0
    results: List[Tuple[np.ndarray, float]] = []

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 取得引擎宣告的取樣率（用於靜音生成和合併）
    engine_sr = tts_engine.sample_rate
    # 實際合成取樣率在首次合成時確認
    actual_sr: Optional[int] = None

    # 全域時間游標：精確追蹤已產生音訊的總時長
    global_cursor = 0.0

    for page in script.pages:
        page_segments: List[np.ndarray] = []

        for i, sentence in enumerate(page.sentences):
            current_sentence += 1
            if progress_callback:
                progress_callback(
                    current_sentence,
                    total_sentences,
                    f"第 {page.page_number} 頁: {sentence.text[:20]}...",
                )

            # 記錄此句在全域時間軸的起始時間（在加入靜音間隔之後）
            if i > 0:
                # 在句子之間插入靜音間隔
                # 使用已確認的取樣率計算靜音長度，確保 sample 精確
                sr_for_silence = actual_sr if actual_sr else engine_sr
                silence_samples = int(pause_sec * sr_for_silence)
                silence_sec = silence_samples / sr_for_silence
                global_cursor += silence_sec

            sentence.start_sec = global_cursor

            try:
                samples, sr = tts_engine.synthesize(sentence.text, speed=speed)

                # 首次合成時確認實際取樣率
                if actual_sr is None:
                    actual_sr = sr
                    if sr != engine_sr:
                        logger.warning(
                            "取樣率修正: engine 宣告=%d, 實際合成=%d，以實際值為準",
                            engine_sr, sr,
                        )

                if sr != actual_sr:
                    logger.warning(
                        "取樣率不一致: 本次合成=%d, 先前=%d",
                        sr, actual_sr,
                    )

                page_segments.append(samples)

                # 直接從 samples 長度算出精確時長（sample 級精度）
                sentence.duration_sec = len(samples) / sr

                # 儲存單句音訊供預覽播放使用
                if output_dir:
                    wav_path = (
                        Path(output_dir)
                        / f"page{page.page_number:03d}_sent{sentence.sentence_index:03d}.wav"
                    )
                    save_wav(str(wav_path), samples, sr)
                    sentence.audio_path = str(wav_path)

                logger.info(
                    "合成完成: P%d S%d (%.4f秒, 起始%.4f秒) %s",
                    page.page_number,
                    sentence.sentence_index + 1,
                    sentence.duration_sec,
                    sentence.start_sec,
                    sentence.text[:30],
                )
            except Exception as e:
                logger.error(
                    "合成失敗: P%d S%d: %s",
                    page.page_number,
                    sentence.sentence_index + 1,
                    e,
                )
                # 使用 1 秒靜音替代
                sr_for_fallback = actual_sr if actual_sr else engine_sr
                silence = generate_silence(1.0, sr_for_fallback)
                sentence.duration_sec = 1.0
                page_segments.append(silence)

            # 累計全域時間游標
            global_cursor += sentence.duration_sec

        # 使用實際取樣率合併該頁所有句子
        merge_sr = actual_sr if actual_sr else engine_sr

        if page_segments:
            combined = concatenate_audio(page_segments, pause_sec, merge_sr)

            if output_dir:
                page_wav = Path(output_dir) / f"page{page.page_number:03d}_full.wav"
                save_wav(str(page_wav), combined, merge_sr)
                # 從實際檔案讀取精確的頁面時長（用於影片合成）
                page_duration = get_wav_duration(str(page_wav))
            else:
                page_duration = calculate_duration(combined, merge_sr)

            results.append((combined, page_duration))
        else:
            results.append((np.array([], dtype=np.float32), 0.0))

    # 最終驗證：列出所有句子的時間戳
    logger.info("=== 字幕時間軸摘要 ===")
    for page in script.pages:
        for s in page.sentences:
            logger.info(
                "  P%d S%d: %.4f ~ %.4f (%s)",
                page.page_number,
                s.sentence_index + 1,
                s.start_sec,
                s.start_sec + s.duration_sec,
                s.text[:20],
            )
    logger.info("字幕總時長: %.4f 秒", global_cursor)

    # 同時記錄合併音訊的實際總時長供比對
    actual_total = sum(dur for _, dur in results)
    logger.info("音訊實際總時長: %.4f 秒", actual_total)
    if abs(global_cursor - actual_total) > 0.1:
        logger.warning(
            "字幕時間 (%.4f) 與音訊時間 (%.4f) 偏差 %.4f 秒",
            global_cursor, actual_total, global_cursor - actual_total,
        )

    return results
