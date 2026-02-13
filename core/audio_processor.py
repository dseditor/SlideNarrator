"""音訊處理模組 -- 合併句子音訊、計算時長、匯出 WAV

支援並行 TTS 合成：先並行產生所有句子的音訊，再按順序計算時間軸。
"""
import logging
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import SENTENCE_PAUSE_SEC, TTS_PARALLEL_WORKERS
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


def _synthesize_one(
    tts_engine,
    text: str,
    speed: float,
    page_num: int,
    sent_idx: int,
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """
    合成單句音訊的 worker 函數（供 ThreadPoolExecutor 呼叫）。

    回傳 (samples, sample_rate, error_msg)
    成功時 error_msg 為 None；失敗時 samples 和 sr 為 None。
    """
    try:
        samples, sr = tts_engine.synthesize(text, speed=speed)
        return samples, sr, None
    except Exception as e:
        logger.error("合成失敗: P%d S%d: %s", page_num, sent_idx + 1, e)
        return None, None, str(e)


def process_all_pages(
    script: Script,
    tts_engine,
    speed: float = 1.0,
    pause_sec: float = SENTENCE_PAUSE_SEC,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Tuple[np.ndarray, float]]:
    """
    處理所有頁面的 TTS 生成與音訊處理（支援並行合成）。

    progress_callback(current, total, message):
        用於 UI 進度條更新。

    回傳:
        [(page_audio, page_duration), ...] 每頁的合併音訊與總時長。
        同時更新 script 中每個 Sentence 的 duration_sec 和 start_sec。

    並行策略：
        1. 先用 ThreadPoolExecutor 並行合成所有句子的音訊
        2. 所有合成完成後，按原始順序計算 global_cursor 時間軸
        3. 字幕時間軸完全不受並行影響，保證與音訊同步
    """
    total_sentences = script.total_sentences
    results: List[Tuple[np.ndarray, float]] = []

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 取得引擎宣告的取樣率
    engine_sr = tts_engine.sample_rate

    # ── 階段 1：收集所有待合成句子 ──
    all_sentences = []
    for page in script.pages:
        for sentence in page.sentences:
            all_sentences.append((page.page_number, sentence))

    if progress_callback:
        progress_callback(0, total_sentences, "開始並行合成音訊...")

    # ── 階段 2：並行合成所有句子 ──
    # 結果按原始索引存放，確保順序性
    synth_results: List[Tuple[Optional[np.ndarray], Optional[int], Optional[str]]] = [
        (None, None, None)
    ] * len(all_sentences)

    workers = max(1, TTS_PARALLEL_WORKERS)
    completed_count = 0

    if workers <= 1:
        # 單執行緒模式：直接循序合成（向後相容）
        for idx, (page_num, sentence) in enumerate(all_sentences):
            completed_count += 1
            if progress_callback:
                progress_callback(
                    completed_count,
                    total_sentences,
                    f"合成中 P{page_num}: {sentence.text[:20]}...",
                )
            synth_results[idx] = _synthesize_one(
                tts_engine, sentence.text, speed, page_num, sentence.sentence_index,
            )
    else:
        # 多執行緒模式：並行合成
        logger.info("啟用並行合成: %d 個 workers", workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}
            for idx, (page_num, sentence) in enumerate(all_sentences):
                future = executor.submit(
                    _synthesize_one,
                    tts_engine,
                    sentence.text,
                    speed,
                    page_num,
                    sentence.sentence_index,
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                synth_results[idx] = future.result()

                completed_count += 1
                page_num, sentence = all_sentences[idx]
                if progress_callback:
                    progress_callback(
                        completed_count,
                        total_sentences,
                        f"已完成 {completed_count}/{total_sentences}: "
                        f"P{page_num} {sentence.text[:15]}...",
                    )

    # ── 階段 3：按順序分配時間軸（字幕同步的關鍵） ──
    # 先確認實際取樣率（使用第一個成功合成的結果）
    actual_sr: Optional[int] = None
    for samples, sr, err in synth_results:
        if samples is not None and sr is not None:
            actual_sr = sr
            if sr != engine_sr:
                logger.warning(
                    "取樣率修正: engine 宣告=%d, 實際合成=%d，以實際值為準",
                    engine_sr, sr,
                )
            break

    global_cursor = 0.0
    sentence_idx = 0  # 全域句子計數器

    for page in script.pages:
        page_segments: List[np.ndarray] = []

        for i, sentence in enumerate(page.sentences):
            samples, sr, err = synth_results[sentence_idx]

            # 記錄此句在全域時間軸的起始時間
            if i > 0:
                sr_for_silence = actual_sr if actual_sr else engine_sr
                silence_samples = int(pause_sec * sr_for_silence)
                silence_sec = silence_samples / sr_for_silence
                global_cursor += silence_sec

            sentence.start_sec = global_cursor

            if samples is not None and sr is not None:
                # 合成成功
                if sr != actual_sr:
                    logger.warning(
                        "取樣率不一致: 本次合成=%d, 先前=%d", sr, actual_sr,
                    )

                page_segments.append(samples)
                sentence.duration_sec = len(samples) / sr

                # 儲存單句音訊
                if output_dir:
                    wav_path = (
                        Path(output_dir)
                        / f"page{page.page_number:03d}_sent{sentence.sentence_index:03d}.wav"
                    )
                    save_wav(str(wav_path), samples, sr)
                    sentence.audio_path = str(wav_path)

                logger.info(
                    "時間軸分配: P%d S%d (%.4f秒, 起始%.4f秒) %s",
                    page.page_number,
                    sentence.sentence_index + 1,
                    sentence.duration_sec,
                    sentence.start_sec,
                    sentence.text[:30],
                )
            else:
                # 合成失敗，使用 1 秒靜音替代
                sr_for_fallback = actual_sr if actual_sr else engine_sr
                silence = generate_silence(1.0, sr_for_fallback)
                sentence.duration_sec = 1.0
                page_segments.append(silence)

            global_cursor += sentence.duration_sec
            sentence_idx += 1

        # 使用實際取樣率合併該頁所有句子
        merge_sr = actual_sr if actual_sr else engine_sr

        if page_segments:
            combined = concatenate_audio(page_segments, pause_sec, merge_sr)

            if output_dir:
                page_wav = Path(output_dir) / f"page{page.page_number:03d}_full.wav"
                save_wav(str(page_wav), combined, merge_sr)
                page_duration = get_wav_duration(str(page_wav))
            else:
                page_duration = calculate_duration(combined, merge_sr)

            results.append((combined, page_duration))
        else:
            results.append((np.array([], dtype=np.float32), 0.0))

    # 最終驗證
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

    actual_total = sum(dur for _, dur in results)
    logger.info("音訊實際總時長: %.4f 秒", actual_total)
    if abs(global_cursor - actual_total) > 0.1:
        logger.warning(
            "字幕時間 (%.4f) 與音訊時間 (%.4f) 偏差 %.4f 秒",
            global_cursor, actual_total, global_cursor - actual_total,
        )

    return results
