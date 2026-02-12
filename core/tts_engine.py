"""TTS 引擎模組 -- 封裝 sherpa-onnx Breeze2-VITS"""
import io
import logging
import re
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sherpa_onnx

from config import (
    DICT_DIR,
    LEXICON_PATH,
    MODEL_PATH,
    TEXT_MAPPING_PATH,
    TOKENS_PATH,
    TTS_MAX_SENTENCES,
    TTS_NUM_THREADS,
    TTS_PROVIDER,
)

logger = logging.getLogger(__name__)


class TextConverter:
    """文本轉換器，將英文和數字轉換為中文發音"""

    def __init__(self, mapping_file: Optional[str] = None):
        if mapping_file is None:
            mapping_file = str(TEXT_MAPPING_PATH)
        self.mapping_file = Path(mapping_file)
        self.conversion_map: dict = {}
        self.load_mapping()

    def load_mapping(self) -> None:
        """載入轉換對照表"""
        try:
            if self.mapping_file.exists():
                with open(self.mapping_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if "|" in line:
                        original, chinese = line.split("|", 1)
                        self.conversion_map[original.strip().lower()] = chinese.strip()

                logger.info("載入 %d 個轉換規則", len(self.conversion_map))
            else:
                logger.warning("轉換對照表不存在: %s", self.mapping_file)
                self._create_default_mapping()
        except Exception as e:
            logger.error("載入轉換對照表失敗: %s", e)
            self._create_default_mapping()

    def _create_default_mapping(self) -> None:
        """建立基本轉換對照表"""
        self.conversion_map = {
            "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
            "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
            "10": "十", "11": "十一", "12": "十二", "13": "十三",
            "14": "十四", "15": "十五", "16": "十六", "17": "十七",
            "18": "十八", "19": "十九", "20": "二十",
            "hello": "哈囉", "hi": "嗨", "bye": "拜拜", "ok": "好的",
            "ai": "人工智慧", "cpu": "中央處理器", "gpu": "圖形處理器",
            "a": "欸", "b": "比", "c": "西", "d": "迪", "e": "伊",
            "f": "艾夫", "g": "吉", "h": "艾奇", "i": "愛", "j": "傑",
            "k": "凱", "l": "艾爾", "m": "艾姆", "n": "艾恩", "o": "歐",
            "p": "皮", "q": "丘", "r": "艾爾", "s": "艾斯", "t": "替",
            "u": "優", "v": "威", "w": "達布爾優", "x": "艾克斯",
            "y": "歪", "z": "萊德",
        }
        logger.info("使用基本轉換規則: %d 個", len(self.conversion_map))

    def convert_numbers(self, text: str) -> str:
        """轉換連續數字為中文"""
        def number_to_chinese(match):
            number = match.group()
            if len(number) <= 2:
                return "".join(
                    self.conversion_map.get(d, d) for d in number
                )
            return self._convert_large_number(number)

        return re.sub(r"\d+", number_to_chinese, text)

    def _convert_large_number(self, number_str: str) -> str:
        """轉換大數字為中文"""
        digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        try:
            num = int(number_str)
            if num == 0:
                return "零"
            if str(num) in self.conversion_map:
                return self.conversion_map[str(num)]
            if num < 10:
                return digits[num]
            elif num < 20:
                return "十" if num == 10 else "十" + digits[num % 10]
            elif num < 100:
                tens, ones = num // 10, num % 10
                result = digits[tens] + "十"
                if ones > 0:
                    result += digits[ones]
                return result
            else:
                return "".join(digits[int(d)] for d in number_str)
        except (ValueError, IndexError):
            return "".join(
                self.conversion_map.get(d, d) for d in number_str if d.isdigit()
            )

    def convert_uppercase_words(self, text: str) -> str:
        """轉換全大寫單字為逐字母發音"""
        def uppercase_to_letters(match):
            return "".join(
                self.conversion_map.get(c.lower(), c) for c in match.group()
            )
        return re.sub(r"\b[A-Z]{2,}\b", uppercase_to_letters, text)

    def convert_english(self, text: str) -> str:
        """轉換英文單詞為中文"""
        sorted_words = sorted(self.conversion_map.keys(), key=len, reverse=True)
        for word in sorted_words:
            if len(word) > 1:
                chinese = self.conversion_map[word]
                pattern = r"\b" + re.escape(word) + r"\b"
                text = re.sub(pattern, chinese, text, flags=re.IGNORECASE)
        return text

    def convert_single_letters(self, text: str) -> str:
        """轉換單個英文字母"""
        def letter_to_chinese(match):
            return self.conversion_map.get(match.group().lower(), match.group())
        return re.sub(r"\b[a-zA-Z]\b", letter_to_chinese, text)

    def preprocess_text(self, text: str) -> str:
        """預處理文本"""
        text = re.sub(r"\bDr\.", "Doctor", text, flags=re.IGNORECASE)
        text = re.sub(r"\bMr\.", "Mister", text, flags=re.IGNORECASE)
        text = re.sub(r"@", " at ", text)
        text = re.sub(r"\.com\b", " dot com", text, flags=re.IGNORECASE)
        return text

    def postprocess_text(self, text: str) -> str:
        """後處理文本"""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([，。！？；：])", r"\1", text)
        return text

    def convert_text(self, text: str) -> str:
        """主要轉換函數"""
        if not text:
            return text

        original = text
        text = self.preprocess_text(text)
        text = self.convert_uppercase_words(text)
        text = self.convert_english(text)
        text = self.convert_numbers(text)
        text = self.convert_single_letters(text)
        text = self.postprocess_text(text)

        if text != original:
            logger.debug("文本轉換: %r -> %r", original, text)
        return text


class TTSEngine:
    """TTS 引擎 -- 封裝 sherpa-onnx 模型"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        lexicon_path: Optional[str] = None,
        tokens_path: Optional[str] = None,
        dict_dir: Optional[str] = None,
        mapping_file: Optional[str] = None,
        num_threads: int = TTS_NUM_THREADS,
    ):
        self._model_path = str(model_path or MODEL_PATH)
        self._lexicon_path = str(lexicon_path or LEXICON_PATH)
        self._tokens_path = str(tokens_path or TOKENS_PATH)
        self._dict_dir = str(dict_dir or DICT_DIR)
        self._num_threads = num_threads
        self._tts: Optional[sherpa_onnx.OfflineTts] = None
        self._text_converter = TextConverter(mapping_file)
        self._setup_model()

    def _setup_model(self) -> None:
        """初始化 sherpa-onnx 模型"""
        # 驗證模型檔案
        for label, path in [
            ("model", self._model_path),
            ("lexicon", self._lexicon_path),
            ("tokens", self._tokens_path),
        ]:
            p = Path(path)
            if not p.exists() or p.stat().st_size == 0:
                raise FileNotFoundError(f"模型檔案缺失: {label} ({path})")

        dict_dir_str = self._dict_dir if Path(self._dict_dir).exists() else ""

        vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=self._model_path,
            lexicon=self._lexicon_path,
            tokens=self._tokens_path,
            dict_dir=dict_dir_str,
            data_dir="",
        )

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits_config,
            num_threads=self._num_threads,
            debug=False,
            provider=TTS_PROVIDER,
        )

        config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts="",
            rule_fars="",
            max_num_sentences=TTS_MAX_SENTENCES,
        )

        logger.info("正在載入 TTS 模型...")
        self._tts = sherpa_onnx.OfflineTts(config)
        logger.info(
            "TTS 模型載入成功 (說話者: %d, 取樣率: %d Hz)",
            self._tts.num_speakers,
            self._tts.sample_rate,
        )

        # 簡單測試
        test = self._tts.generate(text="測試", sid=0, speed=1.0)
        if len(test.samples) == 0:
            raise RuntimeError("模型測試失敗：產生的音訊為空")
        logger.info("模型測試通過")

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        enable_conversion: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        合成單一句子的語音。

        Returns:
            (audio_samples, sample_rate)
            audio_samples: float32 numpy array, 已正規化
            sample_rate: int

        Raises:
            ValueError: 文本為空或合成失敗
        """
        if not text or not text.strip():
            raise ValueError("文本不可為空")

        if self._tts is None:
            raise RuntimeError("TTS 模型未初始化")

        processed = text.strip()
        if enable_conversion:
            processed = self._text_converter.convert_text(processed)

        if len(processed) > 500:
            processed = processed[:500]

        audio = self._tts.generate(text=processed, sid=0, speed=speed)
        samples = np.array(audio.samples, dtype=np.float32)

        if len(samples) == 0:
            raise ValueError(f"語音合成失敗：產生的音訊為空 (文本: {text[:30]})")

        if len(samples.shape) > 1:
            samples = samples.mean(axis=1)

        # 正規化
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val * 0.9

        return samples, audio.sample_rate

    def synthesize_to_wav_bytes(
        self,
        text: str,
        speed: float = 1.0,
    ) -> bytes:
        """合成並回傳 WAV 格式的 bytes（用於 UI 預覽播放）"""
        samples, sample_rate = self.synthesize(text, speed)

        # float32 -> int16 PCM
        int_samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        raw_data = int_samples.tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_data)

        return buf.getvalue()

    @property
    def sample_rate(self) -> int:
        if self._tts is None:
            raise RuntimeError("TTS 模型未初始化")
        return self._tts.sample_rate

    @property
    def is_ready(self) -> bool:
        return self._tts is not None
