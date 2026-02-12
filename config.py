"""全域設定常數"""
import sys
from pathlib import Path

# 判斷是否為 PyInstaller 打包環境
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
    APP_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
    APP_DIR = BASE_DIR

# 模型路徑
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "breeze2-vits.onnx"
LEXICON_PATH = MODEL_DIR / "lexicon.txt"
TOKENS_PATH = MODEL_DIR / "tokens.txt"
DICT_DIR = BASE_DIR / "dict"
TEXT_MAPPING_PATH = BASE_DIR / "text_mapping.txt"

# ffmpeg
FFMPEG_DIR = BASE_DIR / "ffmpeg"
FFMPEG_PATH = FFMPEG_DIR / "ffmpeg.exe"

# TTS 設定
TTS_NUM_THREADS = 4
TTS_PROVIDER = "CPUExecutionProvider"
TTS_MAX_SENTENCES = 5
DEFAULT_SPEED = 1.0
DEFAULT_SID = 0

# 音訊設定
SAMPLE_RATE = 48000
SENTENCE_PAUSE_SEC = 0.5

# 影片設定
DEFAULT_VIDEO_FPS = 1
DEFAULT_VIDEO_RESOLUTION = (1920, 1080)
DEFAULT_SLIDE_DPI = 200

# 輸出路徑
OUTPUT_DIR = APP_DIR / "output"
TEMP_DIR = APP_DIR / "temp"
