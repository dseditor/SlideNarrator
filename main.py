"""簡報自動旁白系統 -- 應用程式進入點"""
import logging
import sys

from config import OUTPUT_DIR, TEMP_DIR


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    # 確保輸出目錄存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化 TTS 引擎
    logger.info("正在初始化 TTS 引擎...")
    try:
        from core.tts_engine import TTSEngine
        engine = TTSEngine()
    except Exception as e:
        logger.error("TTS 引擎初始化失敗: %s", e)
        engine = None

    # 啟動 UI
    logger.info("正在啟動使用者介面...")
    from ui.app import NarratorApp, SharedState

    state = SharedState()
    state.tts_engine = engine
    if engine and engine.is_ready:
        state.sample_rate = engine.sample_rate

    app = NarratorApp(state)
    app.mainloop()


if __name__ == "__main__":
    main()
