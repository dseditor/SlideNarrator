"""
Breeze2-VITS 繁體中文語音合成 - 單說話人版本
專為台灣國語優化的高品質語音合成系統
"""

import gradio as gr
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import torch

try:
    import sherpa_onnx
except ImportError:
    os.system("pip install sherpa-onnx")
    import sherpa_onnx

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    os.system("pip install huggingface_hub")
    from huggingface_hub import hf_hub_download


class TaiwaneseVITSTTS:
    def __init__(self):
        self.tts = None
        self.model_dir = Path("./models")
        self.dict_dir = Path("./dict")
        self.setup_jieba_dict()
        self.setup_model()
    
    def setup_jieba_dict(self):
        """設置 jieba 字典目錄"""
        try:
            print("🔧 設置 jieba 字典...")
            
            # 創建字典目錄
            self.dict_dir.mkdir(exist_ok=True)
            
            # 創建基本的字典文件
            self.create_basic_jieba_dict()
            
            print(f"✅ jieba 字典設置完成: {self.dict_dir}")
            
        except Exception as e:
            print(f"⚠️ jieba 字典設置失敗: {e}")
            # 創建空目錄作為後備
            self.dict_dir.mkdir(exist_ok=True)
    
    def create_basic_jieba_dict(self):
        """創建基本的 jieba 字典文件"""
        try:
            # 創建基本的 jieba 字典文件
            jieba_dict_path = self.dict_dir / "jieba.dict.utf8"
            user_dict_path = self.dict_dir / "user.dict.utf8"
            idf_path = self.dict_dir / "idf.txt.big"
            stop_words_path = self.dict_dir / "stop_words.txt"
            
            # 如果字典文件不存在，創建空文件
            for file_path in [jieba_dict_path, user_dict_path, idf_path, stop_words_path]:
                if not file_path.exists():
                    file_path.touch()
                    print(f"📝 創建字典文件: {file_path.name}")
                
        except Exception as e:
            print(f"⚠️ 創建基本字典文件失敗: {e}")

    def verify_model_files(self):
        """檢查本地模型文件是否存在"""
        required_files = ["breeze2-vits.onnx", "lexicon.txt", "tokens.txt"]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.model_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
            elif file_path.stat().st_size == 0:
                missing_files.append(f"{file_name} (檔案大小為 0)")
        
        if missing_files:
            print(f"❌ 缺少模型文件: {missing_files}")
            return False
        
        print("✅ 所有模型文件都存在")
        for file_name in required_files:
            file_path = self.model_dir / file_name
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  📄 {file_name}: {size_mb:.1f} MB")
        
        return True

    def setup_model(self):
        """設置和初始化模型"""
        try:
            if not self.verify_model_files():
                raise FileNotFoundError("模型文件缺失")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            provider = "cuda" if device == "cuda" else "cpu"
            
            print(f"🔧 使用設備: {device.upper()}")
            if device == "cuda":
                try:
                    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
                    print(f"💾 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                except:
                    print("🎮 GPU 資訊獲取失敗，但將嘗試使用 GPU")
            
            # 配置 VITS 模型
            vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(self.model_dir / "breeze2-vits.onnx"),
                lexicon=str(self.model_dir / "lexicon.txt"),
                tokens=str(self.model_dir / "tokens.txt"),
                dict_dir=str(self.dict_dir),
            )
            
            print(f"📚 字典目錄: {self.dict_dir}")
            
            # 配置 TTS 模型
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_config,
                num_threads=2 if device == "cpu" else 1,
                debug=False,  # 關閉調試模式以減少日誌
                provider=provider,
            )
            
            # 創建 TTS 配置
            config = sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                rule_fsts="",
                max_num_sentences=2,  # 支援較長句子
            )
            
            print("🔄 正在載入 TTS 模型...")
            self.tts = sherpa_onnx.OfflineTts(config)
            
            print("🚀 TTS 模型初始化成功!")
            
            # 測試模型
            print("🧪 進行模型測試...")
            test_audio = self.tts.generate(text="測試", sid=0, speed=1.0)
            if len(test_audio.samples) > 0:
                print("✅ 模型測試通過!")
            else:
                print("⚠️ 模型測試失敗，但模型已載入")
            
        except Exception as e:
            print(f"❌ 模型設置失敗: {e}")
            print(f"錯誤類型: {type(e).__name__}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            raise

    def synthesize(self, text, speed=1.0):
        """合成語音 - 單說話人版本"""
        if not text or not text.strip():
            return None, "❌ 請輸入文本"
        
        # 文本預處理
        text = text.strip()
        if len(text) > 500:  # 增加文本長度限制
            text = text[:500]
            
        try:
            print(f"🎤 正在合成語音: {text[:50]}...")
            print(f"⚡ 語音速度: {speed}x")
            
            # 生成語音 - 固定使用說話人 ID 0
            audio = self.tts.generate(
                text=text,
                sid=0,  # 固定使用第一個說話人
                speed=speed
            )
            
            # 獲取音頻數據
            samples = audio.samples
            sample_rate = audio.sample_rate
            
            if len(samples) == 0:
                return None, "❌ 語音生成失敗：生成的音頻為空"
            
            # 轉換為 numpy 陣列
            audio_array = np.array(samples, dtype=np.float32)
            
            # 確保是單聲道
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # 正規化音頻
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val * 0.9
            
            duration = len(audio_array) / sample_rate
            print(f"✅ 語音合成完成! 長度: {duration:.2f}秒")
            
            return (sample_rate, audio_array), f"✅ 語音合成成功！\n📊 採樣率: {sample_rate}Hz\n⏱️ 時長: {duration:.2f}秒\n🎭 台灣國語聲音"
            
        except Exception as e:
            error_msg = f"❌ 語音合成失敗: {str(e)}"
            print(error_msg)
            return None, error_msg


# 全局 TTS 實例
print("🔧 正在初始化 TTS 模型...")
try:
    tts_model = TaiwaneseVITSTTS()
    print("✅ TTS 系統就緒!")
    model_status = "🟢 模型已載入"
except Exception as e:
    print(f"❌ TTS 初始化失敗: {e}")
    tts_model = None
    model_status = f"🔴 模型載入失敗: {str(e)}"


def generate_speech(text, speed):
    """Gradio 介面函數 - 移除說話人參數"""
    if tts_model is None:
        return None, f"❌ TTS 模型未正確載入\n\n詳情: {model_status}"
    
    return tts_model.synthesize(text, speed)


def create_interface():
    # 預設範例文本 - 移除說話人參數
    examples = [
        ["你好，歡迎使用繁體中文語音合成系統！", 1.0],
        ["今天天氣很好，適合出去走走。", 1.0],
        ["人工智慧技術正在快速發展，為我們的生活帶來許多便利。", 1.1],
        ["台灣是一個美麗的島嶼，有著豐富的文化和美食。", 0.9],
        ["科技改變生活，創新引領未來。讓我們一起擁抱智慧時代的到來。", 1.2],
        ["春天來了，櫻花盛開，微風輕拂，真是個美好的季節。", 0.8],
    ]
    
    # 檢查模型狀態
    device_info = "🎮 GPU" if torch.cuda.is_available() else "💻 CPU"
    
    with gr.Blocks(
        title="繁體中文語音合成 - Breeze2-VITS",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 900px !important;
            margin: auto !important;
        }
        .status-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .feature-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 10px 0;
            text-align: center;
        }
        """
    ) as demo:
        
        gr.HTML(f"""
        <div class="status-box">
            <h1>🎙️ 繁體中文語音合成 - Breeze2-VITS</h1>
            <p><strong>狀態:</strong> {model_status} | <strong>設備:</strong> {device_info}</p>
        </div>
        """)
        
        gr.HTML("""
        <div class="feature-box">
            <strong>🇹🇼 專業台灣國語 TTS</strong> | 由 MediaTek 開發，專為繁體中文優化
        </div>
        """)
        
        if not tts_model:
            gr.Markdown(f"""
            ### ⚠️ 模型載入失敗
            
            **錯誤詳情**: {model_status}
            
            **可能原因**:
            - 模型文件缺失或損壞
            - jieba 字典配置問題
            - 記憶體不足
            
            請檢查日誌獲取更多資訊。
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 文本輸入
                text_input = gr.Textbox(
                    label="📝 輸入文本 (最多500字)",
                    placeholder="請輸入要合成的繁體中文文本...",
                    lines=5,
                    max_lines=8,
                    value="你好，這是一個語音合成測試。歡迎使用繁體中文TTS系統！"
                )
                
                # 只保留語音速度控制
                speed = gr.Slider(
                    label="⚡ 語音速度",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    info="調節語音播放速度 (0.5x 慢速 ↔ 2.0x 快速)"
                )
                
                # 生成按鈕
                generate_btn = gr.Button(
                    "🎵 生成台灣國語語音",
                    variant="primary",
                    size="lg",
                    interactive=tts_model is not None
                )
        
            with gr.Column(scale=1):
                # 音頻輸出
                audio_output = gr.Audio(
                    label="🔊 生成的語音",
                    type="numpy",
                    interactive=False,
                    show_download_button=True
                )
                
                # 狀態訊息
                status_msg = gr.Textbox(
                    label="📊 狀態資訊",
                    interactive=False,
                    lines=4,
                    value="準備就緒，請輸入文本並點擊生成語音" if tts_model else f"模型載入失敗: {model_status}"
                )
        
        # 範例
        if tts_model:  # 只有在模型正常載入時才顯示範例
            gr.Examples(
                examples=examples,
                inputs=[text_input, speed],  # 移除說話人參數
                outputs=[audio_output, status_msg],
                fn=generate_speech,
                cache_examples=False,
                label="📚 範例文本 (點擊即可使用)"
            )
        
        # 使用說明和技術資訊
        with gr.Accordion("📋 使用說明與技術資訊", open=False):
            gr.Markdown(f"""
            ### 🚀 使用說明
            1. 在文本框中輸入繁體中文文本 (支援最多500字)
            2. 調整語音速度 (建議範圍 0.8x - 1.5x)
            3. 點擊「生成台灣國語語音」按鈕
            4. 在右側播放和下載生成的語音
            
            ### 🎯 模型特色
            - **專業台灣國語**: 經過台灣語料訓練，發音自然
            - **高品質合成**: 使用 VITS 架構，語音清晰流暢
            - **移動優化**: 輕量化設計，適合各種設備
            - **即時生成**: 快速推理，支援即時語音合成
            
            ### 🔧 技術資訊
            - **模型**: MediaTek Breeze2-VITS-onnx
            - **語言**: 繁體中文 (台灣國語)
            - **採樣率**: 22050 Hz
            - **推理引擎**: Sherpa-ONNX
            - **運行設備**: {device_info}
            - **模型狀態**: {model_status}
            - **字典配置**: {'✅ 已配置' if Path('./dict').exists() else '❌ 未配置'}
            
            ### 📝 最佳實踐
            - **文本長度**: 建議單次合成 10-100 字，效果最佳
            - **標點符號**: 適當使用逗號和句號來控制語調停頓
            - **語音速度**: 一般對話建議 1.0x，朗讀建議 0.9x，快速播報建議 1.3x
            - **特殊字符**: 避免使用過多英文或特殊符號
            
            ### 🛠️ 故障排除
            如果遇到問題：
            1. 檢查文本是否為繁體中文
            2. 嘗試較短的文本 (10-50字)
            3. 重新整理頁面重新載入模型
            4. 檢查瀏覽器控制台錯誤訊息
            
            ### 📄 授權資訊
            - **模型**: MediaTek Research 開源模型
            - **使用範圍**: 研究和個人使用
            - **商業使用**: 請參考 MediaTek 授權條款
            """)
        
        # 事件綁定 - 移除說話人參數
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, speed],
            outputs=[audio_output, status_msg],
            api_name="generate_speech"
        )
        
        # 鍵盤快捷鍵
        text_input.submit(
            fn=generate_speech,
            inputs=[text_input, speed],
            outputs=[audio_output, status_msg]
        )
    
    return demo


# 啟動應用
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        show_api=True
    )
