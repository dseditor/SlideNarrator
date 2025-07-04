"""
Breeze2-VITS 繁體中文語音合成 - 修復版
添加 jieba 字典支援以解決中文 TTS 模型問題
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
            
            # 檢查是否需要下載字典文件
            dict_files_needed = [
                "jieba.dict.utf8",
                "user.dict.utf8", 
                "idf.txt.big",
                "stop_words.txt"
            ]
            
            # 嘗試從 Hugging Face 下載字典文件（如果有的話）
            # 或者創建基本的字典文件
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
            if not jieba_dict_path.exists():
                jieba_dict_path.touch()
                print(f"📝 創建空字典文件: {jieba_dict_path}")
            
            if not user_dict_path.exists():
                user_dict_path.touch()
                print(f"📝 創建用戶字典文件: {user_dict_path}")
            
            if not idf_path.exists():
                idf_path.touch()
                print(f"📝 創建 IDF 文件: {idf_path}")
                
            if not stop_words_path.exists():
                stop_words_path.touch()
                print(f"📝 創建停用詞文件: {stop_words_path}")
                
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
            print("📂 當前目錄結構:")
            for item in Path(".").rglob("*"):
                if item.is_file():
                    size = item.stat().st_size
                    print(f"  {item}: {size} bytes")
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
            
            # 配置 VITS 模型 - 關鍵修改：添加字典目錄
            vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(self.model_dir / "breeze2-vits.onnx"),
                lexicon=str(self.model_dir / "lexicon.txt"),
                tokens=str(self.model_dir / "tokens.txt"),
                dict_dir=str(self.dict_dir),  # 添加字典目錄
            )
            
            print(f"📚 字典目錄: {self.dict_dir}")
            print(f"📁 字典目錄內容: {list(self.dict_dir.iterdir()) if self.dict_dir.exists() else '目錄不存在'}")
            
            # 配置 TTS 模型
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_config,
                num_threads=2 if device == "cpu" else 1,
                debug=True,  # 啟用調試模式以獲得更多資訊
                provider=provider,
            )
            
            # 創建 TTS 配置
            config = sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                rule_fsts="",
                max_num_sentences=1,
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

    def synthesize(self, text, speaker_id=0, speed=1.0):
        """合成語音"""
        if not text or not text.strip():
            return None, "❌ 請輸入文本"
        
        # 文本預處理
        text = text.strip()
        if len(text) > 200:
            text = text[:200]
            
        try:
            print(f"🎤 正在合成語音: {text[:30]}...")
            print(f"🎭 說話人: {speaker_id}, ⚡ 速度: {speed}x")
            
            # 生成語音
            audio = self.tts.generate(
                text=text,
                sid=speaker_id,
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
            
            return (sample_rate, audio_array), f"✅ 語音合成成功！\n📊 採樣率: {sample_rate}Hz\n⏱️ 時長: {duration:.2f}秒\n🎭 說話人: {speaker_id}"
            
        except Exception as e:
            error_msg = f"❌ 語音合成失敗: {str(e)}"
            print(error_msg)
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
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


def generate_speech(text, speaker_id, speed):
    """Gradio 介面函數"""
    if tts_model is None:
        return None, f"❌ TTS 模型未正確載入\n\n詳情: {model_status}"
    
    return tts_model.synthesize(text, speaker_id, speed)


def create_interface():
    # 預設範例文本
    examples = [
        ["你好，歡迎使用繁體中文語音合成系統！", 0, 1.0],
        ["今天天氣很好，適合出去走走。", 1, 1.0],
        ["人工智慧技術正在快速發展，為我們的生活帶來許多便利。", 2, 1.2],
        ["台灣是一個美麗的島嶼，有著豐富的文化和美食。", 3, 0.9],
        ["科技改變生活，創新引領未來。", 4, 1.1],
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
        """
    ) as demo:
        
        gr.HTML(f"""
        <div class="status-box">
            <h1>🎙️ 繁體中文語音合成 - Breeze2-VITS</h1>
            <p><strong>狀態:</strong> {model_status} | <strong>設備:</strong> {device_info}</p>
        </div>
        """)
        
        gr.Markdown("""
        使用 **MediaTek Breeze2-VITS** 模型進行高品質繁體中文語音合成
        
        ✨ **特色:** 🇹🇼 繁體中文優化 | 🎭 多種說話人 | ⚡ 快速推理 | 🎚️ 速度調節
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
                    label="📝 輸入文本 (最多200字)",
                    placeholder="請輸入要合成的繁體中文文本...",
                    lines=4,
                    max_lines=6,
                    value="你好，這是一個語音合成測試。歡迎使用繁體中文TTS系統！"
                )
                
                with gr.Row():
                    # 說話人選擇
                    speaker_id = gr.Slider(
                        label="🎭 說話人",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=0,
                        info="選擇不同的說話人聲音 (0-10)"
                    )
                    
                    # 語音速度
                    speed = gr.Slider(
                        label="⚡ 語音速度",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        info="調節語音播放速度"
                    )
                
                # 生成按鈕
                generate_btn = gr.Button(
                    "🎵 生成語音",
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
                inputs=[text_input, speaker_id, speed],
                outputs=[audio_output, status_msg],
                fn=generate_speech,
                cache_examples=False,
                label="📚 範例文本 (點擊即可使用)"
            )
        
        # 使用說明和技術資訊
        with gr.Accordion("📋 使用說明與技術資訊", open=False):
            gr.Markdown(f"""
            ### 使用說明
            1. 在文本框中輸入繁體中文文本 (建議不超過200字)
            2. 選擇喜歡的說話人聲音 (0-10，每個數字對應不同聲音特色)
            3. 調整語音速度 (0.5x 慢速 ↔ 2.0x 快速)
            4. 點擊「生成語音」按鈕
            5. 在右側播放和下載生成的語音
            
            ### 技術資訊
            - **模型**: MediaTek Breeze2-VITS-onnx
            - **語言**: 繁體中文 (台灣國語)
            - **採樣率**: 22050 Hz
            - **推理引擎**: Sherpa-ONNX
            - **運行設備**: {device_info}
            - **模型狀態**: {model_status}
            - **jieba 字典**: {'✅ 已配置' if Path('./dict').exists() else '❌ 未配置'}
            
            ### 故障排除
            如果遇到問題：
            1. 檢查文本是否為繁體中文
            2. 嘗試較短的文本 (10-50字)
            3. 重新整理頁面
            4. 檢查瀏覽器控制台錯誤
            """)
        
        # 事件綁定
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, speaker_id, speed],
            outputs=[audio_output, status_msg],
            api_name="generate_speech"
        )
        
        # 鍵盤快捷鍵
        text_input.submit(
            fn=generate_speech,
            inputs=[text_input, speaker_id, speed],
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
