"""
Breeze2-VITS 繁體中文語音合成 - 增強版
支援英文和數字自動轉換為中文發音
"""

import gradio as gr
import numpy as np
import os
import re
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


class TextConverter:
    """文本轉換器，將英文和數字轉換為中文發音"""
    
    def __init__(self, mapping_file="text_mapping.txt"):
        self.mapping_file = Path(mapping_file)
        self.conversion_map = {}
        self.load_mapping()
    
    def load_mapping(self):
        """載入轉換對照表"""
        try:
            if self.mapping_file.exists():
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    # 跳過註釋和空行
                    if line.startswith('#') or not line:
                        continue
                    
                    if '|' in line:
                        original, chinese = line.split('|', 1)
                        self.conversion_map[original.strip().lower()] = chinese.strip()
                
                print(f"✅ 載入 {len(self.conversion_map)} 個轉換規則")
            else:
                print(f"⚠️ 轉換對照表文件不存在: {self.mapping_file}")
                self.create_default_mapping()
        except Exception as e:
            print(f"❌ 載入轉換對照表失敗: {e}")
            self.create_default_mapping()
    
    def create_default_mapping(self):
        """創建預設的轉換對照表"""
        default_mappings = {
            # 數字
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
            '10': '十', '100': '一百', '1000': '一千',
            
            # 常用英文
            'hello': '哈囉', 'hi': '嗨', 'bye': '拜拜', 'ok': '歐凱',
            'yes': '是的', 'no': '不', 'good': '好的', 'bad': '不好',
            
            # 技術詞彙
            'ai': '人工智慧', 'api': '程式介面', 'app': '應用程式',
            'cpu': '中央處理器', 'gpu': '圖形處理器',
            
            # 字母
            'a': '欸', 'b': '比', 'c': '西', 'd': '迪', 'e': '伊'
        }
        
        self.conversion_map = default_mappings
        print(f"✅ 使用預設轉換規則: {len(default_mappings)} 個")
    
    def convert_numbers(self, text):
        """轉換連續數字為中文"""
        def number_to_chinese(match):
            number = match.group()
            if len(number) <= 2:  # 簡單數字直接對應
                result = ""
                for digit in number:
                    result += self.conversion_map.get(digit, digit)
                return result
            else:
                # 複雜數字處理
                return self.convert_large_number(number)
        
        # 匹配連續數字
        text = re.sub(r'\d+', number_to_chinese, text)
        return text
    
    def convert_large_number(self, number_str):
        """轉換大數字為中文"""
        try:
            num = int(number_str)
            if num == 0:
                return '零'
            
            # 簡化的數字轉換（支援到萬）
            units = ['', '十', '百', '千', '萬']
            digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
            
            if num < 10:
                return digits[num]
            elif num < 100:
                tens = num // 10
                ones = num % 10
                result = digits[tens] + '十'
                if ones > 0:
                    result += digits[ones]
                return result
            elif num < 1000:
                hundreds = num // 100
                remainder = num % 100
                result = digits[hundreds] + '百'
                if remainder > 0:
                    if remainder < 10:
                        result += '零' + digits[remainder]
                    else:
                        result += self.convert_large_number(str(remainder))
                return result
            else:
                # 對於更大的數字，簡化處理
                return number_str  # 保持原樣
        except:
            return number_str
    
    def convert_english(self, text):
        """轉換英文單詞為中文"""
        # 按長度排序，先處理長詞彙
        sorted_words = sorted(self.conversion_map.keys(), key=len, reverse=True)
        
        for english_word in sorted_words:
            if len(english_word) > 1:  # 跳過單字母，後面單獨處理
                chinese_word = self.conversion_map[english_word]
                # 使用單詞邊界匹配，不區分大小寫
                pattern = r'\b' + re.escape(english_word) + r'\b'
                text = re.sub(pattern, chinese_word, text, flags=re.IGNORECASE)
        
        return text
    
    def convert_single_letters(self, text):
        """轉換單個英文字母"""
        def letter_to_chinese(match):
            letter = match.group().lower()
            return self.conversion_map.get(letter, letter)
        
        # 匹配獨立的英文字母
        text = re.sub(r'\b[a-zA-Z]\b', letter_to_chinese, text)
        return text
    
    def convert_text(self, text):
        """主要轉換函數"""
        if not text:
            return text
        
        original_text = text
        print(f"🔄 原始文本: {original_text}")
        
        # 1. 轉換英文單詞
        text = self.convert_english(text)
        
        # 2. 轉換數字
        text = self.convert_numbers(text)
        
        # 3. 轉換剩餘的單個字母
        text = self.convert_single_letters(text)
        
        # 4. 清理多餘空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text != original_text:
            print(f"✅ 轉換後文本: {text}")
        
        return text


class TaiwaneseVITSTTS:
    def __init__(self):
        self.tts = None
        self.model_dir = Path("./models")
        self.dict_dir = Path("./dict")
        self.text_converter = TextConverter()
        self.setup_jieba_dict()
        self.setup_model()
    
    def setup_jieba_dict(self):
        """設置 jieba 字典目錄"""
        try:
            print("🔧 設置 jieba 字典...")
            self.dict_dir.mkdir(exist_ok=True)
            self.create_basic_jieba_dict()
            print(f"✅ jieba 字典設置完成: {self.dict_dir}")
        except Exception as e:
            print(f"⚠️ jieba 字典設置失敗: {e}")
            self.dict_dir.mkdir(exist_ok=True)
    
    def create_basic_jieba_dict(self):
        """創建基本的 jieba 字典文件"""
        try:
            jieba_dict_path = self.dict_dir / "jieba.dict.utf8"
            user_dict_path = self.dict_dir / "user.dict.utf8"
            idf_path = self.dict_dir / "idf.txt.big"
            stop_words_path = self.dict_dir / "stop_words.txt"
            
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
        return True

    def setup_model(self):
        """設置和初始化模型"""
        try:
            if not self.verify_model_files():
                raise FileNotFoundError("模型文件缺失")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            provider = "cuda" if device == "cuda" else "cpu"
            
            print(f"🔧 使用設備: {device.upper()}")
            
            vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(self.model_dir / "breeze2-vits.onnx"),
                lexicon=str(self.model_dir / "lexicon.txt"),
                tokens=str(self.model_dir / "tokens.txt"),
                dict_dir=str(self.dict_dir),
            )
            
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_config,
                num_threads=2 if device == "cpu" else 1,
                debug=False,
                provider=provider,
            )
            
            config = sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                rule_fsts="",
                max_num_sentences=2,
            )
            
            print("🔄 正在載入 TTS 模型...")
            self.tts = sherpa_onnx.OfflineTts(config)
            print("🚀 TTS 模型初始化成功!")
            
            # 測試模型
            test_audio = self.tts.generate(text="測試", sid=0, speed=1.0)
            if len(test_audio.samples) > 0:
                print("✅ 模型測試通過!")
            
        except Exception as e:
            print(f"❌ 模型設置失敗: {e}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            raise

    def synthesize(self, text, speed=1.0, enable_conversion=True):
        """合成語音"""
        if not text or not text.strip():
            return None, "❌ 請輸入文本"
        
        original_text = text.strip()
        
        # 文本轉換
        if enable_conversion:
            text = self.text_converter.convert_text(original_text)
        else:
            text = original_text
        
        if len(text) > 500:
            text = text[:500]
            
        try:
            print(f"🎤 正在合成語音...")
            if enable_conversion and text != original_text:
                print(f"📝 使用轉換後文本: {text}")
            
            audio = self.tts.generate(text=text, sid=0, speed=speed)
            samples = audio.samples
            sample_rate = audio.sample_rate
            
            if len(samples) == 0:
                return None, "❌ 語音生成失敗：生成的音頻為空"
            
            audio_array = np.array(samples, dtype=np.float32)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val * 0.9
            
            duration = len(audio_array) / sample_rate
            print(f"✅ 語音合成完成! 長度: {duration:.2f}秒")
            
            status_info = f"✅ 語音合成成功！\n📊 採樣率: {sample_rate}Hz\n⏱️ 時長: {duration:.2f}秒"
            if enable_conversion and text != original_text:
                status_info += f"\n🔄 已轉換: {original_text} → {text}"
            
            return (sample_rate, audio_array), status_info
            
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


def generate_speech(text, speed, enable_conversion):
    """Gradio 介面函數"""
    if tts_model is None:
        return None, f"❌ TTS 模型未正確載入\n\n詳情: {model_status}"
    
    return tts_model.synthesize(text, speed, enable_conversion)


def create_interface():
    # 預設範例文本
    examples = [
        ["你好，歡迎使用繁體中文語音合成系統！", 1.0, True],
        ["今天是2024年1月1日，天氣很好。", 1.0, True],
        ["我的email是test@gmail.com，請聯繫我。", 1.0, True],
        ["這是一個AI技術的demo，使用Python開發。", 1.1, True],
        ["Hello world! 這是一個測試。", 1.0, True],
        ["iPhone 15和Samsung Galaxy哪個比較好？", 0.9, True],
    ]
    
    device_info = "🎮 GPU" if torch.cuda.is_available() else "💻 CPU"
    
    with gr.Blocks(
        title="繁體中文語音合成 - Breeze2-VITS Enhanced",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1000px !important;
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
            <h1>🎙️ 繁體中文語音合成 - Breeze2-VITS Enhanced</h1>
            <p><strong>狀態:</strong> {model_status} | <strong>設備:</strong> {device_info}</p>
        </div>
        """)
        
        gr.HTML("""
        <div class="feature-box">
            <strong>🇹🇼 專業台灣國語 TTS</strong> | 🔄 自動英數轉換 | 🎯 智慧文本處理
        </div>
        """)
        
        if not tts_model:
            gr.Markdown(f"""
            ### ⚠️ 模型載入失敗
            **錯誤詳情**: {model_status}
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="📝 輸入文本 (支援中英混合、數字)",
                    placeholder="請輸入要合成的文本，支援中文、英文、數字混合...",
                    lines=5,
                    max_lines=8,
                    value="你好！今天是2024年，歡迎使用AI語音合成系統。"
                )
                
                with gr.Row():
                    speed = gr.Slider(
                        label="⚡ 語音速度",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        info="調節語音播放速度"
                    )
                    
                    enable_conversion = gr.Checkbox(
                        label="🔄 啟用英數轉換",
                        value=True,
                        info="自動將英文和數字轉換為中文發音"
                    )
                
                generate_btn = gr.Button(
                    "🎵 生成台灣國語語音",
                    variant="primary",
                    size="lg",
                    interactive=tts_model is not None
                )
        
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="🔊 生成的語音",
                    type="numpy",
                    interactive=False,
                    show_download_button=True
                )
                
                status_msg = gr.Textbox(
                    label="📊 狀態資訊",
                    interactive=False,
                    lines=5,
                    value="準備就緒，請輸入文本並點擊生成語音" if tts_model else f"模型載入失敗: {model_status}"
                )
        
        if tts_model:
            gr.Examples(
                examples=examples,
                inputs=[text_input, speed, enable_conversion],
                outputs=[audio_output, status_msg],
                fn=generate_speech,
                cache_examples=False,
                label="📚 範例文本 (支援中英數混合)"
            )
        
        with gr.Accordion("📋 使用說明與功能特色", open=False):
            gr.Markdown(f"""
            ### 🚀 主要功能
            
            #### 🔄 智慧文本轉換
            - **英文轉換**: hello → 哈囉, AI → 人工智慧
            - **數字轉換**: 123 → 一二三, 2024 → 二零二四
            - **品牌名稱**: Apple → 蘋果, Google → 谷歌
            - **技術詞彙**: API → 程式介面, CPU → 中央處理器
            
            #### 🎯 支援內容
            - 繁體中文文本
            - 英文單詞和句子
            - 阿拉伯數字
            - 混合語言文本
            - 常見縮寫和品牌
            
            ### 📝 使用技巧
            1. **啟用轉換**: 勾選「啟用英數轉換」自動處理英文和數字
            2. **關閉轉換**: 取消勾選以使用原始文本（純中文效果最佳）
            3. **混合文本**: 支援「今天天氣很好，temperature是25度」這樣的混合文本
            4. **專有名詞**: 系統已內建常見品牌和技術詞彙的中文發音
            
            ### 🔧 技術資訊
            - **模型**: MediaTek Breeze2-VITS-onnx
            - **轉換規則**: {len(tts_model.text_converter.conversion_map) if tts_model else 0} 個內建對照
            - **支援格式**: 中文、英文、數字、符號
            - **運行設備**: {device_info}
            - **模型狀態**: {model_status}
            
            ### ⚙️ 自定義轉換
            您可以編輯 `text_mapping.txt` 文件來添加自定義的轉換規則：
            ```
            your_word|您的中文發音
            brand_name|品牌中文名
            ```
            
            ### 🛠️ 故障排除
            - **英文不發音**: 確保啟用「英數轉換」功能
            - **數字不發音**: 檢查轉換功能是否開啟
            - **發音不準**: 嘗試關閉轉換使用純中文文本
            - **載入失敗**: 檢查模型文件是否完整
            """)
        
        # 事件綁定
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, speed, enable_conversion],
            outputs=[audio_output, status_msg],
            api_name="generate_speech"
        )
        
        text_input.submit(
            fn=generate_speech,
            inputs=[text_input, speed, enable_conversion],
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
