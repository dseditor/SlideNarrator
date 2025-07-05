"""
Breeze2-VITS 繁體中文語音合成 - 修正版
只修正字典檔載入問題，保持原有功能
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

# 安裝 pypinyin 用於中文發音處理
try:
    from pypinyin import pinyin, Style
except ImportError:
    os.system("pip install pypinyin")
    from pypinyin import pinyin, Style


class TextConverter:
    """文本轉換器，將英文和數字轉換為中文發音"""
    
    def __init__(self, mapping_file="text_mapping.txt"):
        self.mapping_file = Path(mapping_file)
        self.conversion_map = {}
        self.debug_mode = False  # 簡化調試模式
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
                self.create_enhanced_mapping()
        except Exception as e:
            print(f"❌ 載入轉換對照表失敗: {e}")
            self.create_enhanced_mapping()
    
    def create_enhanced_mapping(self):
        """創建基本轉換對照表"""
        default_mappings = {
            # 數字
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
            '10': '十', '11': '十一', '12': '十二', '13': '十三', '14': '十四', '15': '十五',
            '16': '十六', '17': '十七', '18': '十八', '19': '十九', '20': '二十',
            
            # 基本英文
            'hello': '哈囉', 'hi': '嗨', 'bye': '拜拜', 'ok': '好的',
            'ai': '人工智慧', 'cpu': '中央處理器', 'gpu': '圖形處理器',
            
            # 字母
            'a': '欸', 'b': '比', 'c': '西', 'd': '迪', 'e': '伊',
            'f': '艾夫', 'g': '吉', 'h': '艾奇', 'i': '愛', 'j': '傑',
            'k': '凱', 'l': '艾爾', 'm': '艾姆', 'n': '艾恩', 'o': '歐',
            'p': '皮', 'q': '丘', 'r': '艾爾', 's': '艾斯', 't': '替',
            'u': '優', 'v': '威', 'w': '達布爾優', 'x': '艾克斯', 'y': '歪', 'z': '萊德',
        }
        
        self.conversion_map = default_mappings
        print(f"✅ 使用基本轉換規則: {len(default_mappings)} 個")
    
    def debug_print(self, message):
        """調試打印函數"""
        if self.debug_mode:
            print(f"🔍 [DEBUG] {message}")
    
    def convert_numbers(self, text):
        """轉換連續數字為中文"""
        def number_to_chinese(match):
            number = match.group()
            if len(number) <= 2:  
                result = ""
                for digit in number:
                    chinese_digit = self.conversion_map.get(digit, digit)
                    result += chinese_digit
                return result
            else:
                return self.convert_large_number(number)
        
        result = re.sub(r'\d+', number_to_chinese, text)
        return result
    
    def convert_large_number(self, number_str):
        """轉換大數字為中文"""
        try:
            num = int(number_str)
            if num == 0:
                return '零'
            
            if str(num) in self.conversion_map:
                return self.conversion_map[str(num)]
            
            # 簡化的數字轉換
            digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
            
            if num < 10:
                return digits[num]
            elif num < 20:
                if num == 10:
                    return '十'
                else:
                    return '十' + digits[num % 10]
            elif num < 100:
                tens = num // 10
                ones = num % 10
                result = digits[tens] + '十'
                if ones > 0:
                    result += digits[ones]
                return result
            else:
                # 逐位轉換
                result = ""
                for digit in number_str:
                    result += digits[int(digit)]
                return result
        except:
            result = ""
            for digit in number_str:
                if digit.isdigit():
                    result += self.conversion_map.get(digit, digit)
                else:
                    result += digit
            return result
    
    def convert_uppercase_words(self, text):
        """轉換全大寫單字為逐字母發音"""
        def uppercase_to_letters(match):
            word = match.group()
            result = ""
            for letter in word:
                chinese_letter = self.conversion_map.get(letter.lower(), letter)
                result += chinese_letter
            return result
        
        result = re.sub(r'\b[A-Z]{2,}\b', uppercase_to_letters, text)
        return result
    
    def convert_english(self, text):
        """轉換英文單詞為中文"""
        sorted_words = sorted(self.conversion_map.keys(), key=len, reverse=True)
        
        for english_word in sorted_words:
            if len(english_word) > 1:
                chinese_word = self.conversion_map[english_word]
                pattern = r'\b' + re.escape(english_word) + r'\b'
                text = re.sub(pattern, chinese_word, text, flags=re.IGNORECASE)
        
        return text
    
    def convert_single_letters(self, text):
        """轉換單個英文字母"""
        def letter_to_chinese(match):
            letter = match.group().lower()
            chinese = self.conversion_map.get(letter, letter)
            return chinese
        
        result = re.sub(r'\b[a-zA-Z]\b', letter_to_chinese, text)
        return result
    
    def preprocess_text(self, text):
        """預處理文本"""
        text = re.sub(r'\bDr\.', 'Doctor', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMr\.', 'Mister', text, flags=re.IGNORECASE)
        text = re.sub(r'@', ' at ', text)
        text = re.sub(r'\.com\b', ' dot com', text, flags=re.IGNORECASE)
        return text
    
    def postprocess_text(self, text):
        """後處理文本"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s+([，。！？；：])', r'\1', text)
        return text
    
    def convert_text(self, text):
        """主要轉換函數"""
        if not text:
            return text
        
        original_text = text
        print(f"🔄 開始轉換文本: {repr(original_text)}")
        
        # 預處理
        text = self.preprocess_text(text)
        
        # 轉換
        text = self.convert_uppercase_words(text)
        text = self.convert_english(text)
        text = self.convert_numbers(text)
        text = self.convert_single_letters(text)
        
        # 後處理
        text = self.postprocess_text(text)
        
        if text != original_text:
            print(f"✅ 轉換完成: {repr(original_text)} → {repr(text)}")
        else:
            print(f"ℹ️ 文本未發生變化: {repr(text)}")
        
        return text


class TaiwaneseVITSTTS:
    def __init__(self):
        self.tts = None
        self.model_dir = Path("./models")
        self.dict_dir = Path("./dict")  # 保留原邏輯
        self.text_converter = TextConverter()
        self.debug_mode = False
        self.setup_model()
    
    def debug_print(self, message):
        """調試打印函數"""
        if self.debug_mode:
            print(f"🔍 [TTS DEBUG] {message}")
    
    def verify_model_files(self):
        """檢查模型文件 - 修正版本"""
        print("🔍 檢查模型文件...")
        
        # 檢查多種可能的檔案名稱
        model_files = {
            "model": ["breeze2-vits.onnx", "model.onnx", "vits.onnx"],
            "lexicon": ["lexicon.txt"],
            "tokens": ["tokens.txt"]
        }
        
        found_files = {}
        
        for file_type, possible_names in model_files.items():
            found = False
            for name in possible_names:
                file_path = self.model_dir / name
                if file_path.exists() and file_path.stat().st_size > 0:
                    found_files[file_type] = str(file_path)
                    print(f"✅ 找到 {file_type}: {name}")
                    found = True
                    break
            
            if not found:
                print(f"❌ 未找到 {file_type} 文件")
                return False, {}
        
        return True, found_files

    def setup_model(self):
        """設置和初始化模型 - 修正字典檔載入"""
        try:
            # 檢查模型文件
            files_exist, model_files = self.verify_model_files()
            if not files_exist:
                raise FileNotFoundError("模型文件缺失")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
            
            print(f"🔧 使用設備: {device.upper()}")
            print(f"🔧 使用執行提供者: {provider}")
            
            # 參考 SherpaTTS.kt 的配置方式
            vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model_files["model"],
                lexicon=model_files["lexicon"],
                tokens=model_files["tokens"],
                dict_dir=str(self.dict_dir) if self.dict_dir.exists() else "",  # 如果 dict 目錄存在就使用
                data_dir="",  # 根據 Android 版本，這個可以為空
            )
            
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_config,
                num_threads=4 if device == "cpu" else 2,
                debug=False,
                provider=provider,
            )
            
            config = sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                rule_fsts="",  # 參考 Android 版本設為空
                rule_fars="",  # 參考 Android 版本設為空  
                max_num_sentences=5,
            )
            
            print("🔄 正在載入 TTS 模型...")
            self.tts = sherpa_onnx.OfflineTts(config)
            
            # 獲取模型信息
            num_speakers = self.tts.num_speakers
            sample_rate = self.tts.sample_rate
            
            print(f"✅ TTS 模型載入成功!")
            print(f"📊 說話者數量: {num_speakers}")
            print(f"📊 採樣率: {sample_rate} Hz")
            
            # 測試模型
            test_audio = self.tts.generate(text="測試", sid=0, speed=1.0)
            if len(test_audio.samples) > 0:
                print("✅ 模型測試通過!")
            
        except Exception as e:
            print(f"❌ 模型設置失敗: {e}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            raise

    def validate_converted_text(self, text):
        """驗證轉換後的文本是否適合TTS"""
        english_chars = re.findall(r'[a-zA-Z]+', text)
        if english_chars:
            self.debug_print(f"警告：轉換後仍有英文字母: {english_chars}")
        
        unsupported_chars = re.findall(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s\d，。！？；：]', text)
        if unsupported_chars:
            self.debug_print(f"警告：發現可能不支持的字符: {set(unsupported_chars)}")
        
        return text

    def synthesize(self, text, speed=1.0, enable_conversion=True):
        """合成語音"""
        if not text or not text.strip():
            return None, "❌ 請輸入文本"
        
        original_text = text.strip()
        self.debug_print(f"開始語音合成，原始文本: {repr(original_text)}")
        
        # 文本轉換
        if enable_conversion:
            text = self.text_converter.convert_text(original_text)
            text = self.validate_converted_text(text)
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
                status_info += f"\n🔄 文本轉換: {original_text} → {text}"
            
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
        ["你好，歡迎使用繁體中文語音合成系統！", 1.0],
        ["今天天氣晴朗，適合外出踏青。", 1.0],
        ["台灣的夜市文化非常豐富多彩。", 1.0],
        ["人工智慧技術正在快速發展。", 1.1],
        ["這個語音合成系統效果很不錯。", 1.0],
        ["祝您使用愉快，謝謝您的支持。", 0.9],
    ]
    
    device_info = "🎮 GPU" if torch.cuda.is_available() else "💻 CPU"
    
    with gr.Blocks(
        title="繁體中文語音合成 - Breeze2-VITS Enhanced",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
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
            <strong>🇹🇼 輕量台灣發音TTS</strong> 
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
                    label="📝 輸入文本 (以中文為主，英數表現不佳)",
                    placeholder="請輸入要合成的文本",
                    lines=5,
                    max_lines=8,
                    value="你好！歡迎使用繁體中文語音合成系統。"
                )
                
                speed = gr.Slider(
                    label="⚡ 語音速度",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    info="調節語音播放速度"
                )
                
                enable_conversion = gr.State(value=True)
                
                generate_btn = gr.Button(
                    "🎵 生成語音",
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
                    lines=8,
                    value="準備就緒，請輸入文本並點擊生成語音" if tts_model else f"模型載入失敗: {model_status}"
                )
        
        if tts_model:
            gr.Examples(
                examples=examples,
                inputs=[text_input, speed],
                outputs=[audio_output, status_msg],
                fn=lambda text, speed: generate_speech(text, speed, True),
                cache_examples=False,
                label="📚 範例文本"
            )
        
        with gr.Accordion("📋 使用說明與功能特色", open=False):
            gr.Markdown(f"""
            ### 🚀 主要功能
            
            #### 🎯 支援內容
            - 單一語音的繁體中文文本，英文數字支援有限
            - 模型輕量
            
            ### 🔧 技術資訊
            - **模型**: MediaTek Breeze2-VITS-onnx
            - **運行設備**: {device_info}
            """)
        
        # 事件綁定
        generate_btn.click(
            fn=lambda text, speed, conv=True: generate_speech(text, speed, conv),
            inputs=[text_input, speed],
            outputs=[audio_output, status_msg],
            api_name="generate_speech"
        )
        
        text_input.submit(
            fn=lambda text, speed, conv=True: generate_speech(text, speed, conv),
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
