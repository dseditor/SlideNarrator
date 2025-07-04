"""
Breeze2-VITS 繁體中文語音合成 - 英文朗讀問題修復版
增強調試功能和轉換邏輯
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
    """文本轉換器，將英文和數字轉換為中文發音 - 增強調試版"""
    
    def __init__(self, mapping_file="text_mapping.txt"):
        self.mapping_file = Path(mapping_file)
        self.conversion_map = {}
        self.debug_mode = True  # 啟用調試模式
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
                
                # 調試：顯示部分轉換規則
                if self.debug_mode:
                    print("🔍 部分轉換規則:")
                    for i, (k, v) in enumerate(list(self.conversion_map.items())[:10]):
                        print(f"  {k} → {v}")
                    if len(self.conversion_map) > 10:
                        print(f"  ... 還有 {len(self.conversion_map) - 10} 個規則")
                        
            else:
                print(f"⚠️ 轉換對照表文件不存在: {self.mapping_file}")
                self.create_enhanced_mapping()
        except Exception as e:
            print(f"❌ 載入轉換對照表失敗: {e}")
            self.create_enhanced_mapping()
    
    def create_enhanced_mapping(self):
        """創建增強的轉換對照表"""
        default_mappings = {
            # 數字
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
            '10': '十', '11': '十一', '12': '十二', '13': '十三', '14': '十四', '15': '十五',
            '16': '十六', '17': '十七', '18': '十八', '19': '十九', '20': '二十',
            '100': '一百', '1000': '一千', '10000': '一萬',
            
            # 基本英文問候語
            'hello': '哈囉', 'hi': '嗨', 'hey': '嘿', 'bye': '拜拜', 'goodbye': '再見',
            'yes': '是的', 'no': '不', 'ok': '好的', 'okay': '好的',
            'good': '好的', 'bad': '不好', 'nice': '很棒', 'great': '很好',
            'thank': '謝謝', 'thanks': '謝謝', 'please': '請',
            'sorry': '對不起', 'excuse': '不好意思',
            
            # 時間相關
            'today': '今天', 'tomorrow': '明天', 'yesterday': '昨天',
            'morning': '早上', 'afternoon': '下午', 'evening': '晚上', 'night': '晚上',
            'monday': '星期一', 'tuesday': '星期二', 'wednesday': '星期三',
            'thursday': '星期四', 'friday': '星期五', 'saturday': '星期六', 'sunday': '星期日',
            
            # 常用動詞
            'go': '去', 'come': '來', 'see': '看', 'look': '看', 'do': '做', 'make': '做',
            'get': '得到', 'take': '拿', 'give': '給', 'have': '有', 'be': '是',
            'know': '知道', 'think': '想', 'want': '想要', 'need': '需要',
            'like': '喜歡', 'love': '愛', 'help': '幫助', 'work': '工作',
            
            # 技術詞彙
            'ai': '人工智慧', 'api': '程式介面', 'app': '應用程式', 'web': '網路',
            'cpu': '中央處理器', 'gpu': '圖形處理器', 'ram': '記憶體',
            'computer': '電腦', 'laptop': '筆記型電腦', 'phone': '手機', 'mobile': '手機',
            'internet': '網際網路', 'wifi': '無線網路', 'bluetooth': '藍牙',
            'software': '軟體', 'hardware': '硬體', 'program': '程式', 'code': '程式碼',
            'data': '資料', 'database': '資料庫', 'file': '檔案', 'folder': '資料夾',
            
            # 品牌名稱
            'apple': '蘋果', 'google': '谷歌', 'microsoft': '微軟', 'amazon': '亞馬遜',
            'facebook': '臉書', 'twitter': '推特', 'youtube': '油管', 'instagram': 'instagram',
            'samsung': '三星', 'sony': '索尼', 'lg': 'LG', 'htc': 'HTC',
            'iphone': '愛瘋', 'android': '安卓', 'windows': '視窗系統', 'ios': 'iOS',
            
            # 常用形容詞
            'big': '大', 'small': '小', 'new': '新', 'old': '舊',
            'hot': '熱', 'cold': '冷', 'fast': '快', 'slow': '慢',
            'easy': '容易', 'hard': '困難', 'simple': '簡單', 'complex': '複雜',
            'important': '重要', 'useful': '有用', 'interesting': '有趣',
            
            # 字母 (更自然的中文發音)
            'a': '欸', 'b': '比', 'c': '西', 'd': '迪', 'e': '伊',
            'f': '艾夫', 'g': '吉', 'h': '艾奇', 'i': '愛', 'j': '傑',
            'k': '凱', 'l': '艾爾', 'm': '艾姆', 'n': '艾恩', 'o': '歐',
            'p': '皮', 'q': '丘', 'r': '艾爾', 's': '艾斯', 't': '替',
            'u': '優', 'v': '威', 'w': '達布爾優', 'x': '艾克斯', 'y': '歪', 'z': '萊德',
            
            # 縮寫詞
            'ceo': '執行長', 'cto': '技術長', 'cfo': '財務長',
            'usa': '美國', 'uk': '英國', 'eu': '歐盟',
            'nasa': '美國太空總署', 'fbi': '聯邦調查局',
            'covid': '新冠肺炎', 'dna': 'DNA', 'gps': '全球定位系統',
            
            # 網路用語
            'email': '電子郵件', 'www': '全球資訊網', 'http': 'HTTP',
            'url': '網址', 'link': '連結', 'click': '點擊',
            'download': '下載', 'upload': '上傳', 'login': '登入', 'logout': '登出',
            
            # 常見英文片語的關鍵詞
            'how': '如何', 'what': '什麼', 'where': '哪裡', 'when': '什麼時候',
            'why': '為什麼', 'who': '誰', 'which': '哪個',
            'this': '這個', 'that': '那個', 'here': '這裡', 'there': '那裡',
            'and': '和', 'or': '或', 'but': '但是', 'so': '所以',
            'very': '非常', 'much': '很多', 'many': '很多', 'some': '一些',
            'all': '全部', 'every': '每個', 'any': '任何',
        }
        
        self.conversion_map = default_mappings
        print(f"✅ 使用增強轉換規則: {len(default_mappings)} 個")
    
    def debug_print(self, message):
        """調試打印函數"""
        if self.debug_mode:
            print(f"🔍 [DEBUG] {message}")
    
    def convert_numbers(self, text):
        """轉換連續數字為中文 - 增強版"""
        self.debug_print(f"數字轉換前: {repr(text)}")
        
        def number_to_chinese(match):
            number = match.group()
            self.debug_print(f"處理數字: {number}")
            
            if len(number) <= 2:  
                result = ""
                for digit in number:
                    chinese_digit = self.conversion_map.get(digit, digit)
                    result += chinese_digit
                    self.debug_print(f"  {digit} → {chinese_digit}")
                return result
            else:
                # 複雜數字處理
                converted = self.convert_large_number(number)
                self.debug_print(f"  大數字 {number} → {converted}")
                return converted
        
        # 匹配連續數字
        result = re.sub(r'\d+', number_to_chinese, text)
        if result != text:
            self.debug_print(f"數字轉換後: {repr(result)}")
        return result
    
    def convert_large_number(self, number_str):
        """轉換大數字為中文 - 改進版"""
        try:
            num = int(number_str)
            if num == 0:
                return '零'
            
            # 使用更完整的數字轉換
            if str(num) in self.conversion_map:
                return self.conversion_map[str(num)]
            
            # 簡化的數字轉換（支援到萬）
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
            elif num < 1000:
                hundreds = num // 100
                remainder = num % 100
                result = digits[hundreds] + '百'
                if remainder > 0:
                    if remainder < 10:
                        result += '零' + digits[remainder]
                    elif remainder < 20:
                        result += '一十' if remainder == 10 else '一十' + digits[remainder % 10]
                    else:
                        result += self.convert_large_number(str(remainder))
                return result
            elif num < 10000:
                thousands = num // 1000
                remainder = num % 1000
                result = digits[thousands] + '千'
                if remainder > 0:
                    if remainder < 100:
                        result += '零' + self.convert_large_number(str(remainder))
                    else:
                        result += self.convert_large_number(str(remainder))
                return result
            else:
                # 對於更大的數字，逐位轉換
                result = ""
                for digit in number_str:
                    result += digits[int(digit)]
                return result
        except:
            # 如果轉換失敗，逐位轉換數字
            result = ""
            for digit in number_str:
                if digit.isdigit():
                    result += self.conversion_map.get(digit, digit)
                else:
                    result += digit
            return result
    
    def convert_english(self, text):
        """轉換英文單詞為中文 - 增強調試版"""
        self.debug_print(f"英文轉換前: {repr(text)}")
        original_text = text
        
        # 按長度排序，先處理長詞彙
        sorted_words = sorted(self.conversion_map.keys(), key=len, reverse=True)
        
        conversion_count = 0
        for english_word in sorted_words:
            if len(english_word) > 1:  # 跳過單字母，後面單獨處理
                chinese_word = self.conversion_map[english_word]
                # 使用單詞邊界匹配，不區分大小寫
                pattern = r'\b' + re.escape(english_word) + r'\b'
                new_text = re.sub(pattern, chinese_word, text, flags=re.IGNORECASE)
                
                if new_text != text:
                    self.debug_print(f"  轉換: {english_word} → {chinese_word}")
                    conversion_count += 1
                    text = new_text
        
        if conversion_count > 0:
            self.debug_print(f"英文轉換後: {repr(text)} (共轉換 {conversion_count} 個詞)")
        else:
            self.debug_print("沒有找到可轉換的英文詞彙")
            
        return text
    
    def convert_single_letters(self, text):
        """轉換單個英文字母 - 增強版"""
        self.debug_print(f"字母轉換前: {repr(text)}")
        
        def letter_to_chinese(match):
            letter = match.group().lower()
            chinese = self.conversion_map.get(letter, letter)
            self.debug_print(f"  字母轉換: {letter} → {chinese}")
            return chinese
        
        # 匹配獨立的英文字母
        result = re.sub(r'\b[a-zA-Z]\b', letter_to_chinese, text)
        if result != text:
            self.debug_print(f"字母轉換後: {repr(result)}")
        return result
    
    def preprocess_text(self, text):
        """預處理文本 - 處理特殊情況"""
        # 處理常見的英文縮寫
        text = re.sub(r'\bDr\.', 'Doctor', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMr\.', 'Mister', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMrs\.', 'Missis', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMs\.', 'Miss', text, flags=re.IGNORECASE)
        
        # 處理email地址中的@符號
        text = re.sub(r'@', ' at ', text)
        
        # 處理網址中的點
        text = re.sub(r'\.com\b', ' dot com', text, flags=re.IGNORECASE)
        text = re.sub(r'\.org\b', ' dot org', text, flags=re.IGNORECASE)
        text = re.sub(r'\.net\b', ' dot net', text, flags=re.IGNORECASE)
        
        return text
    
    def postprocess_text(self, text):
        """後處理文本 - 清理和優化"""
        # 清理多餘空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 處理標點符號前的空格
        text = re.sub(r'\s+([，。！？；：])', r'\1', text)
        
        return text
    
    def convert_text(self, text):
        """主要轉換函數 - 增強調試版"""
        if not text:
            return text
        
        original_text = text
        print(f"🔄 開始轉換文本: {repr(original_text)}")
        
        # 預處理
        text = self.preprocess_text(text)
        if text != original_text:
            self.debug_print(f"預處理後: {repr(text)}")
        
        # 1. 轉換英文單詞（先處理多字母詞彙）
        text = self.convert_english(text)
        
        # 2. 轉換數字
        text = self.convert_numbers(text)
        
        # 3. 轉換剩餘的單個字母
        text = self.convert_single_letters(text)
        
        # 4. 後處理
        text = self.postprocess_text(text)
        
        if text != original_text:
            print(f"✅ 轉換完成: {repr(original_text)} → {repr(text)}")
        else:
            print(f"ℹ️ 文本未發生變化: {repr(text)}")
        
        return text
    
    def test_conversion(self, test_texts=None):
        """測試轉換功能"""
        if test_texts is None:
            test_texts = [
                "Hello world",
                "I have 123 apples",
                "My email is test@gmail.com",
                "Apple iPhone 15 is good",
                "AI and ML are useful",
                "CPU speed is 3.5 GHz"
            ]
        
        print("\n🧪 測試文本轉換功能:")
        print("=" * 50)
        for text in test_texts:
            converted = self.convert_text(text)
            print(f"原文: {text}")
            print(f"轉換: {converted}")
            print("-" * 50)


class TaiwaneseVITSTTS:
    def __init__(self):
        self.tts = None
        self.model_dir = Path("./models")
        self.dict_dir = Path("./dict")
        self.text_converter = TextConverter()
        self.debug_mode = True  # 啟用調試模式
        self.setup_jieba_dict()
        self.setup_model()
    
    def debug_print(self, message):
        """調試打印函數"""
        if self.debug_mode:
            print(f"🔍 [TTS DEBUG] {message}")
    
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
                
                # 測試轉換功能
                print("\n🧪 測試文本轉換:")
                self.text_converter.test_conversion()
            
        except Exception as e:
            print(f"❌ 模型設置失敗: {e}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            raise

    def validate_converted_text(self, text):
        """驗證轉換後的文本是否適合TTS"""
        # 檢查是否還有英文字母
        english_chars = re.findall(r'[a-zA-Z]+', text)
        if english_chars:
            self.debug_print(f"警告：轉換後仍有英文字母: {english_chars}")
        
        # 檢查是否有不支持的字符
        unsupported_chars = re.findall(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s\d，。！？；：]', text)
        if unsupported_chars:
            self.debug_print(f"警告：發現可能不支持的字符: {set(unsupported_chars)}")
        
        return text

    def synthesize(self, text, speed=1.0, enable_conversion=True):
        """合成語音 - 增強調試版"""
        if not text or not text.strip():
            return None, "❌ 請輸入文本"
        
        original_text = text.strip()
        self.debug_print(f"開始語音合成，原始文本: {repr(original_text)}")
        
        # 文本轉換
        if enable_conversion:
            text = self.text_converter.convert_text(original_text)
            # 驗證轉換結果
            text = self.validate_converted_text(text)
        else:
            text = original_text
            self.debug_print("跳過文本轉換")
        
        if len(text) > 500:
            text = text[:500]
            self.debug_print("文本過長，已截斷至500字符")
            
        try:
            print(f"🎤 正在合成語音...")
            self.debug_print(f"最終TTS輸入文本: {repr(text)}")
            
            if enable_conversion and text != original_text:
                print(f"📝 使用轉換後文本: {text}")
            
            audio = self.tts.generate(text=text, sid=0, speed=speed)
            samples = audio.samples
            sample_rate = audio.sample_rate
            
            self.debug_print(f"TTS輸出 - 樣本數: {len(samples)}, 採樣率: {sample_rate}")
            
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
            
            # 添加調試信息
            if self.debug_mode:
                status_info += f"\n🔍 調試信息:\n  原始長度: {len(original_text)}\n  轉換後長度: {len(text)}"
            
            return (sample_rate, audio_array), status_info
            
        except Exception as e:
            error_msg = f"❌ 語音合成失敗: {str(e)}"
            print(error_msg)
            self.debug_print(f"合成失敗詳情: {e}")
            return None, error_msg


# 初始化時運行測試
def run_initialization_tests():
    """運行初始化測試"""
    print("\n" + "="*60)
    print("🔧 運行系統診斷測試")
    print("="*60)
    
    # 測試文本轉換器
    converter = TextConverter()
    test_cases = [
        "Hello world",
        "I love Apple iPhone 15",
        "AI technology is amazing",
        "My email is user@gmail.com",
        "CPU speed is 2.5 GHz"
    ]
    
    print("\n📝 測試文本轉換功能:")
    for test_text in test_cases:
        result = converter.convert_text(test_text)
        print(f"  輸入: {test_text}")
        print(f"  輸出: {result}")
        print()


# 全局 TTS 實例
print("🔧 正在初始化 TTS 模型...")
try:
    # 運行診斷測試
    run_initialization_tests()
    
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
    # 預設範例文本 - 增加更多測試用例
    examples = [
        ["你好，歡迎使用繁體中文語音合成系統！", 1.0, True],
        ["Hello world! 這是一個測試。", 1.0, True],
        ["I love Apple iPhone 15 and Samsung Galaxy", 1.0, True],
        ["AI technology is amazing, CPU speed is 3.5 GHz", 1.0, True],
        ["My email is test@gmail.com, please contact me", 1.0, True],
        ["今天是2024年1月1日，天氣很好。", 1.0, True],
        ["Google and Microsoft are big tech companies", 1.0, True],
        ["API development with Python is easy", 1.0, True],
    ]
    
    device_info = "🎮 GPU" if torch.cuda.is_available() else "💻 CPU"
    
    with gr.Blocks(
        title="繁體中文語音合成 - Breeze2-VITS Enhanced Debug",
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
        .debug-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        gr.HTML(f"""
        <div class="status-box">
            <h1>🎙️ 繁體中文語音合成 - Breeze2-VITS Enhanced Debug</h1>
            <p><strong>狀態:</strong> {model_status} | <strong>設備:</strong> {device_info}</p>
        </div>
        """)
        
        gr.HTML("""
        <div class="debug-box">
            <strong>🔍 調試增強版</strong> | 詳細轉換日志 | 問題診斷 | 性能分析
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
                    value="Hello world! 今天是2024年，歡迎使用AI語音合成系統。"
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
                    label="📊 狀態資訊與調試信息",
                    interactive=False,
                    lines=8,
                    value="準備就緒，請輸入文本並點擊生成語音" if tts_model else f"模型載入失敗: {model_status}"
                )
        
        if tts_model:
            gr.Examples(
                examples=examples,
                inputs=[text_input, speed, enable_conversion],
                outputs=[audio_output, status_msg],
                fn=generate_speech,
                cache_examples=False,
                label="📚 範例文本 (包含英文朗讀測試)"
            )
        
        with gr.Accordion("🔍 調試信息與故障排除", open=True):
            gr.Markdown(f"""
            ### 🚀 調試功能
            
            #### 🔄 轉換規則狀態
            - **載入規則數**: {len(tts_model.text_converter.conversion_map) if tts_model else 0} 個
            - **調試模式**: {'✅ 已啟用' if tts_model and tts_model.debug_mode else '❌ 未啟用'}
            - **模型狀態**: {model_status}
            
            #### 🧪 常見問題診斷
            
            **問題1: 英文不發音**
            - ✅ 確保啟用「英數轉換」功能
            - ✅ 檢查控制台轉換日志
            - ✅ 測試單獨的英文單詞
            
            **問題2: 轉換後仍有英文**
            - 可能是詞典中缺少該詞彙
            - 查看調試信息中的轉換過程
            - 考慮添加自定義轉換規則
            
            **問題3: 發音不自然**
            - 嘗試調整轉換後的中文用詞
            - 使用更常見的中文表達
            - 關閉轉換使用純中文測試
            
            #### 🔧 調試步驟
            1. 打開瀏覽器開發者工具查看控制台
            2. 輸入測試文本並生成語音
            3. 觀察轉換過程的調試信息
            4. 檢查哪些詞彙被成功轉換
            5. 分析未轉換詞彙的原因
            
            #### 📝 測試建議
            - 先測試純英文: "Hello world"
            - 再測試中英混合: "Hello 世界"
            - 測試數字: "I have 123 apples"
            - 測試品牌: "Apple iPhone Samsung"
            - 測試技術詞彙: "AI CPU GPU API"
            """)
        
        with gr.Accordion("📋 使用說明與功能特色", open=False):
            gr.Markdown(f"""
            ### 🚀 主要功能
            
            #### 🔄 智慧文本轉換 (增強版)
            - **基本英文**: hello → 哈囉, good → 好的, thank → 謝謝
            - **技術詞彙**: AI → 人工智慧, CPU → 中央處理器, API → 程式介面
            - **品牌名稱**: Apple → 蘋果, Google → 谷歌, iPhone → 愛瘋
            - **數字轉換**: 123 → 一二三, 2024 → 二零二四
            - **字母發音**: A → 欸, B → 比, C → 西
            - **縮寫詞**: CEO → 執行長, USA → 美國, GPS → 全球定位系統
            
            #### 🎯 支援內容
            - 繁體中文文本
            - 英文單詞和句子
            - 阿拉伯數字
            - 混合語言文本
            - 常見縮寫和品牌
            - 網路用語和技術術語
            
            ### 📝 使用技巧
            1. **測試英文**: 使用範例中的英文測試案例
            2. **調試轉換**: 查看控制台的詳細轉換過程
            3. **混合文本**: 嘗試「Hello world 這是測試」
            4. **數字處理**: 測試不同長度的數字
            
            ### 🔧 技術資訊
            - **模型**: MediaTek Breeze2-VITS-onnx
            - **轉換規則**: {len(tts_model.text_converter.conversion_map) if tts_model else 0} 個內建對照
            - **調試模式**: {'啟用' if tts_model and tts_model.debug_mode else '未啟用'}
            - **運行設備**: {device_info}
            - **模型狀態**: {model_status}
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
