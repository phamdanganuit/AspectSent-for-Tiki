import os
import sys
import torch
import pandas as pd
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.constants import GOLD_DIR, SILVER_DIR, NORMALIZED_REVIEWS_FILE

# Đường dẫn đến mô hình đã fine-tune
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
MODEL_DIR = os.path.join(MODELS_DIR, "phobert_finetuned")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

# Ánh xạ aspect và sentiment
ASPECT_MAPPING = {
    0: 'other',
    1: 'cskh',
    2: 'quality',
    3: 'price',
    4: 'ship'
}

SENTIMENT_MAPPING = {
    0: 'rất tiêu cực',
    1: 'tiêu cực',
    2: 'trung lập',
    3: 'tích cực',
    4: 'rất tích cực'
}

# Thông số
NUM_SENTIMENT_CLASSES = 5
NUM_ASPECT_CLASSES = 5
MAX_LENGTH = 256

# Mô hình đa nhiệm vụ dựa trên PhoBERT (sao chép từ phobert_finetune.py)
class MultiTaskPhoBERT(nn.Module):
    def __init__(self, num_sentiment_classes, num_aspect_classes):
        super(MultiTaskPhoBERT, self).__init__()
        
        # Tải pretrained PhoBERT
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        
        # Lớp dropout
        self.dropout = nn.Dropout(0.1)
        
        # Các lớp phân loại
        hidden_size = self.phobert.config.hidden_size
        
        # Nhiệm vụ 1: Phân loại sentiment (multi-class)
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_classes)
        
        # Nhiệm vụ 2: Phân loại aspect (multi-label)
        self.aspect_classifier = nn.Linear(hidden_size, num_aspect_classes)
        
    def forward(self, input_ids, attention_mask):
        # Lấy embedding từ PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Lấy embedding của token [CLS] (đầu tiên)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # Dự đoán cho từng nhiệm vụ
        sentiment_logits = self.sentiment_classifier(pooled_output)
        aspect_logits = self.aspect_classifier(pooled_output)
        
        return sentiment_logits, aspect_logits

class SentimentAspectPredictor:
    def __init__(self, model_path, tokenizer_name="vinai/phobert-base"):
        # Thiết lập thiết bị (GPU nếu có)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")
        
        # Tải tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Đã tải tokenizer: {tokenizer_name}")
        
        # Tải mô hình
        self.model = MultiTaskPhoBERT(NUM_SENTIMENT_CLASSES, NUM_ASPECT_CLASSES)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Đã tải mô hình từ: {model_path}")
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, texts, batch_size=16):
        """
        Dự đoán sentiment và aspect cho một danh sách văn bản.
        
        Args:
            texts (list): Danh sách các văn bản cần dự đoán
            batch_size (int): Kích thước batch
            
        Returns:
            tuple: (sentiment_predictions, aspect_predictions)
        """
        # Danh sách kết quả
        all_sentiment_preds = []
        all_aspect_preds = []
        
        # Xử lý theo batch
        for i in tqdm(range(0, len(texts), batch_size), desc="Dự đoán"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encoded_inputs = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoded_inputs['input_ids'].to(self.device)
            attention_mask = encoded_inputs['attention_mask'].to(self.device)
            
            # Dự đoán
            with torch.no_grad():
                sentiment_logits, aspect_logits = self.model(input_ids, attention_mask)
            
            # Lấy nhãn sentiment (xác suất cao nhất)
            sentiment_preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
            
            # Lấy nhãn aspect (multi-label)
            aspect_probs = torch.sigmoid(aspect_logits).cpu().numpy()
            aspect_preds = (aspect_probs >= 0.5).astype(int)
            
            all_sentiment_preds.extend(sentiment_preds)
            all_aspect_preds.extend(aspect_preds)
        
        return all_sentiment_preds, all_aspect_preds
    
    def format_predictions(self, sentiment_preds, aspect_preds):
        """
        Định dạng kết quả dự đoán thành dạng dễ đọc.
        
        Args:
            sentiment_preds (list): Dự đoán sentiment
            aspect_preds (list): Dự đoán aspect
            
        Returns:
            list: Danh sách các dự đoán được định dạng
        """
        results = []
        
        for sentiment_pred, aspect_pred in zip(sentiment_preds, aspect_preds):
            # Lấy nhãn sentiment
            sentiment = SENTIMENT_MAPPING.get(sentiment_pred, "không xác định")
            
            # Lấy danh sách aspect
            aspects = [ASPECT_MAPPING.get(i) for i, pred in enumerate(aspect_pred) if pred == 1]
            if not aspects:
                aspects = ["other"]
            
            results.append({
                "sentiment": sentiment,
                "sentiment_label": int(sentiment_pred),
                "aspects": aspects,
                "aspect_labels": [i for i, pred in enumerate(aspect_pred) if pred == 1]
            })
        
        return results
    
    def predict_and_format(self, texts, batch_size=16):
        """
        Dự đoán và định dạng kết quả cho một danh sách văn bản.
        
        Args:
            texts (list): Danh sách các văn bản cần dự đoán
            batch_size (int): Kích thước batch
            
        Returns:
            list: Danh sách các dự đoán được định dạng
        """
        sentiment_preds, aspect_preds = self.predict(texts, batch_size)
        return self.format_predictions(sentiment_preds, aspect_preds)

def process_file(input_file, output_file, text_column="reviews", batch_size=8):
    """
    Xử lý file CSV để thêm dự đoán sentiment và aspect.
    
    Args:
        input_file (str): Đường dẫn đến file đầu vào
        output_file (str): Đường dẫn đến file đầu ra
        text_column (str): Tên cột chứa văn bản cần dự đoán
        batch_size (int): Kích thước batch
    """
    try:
        # Đọc dữ liệu
        df = pd.read_csv(input_file)
        print(f"Đã đọc {len(df)} dòng từ {input_file}")
        
        if text_column not in df.columns:
            print(f"Lỗi: Không tìm thấy cột '{text_column}' trong file đầu vào")
            return
        
        # Khởi tạo predictor
        predictor = SentimentAspectPredictor(MODEL_PATH)
        
        # Lấy danh sách văn bản cần dự đoán
        texts = df[text_column].astype(str).tolist()
        
        # Dự đoán
        predictions = predictor.predict_and_format(texts, batch_size)
        
        # Thêm kết quả vào DataFrame
        df['predicted_sentiment'] = [pred['sentiment'] for pred in predictions]
        df['predicted_sentiment_label'] = [pred['sentiment_label'] for pred in predictions]
        df['predicted_aspects'] = [','.join(pred['aspects']) for pred in predictions]
        df['predicted_aspect_labels'] = [pred['aspect_labels'] for pred in predictions]
        
        # Lưu kết quả
        df.to_csv(output_file, index=False)
        print(f"Đã lưu kết quả dự đoán vào {output_file}")
        
    except Exception as e:
        print(f"Lỗi khi xử lý file: {e}")

def main():
    print("=== BẮT ĐẦU QUÁ TRÌNH TRÍCH XUẤT SENTIMENT VÀ ASPECT ===")
    
    # Kiểm tra file mô hình tồn tại
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}")
        print("Vui lòng chạy bước fine-tune trước.")
        return
    
    # Đường dẫn file đầu vào/ra
    input_file = NORMALIZED_REVIEWS_FILE
    output_file = os.path.join(GOLD_DIR, "reviews_with_predictions.csv")
    
    # Xử lý file
    process_file(input_file, output_file)
    
    print("=== KẾT THÚC QUÁ TRÌNH TRÍCH XUẤT SENTIMENT VÀ ASPECT ===")

if __name__ == "__main__":
    main() 