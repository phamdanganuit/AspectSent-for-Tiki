import os
import sys
import torch
import pandas as pd
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.constants import GOLD_DIR, SILVER_DIR, TOKENIZED_REVIEWS_FILE

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

# Khối CNN 1D
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 4, 5]):
        super(CNNBlock, self).__init__()
        
        # Các lớp CNN với các kích thước kernel khác nhau
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, 
                     out_channels=out_channels, 
                     kernel_size=k) 
            for k in kernel_sizes
        ])
        
    def forward(self, x):
        # x: batch_size x sequence_length x embedding_dim
        
        # Chuyển đổi sang định dạng cho CNN: batch_size x embedding_dim x sequence_length
        x = x.permute(0, 2, 1)
        
        # Áp dụng các lớp CNN và max-over-time pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = torch.relu(conv_out)
            # Max pooling
            pool_out = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pool_out)
        
        # Ghép các đầu ra từ các kích thước kernel khác nhau
        combined = torch.cat(conv_outputs, dim=1)
        return combined

# Mô hình đa nhiệm vụ dựa trên PhoBERT kết hợp CNN
class MultiTaskPhoBERTCNN(nn.Module):
    def __init__(self, num_sentiment_classes, num_aspect_classes):
        super(MultiTaskPhoBERTCNN, self).__init__()
        
        # Tải pretrained PhoBERT
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        
        # Lớp dropout
        self.dropout = nn.Dropout(0.1)
        
        # Kích thước ẩn của PhoBERT
        hidden_size = self.phobert.config.hidden_size
        
        # Mạng CNN cho trích xuất đặc trưng từ chuỗi đầu ra PhoBERT
        self.cnn_block = CNNBlock(in_channels=hidden_size, out_channels=128)
        
        # Kích thước đầu ra của CNN (128 cho mỗi kích thước kernel, 3 kích thước kernel)
        cnn_output_size = 128 * 3
        
        # Các lớp phân loại
        # Nhiệm vụ 1: Phân loại sentiment (multi-class)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size + cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_sentiment_classes)
        )
        
        # Nhiệm vụ 2: Phân loại aspect (multi-label)
        self.aspect_classifier = nn.Sequential(
            nn.Linear(hidden_size + cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_aspect_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Lấy đầu ra từ PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Lấy embedding của token [CLS] (đầu tiên)
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Lấy đầu ra chuỗi từ PhoBERT để đưa vào CNN
        sequence_output = outputs.last_hidden_state
        
        # Trích xuất đặc trưng bằng CNN
        cnn_features = self.cnn_block(sequence_output)
        
        # Kết hợp đặc trưng từ [CLS] và CNN
        combined_features = torch.cat([cls_output, cnn_features], dim=1)
        combined_features = self.dropout(combined_features)
        
        # Dự đoán cho từng nhiệm vụ
        sentiment_logits = self.sentiment_classifier(combined_features)
        aspect_logits = self.aspect_classifier(combined_features)
        
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
        self.model = MultiTaskPhoBERTCNN(NUM_SENTIMENT_CLASSES, NUM_ASPECT_CLASSES)
        
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

def process_file(input_file, output_file, text_column="tokenized_text", batch_size=8):
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
        df['predicted_aspect_labels'] = [','.join(map(str, pred['aspect_labels'])) for pred in predictions]
        
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
    input_file = TOKENIZED_REVIEWS_FILE  # Sử dụng file đã tách từ thay vì file đã chuẩn hóa
    output_file = os.path.join(GOLD_DIR, "reviews_with_predictions.csv")
    
    # Xử lý file
    process_file(input_file, output_file, text_column="tokenized_text")
    
    print("=== KẾT THÚC QUÁ TRÌNH TRÍCH XUẤT SENTIMENT VÀ ASPECT ===")

if __name__ == "__main__":
    main() 