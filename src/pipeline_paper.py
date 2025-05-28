from pathlib import Path
import os
import pandas as pd
import numpy as np
from underthesea import word_tokenize  # Thay bằng vnTokenizer nếu có
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Định nghĩa đường dẫn
BASE_DIR = Path(__file__).parent.parent
STOPWORDS_PATH = BASE_DIR / "data" / "vietnamese_stopwords.txt"
DATA_PATH = BASE_DIR / "data" / "Gold" / "finetuning_metadata.csv"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_PATH = OUTPUT_DIR / "classification_results.txt"
MODEL_DIR = BASE_DIR / "models"
EP = 20  # Số epoch huấn luyện

# Kiểm tra file tồn tại
if not STOPWORDS_PATH.exists():
    raise FileNotFoundError(f"Stopwords file not found at {STOPWORDS_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

# Hàm tải danh sách stop-words
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(f.read().splitlines())

# Hàm tiền xử lý văn bản
def preprocess_text(text, stopwords):
    if not isinstance(text, str) or not text.strip():
        logging.warning("Invalid or empty text input")
        return ""
    try:
        tokens = word_tokenize(text)  # Thay bằng vnTokenizer.tokenize(text) nếu có
        if len(tokens) <= 5:
            return ' '.join(tokens)
        processed = ' '.join(word for word in tokens if word.lower() not in stopwords)
        return processed if processed.strip() else ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error tokenizing text '{text}': {e}")
        return ""

# Mô hình LSTM với PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.dropout(hn[-1])
        x = self.fc(x)
        return x

# Mô hình CNN với PyTorch
class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * (input_dim // 2), num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Hàm huấn luyện mô hình PyTorch 
def train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, class_weights=None):
    model.train()  # Đặt mô hình ở chế độ huấn luyện
    best_val_acc = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for data, target in train_loader:
            # Kiểm tra dữ liệu đầu vào
            if data.size(0) == 0 or target.size(0) == 0:
                logging.warning("Empty batch detected, skipping...")
                continue
            logging.debug(f"Batch data shape: {data.shape}, target shape: {target.shape}")
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            if class_weights is not None:
                loss = loss * class_weights[target].mean()
            
            try:
                loss.backward()
            except RuntimeError as e:
                logging.error(f"Error during backward pass: {e}")
                logging.error(f"Input data shape: {data.shape}")
                raise
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        val_accuracy, _ = evaluate_pytorch_model(model, val_loader, device)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Đặt lại chế độ huấn luyện sau khi đánh giá
        model.train()
        logging.debug(f"Model training mode after evaluation: {model.training}")
        
        # Lưu mô hình tốt nhất dựa trên validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), MODEL_DIR / f"{model.__class__.__name__}_best.pth")
    
    model.load_state_dict(torch.load(MODEL_DIR / f"{model.__class__.__name__}_best.pth"))
    return model

# Hàm đánh giá mô hình PyTorch
def evaluate_pytorch_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            predictions.extend(predicted.cpu().numpy())
    accuracy = correct / total
    return accuracy, predictions

# Hàm chính
def main():
    # Tải dữ liệu
    df = pd.read_csv(DATA_PATH)
    
    # Làm sạch dữ liệu
    df = df.dropna(subset=['tokenized_text', 'sentiment', 'aspect'])
    df = df[df['tokenized_text'].str.strip() != '']
    df = df[df['tokenized_text'].str.len() > 5]
    
    if df.empty:
        raise ValueError("No valid data after cleaning. Check tokenized_text, sentiment, or aspect columns.")
    
    # Lấy cột text, sentiment, và aspect
    texts = df['tokenized_text'].astype(str)
    sentiments = df['sentiment']
    aspects = df['aspect']
    
    # Mã hóa nhãn
    sentiment_encoder = LabelEncoder()
    aspect_encoder = LabelEncoder()
    sentiments_encoded = sentiment_encoder.fit_transform(sentiments)
    aspects_encoded = aspect_encoder.fit_transform(aspects)
    num_sentiment_classes = len(sentiment_encoder.classes_)
    num_aspect_classes = len(aspect_encoder.classes_)
    
    # Tải stop-words
    stopwords = load_stopwords(STOPWORDS_PATH)
    
    # Tiền xử lý văn bản
    processed_texts = [preprocess_text(text, stopwords) for text in texts]
    
    # Lọc văn bản hợp lệ
    valid_texts = [text for text in processed_texts if text.strip() and len(text.split()) >= 1]
    valid_indices = [i for i, text in enumerate(processed_texts) if text.strip() and len(text.split()) >= 1]
    if not valid_texts:
        raise ValueError("No valid texts after preprocessing. Check data or stopwords.")
    
    # Kiểm tra từ vựng
    logging.info("Sample processed texts: %s", valid_texts[:5])
    logging.info("Number of valid texts: %d", len(valid_texts))
    unique_tokens = set()
    for text in valid_texts:
        unique_tokens.update(text.split())
    logging.info("Unique tokens sample: %s", list(unique_tokens)[:10])
    logging.info("Vocabulary size: %d", len(unique_tokens))
    
    texts = valid_texts
    valid_indices = np.array(valid_indices)
    sentiments = sentiments.iloc[valid_indices]
    aspects = aspects.iloc[valid_indices]
    sentiments_encoded = np.array(sentiments_encoded)[valid_indices]
    aspects_encoded = np.array(aspects_encoded)[valid_indices]
    
    # Trích xuất đặc trưng TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    try:
        X_tfidf = tfidf.fit_transform(texts).toarray()
        logging.info("TF-IDF vocabulary size: %d", len(tfidf.vocabulary_))
    except ValueError as e:
        logging.error(f"Error in TF-IDF: {e}")
        logging.error("Check if texts are valid or adjust stop words.")
        return
    
    # Giảm chiều với SVD
    svd = TruncatedSVD(n_components=50)
    X_svd = svd.fit_transform(X_tfidf)
    
    # Chia dữ liệu: train, validation, test
    X_temp, X_test, sentiment_temp, sentiment_test, aspect_temp, aspect_test = train_test_split(
        X_svd, sentiments_encoded, aspects_encoded, test_size=0.2, random_state=42
    )
    X_train, X_val, sentiment_train, sentiment_val, aspect_train, aspect_val = train_test_split(
        X_temp, sentiment_temp, aspect_temp, test_size=0.1, random_state=42
    )
    
    # Định dạng dữ liệu cho PyTorch
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    sentiment_train_torch = torch.tensor(sentiment_train, dtype=torch.long)
    sentiment_val_torch = torch.tensor(sentiment_val, dtype=torch.long)
    sentiment_test_torch = torch.tensor(sentiment_test, dtype=torch.long)
    aspect_train_torch = torch.tensor(aspect_train, dtype=torch.long)
    aspect_val_torch = torch.tensor(aspect_val, dtype=torch.long)
    aspect_test_torch = torch.tensor(aspect_test, dtype=torch.long)
    
    # Kiểm tra shape của dữ liệu
    logging.info(f"X_train_torch shape: {X_train_torch.shape}")
    logging.info(f"sentiment_train_torch shape: {sentiment_train_torch.shape}")
    
    # Tạo DataLoader cho sentiment
    train_dataset_sentiment = TensorDataset(X_train_torch, sentiment_train_torch)
    val_dataset_sentiment = TensorDataset(X_val_torch, sentiment_val_torch)
    test_dataset_sentiment = TensorDataset(X_test_torch, sentiment_test_torch)
    train_loader_sentiment = DataLoader(train_dataset_sentiment, batch_size=16, shuffle=True)
    val_loader_sentiment = DataLoader(val_dataset_sentiment, batch_size=16, shuffle=False)
    test_loader_sentiment = DataLoader(test_dataset_sentiment, batch_size=16, shuffle=False)
    
    # Tạo DataLoader cho aspect
    train_dataset_aspect = TensorDataset(X_train_torch, aspect_train_torch)
    val_dataset_aspect = TensorDataset(X_val_torch, aspect_val_torch)
    test_dataset_aspect = TensorDataset(X_test_torch, aspect_test_torch)
    train_loader_aspect = DataLoader(train_dataset_aspect, batch_size=16, shuffle=True)
    val_loader_aspect = DataLoader(val_dataset_aspect, batch_size=16, shuffle=False)
    test_loader_aspect = DataLoader(test_dataset_aspect, batch_size=16, shuffle=False)
    
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Tạm thời tắt cuDNN để debug
    torch.backends.cudnn.enabled = False
    
    # Tính trọng số lớp
    sentiment_class_weights = compute_class_weight('balanced', classes=np.unique(sentiment_train), y=sentiment_train)
    aspect_class_weights = compute_class_weight('balanced', classes=np.unique(aspect_train), y=aspect_train)
    sentiment_class_weights_torch = torch.tensor(sentiment_class_weights, dtype=torch.float32).to(device)
    aspect_class_weights_torch = torch.tensor(aspect_class_weights, dtype=torch.float32).to(device)
    
    # Khởi tạo và huấn luyện mô hình LSTM và CNN cho sentiment
    lstm_model_sentiment = LSTMModel(input_dim=1, hidden_dim=128, num_classes=num_sentiment_classes).to(device)
    cnn_model_sentiment = CNNModel(input_dim=X_train.shape[1], num_classes=num_sentiment_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_lstm = optim.Adam(lstm_model_sentiment.parameters(), lr=0.001)
    scheduler_lstm = optim.lr_scheduler.StepLR(optimizer_lstm, step_size=5, gamma=0.1)
    lstm_model_sentiment = train_pytorch_model(lstm_model_sentiment, train_loader_sentiment, val_loader_sentiment, criterion, optimizer_lstm, scheduler_lstm, device, num_epochs=EP, class_weights=sentiment_class_weights_torch)
    
    optimizer_cnn = optim.Adam(cnn_model_sentiment.parameters(), lr=0.001)
    scheduler_cnn = optim.lr_scheduler.StepLR(optimizer_cnn, step_size=5, gamma=0.1)
    cnn_model_sentiment = train_pytorch_model(cnn_model_sentiment, train_loader_sentiment, val_loader_sentiment, criterion, optimizer_cnn, scheduler_cnn, device, num_epochs=EP, class_weights=sentiment_class_weights_torch)
    
    # Khởi tạo và huấn luyện mô hình LSTM và CNN cho aspect
    lstm_model_aspect = LSTMModel(input_dim=1, hidden_dim=128, num_classes=num_aspect_classes).to(device)
    cnn_model_aspect = CNNModel(input_dim=X_train.shape[1], num_classes=num_aspect_classes).to(device)
    
    optimizer_lstm = optim.Adam(lstm_model_aspect.parameters(), lr=0.001)
    scheduler_lstm = optim.lr_scheduler.StepLR(optimizer_lstm, step_size=5, gamma=0.1)
    lstm_model_aspect = train_pytorch_model(lstm_model_aspect, train_loader_aspect, val_loader_aspect, criterion, optimizer_lstm, scheduler_lstm, device, num_epochs=EP, class_weights=aspect_class_weights_torch)
    
    optimizer_cnn = optim.Adam(cnn_model_aspect.parameters(), lr=0.001)
    scheduler_cnn = optim.lr_scheduler.StepLR(optimizer_cnn, step_size=5, gamma=0.1)
    cnn_model_aspect = train_pytorch_model(cnn_model_aspect, train_loader_aspect, val_loader_aspect, criterion, optimizer_cnn, scheduler_cnn, device, num_epochs=EP, class_weights=aspect_class_weights_torch)
    
    # Đánh giá mô hình trên tập test
    lstm_sentiment_accuracy, lstm_sentiment_predictions = evaluate_pytorch_model(lstm_model_sentiment, test_loader_sentiment, device)
    cnn_sentiment_accuracy, cnn_sentiment_predictions = evaluate_pytorch_model(cnn_model_sentiment, test_loader_sentiment, device)
    lstm_aspect_accuracy, lstm_aspect_predictions = evaluate_pytorch_model(lstm_model_aspect, test_loader_aspect, device)
    cnn_aspect_accuracy, cnn_aspect_predictions = evaluate_pytorch_model(cnn_model_aspect, test_loader_aspect, device)
    
    # Ghi log và báo cáo kết quả
    logging.info(f"LSTM Sentiment Accuracy: {lstm_sentiment_accuracy:.4f}")
    logging.info(f"LSTM Sentiment Classification Report:\n%s", 
                 classification_report(sentiment_test, lstm_sentiment_predictions, target_names=[str(i) for i in sentiment_encoder.classes_]))
    logging.info(f"CNN Sentiment Accuracy: {cnn_sentiment_accuracy:.4f}")
    logging.info(f"CNN Sentiment Classification Report:\n%s", 
                 classification_report(sentiment_test, cnn_sentiment_predictions, target_names=[str(i) for i in sentiment_encoder.classes_]))
    
    logging.info(f"LSTM Aspect Accuracy: {lstm_aspect_accuracy:.4f}")
    logging.info(f"LSTM Aspect Classification Report:\n%s", 
                 classification_report(aspect_test, lstm_aspect_predictions, target_names=[str(i) for i in aspect_encoder.classes_]))
    logging.info(f"CNN Aspect Accuracy: {cnn_aspect_accuracy:.4f}")
    logging.info(f"CNN Aspect Classification Report:\n%s", 
                 classification_report(aspect_test, cnn_aspect_predictions, target_names=[str(i) for i in aspect_encoder.classes_]))
    
    # Lưu kết quả
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(f"LSTM Sentiment Accuracy: {lstm_sentiment_accuracy:.4f}\n")
        f.write(f"CNN Sentiment Accuracy: {cnn_sentiment_accuracy:.4f}\n")
        f.write(f"LSTM Aspect Accuracy: {lstm_aspect_accuracy:.4f}\n")
        f.write(f"CNN Aspect Accuracy: {cnn_aspect_accuracy:.4f}\n")
    
    # Lưu mô hình
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(lstm_model_sentiment.state_dict(), MODEL_DIR / "lstm_model_sentiment_final.pth")
    torch.save(cnn_model_sentiment.state_dict(), MODEL_DIR / "cnn_model_sentiment_final.pth")
    torch.save(lstm_model_aspect.state_dict(), MODEL_DIR / "lstm_model_aspect_final.pth")
    torch.save(cnn_model_aspect.state_dict(), MODEL_DIR / "cnn_model_aspect_final.pth")

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()