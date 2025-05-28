from pathlib import Path
import pandas as pd
import numpy as np
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Định nghĩa đường dẫn
BASE_DIR = Path(__file__).parent.parent
STOPWORDS_PATH = BASE_DIR / "data" / "vietnamese_stopwords.txt"
DATA_PATH = BASE_DIR / "data" / "Gold" / "finetuning_metadata.csv"  # Thay bằng tệp dữ liệu kiểm tra nếu cần
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_PATH = OUTPUT_DIR / "evaluation_results.txt"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

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
        tokens = word_tokenize(text)
        if len(tokens) <= 5:
            return ' '.join(tokens)
        processed = ' '.join(word for word in tokens if word.lower() not in stopwords)
        return processed if processed.strip() else ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error during text processing: {e}")
        return ""

# Định nghĩa mô hình LSTM
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

# Định nghĩa mô hình CNN
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

# Hàm đánh giá mô hình
def evaluate_pytorch_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    probabilities = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            probabilities.extend(torch.softmax(output, dim=1).cpu().numpy())
    
    accuracy = correct / total
    cm = confusion_matrix(true_labels, predictions)
    try:
        auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovr')
    except ValueError as e:
        logging.warning(f"Could not compute AUC: {e}")
        auc_score = None
    
    return accuracy, predictions, true_labels, cm, auc_score

# Hàm vẽ confusion matrix
def plot_confusion_matrix(cm, classes, model_name, task, output_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name} {task}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_dir / f'{model_name}_{task}_cm.png')
    plt.close()

# Hàm chính để đánh giá
def main():
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Tải dữ liệu
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['tokenized_text', 'sentiment', 'aspect'])
    df = df[df['tokenized_text'].str.strip() != '']
    df = df[df['tokenized_text'].str.len() > 5]
    
    if df.empty:
        raise ValueError("No valid data after cleaning. Check tokenized_text, sentiment, or aspect columns.")
    
    # Kiểm tra phân bố lớp
    logging.info(f"Sentiment class distribution:\n{df['sentiment'].value_counts()}")
    logging.info(f"Aspect class distribution:\n{df['aspect'].value_counts()}")
    
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
    valid_texts = [text for text in processed_texts if text.strip() and len(text.split()) >= 1]
    valid_indices = [i for i, text in enumerate(processed_texts) if text.strip() and len(text.split()) >= 1]
    
    if not valid_texts:
        raise ValueError("No valid texts after preprocessing. Check data or stopwords.")
    
    texts = valid_texts
    valid_indices = np.array(valid_indices)
    sentiments_encoded = np.array(sentiments_encoded)[valid_indices]
    aspects_encoded = np.array(aspects_encoded)[valid_indices]
    
    # Trích xuất đặc trưng TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(texts).toarray()
    
    # Giảm chiều với SVD
    svd = TruncatedSVD(n_components=50)  # Giữ n_components=50 như trong huấn luyện
    X_svd = svd.fit_transform(X_tfidf)
    
    # Chuyển dữ liệu thành tensor
    X_test_torch = torch.tensor(X_svd, dtype=torch.float32).unsqueeze(-1)
    sentiment_test_torch = torch.tensor(sentiments_encoded, dtype=torch.long)
    aspect_test_torch = torch.tensor(aspects_encoded, dtype=torch.long)
    
    # Tạo DataLoader
    test_dataset_sentiment = TensorDataset(X_test_torch, sentiment_test_torch)
    test_loader_sentiment = DataLoader(test_dataset_sentiment, batch_size=16, shuffle=False)
    test_dataset_aspect = TensorDataset(X_test_torch, aspect_test_torch)
    test_loader_aspect = DataLoader(test_dataset_aspect, batch_size=16, shuffle=False)
    
    # Tải mô hình đã lưu
    lstm_model_sentiment = LSTMModel(input_dim=1, hidden_dim=128, num_classes=num_sentiment_classes).to(device)
    cnn_model_sentiment = CNNModel(input_dim=X_svd.shape[1], num_classes=num_sentiment_classes).to(device)
    lstm_model_aspect = LSTMModel(input_dim=1, hidden_dim=128, num_classes=num_aspect_classes).to(device)
    cnn_model_aspect = CNNModel(input_dim=X_svd.shape[1], num_classes=num_aspect_classes).to(device)
    
    try:
        lstm_model_sentiment.load_state_dict(torch.load(MODEL_DIR / "lstm_model_sentiment_final.pth", map_location=device))
        cnn_model_sentiment.load_state_dict(torch.load(MODEL_DIR / "cnn_model_sentiment_final.pth", map_location=device))
        lstm_model_aspect.load_state_dict(torch.load(MODEL_DIR / "lstm_model_aspect_final.pth", map_location=device))
        cnn_model_aspect.load_state_dict(torch.load(MODEL_DIR / "cnn_model_aspect_final.pth", map_location=device))
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}")
        return
    
    # Đánh giá mô hình
    lstm_sentiment_accuracy, lstm_sentiment_predictions, lstm_sentiment_true, lstm_sentiment_cm, lstm_sentiment_auc = evaluate_pytorch_model(lstm_model_sentiment, test_loader_sentiment, device)
    cnn_sentiment_accuracy, cnn_sentiment_predictions, cnn_sentiment_true, cnn_sentiment_cm, cnn_sentiment_auc = evaluate_pytorch_model(cnn_model_sentiment, test_loader_sentiment, device)
    lstm_aspect_accuracy, lstm_aspect_predictions, lstm_aspect_true, lstm_aspect_cm, lstm_aspect_auc = evaluate_pytorch_model(lstm_model_aspect, test_loader_aspect, device)
    cnn_aspect_accuracy, cnn_aspect_predictions, cnn_aspect_true, cnn_aspect_cm, cnn_aspect_auc = evaluate_pytorch_model(cnn_model_aspect, test_loader_aspect, device)
    
    # Định nghĩa tên lớp (thay bằng tên thực tế nếu biết)
    sentiment_class_names = [str(i) for i in sentiment_encoder.classes_]  # Hoặc ["negative", "neutral", "positive"]
    aspect_class_names = [str(i) for i in aspect_encoder.classes_]  # Hoặc danh sách tên aspect cụ thể
    
    # Ghi kết quả
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(f"LSTM Sentiment Accuracy: {lstm_sentiment_accuracy:.4f}\n")
        f.write(f"LSTM Sentiment AUC: {lstm_sentiment_auc if lstm_sentiment_auc is not None else 'N/A'}\n")
        f.write(f"LSTM Sentiment Confusion Matrix:\n{lstm_sentiment_cm}\n")
        f.write(f"CNN Sentiment Accuracy: {cnn_sentiment_accuracy:.4f}\n")
        f.write(f"CNN Sentiment AUC: {cnn_sentiment_auc if cnn_sentiment_auc is not None else 'N/A'}\n")
        f.write(f"CNN Sentiment Confusion Matrix:\n{cnn_sentiment_cm}\n")
        f.write(f"LSTM Aspect Accuracy: {lstm_aspect_accuracy:.4f}\n")
        f.write(f"LSTM Aspect AUC: {lstm_aspect_auc if lstm_aspect_auc is not None else 'N/A'}\n")
        f.write(f"LSTM Aspect Confusion Matrix:\n{lstm_aspect_cm}\n")
        f.write(f"CNN Aspect Accuracy: {cnn_aspect_accuracy:.4f}\n")
        f.write(f"CNN Aspect AUC: {cnn_aspect_auc if cnn_aspect_auc is not None else 'N/A'}\n")
        f.write(f"CNN Aspect Confusion Matrix:\n{cnn_aspect_cm}\n")
    
    # Ghi báo cáo phân loại
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("LSTM Sentiment Classification Report:\n")
        f.write(classification_report(lstm_sentiment_true, lstm_sentiment_predictions, target_names=sentiment_class_names, zero_division=0))
        f.write("\nCNN Sentiment Classification Report:\n")
        f.write(classification_report(cnn_sentiment_true, cnn_sentiment_predictions, target_names=sentiment_class_names, zero_division=0))
        f.write("\nLSTM Aspect Classification Report:\n")
        f.write(classification_report(lstm_aspect_true, lstm_aspect_predictions, target_names=aspect_class_names, zero_division=0))
        f.write("\nCNN Aspect Classification Report:\n")
        f.write(classification_report(cnn_aspect_true, cnn_aspect_predictions, target_names=aspect_class_names, zero_division=0))
    
    # Vẽ confusion matrix
    plot_confusion_matrix(lstm_sentiment_cm, sentiment_class_names, 'LSTM', 'Sentiment', OUTPUT_DIR)
    plot_confusion_matrix(cnn_sentiment_cm, sentiment_class_names, 'CNN', 'Sentiment', OUTPUT_DIR)
    plot_confusion_matrix(lstm_aspect_cm, aspect_class_names, 'LSTM', 'Aspect', OUTPUT_DIR)
    plot_confusion_matrix(cnn_aspect_cm, aspect_class_names, 'CNN', 'Aspect', OUTPUT_DIR)
    
    # Logging kết quả
    logging.info(f"LSTM Sentiment Accuracy: {lstm_sentiment_accuracy:.4f}, AUC: {lstm_sentiment_auc if lstm_sentiment_auc is not None else 'N/A'}")
    logging.info(f"LSTM Sentiment Confusion Matrix:\n{lstm_sentiment_cm}")
    logging.info(f"LSTM Sentiment Classification Report:\n{classification_report(lstm_sentiment_true, lstm_sentiment_predictions, target_names=sentiment_class_names, zero_division=0)}")
    logging.info(f"CNN Sentiment Accuracy: {cnn_sentiment_accuracy:.4f}, AUC: {cnn_sentiment_auc if cnn_sentiment_auc is not None else 'N/A'}")
    logging.info(f"CNN Sentiment Confusion Matrix:\n{cnn_sentiment_cm}")
    logging.info(f"CNN Sentiment Classification Report:\n{classification_report(cnn_sentiment_true, cnn_sentiment_predictions, target_names=sentiment_class_names, zero_division=0)}")
    logging.info(f"LSTM Aspect Accuracy: {lstm_aspect_accuracy:.4f}, AUC: {lstm_aspect_auc if lstm_aspect_auc is not None else 'N/A'}")
    logging.info(f"LSTM Aspect Confusion Matrix:\n{lstm_aspect_cm}")
    logging.info(f"LSTM Aspect Classification Report:\n{classification_report(lstm_aspect_true, lstm_aspect_predictions, target_names=aspect_class_names, zero_division=0)}")
    logging.info(f"CNN Aspect Accuracy: {cnn_aspect_accuracy:.4f}, AUC: {cnn_aspect_auc if cnn_aspect_auc is not None else 'N/A'}")
    logging.info(f"CNN Aspect Confusion Matrix:\n{cnn_aspect_cm}")
    logging.info(f"CNN Aspect Classification Report:\n{classification_report(cnn_aspect_true, cnn_aspect_predictions, target_names=aspect_class_names, zero_division=0)}")

if __name__ == "__main__":
    main()