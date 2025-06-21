import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import logging
import os
import time

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Cấu hình ---
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
CNN_FILTERS = 128 # Tăng số filter cho mô hình CNN riêng
CNN_KERNEL_SIZE = 5

# --- Đường dẫn lưu file cho 2 mô hình riêng biệt ---
SENTIMENT_MODEL_PATH = "models/sentiment_cnn_model.pth"
ASPECT_MODEL_PATH = "models/aspect_lstm_model.pth"
TOKENIZER_SAVE_PATH = "models/tokenizer.pkl" # Dùng chung tokenizer
ASPECT_LABEL_ENCODER_SAVE_PATH = "models/aspect_label_encoder.pkl"
SENTIMENT_LABEL_ENCODER_SAVE_PATH = "models/sentiment_label_encoder.pkl"

DATA_PATH = "E:/study/NLP/do_an_CK/AspectSent-for-Tiki/data/Gold/finetuning_metadata.csv" # Đường dẫn của bạn
TEXT_COLUMN = "tokenized_text"
ASPECT_COLUMN = "aspect"
SENTIMENT_COLUMN = "sentiment"

EPOCHS = 40 # Có thể cần số epoch khác nhau cho mỗi mô hình
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Sử dụng thiết bị: {DEVICE}")

# Hàm load_and_preprocess_data giữ nguyên như trước
# vì bước chuẩn bị dữ liệu là giống nhau cho cả hai mô hình.
def load_and_preprocess_data(
    data_path, text_column_config, aspect_column_config, sentiment_column_config,
    max_words, max_sequence_length
):
    logging.info(f"Đang tải dữ liệu từ {data_path}...")
    current_text_column = text_column_config
    current_aspect_column = aspect_column_config
    current_sentiment_column = sentiment_column_config

    try:
        df = pd.read_csv(data_path)
        # (Phần xử lý lỗi và dữ liệu giả giữ nguyên như cũ)
    except FileNotFoundError:
        logging.error(f"Không tìm thấy dataset tại {data_path}. Sử dụng dữ liệu giả.")
        # Tạo dữ liệu giả
        data_dummy = {
            "text_dummy": ["Sản phẩm tốt.", "Giao hàng nhanh.", "Giá quá cao."] * 30,
            "aspect_dummy": ["Chất lượng sản phẩm", "Giao hàng", "Giá cả"] * 30,
            "sentiment_dummy": ["Tích cực", "Tích cực", "Tiêu cực"] * 30,
        }
        df = pd.DataFrame(data_dummy)
        current_text_column, current_aspect_column, current_sentiment_column = "text_dummy", "aspect_dummy", "sentiment_dummy"

    df.dropna(subset=[current_text_column, current_aspect_column, current_sentiment_column], inplace=True)
    texts = df[current_text_column].astype(str).values
    aspect_labels_raw = df[current_aspect_column].astype(str).values
    sentiment_labels_raw = df[current_sentiment_column].astype(str).values

    aspect_label_encoder = LabelEncoder()
    y_aspect = aspect_label_encoder.fit_transform(aspect_labels_raw)
    
    sentiment_label_encoder = LabelEncoder()
    y_sentiment = sentiment_label_encoder.fit_transform(sentiment_labels_raw)

    tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_sequence_length, padding="post", truncating="post")
    
    return X, y_aspect, y_sentiment, tokenizer, aspect_label_encoder, sentiment_label_encoder

# --- Định nghĩa các mô hình riêng biệt ---

# 1. Mô hình chỉ dùng CNN
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, cnn_filters, cnn_kernel_size, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, kernel_size=cnn_kernel_size)
        self.relu = nn.ReLU()
        # Global Max Pooling sẽ lấy giá trị lớn nhất trên toàn bộ chuỗi
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(cnn_filters, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded_permuted = embedded.permute(0, 2, 1) # -> (batch, embed_dim, seq_len)
        conv_out = self.conv1d(embedded_permuted)
        conv_out = self.relu(conv_out)
        pooled_out = self.global_max_pool(conv_out).squeeze(2) # (batch, filters, 1) -> (batch, filters)
        pooled_out = self.dropout(pooled_out)
        output = self.fc(pooled_out)
        return output

# 2. Mô hình chỉ dùng LSTM
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, lstm_units, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        # LSTM hai chiều (bidirectional) nên hidden_size * 2
        self.fc = nn.Linear(lstm_units * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        # Lấy output cuối cùng của LSTM
        _, (h_n, _) = self.lstm(embedded)
        # h_n shape: (num_layers*num_directions, batch, hidden_size)
        # Nối hidden state của chiều xuôi và ngược
        lstm_final_output = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        output = self.dropout(lstm_final_output)
        output = self.fc(output)
        return output

# --- Hàm Huấn luyện và Đánh giá chung cho mô hình đơn ---
def train_single_model_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    return total_loss / total_samples, total_correct / total_samples

def evaluate_single_model_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples

def train_single_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, model_save_path, task_name=""):
    logging.info(f"--- Bắt đầu huấn luyện cho tác vụ: {task_name} ---")
    best_val_loss = float('inf')
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = train_single_model_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_single_model_epoch(model, val_loader, criterion, device)
        end_time = time.time()

        logging.info(
            f"Epoch {epoch+1}/{epochs} - Time: {end_time - start_time:.2f}s - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Lưu mô hình {task_name} tốt nhất tại epoch {epoch+1} vào {model_save_path}")
    logging.info(f"--- Hoàn thành huấn luyện cho {task_name} ---")

def main():
    # 1. Tải và xử lý dữ liệu (dùng chung)
    X, y_aspect, y_sentiment, tokenizer, aspect_le, sentiment_le = load_and_preprocess_data(
        DATA_PATH, TEXT_COLUMN, ASPECT_COLUMN, SENTIMENT_COLUMN, MAX_WORDS, MAX_SEQUENCE_LENGTH
    )
    
    # Lưu tokenizer và các label encoder
    os.makedirs(os.path.dirname(TOKENIZER_SAVE_PATH), exist_ok=True)
    with open(TOKENIZER_SAVE_PATH, "wb") as f: pickle.dump(tokenizer, f)
    with open(ASPECT_LABEL_ENCODER_SAVE_PATH, "wb") as f: pickle.dump(aspect_le, f)
    with open(SENTIMENT_LABEL_ENCODER_SAVE_PATH, "wb") as f: pickle.dump(sentiment_le, f)
    logging.info("Đã lưu Tokenizer và Label Encoders.")

    # 2. HUẤN LUYỆN MÔ HÌNH SENTIMENT (DÙNG CNN)
    logging.info("="*50)
    logging.info("CHUẨN BỊ HUẤN LUYỆN MÔ HÌNH SENTIMENT (CNN)")
    num_sentiment_classes = len(sentiment_le.classes_)
    logging.info(f"Số lớp Sentiment: {num_sentiment_classes} - {sentiment_le.classes_}")
    
    X_train, X_test, y_sentiment_train, y_sentiment_test = train_test_split(
        X, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
    )

    train_dataset_sent = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_sentiment_train))
    val_dataset_sent = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_sentiment_test))
    train_loader_sent = DataLoader(train_dataset_sent, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_sent = DataLoader(val_dataset_sent, batch_size=BATCH_SIZE)

    sentiment_model = CNNModel(
        vocab_size=MAX_WORDS,
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_sentiment_classes,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_size=CNN_KERNEL_SIZE
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(sentiment_model.parameters(), lr=LEARNING_RATE)
    
    train_single_model(sentiment_model, train_loader_sent, val_loader_sent, criterion, optimizer, 
                       DEVICE, EPOCHS, SENTIMENT_MODEL_PATH, "Sentiment Classification")


    # 3. HUẤN LUYỆN MÔ HÌNH ASPECT (DÙNG LSTM)
    logging.info("="*50)
    logging.info("CHUẨN BỊ HUẤN LUYỆN MÔ HÌNH ASPECT (LSTM)")
    num_aspect_classes = len(aspect_le.classes_)
    logging.info(f"Số lớp Aspect: {num_aspect_classes} - {aspect_le.classes_}")

    X_train, X_test, y_aspect_train, y_aspect_test = train_test_split(
        X, y_aspect, test_size=0.2, random_state=42, stratify=y_aspect
    )

    train_dataset_asp = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_aspect_train))
    val_dataset_asp = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_aspect_test))
    train_loader_asp = DataLoader(train_dataset_asp, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_asp = DataLoader(val_dataset_asp, batch_size=BATCH_SIZE)

    aspect_model = LSTMModel(
        vocab_size=MAX_WORDS,
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_aspect_classes,
        lstm_units=LSTM_UNITS
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(aspect_model.parameters(), lr=LEARNING_RATE)

    train_single_model(aspect_model, train_loader_asp, val_loader_asp, criterion, optimizer, 
                       DEVICE, EPOCHS, ASPECT_MODEL_PATH, "Aspect Classification")

    logging.info("="*50)
    logging.info("Tất cả các quy trình huấn luyện đã hoàn thành.")

if __name__ == "__main__":
    main()