import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import logging
import os
import time
import matplotlib.pyplot as plt

# Cấu hình logging để theo dõi tiến trình
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Cấu hình chung cho các thử nghiệm ---
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
CNN_FILTERS = 128
CNN_KERNEL_SIZE = 5
EPOCHS = 40 # Bạn có thể điều chỉnh số epoch
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOSS_WEIGHT_ASPECT = 1.0
LOSS_WEIGHT_SENTIMENT = 1.0

# --- Đường dẫn lưu trữ chung ---
TOKENIZER_SAVE_PATH = "models/tokenizer.pkl"
ASPECT_LABEL_ENCODER_SAVE_PATH = "models/aspect_label_encoder.pkl"
SENTIMENT_LABEL_ENCODER_SAVE_PATH = "models/sentiment_label_encoder.pkl"
DATA_PATH = "E:/study/NLP/do_an_CK/AspectSent-for-Tiki/data/Gold/finetuning_metadata.csv" # Đường dẫn tới dữ liệu của bạn
TEXT_COLUMN = "tokenized_text"
ASPECT_COLUMN = "aspect"
SENTIMENT_COLUMN = "sentiment"

# --- Đường dẫn cho Thử nghiệm 1: Multi-Output CNN ---
CNN_MULTI_MODEL_PATH = "models/multi_output_cnn_model.pth"
CNN_MULTI_PLOT_DIR = "results/cnn_multi_plots"
CNN_MULTI_REPORT_PATH = "results/cnn_multi_report.txt"

# --- Đường dẫn cho Thử nghiệm 2: Multi-Output LSTM ---
LSTM_MULTI_MODEL_PATH = "models/multi_output_lstm_model.pth"
LSTM_MULTI_PLOT_DIR = "results/lstm_multi_plots"
LSTM_MULTI_REPORT_PATH = "results/lstm_multi_report.txt"

# Tự động chọn thiết bị (GPU nếu có)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Sử dụng thiết bị: {DEVICE}")


def load_and_preprocess_data(data_path, text_column, aspect_column, sentiment_column, max_words, max_len):
    """Tải và tiền xử lý dữ liệu từ file CSV, có xử lý trường hợp không tìm thấy file."""
    logging.info(f"Đang tải dữ liệu từ {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"Không tìm thấy dataset tại {data_path}. Sử dụng dữ liệu giả để minh họa.")
        data_dummy = {
            "text_dummy": ["Sản phẩm rất tốt, giao hàng nhanh.", "Giá hơi cao nhưng chất lượng ổn.", "Thất vọng về dịch vụ chăm sóc khách hàng.", "Pin dùng được lâu, thiết kế đẹp.", "Giao hàng chậm quá.", "Màu sắc không giống hình.", "Rất hài lòng với sản phẩm này.", "Nhân viên tư vấn nhiệt tình.", "Chất liệu vải không như mong đợi.", "Sẽ ủng hộ shop lần sau."] * 20,
            "aspect_dummy": ["Chất lượng sản phẩm", "Giá cả", "Dịch vụ", "Tính năng sản phẩm", "Giao hàng", "Hình thức sản phẩm", "Chất lượng sản phẩm", "Dịch vụ", "Chất lượng sản phẩm", "Tổng quan"] * 20,
            "sentiment_dummy": ["Tích cực", "Trung tính", "Tiêu cực", "Tích cực", "Tiêu cực", "Tiêu cực", "Tích cực", "Tích cực", "Tiêu cực", "Tích cực"] * 20,
        }
        df = pd.DataFrame(data_dummy)
        text_column, aspect_column, sentiment_column = "text_dummy", "aspect_dummy", "sentiment_dummy"
    
    df.dropna(subset=[text_column, aspect_column, sentiment_column], inplace=True)
    texts = df[text_column].astype(str).values
    
    aspect_le = LabelEncoder()
    y_aspect = aspect_le.fit_transform(df[aspect_column].astype(str))
    
    sentiment_le = LabelEncoder()
    y_sentiment = sentiment_le.fit_transform(df[sentiment_column].astype(str))

    tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    
    return X, y_aspect, y_sentiment, tokenizer, aspect_le, sentiment_le

# --- Định nghĩa các mô hình đa đầu ra ---

class MultiOutputCNN(nn.Module):
    """Mô hình CNN với 2 đầu ra cho Aspect và Sentiment."""
    def __init__(self, vocab_size, embedding_dim, num_aspect_classes, num_sentiment_classes, cnn_filters, cnn_kernel_size, dropout_rate=0.5):
        super(MultiOutputCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, kernel_size=cnn_kernel_size)
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc_common = nn.Linear(cnn_filters, 128)
        self.relu_common = nn.ReLU()
        self.fc_aspect = nn.Linear(128, num_aspect_classes)
        self.fc_sentiment = nn.Linear(128, num_sentiment_classes)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conv_out = self.relu(self.conv1d(embedded))
        pooled_out = self.global_max_pool(conv_out).squeeze(2)
        common_features = self.relu_common(self.fc_common(self.dropout(pooled_out)))
        output_aspect = self.fc_aspect(common_features)
        output_sentiment = self.fc_sentiment(common_features)
        return output_aspect, output_sentiment

class MultiOutputLSTM(nn.Module):
    """Mô hình LSTM với 2 đầu ra cho Aspect và Sentiment."""
    def __init__(self, vocab_size, embedding_dim, num_aspect_classes, num_sentiment_classes, lstm_units, dropout_rate=0.5):
        super(MultiOutputLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc_common = nn.Linear(lstm_units * 2, 128) # LSTM hai chiều
        self.relu_common = nn.ReLU()
        self.fc_aspect = nn.Linear(128, num_aspect_classes)
        self.fc_sentiment = nn.Linear(128, num_sentiment_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        lstm_final_output = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        common_features = self.relu_common(self.fc_common(self.dropout(lstm_final_output)))
        output_aspect = self.fc_aspect(common_features)
        output_sentiment = self.fc_sentiment(common_features)
        return output_aspect, output_sentiment

# --- Các hàm huấn luyện và đánh giá chung ---
def train_multi_output_model(model, train_loader, val_loader, optimizer, device, epochs, model_save_path, model_name=""):
    """Hàm chung để huấn luyện một mô hình đa đầu ra và trả về lịch sử huấn luyện."""
    logging.info(f"--- Bắt đầu huấn luyện cho mô hình: {model_name} ---")
    criterion_aspect = nn.CrossEntropyLoss()
    criterion_sentiment = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc_aspect': [], 'val_acc_aspect': [], 'train_acc_sentiment': [], 'val_acc_sentiment': []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_aspect_correct, train_sentiment_correct, train_samples = 0, 0, 0, 0
        for inputs, labels_aspect, labels_sentiment in train_loader:
            inputs, labels_aspect, labels_sentiment = inputs.to(device), labels_aspect.to(device), labels_sentiment.to(device)
            optimizer.zero_grad()
            out_aspect, out_sentiment = model(inputs)
            loss = (LOSS_WEIGHT_ASPECT * criterion_aspect(out_aspect, labels_aspect)) + (LOSS_WEIGHT_SENTIMENT * criterion_sentiment(out_sentiment, labels_sentiment))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, pred_aspect = torch.max(out_aspect.data, 1)
            train_aspect_correct += (pred_aspect == labels_aspect).sum().item()
            _, pred_sentiment = torch.max(out_sentiment.data, 1)
            train_sentiment_correct += (pred_sentiment == labels_sentiment).sum().item()
            train_samples += labels_aspect.size(0)
        
        history['train_loss'].append(train_loss / train_samples)
        history['train_acc_aspect'].append(train_aspect_correct / train_samples)
        history['train_acc_sentiment'].append(train_sentiment_correct / train_samples)
        
        model.eval()
        val_loss, val_aspect_correct, val_sentiment_correct, val_samples = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels_aspect, labels_sentiment in val_loader:
                inputs, labels_aspect, labels_sentiment = inputs.to(device), labels_aspect.to(device), labels_sentiment.to(device)
                out_aspect, out_sentiment = model(inputs)
                val_loss += ((LOSS_WEIGHT_ASPECT * criterion_aspect(out_aspect, labels_aspect)) + (LOSS_WEIGHT_SENTIMENT * criterion_sentiment(out_sentiment, labels_sentiment))).item() * inputs.size(0)
                
                _, pred_aspect = torch.max(out_aspect.data, 1)
                val_aspect_correct += (pred_aspect == labels_aspect).sum().item()
                _, pred_sentiment = torch.max(out_sentiment.data, 1)
                val_sentiment_correct += (pred_sentiment == labels_sentiment).sum().item()
                val_samples += labels_aspect.size(0)

        val_loss_epoch = val_loss / val_samples
        history['val_loss'].append(val_loss_epoch)
        history['val_acc_aspect'].append(val_aspect_correct / val_samples)
        history['val_acc_sentiment'].append(val_sentiment_correct / val_samples)

        logging.info(
            f"Epoch {epoch+1}/{epochs} [{model_name}] - "
            f"Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f} | "
            f"Aspect Acc (T/V): {history['train_acc_aspect'][-1]:.3f}/{history['val_acc_aspect'][-1]:.3f} | "
            f"Sentiment Acc (T/V): {history['train_acc_sentiment'][-1]:.3f}/{history['val_acc_sentiment'][-1]:.3f}"
        )

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Lưu mô hình {model_name} tốt nhất tại epoch {epoch+1} với Val Loss {best_val_loss:.4f} vào {model_save_path}")
            
    return history

def plot_multi_output_history(history, save_dir, model_name):
    """Vẽ và lưu biểu đồ cho quá trình huấn luyện mô hình đa đầu ra."""
    os.makedirs(save_dir, exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(18, 5))
    plt.suptitle(f'Training and Validation Metrics for {model_name}', fontsize=16)
    
    # Biểu đồ Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Loss'); plt.xlabel('Epochs'); plt.legend(); plt.grid(True)
    
    # Biểu đồ Aspect Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['train_acc_aspect'], 'bo-', label='Training Aspect Acc')
    plt.plot(epochs_range, history['val_acc_aspect'], 'ro-', label='Validation Aspect Acc')
    plt.title('Aspect Accuracy'); plt.xlabel('Epochs'); plt.legend(); plt.grid(True)
    
    # Biểu đồ Sentiment Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['train_acc_sentiment'], 'bo-', label='Training Sentiment Acc')
    plt.plot(epochs_range, history['val_acc_sentiment'], 'ro-', label='Validation Sentiment Acc')
    plt.title('Sentiment Accuracy'); plt.xlabel('Epochs'); plt.legend(); plt.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "training_plots.png"))
    plt.close()
    
    logging.info(f"Đã lưu biểu đồ huấn luyện cho {model_name} tại {save_dir}")

def generate_multi_output_report(model_instance, model_path, dataloader, device, aspect_le, sentiment_le, report_file_path, model_name):
    """Tạo báo cáo chi tiết cho một mô hình đa đầu ra."""
    logging.info(f"Đang tạo báo cáo cho {model_name} từ: {model_path}")
    try:
        # Tải trọng số của mô hình. `weights_only=True` là lựa chọn an toàn hơn theo khuyến nghị.
        model_instance.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        logging.error(f"Lỗi khi tải mô hình {model_path}: {e}. Bỏ qua việc tạo báo cáo.")
        return
        
    model_instance.to(device)
    model_instance.eval()
    
    true_aspects, pred_aspects, true_sentiments, pred_sentiments = [], [], [], []
    with torch.no_grad():
        for inputs, labels_aspect, labels_sentiment in dataloader:
            inputs = inputs.to(device)
            out_aspect, out_sentiment = model_instance(inputs)
            
            _, p_aspect = torch.max(out_aspect.data, 1)
            pred_aspects.extend(p_aspect.cpu().numpy())
            true_aspects.extend(labels_aspect.cpu().numpy())
            
            _, p_sentiment = torch.max(out_sentiment.data, 1)
            pred_sentiments.extend(p_sentiment.cpu().numpy())
            true_sentiments.extend(labels_sentiment.cpu().numpy())
    
    # === ĐOẠN MÃ ĐÃ SỬA LỖI ===
    # Sử dụng toán tử `&` của NumPy để so sánh từng phần tử của hai mảng boolean.
    # Điều này tạo ra một mảng boolean mới, nơi True chỉ xuất hiện nếu cả hai dự đoán đều đúng.
    correct_predictions = (np.array(pred_aspects) == np.array(true_aspects)) & (np.array(pred_sentiments) == np.array(true_sentiments))
    subset_accuracy = np.mean(correct_predictions)
    
    report_aspect = classification_report(true_aspects, pred_aspects, target_names=aspect_le.classes_.astype(str), zero_division=0)
    report_sentiment = classification_report(true_sentiments, pred_sentiments, target_names=sentiment_le.classes_.astype(str), zero_division=0)
    
    os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH: {model_name}\n")
        f.write(f"Đường dẫn mô hình: {model_path}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Subset Accuracy (Cả Aspect và Sentiment đều đúng): {subset_accuracy:.4f}\n\n")
        f.write("--- Báo cáo Phân loại Khía cạnh (Aspect) ---\n")
        f.write(report_aspect + "\n\n")
        f.write("--- Báo cáo Phân loại Cảm xúc (Sentiment) ---\n")
        f.write(report_sentiment)
    
    logging.info(f"Báo cáo chi tiết cho {model_name} đã lưu tại: {report_file_path}")
    print(f"\n--- BÁO CÁO CHO {model_name.upper()} ---")
    print(f"Subset Accuracy: {subset_accuracy:.4f}")
    print("\nBáo cáo Khía cạnh (Aspect Report):\n", report_aspect)
    print("\nBáo cáo Cảm xúc (Sentiment Report):\n", report_sentiment)


def main():
    """Hàm chính điều phối toàn bộ quy trình so sánh hai mô hình."""
    # 1. Tải và tiền xử lý dữ liệu (dùng chung)
    X, y_aspect, y_sentiment, tokenizer, aspect_le, sentiment_le = load_and_preprocess_data(
        DATA_PATH, TEXT_COLUMN, ASPECT_COLUMN, SENTIMENT_COLUMN, MAX_WORDS, MAX_SEQUENCE_LENGTH
    )
    
    # Lưu các công cụ tiền xử lý để có thể tái sử dụng sau này
    for path, obj in [(TOKENIZER_SAVE_PATH, tokenizer), (ASPECT_LABEL_ENCODER_SAVE_PATH, aspect_le), (SENTIMENT_LABEL_ENCODER_SAVE_PATH, sentiment_le)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f: pickle.dump(obj, f)
    logging.info("Đã lưu Tokenizer và Label Encoders.")

    num_aspect_classes = len(aspect_le.classes_)
    num_sentiment_classes = len(sentiment_le.classes_)

    # Chia dữ liệu một lần để đảm bảo cả hai mô hình được huấn luyện và đánh giá trên cùng tập dữ liệu
    X_train, X_test, y_aspect_train, y_aspect_test, y_sentiment_train, y_sentiment_test = train_test_split(
        X, y_aspect, y_sentiment, test_size=0.2, random_state=42, stratify=y_aspect
    )

    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_aspect_train), torch.LongTensor(y_sentiment_train))
    val_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_aspect_test), torch.LongTensor(y_sentiment_test))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- THỬ NGHIỆM 1: MULTI-OUTPUT CNN ---
    logging.info("="*50 + "\nBẮT ĐẦU THỬ NGHIỆM 1: MULTI-OUTPUT CNN\n" + "="*50)
    cnn_model = MultiOutputCNN(MAX_WORDS, EMBEDDING_DIM, num_aspect_classes, num_sentiment_classes, CNN_FILTERS, CNN_KERNEL_SIZE).to(DEVICE)
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    cnn_history = train_multi_output_model(cnn_model, train_loader, val_loader, optimizer_cnn, DEVICE, EPOCHS, CNN_MULTI_MODEL_PATH, "Multi-Output CNN")
    plot_multi_output_history(cnn_history, CNN_MULTI_PLOT_DIR, "Multi-Output CNN")
    generate_multi_output_report(cnn_model, CNN_MULTI_MODEL_PATH, val_loader, DEVICE, aspect_le, sentiment_le, CNN_MULTI_REPORT_PATH, "Multi-Output CNN")
    
    # --- THỬ NGHIỆM 2: MULTI-OUTPUT LSTM ---
    logging.info("="*50 + "\nBẮT ĐẦU THỬ NGHIỆM 2: MULTI-OUTPUT LSTM\n" + "="*50)
    lstm_model = MultiOutputLSTM(MAX_WORDS, EMBEDDING_DIM, num_aspect_classes, num_sentiment_classes, LSTM_UNITS).to(DEVICE)
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    lstm_history = train_multi_output_model(lstm_model, train_loader, val_loader, optimizer_lstm, DEVICE, EPOCHS, LSTM_MULTI_MODEL_PATH, "Multi-Output LSTM")
    plot_multi_output_history(lstm_history, LSTM_MULTI_PLOT_DIR, "Multi-Output LSTM")
    generate_multi_output_report(lstm_model, LSTM_MULTI_MODEL_PATH, val_loader, DEVICE, aspect_le, sentiment_le, LSTM_MULTI_REPORT_PATH, "Multi-Output LSTM")

    logging.info("="*50)
    logging.info("Tất cả các quy trình so sánh đã hoàn thành.")

if __name__ == "__main__":
    main()
