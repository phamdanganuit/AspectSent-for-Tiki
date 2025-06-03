import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer # Sử dụng Tokenizer của Keras
from keras.preprocessing.sequence import pad_sequences # Sử dụng pad_sequences của Keras
import pickle
import logging
import os
import time
import matplotlib.pyplot as plt

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Cấu hình ---
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
CNN_FILTERS = 64
CNN_KERNEL_SIZE = 5
NUM_ASPECT_CLASSES = None
NUM_SENTIMENT_CLASSES = None

MODEL_SAVE_PATH = "models/multi_output_cnn_lstm_pytorch.pth"
TOKENIZER_SAVE_PATH = "models/tokenizer_multi_output_pytorch.pkl"
ASPECT_LABEL_ENCODER_SAVE_PATH = "models/aspect_label_encoder_pytorch.pkl"
SENTIMENT_LABEL_ENCODER_SAVE_PATH = "models/sentiment_label_encoder_pytorch.pkl"
PLOTS_SAVE_DIR = "results/training_plots" # Thư mục lưu biểu đồ
CLASSIFICATION_REPORT_PATH = "E:/study/NLP/do_an_CK/AspectSent-for-Tiki/results/multi_output_classification_report_pytorch.txt" # Đường dẫn lưu báo cáo

DATA_PATH = "E:/study/NLP/do_an_CK/AspectSent-for-Tiki/data/Gold/finetuning_metadata.csv" # Đường dẫn của bạn
TEXT_COLUMN = "tokenized_text"
ASPECT_COLUMN = "aspect"
SENTIMENT_COLUMN = "sentiment"

EPOCHS = 40 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOSS_WEIGHT_ASPECT = 1.0
LOSS_WEIGHT_SENTIMENT = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Sử dụng thiết bị: {DEVICE}")


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
        logging.info(f"Đã tải thành công dữ liệu từ: {data_path}")
        if current_text_column not in df.columns:
            raise ValueError(f"Cột văn bản '{current_text_column}' không tìm thấy trong tệp {data_path}.")
        if current_aspect_column not in df.columns:
            raise ValueError(f"Cột khía cạnh '{current_aspect_column}' không tìm thấy trong tệp {data_path}.")
        if current_sentiment_column not in df.columns:
            raise ValueError(f"Cột cảm xúc '{current_sentiment_column}' không tìm thấy trong tệp {data_path}.")

    except FileNotFoundError:
        logging.error(f"Không tìm thấy dataset tại {data_path}. Vui lòng cung cấp đường dẫn hợp lệ.")
        logging.warning("Sử dụng dữ liệu giả để minh họa vì không tìm thấy dataset.")
        dummy_data_text_col = "text_dummy"
        dummy_data_aspect_col = "aspect_dummy"
        dummy_data_sentiment_col = "sentiment_dummy"
        data_dummy = {
            dummy_data_text_col: [
                "Sản phẩm rất tốt, giao hàng nhanh.", "Giá hơi cao nhưng chất lượng ổn.",
                "Thất vọng về dịch vụ chăm sóc khách hàng.", "Pin dùng được lâu, thiết kế đẹp.",
                "Giao hàng chậm quá.", "Màu sắc không giống hình.",
                "Rất hài lòng với sản phẩm này.", "Nhân viên tư vấn nhiệt tình.",
                "Chất liệu vải không như mong đợi.", "Sẽ ủng hộ shop lần sau."
            ] * 10, # Nhân dữ liệu giả lên để có đủ mẫu cho train/test split
            dummy_data_aspect_col: [
                "Chất lượng sản phẩm", "Giá cả", "Dịch vụ", "Tính năng sản phẩm", "Giao hàng",
                "Hình thức sản phẩm", "Chất lượng sản phẩm", "Dịch vụ", "Chất lượng sản phẩm", "Tổng quan"
            ] * 10,
            dummy_data_sentiment_col: [
                "Tích cực", "Trung tính", "Tiêu cực", "Tích cực", "Tiêu cực",
                "Tiêu cực", "Tích cực", "Tích cực", "Tiêu cực", "Tích cực"
            ] * 10,
        }
        df = pd.DataFrame(data_dummy)
        logging.warning(f"Sử dụng cột '{dummy_data_text_col}' của dữ liệu giả thay cho '{current_text_column}' (cấu hình).")
        current_text_column = dummy_data_text_col
        logging.warning(f"Sử dụng cột '{dummy_data_aspect_col}' của dữ liệu giả thay cho '{current_aspect_column}' (cấu hình).")
        current_aspect_column = dummy_data_aspect_col
        logging.warning(f"Sử dụng cột '{dummy_data_sentiment_col}' của dữ liệu giả thay cho '{current_sentiment_column}' (cấu hình).")
        current_sentiment_column = dummy_data_sentiment_col

    logging.info(f"Số lượng dòng ban đầu: {len(df)}")
    df.dropna(subset=[current_text_column, current_aspect_column, current_sentiment_column], inplace=True)
    logging.info(f"Số lượng dòng sau khi loại bỏ NaN ở các cột ('{current_text_column}', '{current_aspect_column}', '{current_sentiment_column}'): {len(df)}")

    if df.empty:
        raise ValueError("Dataset rỗng sau khi loại bỏ NaN. Vui lòng kiểm tra dữ liệu đầu vào.")

    texts = df[current_text_column].astype(str).values
    aspect_labels_raw = df[current_aspect_column].astype(str).values
    sentiment_labels_raw = df[current_sentiment_column].astype(str).values

    logging.info("Đang mã hóa nhãn Khía cạnh...")
    aspect_label_encoder = LabelEncoder()
    y_aspect = aspect_label_encoder.fit_transform(aspect_labels_raw)
    num_aspect_classes_val = len(aspect_label_encoder.classes_)
    logging.info(f"Số lượng lớp Khía cạnh: {num_aspect_classes_val}")
    logging.info(f"Các lớp Khía cạnh: {aspect_label_encoder.classes_}")
    global NUM_ASPECT_CLASSES # Cập nhật biến global
    NUM_ASPECT_CLASSES = num_aspect_classes_val

    logging.info("Đang mã hóa nhãn Cảm xúc...")
    sentiment_label_encoder = LabelEncoder()
    y_sentiment = sentiment_label_encoder.fit_transform(sentiment_labels_raw)
    num_sentiment_classes_val = len(sentiment_label_encoder.classes_)
    logging.info(f"Số lượng lớp Cảm xúc: {num_sentiment_classes_val}")
    logging.info(f"Các lớp Cảm xúc: {sentiment_label_encoder.classes_}")
    global NUM_SENTIMENT_CLASSES # Cập nhật biến global
    NUM_SENTIMENT_CLASSES = num_sentiment_classes_val

    logging.info("Đang tokenize văn bản...")
    tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
    tokenizer.fit_on_texts(texts)

    logging.info("Đang chuyển văn bản thành chuỗi số và padding...")
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_sequence_length, padding="post", truncating="post")

    logging.info("Đang chia dữ liệu thành tập huấn luyện và tập kiểm tra...")
    stratify_labels = None
    if len(np.unique(y_aspect)) >= 2 and len(np.unique(y_sentiment)) < 2 :
        stratify_labels = y_aspect
    elif len(np.unique(y_sentiment)) >= 2 and len(np.unique(y_aspect)) < 2:
        stratify_labels = y_sentiment
    elif len(np.unique(y_aspect)) >= 2 and len(np.unique(y_sentiment)) >= 2:
        # Ưu tiên stratify theo cột có nhiều lớp hơn, hoặc một cột cố định nếu số lớp bằng nhau
        if num_aspect_classes_val > num_sentiment_classes_val:
            stratify_labels = y_aspect
        elif num_sentiment_classes_val > num_aspect_classes_val:
            stratify_labels = y_sentiment
        else: # Nếu số lớp bằng nhau, có thể chọn một cách nhất quán, ví dụ y_aspect
            stratify_labels = y_aspect
    else: # Trường hợp một hoặc cả hai chỉ có 1 lớp, không cần stratify
        stratify_labels = None


    X_train, X_test, y_aspect_train, y_aspect_test, y_sentiment_train, y_sentiment_test = train_test_split(
        X, y_aspect, y_sentiment, test_size=0.2, random_state=42, stratify=stratify_labels
    )

    X_train_tensor = torch.LongTensor(X_train)
    y_aspect_train_tensor = torch.LongTensor(y_aspect_train)
    y_sentiment_train_tensor = torch.LongTensor(y_sentiment_train)
    X_test_tensor = torch.LongTensor(X_test)
    y_aspect_test_tensor = torch.LongTensor(y_aspect_test)
    y_sentiment_test_tensor = torch.LongTensor(y_sentiment_test)

    logging.info(f"X_train_tensor shape: {X_train_tensor.shape}")
    logging.info(f"y_aspect_train_tensor shape: {y_aspect_train_tensor.shape}")
    logging.info(f"y_sentiment_train_tensor shape: {y_sentiment_train_tensor.shape}")
    logging.info(f"X_test_tensor shape: {X_test_tensor.shape}")
    logging.info(f"y_aspect_test_tensor shape: {y_aspect_test_tensor.shape}")
    logging.info(f"y_sentiment_test_tensor shape: {y_sentiment_test_tensor.shape}")


    return (X_train_tensor, y_aspect_train_tensor, y_sentiment_train_tensor,
            X_test_tensor, y_aspect_test_tensor, y_sentiment_test_tensor,
            tokenizer, aspect_label_encoder, sentiment_label_encoder,
            num_aspect_classes_val, num_sentiment_classes_val)


class MultiOutputCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length,
                 num_aspect_classes, num_sentiment_classes,
                 cnn_filters, cnn_kernel_size, lstm_units, dropout_rate=0.5):
        super(MultiOutputCNNLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # padding_idx=0 nếu 0 là index của padding token
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, kernel_size=cnn_kernel_size)
        self.relu_conv = nn.ReLU()
        # Tính toán kích thước đầu ra của Conv1D để MaxPool1D hoạt động đúng
        # L_out = L_in - kernel_size + 1 (nếu stride=1, padding=0)
        conv_output_length = sequence_length - cnn_kernel_size + 1
        self.maxpool1d = nn.MaxPool1d(kernel_size=2 if conv_output_length >= 2 else 1) # Đảm bảo kernel_size không lớn hơn input

        # Tính toán input_size cho LSTM dựa trên output của MaxPool1D
        # L_out_maxpool = floor((L_in_maxpool - kernel_size_maxpool) / stride_maxpool + 1)
        # Nếu kernel_size=2, stride=2 (mặc định của MaxPool1d)
        # lstm_input_dim = cnn_filters (số channels từ Conv1D)
        # lstm_sequence_length = floor((conv_output_length - 2) / 2 + 1) # Kích thước sequence sau maxpool

        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, batch_first=True)

        self.fc_common = nn.Linear(lstm_units, 128) # Giả sử lấy output cuối cùng của LSTM
        self.relu_common = nn.ReLU()
        self.dropout_common = nn.Dropout(dropout_rate)

        self.fc_aspect = nn.Linear(128, num_aspect_classes)
        self.fc_sentiment = nn.Linear(128, num_sentiment_classes)

    def forward(self, x):
        embedded = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        # Conv1d expects (batch_size, channels, seq_len)
        embedded_permuted = embedded.permute(0, 2, 1) # (batch_size, embedding_dim, seq_len)

        conv_out = self.conv1d(embedded_permuted) # (batch_size, cnn_filters, new_seq_len_conv)
        conv_out = self.relu_conv(conv_out)
        if conv_out.size(2) < self.maxpool1d.kernel_size: # Xử lý trường hợp sequence quá ngắn cho maxpool
             # Bỏ qua maxpool hoặc xử lý khác nếu cần
             pooled_out = conv_out
        else:
            pooled_out = self.maxpool1d(conv_out) # (batch_size, cnn_filters, new_seq_len_pooled)

        # LSTM expects (batch_size, seq_len, input_size)
        lstm_input = pooled_out.permute(0, 2, 1) # (batch_size, new_seq_len_pooled, cnn_filters)

        # h_n: (num_layers * num_directions, batch, hidden_size)
        _, (h_n, _) = self.lstm(lstm_input)

        # Lấy hidden state cuối cùng của layer cuối cùng
        # Nếu LSTM là một chiều và một lớp, h_n có shape (1, batch, lstm_units)
        lstm_final_output = h_n.squeeze(0) # (batch, lstm_units)

        common_features = self.fc_common(lstm_final_output)
        common_features = self.relu_common(common_features)
        common_features = self.dropout_common(common_features)

        output_aspect = self.fc_aspect(common_features)
        output_sentiment = self.fc_sentiment(common_features)

        return output_aspect, output_sentiment


def train_model_epoch(model, dataloader, criterion_aspect, criterion_sentiment, optimizer, device, weight_aspect, weight_sentiment):
    model.train()
    total_loss = 0
    total_aspect_correct = 0
    total_sentiment_correct = 0
    total_samples = 0
    for inputs, labels_aspect, labels_sentiment in dataloader:
        inputs, labels_aspect, labels_sentiment = inputs.to(device), labels_aspect.to(device), labels_sentiment.to(device)
        optimizer.zero_grad()
        outputs_aspect, outputs_sentiment = model(inputs)
        loss_aspect = criterion_aspect(outputs_aspect, labels_aspect)
        loss_sentiment = criterion_sentiment(outputs_sentiment, labels_sentiment)
        loss = (weight_aspect * loss_aspect) + (weight_sentiment * loss_sentiment)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0) # Nhân với batch size hiện tại
        _, predicted_aspect = torch.max(outputs_aspect.data, 1)
        total_aspect_correct += (predicted_aspect == labels_aspect).sum().item()
        _, predicted_sentiment = torch.max(outputs_sentiment.data, 1)
        total_sentiment_correct += (predicted_sentiment == labels_sentiment).sum().item()
        total_samples += labels_aspect.size(0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy_aspect = total_aspect_correct / total_samples if total_samples > 0 else 0
    accuracy_sentiment = total_sentiment_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy_aspect, accuracy_sentiment


def evaluate_model_epoch(model, dataloader, criterion_aspect, criterion_sentiment, device, weight_aspect, weight_sentiment):
    model.eval()
    total_loss = 0
    total_aspect_correct = 0
    total_sentiment_correct = 0
    total_samples = 0
    all_aspect_preds = []
    all_aspect_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []

    with torch.no_grad():
        for inputs, labels_aspect, labels_sentiment in dataloader:
            inputs, labels_aspect, labels_sentiment = inputs.to(device), labels_aspect.to(device), labels_sentiment.to(device)
            outputs_aspect, outputs_sentiment = model(inputs)
            loss_aspect = criterion_aspect(outputs_aspect, labels_aspect)
            loss_sentiment = criterion_sentiment(outputs_sentiment, labels_sentiment)
            loss = (weight_aspect * loss_aspect) + (weight_sentiment * loss_sentiment)
            total_loss += loss.item() * inputs.size(0) # Nhân với batch size hiện tại

            _, predicted_aspect = torch.max(outputs_aspect.data, 1)
            total_aspect_correct += (predicted_aspect == labels_aspect).sum().item()
            all_aspect_preds.extend(predicted_aspect.cpu().numpy())
            all_aspect_labels.extend(labels_aspect.cpu().numpy())

            _, predicted_sentiment = torch.max(outputs_sentiment.data, 1)
            total_sentiment_correct += (predicted_sentiment == labels_sentiment).sum().item()
            all_sentiment_preds.extend(predicted_sentiment.cpu().numpy())
            all_sentiment_labels.extend(labels_sentiment.cpu().numpy())

            total_samples += labels_aspect.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy_aspect = total_aspect_correct / total_samples if total_samples > 0 else 0
    accuracy_sentiment = total_sentiment_correct / total_samples if total_samples > 0 else 0

    # Trả về thêm các dự đoán và nhãn để có thể tính report bên ngoài nếu cần
    return avg_loss, accuracy_aspect, accuracy_sentiment, all_aspect_preds, all_aspect_labels, all_sentiment_preds, all_sentiment_labels


def train_full_model(model, train_loader, val_loader,
                     criterion_aspect, criterion_sentiment, optimizer,
                     device, epochs, model_save_path,
                     weight_aspect, weight_sentiment):
    logging.info("Bắt đầu huấn luyện mô hình đa đầu ra...")
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc_aspect': [], 'val_acc_aspect': [],
        'train_acc_sentiment': [], 'val_acc_sentiment': []
    }
    for epoch in range(epochs):
        start_time = time.time()
        train_loss_epoch, train_acc_aspect_epoch, train_acc_sentiment_epoch = train_model_epoch(
            model, train_loader, criterion_aspect, criterion_sentiment, optimizer, device, weight_aspect, weight_sentiment
        )
        # Không cần lấy predictions ở đây nữa vì evaluate_and_save_report sẽ làm việc này với model tốt nhất
        val_loss_epoch, val_acc_aspect_epoch, val_acc_sentiment_epoch, _, _, _, _ = evaluate_model_epoch(
            model, val_loader, criterion_aspect, criterion_sentiment, device, weight_aspect, weight_sentiment
        )
        end_time = time.time()

        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['train_acc_aspect'].append(train_acc_aspect_epoch)
        history['val_acc_aspect'].append(val_acc_aspect_epoch)
        history['train_acc_sentiment'].append(train_acc_sentiment_epoch)
        history['val_acc_sentiment'].append(val_acc_sentiment_epoch)

        logging.info(
            f"Epoch {epoch+1}/{epochs} - Time: {end_time - start_time:.2f}s - "
            f"Train Loss: {train_loss_epoch:.4f} - "
            f"Train Acc Aspect: {train_acc_aspect_epoch:.4f}, Train Acc Sent: {train_acc_sentiment_epoch:.4f} - "
            f"Val Loss: {val_loss_epoch:.4f} - "
            f"Val Acc Aspect: {val_acc_aspect_epoch:.4f}, Val Acc Sent: {val_acc_sentiment_epoch:.4f}"
        )

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Lưu mô hình tốt nhất tại epoch {epoch+1} với Val Loss: {best_val_loss:.4f} vào {model_save_path}")

    logging.info("Hoàn thành huấn luyện.")
    return history


def plot_and_save_history(history, save_dir):
    if not history or not history['train_loss']: # Kiểm tra xem history có dữ liệu không
        logging.warning("Không có dữ liệu history để vẽ biểu đồ.")
        return

    os.makedirs(save_dir, exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"Biểu đồ Loss đã lưu tại: {loss_plot_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_acc_aspect'], 'bo-', label='Training Aspect Accuracy')
    plt.plot(epochs_range, history['val_acc_aspect'], 'ro-', label='Validation Aspect Accuracy')
    plt.title('Training and Validation Aspect Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    aspect_acc_plot_path = os.path.join(save_dir, "aspect_accuracy_plot.png")
    plt.savefig(aspect_acc_plot_path)
    plt.close()
    logging.info(f"Biểu đồ Aspect Accuracy đã lưu tại: {aspect_acc_plot_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_acc_sentiment'], 'bo-', label='Training Sentiment Accuracy')
    plt.plot(epochs_range, history['val_acc_sentiment'], 'ro-', label='Validation Sentiment Accuracy')
    plt.title('Training and Validation Sentiment Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    sentiment_acc_plot_path = os.path.join(save_dir, "sentiment_accuracy_plot.png")
    plt.savefig(sentiment_acc_plot_path)
    plt.close()
    logging.info(f"Biểu đồ Sentiment Accuracy đã lưu tại: {sentiment_acc_plot_path}")


def save_preprocessing_tools(tokenizer, aspect_encoder, sentiment_encoder):
    os.makedirs(os.path.dirname(TOKENIZER_SAVE_PATH), exist_ok=True)
    with open(TOKENIZER_SAVE_PATH, "wb") as f: pickle.dump(tokenizer, f)
    logging.info(f"Tokenizer đã lưu vào {TOKENIZER_SAVE_PATH}")

    os.makedirs(os.path.dirname(ASPECT_LABEL_ENCODER_SAVE_PATH), exist_ok=True)
    with open(ASPECT_LABEL_ENCODER_SAVE_PATH, "wb") as f: pickle.dump(aspect_encoder, f)
    logging.info(f"Aspect Label Encoder đã lưu vào {ASPECT_LABEL_ENCODER_SAVE_PATH}")

    os.makedirs(os.path.dirname(SENTIMENT_LABEL_ENCODER_SAVE_PATH), exist_ok=True)
    with open(SENTIMENT_LABEL_ENCODER_SAVE_PATH, "wb") as f: pickle.dump(sentiment_encoder, f)
    logging.info(f"Sentiment Label Encoder đã lưu vào {SENTIMENT_LABEL_ENCODER_SAVE_PATH}")


def generate_classification_report(model_path, dataloader, device,
                                   aspect_label_encoder, sentiment_label_encoder,
                                   num_aspect_classes, num_sentiment_classes,
                                   report_file_path):
    logging.info(f"Đang tải mô hình từ {model_path} để tạo báo cáo phân loại...")
    # Khởi tạo lại mô hình với cấu trúc chính xác
    model = MultiOutputCNNLSTM(
        vocab_size=MAX_WORDS, embedding_dim=EMBEDDING_DIM, sequence_length=MAX_SEQUENCE_LENGTH,
        num_aspect_classes=num_aspect_classes, num_sentiment_classes=num_sentiment_classes,
        cnn_filters=CNN_FILTERS, cnn_kernel_size=CNN_KERNEL_SIZE, lstm_units=LSTM_UNITS
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"Không tìm thấy tệp mô hình đã lưu tại: {model_path}. Bỏ qua việc tạo báo cáo.")
        return
    except Exception as e:
        logging.error(f"Lỗi khi tải mô hình: {e}. Bỏ qua việc tạo báo cáo.")
        return

    model.eval()
    all_aspect_preds = []
    all_aspect_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []

    logging.info("Đang thực hiện dự đoán trên tập dữ liệu kiểm tra để tạo báo cáo...")
    with torch.no_grad():
        for inputs, labels_aspect, labels_sentiment in dataloader:
            inputs = inputs.to(device)
            # Không cần chuyển labels sang device ở đây vì chúng dùng để so sánh với output trên CPU
            # labels_aspect = labels_aspect.to(device)
            # labels_sentiment = labels_sentiment.to(device)

            outputs_aspect, outputs_sentiment = model(inputs)

            _, predicted_aspect = torch.max(outputs_aspect.data, 1)
            all_aspect_preds.extend(predicted_aspect.cpu().numpy())
            all_aspect_labels.extend(labels_aspect.cpu().numpy()) # labels_aspect đã là tensor CPU từ DataLoader

            _, predicted_sentiment = torch.max(outputs_sentiment.data, 1)
            all_sentiment_preds.extend(predicted_sentiment.cpu().numpy())
            all_sentiment_labels.extend(labels_sentiment.cpu().numpy()) # labels_sentiment đã là tensor CPU

    if not all_aspect_labels or not all_sentiment_labels:
        logging.warning("Không có dữ liệu nhãn hoặc dự đoán nào được thu thập. Không thể tạo báo cáo.")
        return

    # Chuyển đổi sang numpy array
    y_true_aspect = np.array(all_aspect_labels)
    y_pred_aspect = np.array(all_aspect_preds)
    y_true_sentiment = np.array(all_sentiment_labels)
    y_pred_sentiment = np.array(all_sentiment_preds)

    # Tính toán Subset Accuracy (cả aspect và sentiment đều phải đúng)
    correct_predictions = 0
    for i in range(len(y_true_aspect)):
        if y_true_aspect[i] == y_pred_aspect[i] and y_true_sentiment[i] == y_pred_sentiment[i]:
            correct_predictions += 1
    subset_accuracy = correct_predictions / len(y_true_aspect) if len(y_true_aspect) > 0 else 0

    # Lấy tên các lớp từ encoders
    aspect_target_names = aspect_label_encoder.classes_.astype(str)
    sentiment_target_names = sentiment_label_encoder.classes_.astype(str)

    # Tạo báo cáo
    report_aspect = classification_report(y_true_aspect, y_pred_aspect, target_names=aspect_target_names, zero_division=0)
    accuracy_aspect = accuracy_score(y_true_aspect, y_pred_aspect)

    report_sentiment = classification_report(y_true_sentiment, y_pred_sentiment, target_names=sentiment_target_names, zero_division=0)
    accuracy_sentiment = accuracy_score(y_true_sentiment, y_pred_sentiment)

    # Lưu báo cáo vào tệp
    os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("Báo cáo Phân loại Đa Đầu ra (PyTorch CNN-LSTM)\n")
        f.write("======================================================\n\n")
        f.write(f"Đường dẫn mô hình đã đánh giá: {model_path}\n\n")

        f.write(f"Subset Accuracy (Cả Khía cạnh và Cảm xúc đều đúng): {subset_accuracy:.4f}\n\n")

        f.write("--- Báo cáo cho Khía cạnh (Aspect) ---\n")
        f.write(f"Accuracy Khía cạnh: {accuracy_aspect:.4f}\n")
        f.write(report_aspect)
        f.write("\n\n")

        f.write("--- Báo cáo cho Cảm xúc (Sentiment) ---\n")
        f.write(f"Accuracy Cảm xúc: {accuracy_sentiment:.4f}\n")
        f.write(report_sentiment)
        f.write("\n\n")

        f.write("Giải thích các chỉ số:\n")
        f.write("- Precision (Độ chính xác): Trong tất cả các mẫu được dự đoán là thuộc một lớp, bao nhiêu mẫu thực sự thuộc lớp đó.\n")
        f.write("  Công thức: TP / (TP + FP)\n")
        f.write("- Recall (Độ phủ / Độ nhạy): Trong tất cả các mẫu thực sự thuộc một lớp, bao nhiêu mẫu được dự đoán chính xác là thuộc lớp đó.\n")
        f.write("  Công thức: TP / (TP + FN)\n")
        f.write("- F1-score: Trung bình điều hòa của Precision và Recall. Là một chỉ số cân bằng giữa Precision và Recall.\n")
        f.write("  Công thức: 2 * (Precision * Recall) / (Precision + Recall)\n")
        f.write("- Support: Số lượng mẫu thực tế của lớp đó trong tập dữ liệu.\n")
        f.write("- Accuracy (Tổng thể cho một tác vụ): Tỷ lệ các mẫu được phân loại đúng cho tác vụ đó.\n")
        f.write("- Macro Avg: Tính trung bình các chỉ số (Precision, Recall, F1) trên tất cả các lớp mà không tính đến trọng số (số lượng support của mỗi lớp).\n")
        f.write("- Weighted Avg: Tính trung bình các chỉ số có trọng số theo support của mỗi lớp. Hữu ích khi các lớp mất cân bằng.\n")

    logging.info(f"Báo cáo phân loại chi tiết đã được lưu vào: {report_file_path}")
    print(f"\nBáo cáo chi tiết cho Khía cạnh:\nAccuracy: {accuracy_aspect:.4f}\n{report_aspect}")
    print(f"\nBáo cáo chi tiết cho Cảm xúc:\nAccuracy: {accuracy_sentiment:.4f}\n{report_sentiment}")
    print(f"\nSubset Accuracy (Cả Khía cạnh và Cảm xúc đều đúng): {subset_accuracy:.4f}")
    print(f"Báo cáo đầy đủ đã được lưu tại: {report_file_path}")


def main():
    logging.info("Bắt đầu Pipeline Phân loại Văn bản Tiếng Việt (PyTorch CNN-LSTM - Đa Đầu Ra)...")

    try:
        (X_train_tensor, y_aspect_train_tensor, y_sentiment_train_tensor,
         X_test_tensor, y_aspect_test_tensor, y_sentiment_test_tensor,
         tokenizer, aspect_label_encoder, sentiment_label_encoder,
         num_aspect_val, num_sentiment_val) = load_and_preprocess_data(
            DATA_PATH, TEXT_COLUMN, ASPECT_COLUMN, SENTIMENT_COLUMN, MAX_WORDS, MAX_SEQUENCE_LENGTH
        )
    except ValueError as e:
        logging.error(f"Lỗi trong quá trình tải và tiền xử lý dữ liệu: {e}")
        return # Kết thúc nếu không tải được dữ liệu
    except Exception as e:
        logging.error(f"Lỗi không xác định trong load_and_preprocess_data: {e}")
        return


    global NUM_ASPECT_CLASSES, NUM_SENTIMENT_CLASSES # Đảm bảo các biến global được cập nhật
    if NUM_ASPECT_CLASSES is None: NUM_ASPECT_CLASSES = num_aspect_val
    if NUM_SENTIMENT_CLASSES is None: NUM_SENTIMENT_CLASSES = num_sentiment_val

    if NUM_ASPECT_CLASSES is None or NUM_ASPECT_CLASSES <= 0 or \
       NUM_SENTIMENT_CLASSES is None or NUM_SENTIMENT_CLASSES <= 0:
        logging.error("Không xác định được số lượng lớp cho khía cạnh hoặc cảm xúc sau khi tải dữ liệu. Kết thúc.")
        return

    logging.info(f"Số lớp Khía cạnh cuối cùng được sử dụng: {NUM_ASPECT_CLASSES}")
    logging.info(f"Số lớp Cảm xúc cuối cùng được sử dụng: {NUM_SENTIMENT_CLASSES}")


    train_dataset = TensorDataset(X_train_tensor, y_aspect_train_tensor, y_sentiment_train_tensor)
    # Sử dụng X_test_tensor, y_aspect_test_tensor, y_sentiment_test_tensor cho val_dataset
    val_dataset = TensorDataset(X_test_tensor, y_aspect_test_tensor, y_sentiment_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # Không shuffle val_loader

    model = MultiOutputCNNLSTM(
        vocab_size=MAX_WORDS, # Phải là kích thước từ điển thực tế + 1 nếu dùng oov_token và index 0 cho padding
        embedding_dim=EMBEDDING_DIM,
        sequence_length=MAX_SEQUENCE_LENGTH,
        num_aspect_classes=NUM_ASPECT_CLASSES,
        num_sentiment_classes=NUM_SENTIMENT_CLASSES,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_size=CNN_KERNEL_SIZE,
        lstm_units=LSTM_UNITS
    ).to(DEVICE)
    logging.info(model)

    criterion_aspect = nn.CrossEntropyLoss()
    criterion_sentiment = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = train_full_model(model, train_loader, val_loader,
                               criterion_aspect, criterion_sentiment, optimizer, DEVICE, EPOCHS, MODEL_SAVE_PATH,
                               LOSS_WEIGHT_ASPECT, LOSS_WEIGHT_SENTIMENT)

    save_preprocessing_tools(tokenizer, aspect_label_encoder, sentiment_label_encoder)

    if history:
        plot_and_save_history(history, PLOTS_SAVE_DIR)

    # Tạo và lưu báo cáo phân loại sau khi huấn luyện xong
    # Đảm bảo NUM_ASPECT_CLASSES và NUM_SENTIMENT_CLASSES đã được cập nhật chính xác
    if os.path.exists(MODEL_SAVE_PATH) and NUM_ASPECT_CLASSES > 0 and NUM_SENTIMENT_CLASSES > 0:
         generate_classification_report(
            model_path=MODEL_SAVE_PATH,
            dataloader=val_loader, # Sử dụng val_loader chứa dữ liệu test
            device=DEVICE,
            aspect_label_encoder=aspect_label_encoder,
            sentiment_label_encoder=sentiment_label_encoder,
            num_aspect_classes=NUM_ASPECT_CLASSES, # Truyền số lớp đã xác định
            num_sentiment_classes=NUM_SENTIMENT_CLASSES, # Truyền số lớp đã xác định
            report_file_path=CLASSIFICATION_REPORT_PATH
        )
    else:
        logging.warning("Không thể tạo báo cáo phân loại do mô hình chưa được lưu hoặc số lớp không hợp lệ.")


    logging.info("Pipeline hoàn thành thành công.")

if __name__ == "__main__":
    main()