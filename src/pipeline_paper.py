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

# --- Cấu hình --- (Giữ nguyên các cấu hình đã có)
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
    
    # Sử dụng tên cột từ cấu hình ban đầu
    current_text_column = text_column_config
    current_aspect_column = aspect_column_config
    current_sentiment_column = sentiment_column_config

    try:
        df = pd.read_csv(data_path)
        logging.info(f"Đã tải thành công dữ liệu từ: {data_path}")
        # Kiểm tra sự tồn tại của các cột được cấu hình trong dữ liệu thật
        if current_text_column not in df.columns:
            raise ValueError(f"Cột văn bản '{current_text_column}' không tìm thấy trong tệp {data_path}.")
        if current_aspect_column not in df.columns:
            raise ValueError(f"Cột khía cạnh '{current_aspect_column}' không tìm thấy trong tệp {data_path}.")
        if current_sentiment_column not in df.columns:
            raise ValueError(f"Cột cảm xúc '{current_sentiment_column}' không tìm thấy trong tệp {data_path}.")

    except FileNotFoundError:
        logging.error(f"Không tìm thấy dataset tại {data_path}. Vui lòng cung cấp đường dẫn hợp lệ.")
        logging.warning("Sử dụng dữ liệu giả để minh họa vì không tìm thấy dataset.")
        # Tạo dữ liệu giả cho multi-output
        dummy_data_text_col = "text_dummy"
        dummy_data_aspect_col = "aspect_dummy"
        dummy_data_sentiment_col = "sentiment_dummy"
        
        data_dummy = {
            dummy_data_text_col: [ # Sử dụng tên cột rõ ràng cho dữ liệu giả
                "Sản phẩm rất tốt, giao hàng nhanh.", "Giá hơi cao nhưng chất lượng ổn.",
                "Thất vọng về dịch vụ chăm sóc khách hàng.", "Pin dùng được lâu, thiết kế đẹp.",
                "Giao hàng chậm quá.", "Màu sắc không giống hình.",
                "Rất hài lòng với sản phẩm này.", "Nhân viên tư vấn nhiệt tình.",
                "Chất liệu vải không như mong đợi.", "Sẽ ủng hộ shop lần sau."
            ] * 10,
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

        # Cập nhật tên cột hiện tại để sử dụng các cột của dữ liệu giả
        logging.warning(f"Sử dụng cột '{dummy_data_text_col}' của dữ liệu giả thay cho '{current_text_column}' (cấu hình).")
        current_text_column = dummy_data_text_col
        logging.warning(f"Sử dụng cột '{dummy_data_aspect_col}' của dữ liệu giả thay cho '{current_aspect_column}' (cấu hình).")
        current_aspect_column = dummy_data_aspect_col
        logging.warning(f"Sử dụng cột '{dummy_data_sentiment_col}' của dữ liệu giả thay cho '{current_sentiment_column}' (cấu hình).")
        current_sentiment_column = dummy_data_sentiment_col

    logging.info(f"Số lượng dòng ban đầu: {len(df)}")
    # Loại bỏ NaN dựa trên các tên cột hiện tại (có thể là từ config hoặc từ dummy data)
    df.dropna(subset=[current_text_column, current_aspect_column, current_sentiment_column], inplace=True)
    logging.info(f"Số lượng dòng sau khi loại bỏ NaN ở các cột ('{current_text_column}', '{current_aspect_column}', '{current_sentiment_column}'): {len(df)}")

    if df.empty:
        raise ValueError("Dataset rỗng sau khi loại bỏ NaN. Vui lòng kiểm tra dữ liệu đầu vào.")

    texts = df[current_text_column].astype(str).values
    aspect_labels_raw = df[current_aspect_column].astype(str).values
    sentiment_labels_raw = df[current_sentiment_column].astype(str).values

    # Mã hóa nhãn Khía cạnh
    logging.info("Đang mã hóa nhãn Khía cạnh...")
    aspect_label_encoder = LabelEncoder()
    y_aspect = aspect_label_encoder.fit_transform(aspect_labels_raw)
    num_aspect_classes_val = len(aspect_label_encoder.classes_)
    logging.info(f"Số lượng lớp Khía cạnh: {num_aspect_classes_val}")
    logging.info(f"Các lớp Khía cạnh: {aspect_label_encoder.classes_}")
    global NUM_ASPECT_CLASSES
    NUM_ASPECT_CLASSES = num_aspect_classes_val

    # Mã hóa nhãn Cảm xúc
    logging.info("Đang mã hóa nhãn Cảm xúc...")
    sentiment_label_encoder = LabelEncoder()
    y_sentiment = sentiment_label_encoder.fit_transform(sentiment_labels_raw)
    num_sentiment_classes_val = len(sentiment_label_encoder.classes_)
    logging.info(f"Số lượng lớp Cảm xúc: {num_sentiment_classes_val}")
    logging.info(f"Các lớp Cảm xúc: {sentiment_label_encoder.classes_}")
    global NUM_SENTIMENT_CLASSES
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
        # Nếu cả hai đều có nhiều hơn 1 lớp, có thể ưu tiên stratify theo cái có nhiều lớp hơn
        # hoặc tạo nhãn kết hợp để stratify nếu thực sự cần thiết.
        # Tạm thời stratify theo aspect nếu nó có vẻ "đa dạng" hơn.
        stratify_labels = y_aspect if num_aspect_classes_val >= num_sentiment_classes_val else y_sentiment
    # Nếu stratify_labels vẫn là None, train_test_split sẽ không stratify.

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

    return (X_train_tensor, y_aspect_train_tensor, y_sentiment_train_tensor,
            X_test_tensor, y_aspect_test_tensor, y_sentiment_test_tensor,
            tokenizer, aspect_label_encoder, sentiment_label_encoder,
            num_aspect_classes_val, num_sentiment_classes_val)


class MultiOutputCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length,
                 num_aspect_classes, num_sentiment_classes,
                 cnn_filters, cnn_kernel_size, lstm_units, dropout_rate=0.5):
        super(MultiOutputCNNLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, kernel_size=cnn_kernel_size)
        self.relu_conv = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, batch_first=True)
        
        # Lớp trung gian chung trước khi tách đầu ra
        self.fc_common = nn.Linear(lstm_units, 128)
        self.relu_common = nn.ReLU()
        self.dropout_common = nn.Dropout(dropout_rate)
        
        # Đầu ra cho Khía cạnh
        self.fc_aspect = nn.Linear(128, num_aspect_classes)
        # Đầu ra cho Cảm xúc
        self.fc_sentiment = nn.Linear(128, num_sentiment_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_permuted = embedded.permute(0, 2, 1)
        conv_out = self.conv1d(embedded_permuted)
        conv_out = self.relu_conv(conv_out)
        pooled_out = self.maxpool1d(conv_out)
        lstm_input = pooled_out.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(lstm_input)
        lstm_final_output = h_n.squeeze(0)

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
        inputs = inputs.to(device)
        labels_aspect = labels_aspect.to(device)
        labels_sentiment = labels_sentiment.to(device)

        optimizer.zero_grad()
        outputs_aspect, outputs_sentiment = model(inputs)
        
        loss_aspect = criterion_aspect(outputs_aspect, labels_aspect)
        loss_sentiment = criterion_sentiment(outputs_sentiment, labels_sentiment)
        
        # Kết hợp loss, có thể có trọng số
        loss = (weight_aspect * loss_aspect) + (weight_sentiment * loss_sentiment)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        
        _, predicted_aspect = torch.max(outputs_aspect.data, 1)
        total_aspect_correct += (predicted_aspect == labels_aspect).sum().item()
        
        _, predicted_sentiment = torch.max(outputs_sentiment.data, 1)
        total_sentiment_correct += (predicted_sentiment == labels_sentiment).sum().item()
        
        total_samples += labels_aspect.size(0) # Hoặc labels_sentiment.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy_aspect = total_aspect_correct / total_samples
    accuracy_sentiment = total_sentiment_correct / total_samples
    return avg_loss, accuracy_aspect, accuracy_sentiment

def evaluate_model_epoch(model, dataloader, criterion_aspect, criterion_sentiment, device, weight_aspect, weight_sentiment):
    model.eval()
    total_loss = 0
    total_aspect_correct = 0
    total_sentiment_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels_aspect, labels_sentiment in dataloader:
            inputs = inputs.to(device)
            labels_aspect = labels_aspect.to(device)
            labels_sentiment = labels_sentiment.to(device)

            outputs_aspect, outputs_sentiment = model(inputs)
            loss_aspect = criterion_aspect(outputs_aspect, labels_aspect)
            loss_sentiment = criterion_sentiment(outputs_sentiment, labels_sentiment)
            loss = (weight_aspect * loss_aspect) + (weight_sentiment * loss_sentiment)
            
            total_loss += loss.item() * inputs.size(0)

            _, predicted_aspect = torch.max(outputs_aspect.data, 1)
            total_aspect_correct += (predicted_aspect == labels_aspect).sum().item()
            
            _, predicted_sentiment = torch.max(outputs_sentiment.data, 1)
            total_sentiment_correct += (predicted_sentiment == labels_sentiment).sum().item()
            
            total_samples += labels_aspect.size(0)

    avg_loss = total_loss / total_samples
    accuracy_aspect = total_aspect_correct / total_samples
    accuracy_sentiment = total_sentiment_correct / total_samples
    return avg_loss, accuracy_aspect, accuracy_sentiment


def train_full_model(model, train_loader, val_loader, 
                     criterion_aspect, criterion_sentiment, optimizer, 
                     device, epochs, model_save_path,
                     weight_aspect, weight_sentiment):
    logging.info("Bắt đầu huấn luyện mô hình đa đầu ra...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc_aspect, train_acc_sentiment = train_model_epoch(
            model, train_loader, criterion_aspect, criterion_sentiment, optimizer, device, weight_aspect, weight_sentiment
        )
        val_loss, val_acc_aspect, val_acc_sentiment = evaluate_model_epoch(
            model, val_loader, criterion_aspect, criterion_sentiment, device, weight_aspect, weight_sentiment
        )
        end_time = time.time()

        logging.info(
            f"Epoch {epoch+1}/{epochs} - Time: {end_time - start_time:.2f}s - "
            f"Train Loss: {train_loss:.4f} - "
            f"Train Acc Aspect: {train_acc_aspect:.4f}, Train Acc Sent: {train_acc_sentiment:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Acc Aspect: {val_acc_aspect:.4f}, Val Acc Sent: {val_acc_sentiment:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Lưu mô hình tốt nhất tại epoch {epoch+1} vào {model_save_path}")
    logging.info("Hoàn thành huấn luyện.")


def save_preprocessing_tools(tokenizer, aspect_encoder, sentiment_encoder):
    os.makedirs(os.path.dirname(TOKENIZER_SAVE_PATH), exist_ok=True)
    with open(TOKENIZER_SAVE_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    logging.info(f"Tokenizer đã lưu vào {TOKENIZER_SAVE_PATH}")

    os.makedirs(os.path.dirname(ASPECT_LABEL_ENCODER_SAVE_PATH), exist_ok=True)
    with open(ASPECT_LABEL_ENCODER_SAVE_PATH, "wb") as f:
        pickle.dump(aspect_encoder, f)
    logging.info(f"Aspect Label Encoder đã lưu vào {ASPECT_LABEL_ENCODER_SAVE_PATH}")

    os.makedirs(os.path.dirname(SENTIMENT_LABEL_ENCODER_SAVE_PATH), exist_ok=True)
    with open(SENTIMENT_LABEL_ENCODER_SAVE_PATH, "wb") as f:
        pickle.dump(sentiment_encoder, f)
    logging.info(f"Sentiment Label Encoder đã lưu vào {SENTIMENT_LABEL_ENCODER_SAVE_PATH}")


def main():
    logging.info("Bắt đầu Pipeline Phân loại Văn bản Tiếng Việt (PyTorch CNN-LSTM - Đa Đầu Ra)...")

    (X_train_tensor, y_aspect_train_tensor, y_sentiment_train_tensor,
     X_test_tensor, y_aspect_test_tensor, y_sentiment_test_tensor,
     tokenizer, aspect_label_encoder, sentiment_label_encoder,
     num_aspect_val, num_sentiment_val) = load_and_preprocess_data(
        DATA_PATH, TEXT_COLUMN, ASPECT_COLUMN, SENTIMENT_COLUMN, MAX_WORDS, MAX_SEQUENCE_LENGTH
    )

    global NUM_ASPECT_CLASSES, NUM_SENTIMENT_CLASSES
    if NUM_ASPECT_CLASSES is None: NUM_ASPECT_CLASSES = num_aspect_val
    if NUM_SENTIMENT_CLASSES is None: NUM_SENTIMENT_CLASSES = num_sentiment_val

    if NUM_ASPECT_CLASSES is None or NUM_ASPECT_CLASSES <= 0 or \
       NUM_SENTIMENT_CLASSES is None or NUM_SENTIMENT_CLASSES <= 0:
        logging.error("Không xác định được số lượng lớp cho khía cạnh hoặc cảm xúc. Kết thúc.")
        return

    train_dataset = TensorDataset(X_train_tensor, y_aspect_train_tensor, y_sentiment_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_aspect_test_tensor, y_sentiment_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiOutputCNNLSTM(
        vocab_size=MAX_WORDS,
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

    train_full_model(model, train_loader, val_loader, 
                     criterion_aspect, criterion_sentiment, optimizer, DEVICE, EPOCHS, MODEL_SAVE_PATH,
                     LOSS_WEIGHT_ASPECT, LOSS_WEIGHT_SENTIMENT)
    
    save_preprocessing_tools(tokenizer, aspect_label_encoder, sentiment_label_encoder)

    logging.info("Pipeline hoàn thành thành công.")


if __name__ == "__main__":
    main()