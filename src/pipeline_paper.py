from pathlib import Path
import os
import pandas as pd
import numpy as np
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout

# Định nghĩa đường dẫn
BASE_DIR = Path(__file__).parent.parent  # Thư mục gốc của dự án: AspectSent-for-Tiki
STOPWORDS_PATH = BASE_DIR / "src" / "vietnamese_stopwords.txt"  # Đường dẫn tới file stopwords
DATA_PATH = BASE_DIR / "data" / "Gold" / "finetuning_metadata.csv"  # Đường dẫn tới file dữ liệu
OUTPUT_DIR = BASE_DIR / "results"  # Thư mục lưu kết quả
OUTPUT_PATH = OUTPUT_DIR / "classification_results.txt"  # Đường dẫn file kết quả

# Kiểm tra file tồn tại
if not STOPWORDS_PATH.exists():
    raise FileNotFoundError(f"Stopwords file not found at {STOPWORDS_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

# Hàm tải danh sách stop-words
def load_stopwords(filepath):
    """Đọc danh sách stop-words từ file và trả về tập hợp các từ."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(f.read().splitlines())

# Hàm tiền xử lý văn bản
def preprocess_text(text, stopwords):
    """Tiền xử lý văn bản: tokenize và loại bỏ stop-words."""
    tokens = word_tokenize(text, format="text")
    return ' '.join(word for word in tokens if word.lower() not in stopwords)

# Hàm xây dựng mô hình LSTM
def build_lstm_model(input_dim, num_classes):
    """Xây dựng và biên dịch mô hình LSTM."""
    model = Sequential([
        LSTM(128, input_shape=(input_dim, 1), return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Hàm xây dựng mô hình CNN
def build_cnn_model(input_dim, num_classes):
    """Xây dựng và biên dịch mô hình CNN."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Hàm chính của pipeline
def main():
    # Tải dữ liệu
    df = pd.read_csv(DATA_PATH)
    
    # Lấy cột text và sentiment
    texts = df['tokenized_text'].astype(str)
    labels = df['sentiment']
    
    # Mã hóa nhãn (1, 2, 3, 4, 5 thành 0, 1, 2, 3, 4)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # Tải stop-words
    stopwords = load_stopwords(STOPWORDS_PATH)
    
    # Tiền xử lý văn bản
    processed_texts = [preprocess_text(text, stopwords) for text in texts]
    
    # Trích xuất đặc trưng với TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(processed_texts).toarray()
    
    # Giảm chiều với SVD
    svd = TruncatedSVD(n_components=50)
    X_svd = svd.fit_transform(X_tfidf)
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X_svd, labels_encoded, test_size=0.2, random_state=42
    )
    
    # Định dạng dữ liệu cho LSTM và CNN
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Tính trọng số lớp để xử lý mất cân bằng dữ liệu
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    # Huấn luyện và đánh giá mô hình machine learning
    ml_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(C=1.0, class_weight='balanced')
    }
    
    results = {}
    
    for name, model in ml_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:\n", 
              classification_report(y_test, y_pred, target_names=[str(i) for i in label_encoder.classes_]))
    
    # Huấn luyện và đánh giá mô hình LSTM
    lstm_model = build_lstm_model(X_train.shape[1], num_classes)
    lstm_model.fit(
        X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0, class_weight=class_weights_dict
    )
    lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)
    results['LSTM'] = lstm_accuracy
    print(f"LSTM Accuracy: {lstm_accuracy:.4f}")
    
    # Dự đoán để tạo classification report cho LSTM
    y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
    print(f"LSTM Classification Report:\n", 
          classification_report(y_test, y_pred_lstm, target_names=[str(i) for i in label_encoder.classes_]))
    
    # Huấn luyện và đánh giá mô hình CNN
    cnn_model = build_cnn_model(X_train.shape[1], num_classes)
    cnn_model.fit(
        X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0, class_weight=class_weights_dict
    )
    cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_lstm, y_test, verbose=0)
    results['CNN'] = cnn_accuracy
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    
    # Dự đoán để tạo classification report cho CNN
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_lstm), axis=1)
    print(f"CNN Classification Report:\n", 
          classification_report(y_test, y_pred_cnn, target_names=[str(i) for i in label_encoder.classes_]))
    
    # Tạo thư mục results nếu chưa tồn tại
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Lưu kết quả
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for model, accuracy in results.items():
            f.write(f"{model}: {accuracy:.4f}\n")

if __name__ == "__main__":
    main()