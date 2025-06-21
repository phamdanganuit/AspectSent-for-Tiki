import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import pickle
import os
import sys

# Thêm đường dẫn thư mục gốc của dự án vào Python Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import tuyệt đối
MAX_LEN =150 
WORD_EMBEDDING_DIM = 300
from src.data_preprocessing.normalize_text import normalize_text
from src.data_preprocessing.tokenize_text import tokenize_text

def _train_and_evaluate_model(
    X_train, y_train_aspect, y_train_sentiment,
    X_val, y_val_aspect, y_val_sentiment,
    X_test, y_test_aspect, y_test_sentiment, y_test_sentiment_le,
    vocab_size, num_aspects, num_sentiments,
    aspect_labels, sentiment_labels, model_type
):
    """
    Hàm nội bộ để xây dựng, huấn luyện và đánh giá một mô hình duy nhất.
    """
    input_layer = Input(shape=(MAX_LEN,))
    embedding_layer = Embedding(vocab_size, WORD_EMBEDDING_DIM, input_length=MAX_LEN)(input_layer)

    if model_type == 'cnn_multi':
        x = Conv1D(128, 5, activation='relu')(embedding_layer)
        x = MaxPooling1D(5)(x)
        x = Flatten()(x)
        model_name_str = "CNN"
    elif model_type == 'lstm_multi':
        x = Bidirectional(LSTM(128, return_sequences=False))(embedding_layer)
        model_name_str = "LSTM"
    else:
        raise ValueError(f"Loại mô hình không được hỗ trợ: {model_type}")

    aspect_output = Dense(num_aspects, activation='sigmoid', name='aspect')(x)
    sentiment_output = Dense(num_sentiments, activation='softmax', name='sentiment')(x)

    model = Model(inputs=input_layer, outputs=[aspect_output, sentiment_output])
    model.compile(optimizer='adam',
                  loss={'aspect': 'binary_crossentropy', 'sentiment': 'categorical_crossentropy'},
                  metrics={'aspect': 'accuracy', 'sentiment': 'accuracy'})

    print(f"--- Bắt đầu huấn luyện mô hình {model_name_str} ---")
    model.summary()

    model.fit(
        X_train, {'aspect': y_train_aspect, 'sentiment': y_train_sentiment},
        epochs=10,
        batch_size=32,
        validation_data=(X_val, {'aspect': y_val_aspect, 'sentiment': y_val_sentiment}),
        verbose=1
    )
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_filename = f'{model_type}_model.h5'
    model_path = os.path.join(results_dir, model_filename)
    model.save(model_path)
    print(f"Mô hình đã được lưu tại: {model_path}")

    print(f"\n--- Đánh giá mô hình {model_name_str} trên tập Test ---")
    loss, aspect_loss, sentiment_loss, aspect_acc, sentiment_acc = model.evaluate(
        X_test, {'aspect': y_test_aspect, 'sentiment': y_test_sentiment}, verbose=0)
    print(f"Độ chính xác trên tập test - Aspect: {aspect_acc*100:.2f}%, Sentiment: {sentiment_acc*100:.2f}%")

    predictions = model.predict(X_test)
    y_pred_aspect = (predictions[0] > 0.5).astype(int)
    y_pred_sentiment_le = np.argmax(predictions[1], axis=1)

    report_aspect = classification_report(y_test_aspect, y_pred_aspect, target_names=aspect_labels, zero_division=0)
    
    # Dòng `report_sentiment` gây ra lỗi đã được sửa bằng cách đảm bảo sentiment_labels là list of strings
    report_sentiment = classification_report(y_test_sentiment_le, y_pred_sentiment_le, target_names=sentiment_labels, zero_division=0)
    
    report_filename = f'{model_type}_training_report.txt'
    report_path = os.path.join(results_dir, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"--- Báo cáo huấn luyện và đánh giá mô hình {model_name_str} ---\n\n")
        f.write("Aspect Classification Report:\n")
        f.write(report_aspect)
        f.write("\n\nSentiment Classification Report:\n")
        f.write(report_sentiment)
    print(f"Báo cáo huấn luyện đã được lưu tại: {report_path}")
    print(f"--- Hoàn thành xử lý cho mô hình {model_name_str} ---\n")


def run_pipeline_paper(train_path, test_val_path, val_split_size=0.5, random_state=42):
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("--- Bước 1: Tải dữ liệu từ các file nguồn ---")
    df_train = pd.read_csv(train_path)
    df_test_val = pd.read_csv(test_val_path)
    
    # Cập nhật để sử dụng cột 'reviews'
    df_train.dropna(subset=['reviews', 'aspect', 'sentiment'], inplace=True)
    df_test_val.dropna(subset=['tokenized_text', 'aspect', 'sentiment'], inplace=True)

    df_train['normalized_text'] = df_train['reviews'].apply(normalize_text)
    df_test_val['normalized_text'] = df_test_val['tokenized_text'].apply(normalize_text)

    print("--- Bước 2: Tạo và huấn luyện Tokenizer ---")
    combined_corpus = pd.concat([df_train['normalized_text'], df_test_val['normalized_text']], ignore_index=True)
    corpus_tokenized = [tokenize_text(text) for text in combined_corpus]
    
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(corpus_tokenized)
    vocab_size = len(tokenizer.word_index) + 1
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer đã được lưu vào file 'tokenizer.pickle'")

    print("--- Bước 3: Chuyển văn bản thành chuỗi số ---")
    X_train = pad_sequences(tokenizer.texts_to_sequences(df_train['normalized_text']), maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_val = pad_sequences(tokenizer.texts_to_sequences(df_test_val['normalized_text']), maxlen=MAX_LEN, padding='post', truncating='post')

    print("--- Bước 4: Xử lý nhãn (Aspect và Sentiment) ---")
    df_train_aspect_dummies = df_train['aspect'].str.get_dummies(sep=', ')
    aspect_labels = df_train_aspect_dummies.columns.tolist()
    y_train_aspect = df_train_aspect_dummies.values

    df_test_val_aspect_dummies = df_test_val['aspect'].str.get_dummies(sep=', ').reindex(columns=aspect_labels, fill_value=0)
    y_test_val_aspect = df_test_val_aspect_dummies.values
    
    with open(os.path.join(results_dir, 'aspect_labels.pickle'), 'wb') as handle:
        pickle.dump(aspect_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Aspect labels đã được lưu vào '{os.path.join(results_dir, 'aspect_labels.pickle')}'")

    sentiment_le = LabelEncoder()
    y_train_sentiment_le = sentiment_le.fit_transform(df_train['sentiment'])
    y_test_val_sentiment_le = sentiment_le.transform(df_test_val['sentiment'])
    num_sentiments = len(sentiment_le.classes_)

    y_train_sentiment = to_categorical(y_train_sentiment_le, num_classes=num_sentiments)
    y_test_val_sentiment = to_categorical(y_test_val_sentiment_le, num_classes=num_sentiments)

    # === SỬA LỖI TYPEERROR TẠI ĐÂY ===
    # Chuyển các lớp của sentiment (vốn là số) thành chuỗi ký tự
    sentiment_class_labels = [str(label) for label in sentiment_le.classes_]

    sentiment_mapping = {i: label for i, label in enumerate(sentiment_class_labels)}
    with open('sentiment_mapping.pickle', 'wb') as handle:
        pickle.dump(sentiment_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Sentiment mapping đã được lưu vào 'sentiment_mapping.pickle'")

    print("--- Bước 5: Chia tập dữ liệu thành Validation và Test ---")
    X_val, X_test, y_val_aspect, y_test_aspect, y_val_sentiment, y_test_sentiment, y_val_sentiment_le, y_test_sentiment_le = train_test_split(
        X_test_val, y_test_val_aspect, y_test_val_sentiment, y_test_val_sentiment_le,
        test_size=val_split_size, random_state=random_state
    )
    
    np.save(os.path.join(results_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(results_dir, 'y_test_aspect.npy'), y_test_aspect)
    np.save(os.path.join(results_dir, 'y_test_sentiment.npy'), y_test_sentiment)
    np.save(os.path.join(results_dir, 'y_test_sentiment_le.npy'), y_test_sentiment_le)
    print(f"Dữ liệu Test cuối cùng đã được lưu vào thư mục '{results_dir}/'")

    print("\n--- Bước 6: Huấn luyện và Đánh giá các mô hình ---")
    
    _train_and_evaluate_model(
        X_train, y_train_aspect, y_train_sentiment,
        X_val, y_val_aspect, y_val_sentiment,
        X_test, y_test_aspect, y_test_sentiment, y_test_sentiment_le,
        vocab_size, len(aspect_labels), num_sentiments,
        aspect_labels, sentiment_class_labels, 'cnn_multi'
    )

    _train_and_evaluate_model(
        X_train, y_train_aspect, y_train_sentiment,
        X_val, y_val_aspect, y_val_sentiment,
        X_test, y_test_aspect, y_test_sentiment, y_test_sentiment_le,
        vocab_size, len(aspect_labels), num_sentiments,
        aspect_labels, sentiment_class_labels, 'lstm_multi'
    )
    
    print("\n--- Pipeline huấn luyện đã hoàn tất ---")

if __name__ == '__main__':
    TRAIN_PATH = 'data/Silver/normalized_reviews.csv'
    TEST_VAL_PATH = 'data/Gold/finetuning_metadata.csv'
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_VAL_PATH):
        print(f"Lỗi: Không tìm thấy file '{TRAIN_PATH}' hoặc '{TEST_VAL_PATH}'")
        print("Vui lòng đảm bảo 2 file dữ liệu này nằm ở thư mục gốc của dự án.")
    else:
        run_pipeline_paper(train_path=TRAIN_PATH, test_val_path=TEST_VAL_PATH)