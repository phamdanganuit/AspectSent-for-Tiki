import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
# Định nghĩa các hằng số trực tiếp trong mã nguồn
MAX_LEN = 100  # Độ dài tối đa của chuỗi
WORD_EMBEDDING_DIM = 300  # Kích thước của vector nhúng từ
from src.data_preprocessing.normalize_text import normalize_text
from src.data_preprocessing.tokenize_text import tokenize_text


def _train_and_evaluate_model(X_train, X_test, y_train_aspect, y_test_aspect,
                              y_train_sentiment, y_test_sentiment, y_test_sentiment_le,
                              vocab_size, num_aspects, num_sentiments,
                              aspect_labels, sentiment_labels, model_type):
    # (Nội dung hàm này không thay đổi)
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

    model.fit(X_train, {'aspect': y_train_aspect, 'sentiment': y_train_sentiment},
              epochs=40,
              batch_size=32,
              validation_split=0.1,
              verbose=1)
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_filename = f'{model_type}_model.h5'
    model_path = os.path.join(results_dir, model_filename)
    model.save(model_path)
    print(f"Mô hình đã được lưu tại: {model_path}")

    print(f"\n--- Đánh giá mô hình {model_name_str} ---")
    loss, aspect_loss, sentiment_loss, aspect_acc, sentiment_acc = model.evaluate(
        X_test, {'aspect': y_test_aspect, 'sentiment': y_test_sentiment}, verbose=0)
    print(f"Độ chính xác trên tập test - Aspect: {aspect_acc*100:.2f}%, Sentiment: {sentiment_acc*100:.2f}%")

    predictions = model.predict(X_test)
    y_pred_aspect = (predictions[0] > 0.5).astype(int)
    y_pred_sentiment_le = np.argmax(predictions[1], axis=1)

    report_aspect = classification_report(y_test_aspect, y_pred_aspect, target_names=aspect_labels, zero_division=0)
    report_sentiment = classification_report(y_test_sentiment_le, y_pred_sentiment_le, target_names=sentiment_labels, zero_division=0)
    
    report_filename = f'{model_type}_training_report.txt'
    report_path = os.path.join(results_dir, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"--- Báo cáo huấn luyện và đánh giá mô hình {model_name_str} ---\n\n")
        f.write(report_aspect)
        f.write("\n\n")
        f.write(report_sentiment)
    print(f"Báo cáo huấn luyện đã được lưu tại: {report_path}")
    print(f"--- Hoàn thành xử lý cho mô hình {model_name_str} ---\n")


def run_pipeline_paper(data_path, test_size=0.2, random_state=42):
    """
    Pipeline chính để tải, xử lý dữ liệu, huấn luyện mô hình và lưu tất cả các tạo tác.
    """
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("--- Bước 1: Tải và xử lý dữ liệu ---")
    df = pd.read_csv(data_path)
    
    # === SỬA LỖI KEYERROR TẠI ĐÂY ===
    
    df.dropna(subset=['tokenized_text', 'aspect', 'sentiment'], inplace=True)
    df['normalized_text'] = df['tokenized_text'].apply(normalize_text)

    print("--- Bước 2: Tokenize văn bản và lưu tokenizer ---")
    corpus = [tokenize_text(text) for text in df['normalized_text']]
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(corpus)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer đã được lưu vào file 'tokenizer.pickle'")

    print("--- Bước 3: Chuẩn bị và lưu các loại nhãn ---")
    df_aspect = df['aspect'].str.get_dummies(sep=', ')
    aspect_labels = df_aspect.columns.tolist()
    num_aspects = len(aspect_labels)
    y_aspect = df_aspect.values

    df['sentiment_encoded'] = df['sentiment'].astype('category').cat.codes
    num_sentiments = df['sentiment'].nunique()
    y_sentiment = to_categorical(df['sentiment_encoded'].values, num_classes=num_sentiments)
    
    sentiment_categories = df['sentiment'].astype('category').cat.categories
    sentiment_labels = [str(cat) for cat in sentiment_categories]
    sentiment_mapping = {i: label for i, label in enumerate(sentiment_labels)}

    with open('sentiment_mapping.pickle', 'wb') as handle:
        pickle.dump(sentiment_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Sentiment mapping đã được lưu vào 'sentiment_mapping.pickle'")
    with open(os.path.join(results_dir, 'aspect_labels.pickle'), 'wb') as handle:
        pickle.dump(aspect_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Aspect labels đã được lưu vào '{os.path.join(results_dir, 'aspect_labels.pickle')}'")

    print("--- Bước 4: Chia và lưu dữ liệu Test ---")
    X_train, X_test, y_train_aspect, y_test_aspect, y_train_sentiment, y_test_sentiment, y_train_sentiment_le, y_test_sentiment_le = train_test_split(
        padded_sequences, y_aspect, y_sentiment, df['sentiment_encoded'].values, test_size=test_size, random_state=random_state
    )

    np.save(os.path.join(results_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(results_dir, 'y_test_aspect.npy'), y_test_aspect)
    np.save(os.path.join(results_dir, 'y_test_sentiment.npy'), y_test_sentiment)
    np.save(os.path.join(results_dir, 'y_test_sentiment_le.npy'), y_test_sentiment_le)
    print(f"Dữ liệu Test đã được lưu vào thư mục '{results_dir}/'")

    print("\n--- Bước 5: Huấn luyện và Đánh giá các mô hình ---")
    
    _train_and_evaluate_model(
        X_train, X_test, y_train_aspect, y_test_aspect,
        y_train_sentiment, y_test_sentiment, y_test_sentiment_le,
        vocab_size, num_aspects, num_sentiments,
        aspect_labels, sentiment_labels,
        model_type='cnn_multi'
    )

    _train_and_evaluate_model(
        X_train, X_test, y_train_aspect, y_test_aspect,
        y_train_sentiment, y_test_sentiment, y_test_sentiment_le,
        vocab_size, num_aspects, num_sentiments,
        aspect_labels, sentiment_labels,
        model_type='lstm_multi'
    )
    
    print("\n--- Pipeline huấn luyện đã hoàn tất ---")


if __name__ == '__main__':
    DATA_PATH = "E:/study/NLP/do_an_CK/AspectSent-for-Tiki/data/Gold/finetuning_metadata.csv"
    
    if not os.path.exists(DATA_PATH):
        alt_data_path = os.path.join(project_root, DATA_PATH)
        if os.path.exists(alt_data_path):
            DATA_PATH = alt_data_path
        else:
             print(f"Lỗi: Không tìm thấy file dữ liệu tại '{DATA_PATH}' hoặc '{alt_data_path}'")
    
    run_pipeline_paper(data_path=DATA_PATH)