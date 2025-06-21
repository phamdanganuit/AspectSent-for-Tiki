import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
import pickle
import os
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Thêm đường dẫn thư mục gốc của dự án vào Python Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def plot_sentiment_confusion_matrix(y_true, y_pred, class_labels, model_type):
    """Vẽ và lưu ma trận nhầm lẫn cho tác vụ phân loại cảm xúc."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix for Sentiment - {model_type.upper()}', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        filename = os.path.join('results', f'{model_type}_sentiment_confusion_matrix.png')
        plt.savefig(filename)
        plt.close()
        print(f"Biểu đồ Confusion Matrix cho Sentiment đã được lưu tại: {filename}")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi vẽ biểu đồ Sentiment: {e}")

def plot_aspect_confusion_matrices(y_true, y_pred, class_labels, model_type):
    """Vẽ và lưu các ma trận nhầm lẫn cho từng nhãn của tác vụ trích xuất khía cạnh."""
    try:
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        n_labels = len(class_labels)
        
        n_cols = 3
        n_rows = (n_labels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        fig.suptitle(f'Confusion Matrices for Aspects - {model_type.upper()}', fontsize=20)
        axes = axes.flatten()

        for i, (matrix, label) in enumerate(zip(mcm, class_labels)):
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['Not Present', 'Present'], yticklabels=['Not Present', 'Present'],
                        annot_kws={"size": 14})
            axes[i].set_title(f'Aspect: {label}', fontsize=14)
            axes[i].set_ylabel('True', fontsize=12)
            axes[i].set_xlabel('Predicted', fontsize=12)

        for i in range(n_labels, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = os.path.join('results', f'{model_type}_aspect_confusion_matrices.png')
        plt.savefig(filename)
        plt.close()
        print(f"Biểu đồ Confusion Matrix cho Aspect đã được lưu tại: {filename}")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi vẽ biểu đồ Aspect: {e}")


def evaluate_saved_model(model_type):
    """
    Hàm để tải, đánh giá và vẽ biểu đồ cho một loại mô hình cụ thể.
    """
    print(f"--- Bắt đầu đánh giá mô hình: {model_type.upper()} ---")
    
    results_dir = 'results'
    model_path = os.path.join(results_dir, f'{model_type}_model.h5')
    report_path = os.path.join(results_dir, f'{model_type}_evaluation_report.txt')

    required_files = [
        model_path, os.path.join(results_dir, 'X_test.npy'),
        os.path.join(results_dir, 'y_test_aspect.npy'), os.path.join(results_dir, 'y_test_sentiment_le.npy'),
        os.path.join(results_dir, 'aspect_labels.pickle'), 'sentiment_mapping.pickle'
    ]

    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"LỖI: Không tìm thấy file '{f_path}'.")
            print("Hãy chắc chắn rằng bạn đã chạy thành công 'src/pipeline_paper.py' trước.")
            return

    # Tải dữ liệu và các nhãn
    print("Đang tải dữ liệu kiểm thử và các file nhãn...")
    X_test = np.load(os.path.join(results_dir, 'X_test.npy'))
    y_test_aspect = np.load(os.path.join(results_dir, 'y_test_aspect.npy'))
    y_test_sentiment_le = np.load(os.path.join(results_dir, 'y_test_sentiment_le.npy'))
    with open(os.path.join(results_dir, 'aspect_labels.pickle'), 'rb') as f:
        aspect_labels = pickle.load(f)
    with open('sentiment_mapping.pickle', 'rb') as f:
        sentiment_mapping = pickle.load(f)
    sentiment_labels = [str(cat) for cat in sentiment_mapping.values()]

    # Tải mô hình
    print(f"Đang tải mô hình từ: {model_path}")
    model = load_model(model_path)
    model.summary()

    # Dự đoán
    print("Đang thực hiện dự đoán trên tập test...")
    predictions = model.predict(X_test)
    y_pred_aspect = (predictions[0] > 0.5).astype(int)
    y_pred_sentiment_le = np.argmax(predictions[1], axis=1)

    # Tạo và lưu báo cáo dạng text
    print("Đang tạo và lưu báo cáo dạng text...")
    aspect_report = classification_report(y_test_aspect, y_pred_aspect, target_names=aspect_labels, zero_division=0)
    sentiment_report = classification_report(y_test_sentiment_le, y_pred_sentiment_le, target_names=sentiment_labels, zero_division=0)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH: {model_type.upper()}\n")
        f.write("="*50 + "\n\n")
        f.write("Báo cáo Phân loại Khía cạnh (Aspect):\n")
        f.write(aspect_report)
        f.write("\n\n" + "="*50 + "\n\n")
        f.write("Báo cáo Phân loại Cảm xúc (Sentiment):\n")
        f.write(sentiment_report)
    print(f"Báo cáo đánh giá dạng text đã được lưu tại: {report_path}")

    # Vẽ và lưu các biểu đồ
    print("\n--- Đang vẽ các biểu đồ trực quan ---")
    plot_sentiment_confusion_matrix(y_test_sentiment_le, y_pred_sentiment_le, sentiment_labels, model_type)
    plot_aspect_confusion_matrices(y_test_aspect, y_pred_aspect, aspect_labels, model_type)


if __name__ == '__main__':
    # Bỏ phần đọc tham số từ command line
    # parser = argparse.ArgumentParser(...)
    # args = parser.parse_args()
    
    print("="*70)
    print("===== BẮT ĐẦU QUÁ TRÌNH ĐÁNH GIÁ CHO CẢ 2 MÔ HÌNH =====")
    print("="*70)
    
    # 1. Đánh giá mô hình CNN
    evaluate_saved_model('cnn_multi')
    
    print("\n\n" + "#"*70 + "\n")
    
    # 2. Đánh giá mô hình LSTM
    evaluate_saved_model('lstm_multi')
    
    print("\n" + "="*70)
    print("===== HOÀN TẤT ĐÁNH GIÁ CẢ 2 MÔ HÌNH =====")
    print("="*70)