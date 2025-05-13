import pandas as pd
import sys
import os
from pyvi import ViTokenizer

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.constants import NORMALIZED_REVIEWS_FILE, GOLD_DIR

# Output file sẽ ở thư mục Gold thay vì Silver
TOKENIZED_REVIEWS_FILE = os.path.join(GOLD_DIR, "tokenized_reviews.csv")

def load_normalized_data():
    """
    Đọc dữ liệu đã chuẩn hóa từ file CSV.
    
    Returns:
        pandas.DataFrame: DataFrame chứa dữ liệu đã chuẩn hóa.
    """
    try:
        df = pd.read_csv(NORMALIZED_REVIEWS_FILE)
        print(f"Đã đọc {len(df)} đánh giá đã chuẩn hóa từ {NORMALIZED_REVIEWS_FILE}")
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {NORMALIZED_REVIEWS_FILE}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Lỗi khi đọc file {NORMALIZED_REVIEWS_FILE}: {e}")
        return pd.DataFrame()

def tokenize_text(text):
    """
    Tách từ tiếng Việt sử dụng pyvi ViTokenizer.
    
    Args:
        text (str): Chuỗi văn bản cần tách từ.
        
    Returns:
        str: Chuỗi văn bản đã được tách từ.
    """
    if not isinstance(text, str) or pd.isna(text) or not text:
        return ""
    
    try:
        # ViTokenizer sẽ tách từ và nối các từ đơn lại bằng dấu gạch dưới
        tokenized_text = ViTokenizer.tokenize(text)
        
        # Đảm bảo kết quả trả về là chuỗi
        if not isinstance(tokenized_text, str):
            tokenized_text = str(tokenized_text)
            
        return tokenized_text
    except Exception as e:
        print(f"Lỗi khi tách từ: {e}")
        return text

def tokenize_data(df):
    """
    Tách từ cho dữ liệu đã chuẩn hóa.
    
    Args:
        df (pandas.DataFrame): DataFrame chứa dữ liệu đã chuẩn hóa.
        
    Returns:
        pandas.DataFrame: DataFrame với cột chứa dữ liệu đã tách từ.
    """
    if df.empty:
        print("DataFrame đầu vào rỗng.")
        return df
    
    print("Bắt đầu tách từ cho dữ liệu...")
    
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
    tokenized_df = df.copy()
    
    # Tách từ cho cột normalized_text (đã được chuẩn hóa)
    if 'normalized_text' in tokenized_df.columns:
        tokenized_df['tokenized_text'] = tokenized_df['normalized_text'].apply(tokenize_text)
        print(f"Đã tách từ cho {len(tokenized_df)} đánh giá.")
    else:
        print("Cảnh báo: Không tìm thấy cột 'normalized_text'.")
    
    return tokenized_df

def save_tokenized_data(df):
    """
    Lưu dữ liệu đã tách từ vào file CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame chứa dữ liệu đã tách từ.
    """
    if df.empty:
        print("DataFrame rỗng, không có dữ liệu để lưu.")
        return
    
    # Đảm bảo thư mục Gold tồn tại
    os.makedirs(GOLD_DIR, exist_ok=True)
    
    try:
        df.to_csv(TOKENIZED_REVIEWS_FILE, index=False)
        print(f"Đã lưu {len(df)} đánh giá đã tách từ vào {TOKENIZED_REVIEWS_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu đã tách từ: {e}")

def main():
    """Hàm chính để tách từ cho dữ liệu đánh giá."""
    df = load_normalized_data()
    if not df.empty:
        tokenized_df = tokenize_data(df)
        save_tokenized_data(tokenized_df)
        print(f"Đã hoàn thành tách từ và lưu vào thư mục Gold.")
    else:
        print("Không tải được dữ liệu đã chuẩn hóa.")

if __name__ == "__main__":
    main() 