import pandas as pd
import numpy as np
import re
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.constants import RAW_REVIEWS_FILE, CLEANED_REVIEWS_FILE

def load_raw_data():
    """
    Đọc dữ liệu đánh giá thô từ file CSV.
    
    Returns:
        pandas.DataFrame: DataFrame chứa dữ liệu đánh giá thô.
    """
    try:
        df = pd.read_csv(RAW_REVIEWS_FILE)
        print(f"Đã đọc {len(df)} đánh giá từ {RAW_REVIEWS_FILE}")
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {RAW_REVIEWS_FILE}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Lỗi khi đọc file {RAW_REVIEWS_FILE}: {e}")
        return pd.DataFrame()

def clean_data(df):
    """
    Làm sạch dữ liệu đánh giá.
    
    Args:
        df (pandas.DataFrame): DataFrame chứa dữ liệu đánh giá thô.
        
    Returns:
        pandas.DataFrame: DataFrame đã được làm sạch.
    """
    if df.empty:
        return df
    
    print("Bắt đầu làm sạch dữ liệu...")
    
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
    cleaned_df = df.copy()
    
    # Đổi tên các cột theo yêu cầu
    # 'title' -> 'sentiment', 'content' -> 'reviews', 'type' -> 'type_product'
    column_mapping = {
        'title': 'sentiment',
        'content': 'reviews', 
        'type': 'type_product'
    }
    
    # Kiểm tra xem các cột cần đổi tên có tồn tại không
    for old_col, new_col in column_mapping.items():
        if old_col in cleaned_df.columns:
            cleaned_df.rename(columns={old_col: new_col}, inplace=True)
    
    print(f"Đã đổi tên các cột: {', '.join(column_mapping.keys())} thành {', '.join(column_mapping.values())}")
    
    # Xử lý giá trị thiếu
    print(f"Số lượng giá trị thiếu trước khi xử lý: {cleaned_df.isnull().sum().sum()}")
    
    # Loại bỏ các hàng có cả tiêu đề và nội dung trống
    mask = (cleaned_df['sentiment'].isnull() | (cleaned_df['sentiment'] == "N/A")) & \
           (cleaned_df['reviews'].isnull() | (cleaned_df['reviews'] == "N/A"))
    cleaned_df = cleaned_df[~mask]
    
    # Điền giá trị N/A cho các ô còn thiếu
    cleaned_df.fillna("N/A", inplace=True)
    
    # Thay thế các giá trị trống
    cleaned_df.replace("", "N/A", inplace=True)
    
    print(f"Số lượng dòng sau khi xử lý giá trị thiếu: {len(cleaned_df)}")
    
    # Làm sạch cột reviews
    cleaned_df['reviews'] = cleaned_df['reviews'].apply(clean_text)
    
    # Loại bỏ các đánh giá quá ngắn hoặc không có ý nghĩa
    min_content_length = 5  # Độ dài tối thiểu cho nội dung đánh giá
    cleaned_df = cleaned_df[cleaned_df['reviews'].str.len() >= min_content_length]
    
    print(f"Số lượng dòng sau khi lọc đánh giá quá ngắn: {len(cleaned_df)}")
    
    # Loại bỏ các đánh giá trùng lặp
    cleaned_df.drop_duplicates(subset=['reviews'], keep='first', inplace=True)
    
    print(f"Số lượng dòng sau khi loại bỏ trùng lặp: {len(cleaned_df)}")
    
    return cleaned_df

def clean_text(text):
    """
    Làm sạch một chuỗi văn bản.
    
    Args:
        text (str): Chuỗi văn bản cần làm sạch.
        
    Returns:
        str: Chuỗi văn bản đã được làm sạch.
    """
    if not isinstance(text, str) or text == "N/A":
        return "N/A"
    
    # Loại bỏ các ký tự đặc biệt và khoảng trắng thừa
    text = re.sub(r'[^\w\s\.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Loại bỏ các ký tự xuống dòng và tab
    text = re.sub(r'[\r\n\t]', ' ', text)
    
    # Loại bỏ các từ khóa không cần thiết
    text = re.sub(r'\bxem thêm\b', '', text, flags=re.IGNORECASE)
    
    # Loại bỏ các URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Loại bỏ chuỗi HTML
    text = re.sub(r'<.*?>', '', text)
    
    # Loại bỏ khoảng trắng đầu cuối
    text = text.strip()
    
    # Nếu sau khi làm sạch, chuỗi rỗng thì trả về N/A
    if not text:
        return "N/A"
    
    return text

def save_cleaned_data(df):
    """
    Lưu dữ liệu đã làm sạch vào file CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame chứa dữ liệu đã làm sạch.
    """
    if df.empty:
        print("Không có dữ liệu để lưu.")
        return
    
    try:
        df.to_csv(CLEANED_REVIEWS_FILE, index=False)
        print(f"Đã lưu {len(df)} đánh giá đã làm sạch vào {CLEANED_REVIEWS_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu đã làm sạch: {e}")

def main():
    """Hàm chính để làm sạch dữ liệu đánh giá."""
    df = load_raw_data()
    if not df.empty:
        cleaned_df = clean_data(df)
        save_cleaned_data(cleaned_df)

if __name__ == "__main__":
    main() 